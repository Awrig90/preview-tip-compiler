import re
import html
import json
from datetime import date, datetime, time as dt_time, timedelta, timezone
from zoneinfo import ZoneInfo
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup


DEFAULT_TOMORROW_URL = "https://www.freesupertips.com/predictions/tomorrows-football-predictions/"
UK_TZ = ZoneInfo("Europe/London")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}

JUNK_REASONING_LINES = {
    "Reasoning",
    "Reason for tip",
    "PLACE BET",
    "Bookie",
    "Bet Tip",
    "Sign Up",
    "BET HERE",
    "Claim Free Bets",
}


@dataclass
class PreviewLink:
    url: str
    link_text: str
    listing_time: str = ""
    listing_match: str = ""


@dataclass
class TipRow:
    date: str
    time: str
    match: str
    market: str
    market_type: str
    market_tags: str
    selection: str
    odds_when_tipped: str
    odds_decimal_from_tip: Optional[float]
    current_decimal_from_returns: Optional[float]
    reasoning: str
    url: str
    status: str = "OK"


def clean_space(value: str) -> str:
    value = html.unescape(value or "")
    value = value.replace("\xa0", " ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n\s*\n+", "\n", value)
    return value.strip()


def normalise_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://www.freesupertips.com" + ("" if url.startswith("/") else "/") + url
    return url


@st.cache_data(ttl=900, show_spinner=False)
def fetch_html(url: str) -> str:
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=25)
    response.raise_for_status()
    return response.text


def is_preview_url(href: str) -> bool:
    if not href:
        return False
    path = urlparse(href).path.lower()
    if "/predictions/" not in path:
        return False
    # Most match-preview pages use this slug pattern. This keeps category/nav links out.
    if "predictions-betting-tips-match-previews" in path:
        return True
    return False


def extract_time_from_text(text: str) -> str:
    # Handles 23:00, 02:00 + 1, 00:00+1 etc.
    match = re.search(r"\b(\d{1,2}:\d{2})(?:\s*\+\s*(\d+)|\+(\d+))?\b", text or "")
    if not match:
        return ""
    base = match.group(1)
    plus = match.group(2) or match.group(3)
    return f"{base}+{plus}" if plus else base


def _normalise_iso_candidate(value: str) -> str:
    value = clean_space(value)
    # datetime.fromisoformat accepts +00:00 but not Z or +0000 consistently.
    value = value.replace("Z", "+00:00")
    value = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", value)
    return value


def parse_uk_time_from_datetime_text(text: str) -> str:
    """Find a timezone-aware datetime in HTML/text and return Europe/London time.

    FST often stores fixture kickoffs as UTC/GMT in the raw HTML, while the
    frontend renders the local UK time. Converting timezone-aware datetimes is
    safer than reading the raw HH:MM text directly during BST.
    """
    if not text:
        return ""

    # Match ISO/RFC3339-ish datetime values with an explicit timezone.
    pattern = re.compile(
        r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})\b"
    )
    for raw in pattern.findall(str(text)):
        candidate = _normalise_iso_candidate(raw.replace(" ", "T"))
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if dt.tzinfo is None:
            continue
        return dt.astimezone(UK_TZ).strftime("%H:%M")
    return ""


def extract_uk_time_from_tag(tag) -> str:
    """Extract UK-local kickoff time from a listing/fixture card tag.

    Prefer timezone-aware datetime attributes because raw HTML text can be UTC
    while the visible frontend card displays BST-adjusted UK time.
    """
    if not tag:
        return ""

    attr_priority = ("datetime", "data-time", "data-date", "data-start", "data-kickoff", "content")

    # First inspect datetime-ish attributes on the tag and descendants.
    for node in [tag] + list(tag.find_all(True)):
        attrs = getattr(node, "attrs", {}) or {}
        for attr in attr_priority:
            if attr in attrs:
                value = attrs.get(attr)
                if isinstance(value, (list, tuple)):
                    value = " ".join(str(x) for x in value)
                found = parse_uk_time_from_datetime_text(str(value))
                if found:
                    return found

    # Then inspect the compact HTML for embedded ISO datetime strings.
    found = parse_uk_time_from_datetime_text(str(tag))
    if found:
        return found

    # Final fallback: visible raw text such as 18:00 or 00:00+1.
    return extract_time_from_text(clean_space(tag.get_text(" ", strip=True)))


def guess_match_from_link_text(text: str) -> str:
    text = clean_space(text)
    if not text:
        return ""
    # Remove common timing/TV cruft from listing cards.
    text = re.sub(r"\b\d+h\s*\d+m\b", "", text, flags=re.I)
    text = re.sub(r"\b\d+\s*days?\b", "", text, flags=re.I)
    text = re.sub(r"\b\d{1,2}:\d{2}(?:\s*\+\s*\d+|\+\d+)?\b", "", text)
    text = re.sub(r"\b(BBC One|BBC Two|ITV|Sky Sports|TNT Sports|Premier Sports)\b", "", text, flags=re.I)
    text = clean_space(text)
    # Some cards omit "vs" visually, but the article itself will supply the proper match name.
    return text


def get_preview_href_count(tag) -> int:
    """Count match-preview links inside a listing fragment."""
    if not tag:
        return 0
    return sum(1 for link in tag.find_all("a", href=True) if is_preview_url(urljoin("https://www.freesupertips.com", link.get("href", ""))))


def get_listing_context_tag(a):
    """Return the smallest useful fixture-card tag around a preview link."""
    best = None

    for parent in a.parents:
        if getattr(parent, "name", None) in {"body", "html", "main"}:
            break
        text = clean_space(parent.get_text(" ", strip=True))
        if not text:
            continue

        has_time = bool(extract_time_from_text(text) or extract_uk_time_from_tag(parent))
        if not has_time:
            continue

        preview_count = get_preview_href_count(parent)
        # Prefer the smallest parent that looks like a single fixture/card.
        if preview_count <= 1 and len(text) <= 500:
            return parent

        if best is None and len(text) <= 700:
            best = parent

    return best


def get_listing_context_text(a) -> str:
    """Return the smallest useful listing-card text around a preview link."""
    context_tag = get_listing_context_tag(a)
    if context_tag is not None:
        return clean_space(context_tag.get_text(" ", strip=True))
    return clean_space(a.get_text(" ", strip=True))


def extract_nearest_time_for_anchor(a) -> str:
    """Extract the raw kickoff clock shown in the server HTML.

    FreeSuperTips currently sends fixture clocks in GMT/UTC in the HTML and
    converts them to the browser's UK-local time on the frontend.  We keep the
    raw clock here and perform the GMT/BST conversion only after the preview
    page has supplied the fixture date.
    """
    context_tag = get_listing_context_tag(a)
    if context_tag is not None:
        found = extract_time_from_text(clean_space(context_tag.get_text(" ", strip=True)))
        if found:
            return found

    found = extract_time_from_text(clean_space(a.get_text(" ", strip=True)))
    if found:
        return found

    parent = a.parent
    for _ in range(5):
        if not parent:
            break
        found = extract_time_from_text(clean_space(parent.get_text(" ", strip=True)))
        if found:
            return found
        parent = parent.parent

    return ""


def make_preview_link(listing_url: str, a) -> Optional[PreviewLink]:
    absolute = urljoin(listing_url, a.get("href", ""))
    if not is_preview_url(absolute):
        return None
    link_text = clean_space(a.get_text(" ", strip=True))
    context_text = get_listing_context_text(a)
    listing_time = extract_nearest_time_for_anchor(a)
    return PreviewLink(
        url=absolute,
        link_text=link_text,
        listing_time=listing_time,
        listing_match=guess_match_from_link_text(context_text or link_text),
    )


def is_plain_see_all_link(a) -> bool:
    text = clean_space(a.get_text(" ", strip=True)).lower()
    if text != "see all":
        return False
    href = (a.get("href") or "").lower()
    # The desired league-section marker links to a league page, not to a match preview.
    return "/predictions/" in href and "predictions-betting-tips-match-previews" not in href


def is_league_footer_link(tag) -> bool:
    """Return True for footer links like 'See All UEFA Champions League Predictions'.

    Those appear inside each league block. They are useful visual separators on the
    frontend, but they should not stop extraction because the tomorrow page can
    contain several league blocks one after another.
    """
    text = clean_space(tag.get_text(" ", strip=True)).lower()
    return bool(
        tag.name == "a"
        and text.startswith("see all")
        and "prediction" in text
        and not is_plain_see_all_link(tag)
    )


def is_end_of_main_preview_section(tag) -> bool:
    text = clean_space(tag.get_text(" ", strip=True)).lower()
    if not text:
        return False
    # These guardrails mark the end of the whole fixture listing area, not the end
    # of a single league card. Per-league footer links are skipped, not used as an
    # end marker, so pages with several competitions still return every shown match.
    if "select league" in text:
        return True
    if "tomorrow’s football predictions faqs" in text or "tomorrow's football predictions faqs" in text:
        return True
    if "football betting tips faqs" in text:
        return True
    if "free bets and offers" in text:
        return True
    return False


def extract_preview_links(listing_url: str, html_text: str) -> List[PreviewLink]:
    soup = BeautifulSoup(html_text, "html.parser")
    tags = soup.find_all(["a", "h1", "h2", "h3", "div", "section", "select"])

    # The page has a "Next Up" carousel near the top containing many future previews.
    # The actual tomorrow-listing block begins after a plain "See All" link under the
    # current league header and ends before "See All {League} Predictions" / filters.
    main_start_idx = None
    for idx, tag in enumerate(tags):
        if tag.name == "a" and is_plain_see_all_link(tag):
            main_start_idx = idx + 1
            break

    search_tags = tags[main_start_idx:] if main_start_idx is not None else tags
    seen = set()
    links: List[PreviewLink] = []

    for tag in search_tags:
        if is_end_of_main_preview_section(tag):
            break
        # Do not break on links like "See All UEFA Champions League Predictions".
        # They appear between league blocks, so breaking here would only collect the
        # first competition on the page.
        if is_league_footer_link(tag):
            continue
        if tag.name != "a" or not tag.get("href"):
            continue
        preview_link = make_preview_link(listing_url, tag)
        if preview_link is None:
            continue
        if preview_link.url in seen:
            continue
        seen.add(preview_link.url)
        links.append(preview_link)

    # Fallback: if the scoped scrape finds nothing, use the broad scrape so the app
    # still returns something rather than silently failing after a page layout change.
    if not links:
        for a in soup.find_all("a", href=True):
            preview_link = make_preview_link(listing_url, a)
            if preview_link is None or preview_link.url in seen:
                continue
            seen.add(preview_link.url)
            links.append(preview_link)

    return links


def parse_fractional_odds(odds: str) -> Optional[float]:
    text = clean_space(odds).lower()
    if not text:
        return None
    if text in {"evs", "evens"}:
        return 2.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if not m:
        # If it is already decimal, preserve it.
        m2 = re.search(r"\d+(?:\.\d+)?", text)
        if m2:
            try:
                return round(float(m2.group(0)), 2)
            except ValueError:
                return None
        return None
    num, den = float(m.group(1)), float(m.group(2))
    if den == 0:
        return None
    return round((num / den) + 1, 2)


def parse_money(value: str) -> Optional[float]:
    if not value:
        return None
    m = re.search(r"£\s*([\d,]+)(?:\s*\.\s*(\d{1,2}))?", value)
    if not m:
        return None
    whole = m.group(1).replace(",", "")
    decimals = m.group(2) or "00"
    try:
        return float(f"{whole}.{decimals}")
    except ValueError:
        return None


def get_selected_stake(block) -> Optional[float]:
    selected = block.select_one("select option[selected]")
    if selected and selected.get("value"):
        try:
            return float(selected["value"])
        except ValueError:
            pass
    # Fallback: infer from text like "£10 Returns".
    text = clean_space(block.get_text(" ", strip=True))
    m = re.search(r"£\s*(\d+(?:\.\d+)?)\s*Returns", text, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    if "50p Returns" in text:
        return 0.5
    return None


def get_return_decimal(block) -> Optional[float]:
    stake = get_selected_stake(block)
    if not stake:
        return None
    return_div = block.select_one(".BetGrid__returns")
    if not return_div:
        return None
    returns = parse_money(return_div.get_text("", strip=True))
    if returns is None:
        return None
    return round(returns / stake, 2)


def extract_reasoning(block) -> str:
    # The reasoning usually lives in a WYSIWYG div inside the hidden accordion.
    candidates = block.select(".Html-module__wysiwyg")
    if candidates:
        text = candidates[0].get_text("\n", strip=True)
    else:
        # Conservative fallback: take the text after "Reason for tip" and before odds/bookie table.
        text = block.get_text("\n", strip=True)
        if "Reason for tip" in text:
            text = text.split("Reason for tip", 1)[-1]
        text = re.split(r"\d+\s*/\s*\d+\s*odds when tipped|PLACE BET|Bookie", text, maxsplit=1)[0]

    lines = []
    for line in clean_space(text).splitlines():
        line = clean_space(line)
        if not line:
            continue
        if line in JUNK_REASONING_LINES:
            continue
        lines.append(line)
    return clean_space(" ".join(lines))


MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def resolve_fixture_date(date_label: str) -> Optional[date]:
    """Resolve article labels such as Today, Tomorrow or Thursday 23rd July."""
    label = clean_space(date_label).lower()
    today_uk = datetime.now(UK_TZ).date()

    if not label:
        return None
    if label == "today":
        return today_uk
    if label == "tomorrow":
        return today_uk + timedelta(days=1)

    # Examples: Thursday 23rd July, 23 July, Tue 21 Jul 2026.
    match = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"(?:\s+(\d{4}))?\b",
        label,
        flags=re.I,
    )
    if not match:
        return None

    day = int(match.group(1))
    month = MONTHS.get(match.group(2).lower())
    if not month:
        return None
    year = int(match.group(3)) if match.group(3) else today_uk.year

    try:
        candidate = date(year, month, day)
    except ValueError:
        return None

    # A yearless page around New Year may refer to the following calendar year.
    if not match.group(3) and candidate < today_uk - timedelta(days=180):
        try:
            candidate = date(year + 1, month, day)
        except ValueError:
            return None
    return candidate


def convert_gmt_clock_to_london(raw_clock: str, fixture_date: Optional[date]) -> str:
    """Convert a server-rendered GMT/UTC clock to Europe/London.

    The visible site adjusts these raw clocks for BST in the browser.  Parsing
    the fixture date lets Python make the same adjustment without Playwright.
    """
    raw_clock = clean_space(raw_clock)
    if not raw_clock or fixture_date is None:
        return raw_clock

    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?:\+(\d+))?", raw_clock)
    if not match:
        return raw_clock

    hour = int(match.group(1))
    minute = int(match.group(2))
    extra_days = int(match.group(3) or 0)
    if hour > 23 or minute > 59:
        return raw_clock

    utc_date = fixture_date + timedelta(days=extra_days)
    utc_dt = datetime.combine(utc_date, dt_time(hour, minute), tzinfo=timezone.utc)
    local_dt = utc_dt.astimezone(UK_TZ)
    return local_dt.strftime("%H:%M")


def extract_match_metadata(soup: BeautifulSoup) -> Dict[str, str]:
    h1 = soup.find("h1")
    raw_title = clean_space(h1.get_text(" ", strip=True)) if h1 else ""
    match_name = re.sub(r"\s+Predictions\s*$", "", raw_title, flags=re.I).strip()

    page_text = soup.get_text("\n", strip=True)
    lines = [clean_space(x) for x in page_text.splitlines() if clean_space(x)]

    published = ""
    kickoff_time = ""
    date_label = ""
    stadium = ""

    for line in lines:
        if line.lower().startswith("published on"):
            published = line
            break

    # Look near the title for time/date/stadium. This is intentionally loose because the page markup may change.
    start_idx = 0
    if raw_title in lines:
        start_idx = lines.index(raw_title)
    window = lines[start_idx : start_idx + 25]
    for idx, line in enumerate(window):
        if not kickoff_time and re.fullmatch(r"\d{1,2}:\d{2}(?:\s*\+\s*\d+|\+\d+)?", line):
            kickoff_time = line.replace(" ", "")
            # Usually the next line is Today/Tomorrow/date and the one after that is the stadium.
            if idx + 1 < len(window):
                date_label = window[idx + 1]
            if idx + 2 < len(window):
                stadium = window[idx + 2]
            break

    return {
        "match": match_name,
        "published": published,
        "time": kickoff_time,
        "date": date_label,
        "stadium": stadium,
    }


def classify_market(selection: str, index: int) -> str:
    """Role/order within the preview article, not the filterable betting market."""
    selection_l = selection.lower()
    if index == 1:
        return "Correct Score"
    if re.search(r"\b\d+\s*-\s*\d+\b", selection):
        return "Correct Score"
    if "score anytime" in selection_l or "to score anytime" in selection_l or "goalscorer" in selection_l:
        return "Goalscorer"
    if index == 0:
        return "Main Tip"
    return "Other Tip"


def split_match_teams(match_name: str) -> List[str]:
    """Return likely team names from a fixture string such as 'Ghana vs Panama'."""
    text = clean_space(match_name)
    if not text:
        return []
    text = re.sub(r"\s+Predictions$", "", text, flags=re.I)
    parts = re.split(r"\s+(?:vs?\.?|v)\s+", text, flags=re.I)
    if len(parts) >= 2:
        return [clean_space(x) for x in parts[:2] if clean_space(x)]
    return []


def strip_leading_team(selection: str, match_name: str) -> str:
    """Remove the fixture team name from the start of a tip so team-specific tips can share buckets."""
    selection_clean = clean_space(selection)
    selection_l = selection_clean.lower()
    for team in sorted(split_match_teams(match_name), key=len, reverse=True):
        team_clean = clean_space(team)
        team_l = team_clean.lower()
        if selection_l == team_l:
            return ""
        for prefix in (team_l + " ", team_l + "-", team_l + " –", team_l + " —"):
            if selection_l.startswith(prefix):
                return clean_space(selection_clean[len(team_clean):].lstrip())
    return selection_clean


def classify_market_type(selection: str, match_name: str, index: int) -> str:
    """Filterable market bucket. This deliberately strips team names where appropriate."""
    selection_clean = clean_space(selection)
    selection_l = selection_clean.lower()
    stripped = clean_space(strip_leading_team(selection_clean, match_name))
    stripped_l = stripped.lower().strip()

    # Exact scorelines, e.g. England 2-0 or 2-1.
    if re.search(r"\b\d+\s*-\s*\d+\b", selection_clean):
        return "Correct Score"

    has_fixture_team_prefix = bool(split_match_teams(match_name)) and stripped_l != selection_l

    # Team-neutral markets. Only treat totals as pure totals if the selection is not
    # prefixed by a fixture team, otherwise combined team-win markets would be hidden
    # from the Team To Win filters.
    if selection_l in {"both teams to score", "btts", "yes - both teams to score"}:
        return "Both Teams To Score"
    if "both teams to score" in selection_l and not split_match_teams(match_name):
        return "Both Teams To Score"
    if re.search(r"\bover\s+2\.5\b", selection_l) and not has_fixture_team_prefix:
        return "Over 2.5 Goals"
    if re.search(r"\bunder\s+2\.5\b", selection_l) and not has_fixture_team_prefix:
        return "Under 2.5 Goals"
    if selection_l == "draw" or stripped_l == "draw":
        return "Draw"

    # Team-specific markets normalised into shared buckets.
    if "both teams to score" in stripped_l and ("win" in stripped_l or stripped_l.startswith("and ")):
        return "Team To Win & BTTS"
    if re.search(r"\bover\s+2\.5\b", stripped_l) and ("win" in stripped_l or stripped_l.startswith("and ")):
        return "Team To Win & Over 2.5 Goals"
    if "win to nil" in stripped_l or "to win to nil" in stripped_l:
        return "Team To Win To Nil"
    if stripped_l in {"to win", "win", "winner"}:
        return "Team To Win"
    if re.fullmatch(r"[+-]\d+(?:\.\d+)?", stripped_l):
        return "Handicap"
    if re.search(r"\b[+-]\d+(?:\.\d+)?\b", stripped_l) and len(stripped_l) <= 8:
        return "Handicap"

    # Player/scorer markets.
    if "to score anytime" in selection_l or "score anytime" in selection_l:
        return "Anytime Goalscorer"
    if "to score or assist" in selection_l or "score or assist" in selection_l:
        return "Player To Score Or Assist"
    if "to be shown a card" in selection_l or "to be carded" in selection_l:
        return "Player To Be Carded"
    if "shots" in selection_l:
        return "Player Shots"
    if "fouls" in selection_l:
        return "Player Fouls"

    # Fall back to the article role if that is more helpful than 'Other'.
    if index == 0:
        return "Main Tip - Other"
    if index == 1:
        return "Correct Score"
    return "Other"


def classify_market_tags(selection: str, match_name: str, index: int) -> List[str]:
    """Return all filter buckets a tip should appear under.

    Example: "Portugal and Both Teams To Score" should be visible under
    Team To Win & BTTS, Team To Win and Both Teams To Score.
    """
    primary = classify_market_type(selection, match_name, index)
    selection_clean = clean_space(selection)
    selection_l = selection_clean.lower()
    stripped = clean_space(strip_leading_team(selection_clean, match_name))
    stripped_l = stripped.lower().strip()

    tags: List[str] = []

    def add(tag: str):
        if tag and tag not in tags:
            tags.append(tag)

    add(primary)

    has_team = bool(split_match_teams(match_name)) and stripped_l != selection_l
    has_btts = "both teams to score" in selection_l or "btts" in selection_l
    has_over_25 = bool(re.search(r"\bover\s+2\.5\b", selection_l))
    has_under_25 = bool(re.search(r"\bunder\s+2\.5\b", selection_l))
    has_win_to_nil = "win to nil" in stripped_l or "to win to nil" in stripped_l
    looks_like_team_win = (
        has_team
        and not has_win_to_nil
        and (
            stripped_l in {"", "to win", "win", "winner"}
            or stripped_l.startswith("and ")
            or " to win" in stripped_l
            or " win " in f" {stripped_l} "
        )
    )

    if looks_like_team_win or has_win_to_nil:
        add("Team To Win")
    if has_btts:
        add("Both Teams To Score")
    if has_over_25:
        add("Over 2.5 Goals")
    if has_under_25:
        add("Under 2.5 Goals")

    if (looks_like_team_win or has_win_to_nil) and has_btts:
        add("Team To Win & BTTS")
    if (looks_like_team_win or has_win_to_nil) and has_over_25:
        add("Team To Win & Over 2.5 Goals")

    return tags


def market_tags_to_text(tags: List[str]) -> str:
    return "; ".join(tags)


def get_export_odds(row: pd.Series, odds_source: str) -> str:
    odds_col = "odds_when_tipped" if odds_source == "Odds when tipped" else "current_decimal_from_returns"
    odds = row.get(odds_col, "")
    if pd.isna(odds):
        return ""
    return clean_space(str(odds))


def make_cms_line(row: pd.Series, odds_source: str) -> str:
    match = clean_space(str(row.get("match", "")))
    time = clean_space(str(row.get("time", "")))
    selection = clean_space(str(row.get("selection", "")))
    odds = get_export_odds(row, odds_source)
    reasoning = clean_space(str(row.get("reasoning", "")))
    left = clean_space(f"{match} {time}")
    odds_text = f" at {odds}" if odds else ""
    heading = clean_space(f"{left} - {selection}{odds_text}")
    return clean_space(f"{heading}\n{reasoning}")


def extract_tips_from_preview(url: str, html_text: str, listing: Optional[PreviewLink] = None) -> List[TipRow]:
    soup = BeautifulSoup(html_text, "html.parser")
    meta = extract_match_metadata(soup)

    # Use the kickoff clock from each individual preview page, not from the
    # listing card. The listing scraper can accidentally pick up the first card
    # in a competition block and reuse it across the whole league section. The
    # article's GameBullets block is the reliable per-fixture source:
    #   <ul class="GameBullets"><li>16:00</li><li>Tomorrow</li>...</ul>
    # The raw clock is GMT/UTC, so convert it to Europe/London using the
    # article's own fixture date label. This handles BST/GMT automatically.
    raw_kickoff = meta.get("time", "")
    fixture_date = resolve_fixture_date(meta.get("date", ""))
    meta["time"] = convert_gmt_clock_to_london(raw_kickoff, fixture_date)

    if listing and not meta["match"] and listing.listing_match:
        meta["match"] = listing.listing_match

    blocks = soup.select(".IndividualTipPrediction")
    rows: List[TipRow] = []

    for idx, block in enumerate(blocks):
        h4 = block.find("h4")
        if not h4:
            continue
        selection = clean_space(h4.get_text(" ", strip=True))
        if not selection:
            continue

        odds_el = block.select_one(".BetExpand__odds span")
        odds_when_tipped = clean_space(odds_el.get_text(" ", strip=True)) if odds_el else ""
        odds_decimal = parse_fractional_odds(odds_when_tipped)
        current_decimal = get_return_decimal(block)
        reasoning = extract_reasoning(block)

        rows.append(
            TipRow(
                date=meta.get("date", ""),
                time=meta.get("time", ""),
                match=meta.get("match", ""),
                market=classify_market(selection, idx),
                market_type=classify_market_type(selection, meta.get("match", ""), idx),
                market_tags=market_tags_to_text(classify_market_tags(selection, meta.get("match", ""), idx)),
                selection=selection,
                odds_when_tipped=odds_when_tipped,
                odds_decimal_from_tip=odds_decimal,
                current_decimal_from_returns=current_decimal,
                reasoning=reasoning,
                url=url,
                status="OK" if reasoning else "Missing reasoning",
            )
        )

    if not rows:
        rows.append(
            TipRow(
                date=meta.get("date", ""),
                time=meta.get("time", ""),
                match=meta.get("match", ""),
                market="",
                market_type="",
                market_tags="",
                selection="",
                odds_when_tipped="",
                odds_decimal_from_tip=None,
                current_decimal_from_returns=None,
                reasoning="",
                url=url,
                status="No tip blocks found",
            )
        )

    return rows


def rows_to_dataframe(rows: List[TipRow]) -> pd.DataFrame:
    df = pd.DataFrame([asdict(row) for row in rows])
    if df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "time",
                "match",
                "market",
                "market_type",
                "market_tags",
                "selection",
                "odds_when_tipped",
                "odds_decimal_from_tip",
                "current_decimal_from_returns",
                "reasoning",
                "url",
                "status",
            ]
        )
    return df


def make_tsv(df: pd.DataFrame, odds_source: str) -> str:
    if df.empty:
        return ""
    odds_col = "odds_when_tipped" if odds_source == "Odds when tipped" else "current_decimal_from_returns"
    export = df.copy()
    export["odds"] = export[odds_col].fillna("")
    cols = ["time", "match", "market_type", "selection", "odds", "reasoning", "url"]
    export = export[cols].fillna("")
    export = export.rename(columns={"market_type": "market"})
    return export.to_csv(sep="\t", index=False, header=True)


def make_cms_text(df: pd.DataFrame, odds_source: str) -> str:
    if df.empty:
        return ""
    chunks = [make_cms_line(row, odds_source) for _, row in df.iterrows()]
    return "\n\n".join([x for x in chunks if x])


def make_readable_shortlist_html(df: pd.DataFrame, odds_source: str) -> str:
    if df.empty:
        body = '<p class="muted">No tips match the current filters.</p>'
    else:
        cards = []
        plain_chunks = []
        for _, row in df.iterrows():
            odds = get_export_odds(row, odds_source)
            odds_text = f" at {odds}" if odds else ""
            heading = clean_space(f"{row.get('match', '')} {row.get('time', '')} - {row.get('selection', '')}{odds_text}")
            reasoning = clean_space(str(row.get("reasoning", "")))
            meta = clean_space(f"{row.get('market_type', '')} · {row.get('market', '')}")
            url = clean_space(str(row.get("url", "")))
            plain_chunks.append(f"{heading}\n{reasoning}")
            link = f'<a href="{html.escape(url)}" target="_blank" rel="noopener">Open preview</a>' if url else ""
            reasoning_id = f"reasoning-{len(cards)}"
            reasoning_json = html.escape(json.dumps(reasoning), quote=True)
            cards.append(
                '<article class="tip-card">'
                f'<h3>{html.escape(heading)}</h3>'
                f'<div class="meta">{html.escape(meta)}</div>'
                '<div class="card-actions">'
                f'<button class="copy-reasoning" onclick="copyText({reasoning_json}, this)">Copy reasoning</button>'
                f'{link}'
                '</div>'
                f'<p id="{reasoning_id}">{html.escape(reasoning)}</p>'
                '</article>'
            )
        plain = html.escape("\n\n".join(plain_chunks))
        body = f'<div id="copyText">{"".join(cards)}</div><textarea id="plainText" aria-label="plain text for copying">{plain}</textarea>'

    return f'''
    <html>
    <head>
      <style>
        body {{ font-family: sans-serif; margin: 0; padding: 0 2px 12px 2px; color: #111827; }}
        .toolbar {{ position: sticky; top: 0; background: white; padding: 0 0 10px 0; z-index: 2; }}
        button {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 8px 12px; background: #f9fafb; cursor: pointer; }}
        button:hover {{ background: #f3f4f6; }}
        .tip-card {{ border-bottom: 1px solid #e5e7eb; padding: 14px 0; user-select: text; }}
        .tip-card h3 {{ font-size: 16px; line-height: 1.35; margin: 0 0 4px 0; font-weight: 700; }}
        .tip-card .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 8px; }}
        .tip-card p {{ font-size: 15px; line-height: 1.5; margin: 8px 0 0 0; white-space: pre-wrap; }}
        .tip-card a {{ color: #2563eb; font-size: 13px; }}
        .card-actions {{ display: flex; align-items: center; gap: 10px; margin: 7px 0 0 0; }}
        .copy-reasoning {{ font-size: 13px; padding: 5px 9px; border-radius: 7px; }}
        .muted {{ color: #6b7280; }}
        #plainText {{ position: absolute; left: -9999px; top: -9999px; width: 1px; height: 1px; }}
        @media (prefers-color-scheme: dark) {{
          body {{ background: #0e1117; color: #f9fafb; }}
          .toolbar {{ background: #0e1117; }}
          button {{ background: #1f2937; color: #f9fafb; border-color: #374151; }}
          button:hover {{ background: #374151; }}
          .tip-card {{ border-bottom-color: #374151; }}
          .tip-card .meta {{ color: #9ca3af; }}
          .tip-card a {{ color: #93c5fd; }}
        }}
      </style>
    </head>
    <body>
      <div class="toolbar"><button onclick="copyAll()">Copy all visible tips</button> <span id="copyStatus" class="muted"></span></div>
      {body}
      <script>
        document.addEventListener('keydown', function(e) {{ e.stopPropagation(); }}, true);
        document.addEventListener('copy', function(e) {{ e.stopPropagation(); }}, true);
        async function writeToClipboard(text) {{
          try {{
            await navigator.clipboard.writeText(text);
            return true;
          }} catch (err) {{
            const helper = document.createElement('textarea');
            helper.value = text;
            helper.setAttribute('readonly', '');
            helper.style.position = 'fixed';
            helper.style.left = '-9999px';
            helper.style.top = '-9999px';
            document.body.appendChild(helper);
            helper.select();
            const ok = document.execCommand('copy');
            document.body.removeChild(helper);
            return ok;
          }}
        }}
        async function copyText(text, button) {{
          const original = button.innerText;
          const ok = await writeToClipboard(text);
          button.innerText = ok ? 'Copied' : 'Copy failed';
          setTimeout(() => {{ button.innerText = original; }}, 1200);
        }}
        async function copyAll() {{
          const textEl = document.getElementById('plainText');
          const status = document.getElementById('copyStatus');
          if (!textEl) return;
          const ok = await writeToClipboard(textEl.value);
          if (ok) {{
            status.innerText = 'Copied';
          }} else {{
            textEl.style.position = 'fixed';
            textEl.style.left = '8px';
            textEl.style.top = '8px';
            textEl.style.width = '95%';
            textEl.style.height = '220px';
            textEl.select();
            status.innerText = 'Clipboard blocked. Press Ctrl+C now.';
          }}
        }}
      </script>
    </body>
    </html>
    '''


def render_readable_shortlist(df: pd.DataFrame, odds_source: str):
    height = min(900, max(260, 155 * max(1, len(df)) + 70))
    components.html(make_readable_shortlist_html(df, odds_source), height=height, scrolling=True)


def build_filtered_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    filtered = df.copy()
    tag_values = []
    for raw_tags in filtered.get("market_tags", pd.Series(dtype=str)).fillna(""):
        for tag in str(raw_tags).split(";"):
            tag = clean_space(tag)
            if tag:
                tag_values.append(tag)
    market_options = sorted(set(tag_values))
    markets = st.sidebar.multiselect("Market type", market_options, default=market_options)
    role_options = sorted([x for x in filtered["market"].dropna().unique().tolist() if x])
    roles = st.sidebar.multiselect("Article role", role_options, default=role_options)
    search = st.sidebar.text_input("Search fixture/selection/reasoning", "")
    status_options = sorted([x for x in filtered["status"].dropna().unique().tolist() if x])
    statuses = st.sidebar.multiselect("Status", status_options, default=status_options)

    if markets:
        market_set = set(markets)
        filtered = filtered[
            filtered["market_tags"].fillna("").apply(
                lambda raw: bool(market_set.intersection({clean_space(x) for x in str(raw).split(";") if clean_space(x)}))
            )
        ]
    if roles:
        filtered = filtered[filtered["market"].isin(roles)]
    if statuses:
        filtered = filtered[filtered["status"].isin(statuses)]
    if search.strip():
        needle = search.strip().lower()
        haystack = (
            filtered["match"].fillna("")
            + " "
            + filtered["selection"].fillna("")
            + " "
            + filtered["reasoning"].fillna("")
        ).str.lower()
        filtered = filtered[haystack.str.contains(re.escape(needle), na=False)]
    return filtered


def main():
    st.set_page_config(page_title="Preview Tip Compiler", layout="wide")
    st.title("Preview Tip Compiler")
    st.caption("Pull preview tips into readable, market-filterable shortlists with CMS-ready copy blocks.")

    with st.sidebar:
        st.header("Input")
        mode = st.radio("Source", ["Tomorrow page URL", "Specific preview URLs"], index=0)
        max_previews = st.number_input("Max previews to fetch", min_value=1, max_value=100, value=30, step=1)
        odds_source = st.radio("Export odds", ["Odds when tipped", "Current decimal from returns"], index=0)

    links: List[PreviewLink] = []
    errors: List[str] = []

    if mode == "Tomorrow page URL":
        listing_url = st.text_input("Tomorrow predictions URL", value=DEFAULT_TOMORROW_URL)
        run = st.button("Fetch previews", type="primary")
        if run:
            try:
                listing_html = fetch_html(normalise_url(listing_url))
                links = extract_preview_links(normalise_url(listing_url), listing_html)[: int(max_previews)]
                st.session_state["links"] = [asdict(x) for x in links]
            except Exception as exc:
                st.error(f"Could not fetch listing page: {exc}")
                st.stop()
        elif "links" in st.session_state:
            links = [PreviewLink(**item) for item in st.session_state["links"]]

    else:
        urls_raw = st.text_area(
            "Preview URLs, one per line",
            value="https://www.freesupertips.com/predictions/ghana-vs-panama-predictions-betting-tips-match-previews/",
            height=130,
        )
        run = st.button("Fetch previews", type="primary")
        if run:
            links = [PreviewLink(url=normalise_url(x), link_text="") for x in urls_raw.splitlines() if x.strip()]
            links = links[: int(max_previews)]
            st.session_state["links"] = [asdict(x) for x in links]
        elif "links" in st.session_state:
            links = [PreviewLink(**item) for item in st.session_state["links"]]

    if not links:
        st.info("Enter a URL and click **Fetch previews**.")
        st.stop()

    with st.expander(f"Preview links found ({len(links)})", expanded=False):
        st.dataframe(pd.DataFrame([asdict(x) for x in links]), use_container_width=True)

    rows: List[TipRow] = []
    progress = st.progress(0, text="Fetching preview pages...")
    for i, link in enumerate(links, start=1):
        try:
            preview_html = fetch_html(link.url)
            rows.extend(extract_tips_from_preview(link.url, preview_html, listing=link))
        except Exception as exc:
            errors.append(f"{link.url}: {exc}")
            rows.append(
                TipRow(
                    date="",
                    time=link.listing_time,
                    match=link.listing_match,
                    market="",
                    market_type="",
                    market_tags="",
                    selection="",
                    odds_when_tipped="",
                    odds_decimal_from_tip=None,
                    current_decimal_from_returns=None,
                    reasoning="",
                    url=link.url,
                    status=f"Fetch error: {exc}",
                )
            )
        progress.progress(i / len(links), text=f"Fetched {i}/{len(links)} previews")
    progress.empty()

    df = rows_to_dataframe(rows)
    st.session_state["compiled_df"] = df

    ok_count = int((df["status"] == "OK").sum()) if not df.empty else 0
    st.success(f"Extracted {ok_count} usable tips from {len(links)} preview pages.")
    if errors:
        with st.expander("Fetch errors", expanded=True):
            for err in errors:
                st.warning(err)

    filtered = build_filtered_df(df)

    st.subheader("Readable shortlist")
    st.caption("Filtered tips are shown in the same basic shape as your notes/CMS copy: fixture, time, selection, odds and reasoning.")
    with st.expander("Show readable filtered tips", expanded=True):
        render_readable_shortlist(filtered, odds_source)

    st.subheader("Select tips")
    display_cols = [
        "time",
        "match",
        "market_type",
        "market_tags",
        "selection",
        "odds_when_tipped",
        "reasoning",
        "url",
        "status",
        "market",
        "date",
        "odds_decimal_from_tip",
        "current_decimal_from_returns",
    ]
    filtered = filtered[display_cols]
    editable = filtered.copy()
    editable.insert(0, "select", True)
    edited = st.data_editor(
        editable,
        use_container_width=True,
        hide_index=True,
        column_config={
            "select": st.column_config.CheckboxColumn("Select"),
            "time": st.column_config.TextColumn("Time", width="small"),
            "match": st.column_config.TextColumn("Fixture", width="medium"),
            "market_type": st.column_config.TextColumn("Primary market", width="medium"),
            "market_tags": st.column_config.TextColumn("Filter buckets", width="medium"),
            "selection": st.column_config.TextColumn("Selection", width="medium"),
            "odds_when_tipped": st.column_config.TextColumn("Fractional odds", width="small"),
            "reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            "url": st.column_config.LinkColumn("URL", width="medium"),
            "status": st.column_config.TextColumn("Status", width="small"),
            "market": st.column_config.TextColumn("Article role", width="small"),
        },
    )

    selected = edited[edited["select"]].drop(columns=["select"])

    st.subheader("Copy/export selected tips")
    tsv = make_tsv(selected, odds_source)
    cms_text = make_cms_text(selected, odds_source)

    st.markdown("**CMS / notepad style**")
    st.text_area("Copy this into notes/CMS", value=cms_text, height=320)

    with st.expander("Google Sheets / TSV and CSV export", expanded=False):
        st.text_area("Copy this into Sheets", value=tsv, height=220)
        st.download_button(
            "Download selected CSV",
            data=selected.to_csv(index=False).encode("utf-8"),
            file_name="preview_tips.csv",
            mime="text/csv",
        )

    with st.expander("Debug notes"):
        st.markdown(
            """
- This version expects preview pages to contain `.IndividualTipPrediction` blocks.
- Article role is order-based, but Market type is normalised for filtering. Combined tips also carry multiple filter buckets. For example, "Portugal and Both Teams To Score" appears under "Team To Win & BTTS", "Team To Win" and "Both Teams To Score".
- The tomorrow/upcoming-page scraper starts after the first main section "See All" marker, skips per-league footer links like "See All UEFA Champions League Predictions", and stops at the league filter/FAQ/footer area. Kickoff time is taken from each individual preview page's `GameBullets` block, then converted from GMT/UTC to Europe/London using the fixture date, so BST/GMT changes are handled automatically.
- `Odds when tipped` comes from the visible odds label, e.g. `15/4 odds when tipped`.
- `Current decimal from returns` is calculated from the selected return table stake when available.
"""
        )


if __name__ == "__main__":
    main()
