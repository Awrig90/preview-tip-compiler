# Preview Tip Compiler

Streamlit tool for compiling Free Super Tips match-preview tips into a readable, filterable shortlist.

## What it does

- Fetches links from the main block of the tomorrow predictions page.
- Opens each match preview.
- Extracts each `.IndividualTipPrediction` block.
- Pulls the selection, reasoning, fractional odds, current return-derived decimal odds, fixture and kick-off time.
- Adds a normalised `Market type` column so team-specific selections can be filtered together, e.g.:
  - `Portugal and Both Teams To Score` -> `Team To Win & BTTS`
  - `England to Win` -> `Team To Win`
  - `Portugal to Win To Nil` -> `Team To Win To Nil`
- Displays readable CMS/notepad-style copy blocks.
- Exports selected tips as CMS text, TSV or CSV.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deploy to Streamlit Community Cloud with `app.py` as the main file.
