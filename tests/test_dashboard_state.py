"""Offline regression checks: python -m unittest discover -s tests -v"""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import bs4
import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APPS = ('app.py', 'preview_tip_compiler_app.py')
BASE = 'https://www.freesupertips.com/predictions/'
URLS = [BASE + name + '-predictions-betting-tips-match-previews/' for name in ('iq', 'human', 'unknown')]
LISTING = BASE + 'tomorrows-football-predictions/'


def page(url, author, clock='15:00'):
    data = {'props': {'pageProps': {'responses': {'predictionsSingle': [
        {'url': url, 'startString': '2026-04-11 ' + clock + ':00'}
    ]}}}}
    return f'''<script id="__NEXT_DATA__" type="application/json">{json.dumps(data)}</script>
    <h1>{url.split('/')[-2]} Predictions</h1>
    <ul class="GameBullets"><li>14:00</li><li>Expired</li><li>Stadium</li></ul>
    <div class="AuthSocial"><span class="Author__name">{author}</span></div>
    <div class="IndividualTipPrediction"><h4>Both Teams To Score</h4>
    <div class="BetExpand__odds"><span>4/5</span></div>
    <div class="Html-module__wysiwyg">Both teams have scored regularly.</div></div>'''


def widget(items, label):
    return next(item for item in items if item.label == label)


class DashboardTests(unittest.TestCase):
    def test_fetch_refresh_and_filter_reruns(self):
        for filename in APPS:
            with self.subTest(app=filename):
                st.cache_data.clear()
                pages = dict(zip(URLS, [page(URLS[0], 'SpotlightiQ'), page(URLS[1], 'Alex'), page(URLS[2], '')]))
                pages[LISTING] = '<a href="/predictions/league/">See All</a>' + ''.join(
                    f'<a href="{url}">Match Predictions</a>' for url in URLS)

                def get(url, **kwargs):
                    value = pages[url]
                    if isinstance(value, Exception):
                        raise value
                    return Mock(text=value, raise_for_status=Mock())

                parse = Mock()
                original_init = bs4.BeautifulSoup.__init__

                def counted_init(soup, *args, **kwargs):
                    parse()
                    original_init(soup, *args, **kwargs)

                with patch('requests.get', side_effect=get) as requests, patch.object(bs4.BeautifulSoup, '__init__', counted_init):
                    at = AppTest.from_file(str(ROOT / filename)).run()
                    self.assertFalse(at.exception)
                    self.assertEqual(requests.call_count, 0)
                    at.button(key='fetch_previews').click().run()
                    self.assertFalse(at.exception)
                    self.assertEqual(requests.call_count, 4)
                    self.assertEqual(len(at.session_state['compiled_snapshot']['df']), 3)
                    before_parse = parse.call_count
                    # Remove HTML cache entirely: filters still must not fetch OR parse.
                    st.cache_data.clear()
                    widget(at.checkbox, 'Exclude SpotlightIQ').check().run()
                    self.assertFalse(at.exception)
                    self.assertEqual(len(at.dataframe[-1].value), 2)
                    self.assertNotIn('SpotlightiQ', at.dataframe[-1].value['author'].tolist())
                    widget(at.text_input, 'Search fixture/selection/reasoning').set_value('human').run()
                    self.assertEqual(len(at.dataframe[-1].value), 1)
                    widget(at.multiselect, 'Article role').set_value(['Main tip']).run()
                    widget(at.multiselect, 'Status').set_value(['OK']).run()
                    widget(at.multiselect, 'Market type').set_value(['Both Teams To Score']).run()
                    widget(at.radio, 'Export odds').set_value('Current decimal from returns').run()
                    self.assertFalse(at.exception)
                    self.assertEqual(requests.call_count, 4)
                    self.assertEqual(parse.call_count, before_parse)
                    widget(at.text_input, 'Search fixture/selection/reasoning').set_value('no match').run()
                    self.assertEqual(len(at.dataframe[-1].value), 0)
                    widget(at.text_input, 'Search fixture/selection/reasoning').set_value('').run()
                    widget(at.checkbox, 'Exclude SpotlightIQ').uncheck().run()
                    self.assertEqual(len(at.dataframe[-1].value), 3)
                    self.assertEqual(requests.call_count, 4)
                    # Refresh really goes to the source, not the 15-minute cache.
                    pages[URLS[1]] = page(URLS[1], 'Alex', '16:00')
                    at.button(key='fetch_previews').click().run()
                    self.assertFalse(at.exception)
                    self.assertEqual(requests.call_count, 8)
                    self.assertEqual(at.session_state['compiled_snapshot']['df']['time'].tolist(), ['15:00', '16:00', '15:00'])
                    at.button(key='fetch_previews').click().run()
                    self.assertEqual(requests.call_count, 12)
                    # Input changes are pending until an explicit refresh.
                    widget(at.radio, 'Source').set_value('Specific preview URLs').run()
                    widget(at.text_area, 'Preview URLs, one per line').set_value(URLS[1]).run()
                    widget(at.number_input, 'Max previews to fetch').set_value(1).run()
                    self.assertEqual(requests.call_count, 12)
                    self.assertEqual(len(at.session_state['compiled_snapshot']['df']), 3)
                    self.assertTrue(any('Inputs have changed' in x.value for x in at.info))
                    at.button(key='fetch_previews').click().run()
                    self.assertEqual(requests.call_count, 13)
                    self.assertEqual(len(at.session_state['compiled_snapshot']['df']), 1)
                    # Preview failure stays visible and is not retried on filtering.
                    pages[URLS[1]] = RuntimeError('offline')
                    at.button(key='fetch_previews').click().run()
                    self.assertFalse(at.exception)
                    self.assertEqual(requests.call_count, 14)
                    self.assertEqual(at.session_state['compiled_snapshot']['df']['time'].tolist(), [''])
                    self.assertTrue(at.session_state['compiled_snapshot']['errors'])
                    widget(at.checkbox, 'Exclude SpotlightIQ').check().run()
                    self.assertEqual(requests.call_count, 14)
                    self.assertTrue(any('offline' in x.value for x in at.warning))
                    # Failed or empty listing refresh preserves the previous snapshot.
                    widget(at.radio, 'Source').set_value('Tomorrow page URL').run()
                    previous = at.session_state['compiled_snapshot']
                    pages[LISTING] = RuntimeError('listing offline')
                    at.button(key='fetch_previews').click().run()
                    self.assertFalse(at.exception)
                    pd.testing.assert_frame_equal(at.session_state['compiled_snapshot']['df'], previous['df'])
                    self.assertTrue(at.error)
                    pages[LISTING] = '<html></html>'
                    at.button(key='fetch_previews').click().run()
                    pd.testing.assert_frame_equal(at.session_state['compiled_snapshot']['df'], previous['df'])
                    self.assertTrue(any('No preview links' in x.value for x in at.warning))

    def test_exact_author_exclusion_and_missing_authors(self):
        for filename in APPS:
            with self.subTest(app=filename):
                name = 'test_' + filename[:-3]
                spec = importlib.util.spec_from_file_location(name, ROOT / filename)
                app = importlib.util.module_from_spec(spec)
                sys.modules[name] = app
                spec.loader.exec_module(app)
                authors = ['SpotlightiQ', ' SPOTLIGHTIQ ', 'Alex', None, '', 'SpotlightIQ guest']
                df = pd.DataFrame({'author': authors, 'market_tags': ['BTTS'] * 6, 'market': ['Main tip'] * 6,
                                   'status': ['OK'] * 6, 'match': ['Match'] * 6, 'selection': ['BTTS'] * 6, 'reasoning': ['Reason'] * 6})
                sidebar = Mock()
                sidebar.multiselect.side_effect = lambda label, options, default: default
                sidebar.text_input.return_value = ''
                sidebar.checkbox.return_value = True
                with patch.object(app.st, 'sidebar', sidebar):
                    self.assertEqual(app.build_filtered_df(df).index.tolist(), [2, 3, 4, 5])
                    self.assertEqual(len(app.build_filtered_df(df.drop(columns='author'))), 6)
                    self.assertTrue(app.build_filtered_df(df.iloc[:2]).empty)
                    sidebar.checkbox.return_value = False
                    self.assertEqual(len(app.build_filtered_df(df)), 6)


if __name__ == '__main__':
    unittest.main()
