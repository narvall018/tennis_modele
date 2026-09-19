from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from src.app import wta_kernel_page as page, wta_kernel_ledger as ledger
from test_wta_kernel_app import make_bundle


def snapshot():
    now = page.engine.utc()
    event = {'id': 'one', 'sport_key': 'tennis_wta_test', 'sport_title': 'WTA Test',
        'home_team': 'Alice Alpha', 'away_team': 'Bella Beta', 'commence_time': (now+pd.Timedelta(hours=2)).isoformat(),
        'bookmakers': [{'key': 'betclic_fr', 'markets': [{'key': 'h2h', 'last_update': now.isoformat(),
            'outcomes': [{'name': 'Alice Alpha', 'price': 1.9}, {'name': 'Bella Beta', 'price': 1.9}]}]}]}
    return {'events': [event], 'errors': [], 'competitions': 1, 'omitted': 0, 'at': now.isoformat(), 'remaining': 100}


def app(root):
    return AppTest.from_string('from pathlib import Path\nfrom src.app.wta_kernel_page import render_wta_kernel_page\n'
        f'render_wta_kernel_page(Path({str(root)!r}),7,"test")', default_timeout=20)


def button(ui, name):
    return next(b for b in ui.button if b.label == name)


def test_full_scan_confirmation_record_no_reference_no_implicit_api(tmp_path, monkeypatch):
    make_bundle(tmp_path, page.engine.utc()); calls = []
    def collect(root): calls.append(root); return snapshot()
    monkeypatch.setattr(page, 'collect', collect)
    ui = app(tmp_path).run(); assert not ui.exception and not calls
    button(ui, 'Initialiser la simulation').click().run(); assert not ui.exception and not calls
    button(ui, 'Consulter les cotes WTA françaises (consomme du quota API)').click().run()
    assert not ui.exception and len(calls) == 1
    button(ui, 'Analyser automatiquement les matchs et les mises').click().run()
    assert not ui.exception and any('Analyse impossible' in t.value for t in ui.info)
    assert not any(b.label.startswith('Enregistrer la sélection') for b in ui.button)
    next(s for s in ui.selectbox if s.label == 'Surface confirmée').select('Hard')
    next(c for c in ui.checkbox if c.label.startswith('Je confirme :')).check()
    button(ui, 'Analyser automatiquement les matchs et les mises').click().run()
    assert not ui.exception and len(calls) == 1
    assert not ui.number_input  # no odds references or rankings to enter
    button(ui, 'Enregistrer la sélection prioritaire en simulation').click().run()
    assert not ui.exception and len(calls) == 1
    state = ledger.state(tmp_path/'bets/wta_kernel_strategy.sqlite3', '7:test')
    assert state['reserved_cents'] == 250 and len(state['bets']) == 1
    assert not (tmp_path/'bets/wta_strategy.sqlite3').exists()
    assert not (tmp_path/'bets/atp_reference_price.sqlite3').exists()


def test_expired_scan_no_record_and_no_api_refresh(tmp_path, monkeypatch):
    make_bundle(tmp_path, page.engine.utc()); ledger.initialise(tmp_path/'bets/wta_kernel_strategy.sqlite3', '7:test', 1000)
    data = snapshot(); data['at'] = (page.engine.utc()-pd.Timedelta(minutes=6)).isoformat()
    monkeypatch.setattr(page, 'collect', lambda _: data)
    ui = app(tmp_path).run()
    button(ui, 'Consulter les cotes WTA françaises (consomme du quota API)').click().run()
    button(ui, 'Analyser automatiquement les matchs et les mises').click().run()
    assert not ui.exception and any('Relevé expiré' in t.value for t in ui.warning)
    assert not any(b.label.startswith('Enregistrer la sélection') for b in ui.button)
