from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from src.app import atp_reference_page as page, atp_reference_ledger as ledger
from src.app import tennis_strategy_ledger as old_atp, wta_strategy_ledger as old_wta


def snapshot():
    now = page.engine.utc()
    def book(name, prices):
        return {'key': name, 'markets': [{'key': 'h2h', 'last_update': now.isoformat(),
                'outcomes': [{'name': n, 'price': p} for n, p in zip(['Alpha One', 'Beta Two'], prices)]}]}
    event = {'id': 'one', 'sport_key': 'tennis_atp_test', 'sport_title': 'ATP Test',
             'home_team': 'Alpha One', 'away_team': 'Beta Two',
             'commence_time': (now+pd.Timedelta(hours=2)).isoformat(),
             'bookmakers': [book('pinnacle', [5/3, 2.5]), book('betclic_fr', [1.9, 1.9])]}
    return {'at': now.isoformat(), 'events': [event], 'errors': [], 'catalogue_count': 1,
            'omitted_competitions': 0, 'remaining': 100,
            'coverage': [{'Compétition': 'tennis_atp_test', 'Réponse': 'OK', 'Matchs': 1}]}


def app(tmp_path):
    return AppTest.from_string('from pathlib import Path\nfrom src.app.atp_reference_page import render_atp_reference_page\n'
        f'render_atp_reference_page(Path({str(tmp_path)!r}),7,"test")', default_timeout=20)


def button(ui, label):
    return next(b for b in ui.button if b.label == label)


def test_automatic_scan_record_no_manual_context_no_extra_api_calls(tmp_path, monkeypatch):
    calls = []
    def collect(root):
        calls.append(root)
        return snapshot()
    monkeypatch.setattr(page, 'collect', collect)
    ui = app(tmp_path).run()
    assert not ui.exception and not calls
    button(ui, 'Initialiser la nouvelle simulation').click().run()
    assert not ui.exception and not calls
    assert not ui.selectbox and not ui.number_input
    button(ui, 'Scanner les opportunités ATP').click().run()
    assert not ui.exception and len(calls) == 1
    assert any('Signal théorique détecté' in s.value for s in ui.success)
    assert any('2.50 €' in m.value for m in ui.markdown)
    button(ui, 'Enregistrer la sélection prioritaire en simulation').click().run()
    assert not ui.exception and len(calls) == 1
    state = ledger.state(tmp_path/'bets/atp_reference_price.sqlite3', '7:test')
    assert state['reserved_cents'] == 250 and len(state['bets']) == 1
    assert not any(b.label == 'Enregistrer la sélection prioritaire en simulation' for b in ui.button)
    assert not (tmp_path/'bets/tennis_strategy.sqlite3').exists()
    assert not (tmp_path/'bets/wta_strategy.sqlite3').exists()


def test_expired_scan_cannot_display_actionable_candidate(tmp_path, monkeypatch):
    ledger.initialise(tmp_path/'bets/atp_reference_price.sqlite3', '7:test', 1000)
    data = snapshot(); data['at'] = (page.engine.utc()-pd.Timedelta(minutes=6)).isoformat()
    monkeypatch.setattr(page, 'collect', lambda _: data)
    ui = app(tmp_path).run()
    button(ui, 'Scanner les opportunités ATP').click().run()
    assert not ui.exception
    assert any('Relevé expiré' in w.value for w in ui.warning)
    assert not any(b.label.startswith('Enregistrer la sélection') for b in ui.button)


def test_blocked_not_confused_with_no_signal(tmp_path, monkeypatch):
    ledger.initialise(tmp_path/'bets/atp_reference_price.sqlite3', '7:test', 1000)
    data = snapshot(); data['events'][0]['bookmakers'].pop(0)
    monkeypatch.setattr(page, 'collect', lambda _: data)
    ui = app(tmp_path).run()
    button(ui, 'Scanner les opportunités ATP').click().run()
    assert not ui.exception
    assert any('Analyse impossible' in w.value for w in ui.warning)
    assert not any('Aucune opportunité selon' in i.value for i in ui.info)


def test_old_atp_and_wta_accounts_survive_in_archives(tmp_path):
    old_atp.initialise(tmp_path/'bets/tennis_strategy.sqlite3', '7:test', 500)
    old_wta.initialise(tmp_path/'bets/wta_strategy.sqlite3', '7:test', 700)
    ui = app(tmp_path).run()
    assert not ui.exception
    assert any('Ancien carnet ATP' in e.label for e in ui.expander)
    assert any('Ancien carnet WTA' in e.label for e in ui.expander)
    button(ui, 'Initialiser la nouvelle simulation').click().run()
    assert not ui.exception
    assert old_atp.state(tmp_path/'bets/tennis_strategy.sqlite3', '7:test')['initial_cents'] == 50000
    assert old_wta.state(tmp_path/'bets/wta_strategy.sqlite3', '7:test')['initial_cents'] == 70000
    assert ledger.state(tmp_path/'bets/atp_reference_price.sqlite3', '7:test')['initial_cents'] == 100000


def test_collect_queries_atp_only_and_requests_french_and_reference_regions(tmp_path, monkeypatch):
    data = snapshot(); calls = []
    monkeypatch.setattr(page, 'active_sports', lambda _: SimpleNamespace(ok=True, events=[
        {'key': 'tennis_wta_test', 'active': True}, {'key': 'tennis_atp_test', 'active': True},
        {'key': 'soccer_epl', 'active': True}]))
    def fetch(root, key, regions):
        calls.append((key, regions))
        return SimpleNamespace(ok=True, events=data['events'], remaining=90)
    monkeypatch.setattr(page, 'fetch_h2h_odds', fetch)
    result = page.collect(tmp_path)
    assert calls == [('tennis_atp_test', 'eu,fr')]
    assert len(result['events']) == 1 and not result['errors']


def test_scan_failure_does_not_keep_previous_signals(tmp_path, monkeypatch):
    ledger.initialise(tmp_path/'bets/atp_reference_price.sqlite3', '7:test', 1000)
    monkeypatch.setattr(page, 'collect', lambda _: snapshot())
    ui = app(tmp_path).run()
    button(ui, 'Scanner les opportunités ATP').click().run()
    assert any(b.label.startswith('Enregistrer la sélection') for b in ui.button)
    empty = snapshot(); empty.update(events=[], errors=['Quota épuisé'], coverage=[])
    monkeypatch.setattr(page, 'collect', lambda _: empty)
    button(ui, 'Scanner les opportunités ATP').click().run()
    assert not ui.exception
    assert not any(b.label.startswith('Enregistrer la sélection') for b in ui.button)
    assert any('Quota épuisé' in w.value for w in ui.warning)


def test_provider_exception_never_exposes_a_secret_url(tmp_path, monkeypatch):
    def fail(_):
        raise ValueError('https://example/?apiKey=SECRET_VALUE')
    monkeypatch.setattr(page, 'active_sports', fail)
    result = page.collect(tmp_path)
    assert 'SECRET_VALUE' not in str(result)
    assert result['errors']
