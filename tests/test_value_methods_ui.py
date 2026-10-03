from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest

from src.app import value_methods as engine
from src.app import value_methods_data as data
from src.app import value_methods_ledger as ledger
from test_value_methods import event, ufc_inputs


def app(root, sport):
    return AppTest.from_string('from pathlib import Path\nfrom src.app.value_methods_page import render_value_method_page\n'
        f'render_value_method_page(Path({str(root)!r}),{sport!r},7,"test")', default_timeout=30)


def button(ui, label):
    return next(b for b in ui.button if b.label == label)


def provider(monkeypatch):
    calls = []
    monkeypatch.setattr(data, 'resolve_api_key', lambda root: ('PRIVATE', 'test'))
    def request(path, key, params):
        calls.append(path)
        if path == 'sports/':
            return SimpleNamespace(ok=True, remaining=100, events=[{'key': k, 'active': True} for k in [*data.DEFAULT_LEAGUES, engine.MMA_KEY]])
        now = engine.utc()
        sport = 'ufc' if engine.MMA_KEY in path else 'football'
        e = event(sport, now)
        return SimpleNamespace(ok=True, remaining=99, events=[e] if e['sport_key'] in path else [])
    monkeypatch.setattr(data, '_request', request)
    return calls


def test_football_is_automatic_without_bankroll_then_records_draw_privately(tmp_path, monkeypatch):
    calls = provider(monkeypatch)
    ui = app(tmp_path, 'football').run()
    assert not ui.exception and len(calls) == 6
    assert any('Signal théorique détecté' in s.value for s in ui.success)
    assert not any(b.label.startswith('Enregistrer la sélection') for b in ui.button)
    button(ui, 'Initialiser la simulation').click().run()
    assert not ui.exception and len(calls) == 6
    button(ui, 'Enregistrer la sélection prioritaire en simulation').click().run()
    assert not ui.exception and len(calls) == 6
    state = ledger.state(ledger.database(tmp_path, 'football'), '7:test')
    assert state['reserved_cents'] == 250 and state['bets'][0]['pick'] == 'Match nul'
    assert not ledger.database(tmp_path, 'ufc').exists()
    assert not any(b.label.startswith('Enregistrer la sélection') for b in ui.button)


def test_ufc_automatically_simulates_both_durations_with_own_ledger(tmp_path, monkeypatch):
    ufc_inputs(tmp_path, engine.utc())
    monkeypatch.setattr(data, 'refresh_due', lambda *args: False)
    calls = provider(monkeypatch)
    ui = app(tmp_path, 'ufc').run()
    assert not ui.exception and len(calls) == 2
    button(ui, 'Initialiser la simulation').click().run()
    assert not ui.exception and len(calls) == 2
    assert any('Probabilités UFC' in exp.label for exp in ui.expander)
    button(ui, 'Enregistrer la sélection prioritaire en simulation').click().run()
    assert not ui.exception and len(calls) == 2
    assert ledger.state(ledger.database(tmp_path, 'ufc'), '7:test')['reserved_cents'] == 250
    assert not ledger.database(tmp_path, 'football').exists()
    assert any('adaptation expérimentale' in m.value for m in ui.markdown)


def test_stale_scan_removes_actions_without_hiding_journal(tmp_path, monkeypatch):
    provider(monkeypatch)
    ledger.initialise(ledger.database(tmp_path, 'football'), '7:test', 1000)
    snapshot = data.collect_live(tmp_path, 'football')
    snapshot['at'] = (engine.utc()-pd.Timedelta(minutes=6)).isoformat()
    data.atomic_json(tmp_path/'bets/value_methods_runtime/football.json', snapshot)
    ui = app(tmp_path, 'football').run()
    assert not ui.exception
    assert any('Le relevé a plus de cinq minutes' in info.value for info in ui.info)
    assert not any(b.label.startswith('Enregistrer la sélection') for b in ui.button)
    assert any('Sauvegarder la bankroll' in b.label for b in ui.get('download_button'))


def test_failed_manual_scan_replaces_previous_candidates(tmp_path, monkeypatch):
    provider(monkeypatch)
    ledger.initialise(ledger.database(tmp_path, 'football'), '7:test', 1000)
    ui = app(tmp_path, 'football').run()
    assert not ui.exception and any(b.label.startswith('Enregistrer la sélection') for b in ui.button)
    monkeypatch.setattr(data, '_request', lambda *args: SimpleNamespace(ok=False, error='quota mensuel épuisé (429)', remaining=0, events=[]))
    button(ui, 'Scanner maintenant — Football').click().run()
    assert not ui.exception
    assert not any(b.label.startswith('Enregistrer la sélection') for b in ui.button)
    assert any('quota mensuel épuisé' in w.value for w in ui.warning)


def test_switching_off_automation_does_not_call_provider_again(tmp_path, monkeypatch):
    calls = provider(monkeypatch)
    ui = app(tmp_path, 'football').run()
    ui.toggle[0].set_value(False).run()
    assert not ui.exception and len(calls) == 6
    button(ui, 'Scanner maintenant — Football').click().run()
    assert not ui.exception and len(calls) == 12


@pytest.mark.parametrize('sport', ['football', 'ufc'])
def test_old_daily_cap_warning_and_none_quota_disappear_on_upgrade(tmp_path, monkeypatch, sport):
    calls = provider(monkeypatch)
    now = engine.utc()
    if sport == 'ufc':
        ufc_inputs(tmp_path, now)
        monkeypatch.setattr(data, 'refresh_due', lambda *args: False)
    folder = tmp_path / 'bets/value_methods_runtime'
    data.atomic_json(folder / 'budget.json', {'day': now.tz_convert('Europe/Paris').date().isoformat(), 'used': 12})
    data.atomic_json(folder / f'{sport}.json', {'at': now.isoformat(),
        'sports': data.DEFAULT_LEAGUES if sport == 'football' else [engine.MMA_KEY], 'events': [],
        'daily_used': 12, 'remaining': None, 'errors': ['Plafond quotidien partagé atteint (12 consultations). Reprise demain.']})
    ui = app(tmp_path, sport).run()
    assert not ui.exception and len(calls) == (6 if sport == 'football' else 2)
    text = '\n'.join(element.value for element in [*ui.caption, *ui.warning, *ui.info])
    assert '12/12' not in text and 'Reprise demain' not in text
    assert 'None' not in text and 'consultations aujourd’hui' not in text
    assert any('Dernier scan' in caption.value for caption in ui.caption)


def test_missing_ufc_bundle_and_failed_refresh_do_not_crash_or_repeat_every_rerun(tmp_path, monkeypatch):
    calls = []
    def fail(root):
        calls.append(root)
        raise RuntimeError('https://example/?apiKey=PRIVATE')
    monkeypatch.setattr(data, 'refresh_ufc', fail)
    ui = app(tmp_path, 'ufc').run()
    assert not ui.exception and len(calls) == 1
    assert any('Paquet UFC indisponible' in error.value for error in ui.error)
    assert 'PRIVATE' not in str([w.value for w in ui.warning])
    ui.run()
    assert not ui.exception and len(calls) == 1


def test_navigation_contains_both_methods():
    source = (Path(__file__).resolve().parents[1]/'unified_app.py').read_text()
    assert 'elif section == "Stratégie Football"' in source
    assert 'elif section == "Stratégie UFC"' in source
    assert 'render_football_value_page(PROJECT_ROOT' in source
    assert 'render_ufc_markov_page(PROJECT_ROOT' in source
