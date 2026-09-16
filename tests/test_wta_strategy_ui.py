import json
from types import SimpleNamespace
import pandas as pd
import pytest

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
from src.app import wta_strategy_page as page, wta_strategy as engine, wta_strategy_ledger as ledger


def setup(tmp_path, monkeypatch, stale=False):
    folder = tmp_path/'models/wta_live_strategy'; folder.mkdir(parents=True)
    (folder/'metadata.json').write_text('{}')
    today = engine.utc().tz_convert('Europe/Paris').tz_localize(None).normalize()
    history = pd.DataFrame([{'_date': today-pd.Timedelta(days=1), '_p1': 'Alpha A.', '_p2': 'Beta B.'}])
    meta = {'model_year': today.year, 'history_last_date': '2020-01-01' if stale else str((today-pd.Timedelta(days=1)).date()),
        'files': {'booster.ubj': 'model', 'history.csv.gz': 'history'},
        'evidence': {'roi': {'0.02': .146}, 'settled': 114, 'uncertainty': {'ci95': [-.148, .369]}}}
    monkeypatch.setattr(page, '_bundle', lambda *_: (meta, history, {}, None))
    def score(*args):
        f = json.loads(args[-1])
        return {'strategy_id': engine.STRATEGY_ID, 'model_sha256': 'model', 'history_sha256': 'history',
            'fixture': f, 'event_key': 'ui-test', 'computed_at': engine.utc().isoformat(),
            'selected_side': 0, 'pick': f['player_1'], 'odds': f['odds_1'], 'probability': .6,
            'probabilities': [.6, .4], 'expected_returns': [.188, -.208], 'eligible': True,
            'reason': 'Simulation de test', 'real_money_authorised': False}
    monkeypatch.setattr(page, '_score', score)
    return AppTest.from_string('from pathlib import Path\nfrom src.app.wta_strategy_page import render_wta_strategy_page\n'
        f'render_wta_strategy_page(Path({str(tmp_path)!r}),7,"test")', default_timeout=15)


def button(app, label):
    return next(b for b in app.button if b.label == label)


def test_wta_ui_requires_six_prices_and_records_separate_bankroll(tmp_path, monkeypatch):
    app = setup(tmp_path, monkeypatch).run()
    assert not app.exception
    button(app, 'Initialiser la simulation WTA').click().run()
    assert app.metric[0].value == '1000.00 €'
    button(app, 'Calculer selon la stratégie WTA').click().run()
    assert any('six cotes' in e.value for e in app.error)
    for side, name in [(1, 'Alpha A.'), (2, 'Beta B.')]:
        next(s for s in app.selectbox if s.label == f'Joueuse côté {side}').select(name)
        next(n for n in app.number_input if n.label == f'Classement WTA actuel — côté {side}').set_value(10*side)
        for label in ['Cote française', 'Référence Bet365', 'Référence Pinnacle']:
            next(n for n in app.number_input if n.label == f'{label} — côté {side}').set_value(2. if side == 1 else 1.85)
    next(s for s in app.selectbox if s.label == 'Surface WTA confirmée').select('Hard')
    app.text_input[0].set_value('Test')
    app.checkbox[0].check()
    button(app, 'Calculer selon la stratégie WTA').click().run()
    assert not app.exception
    button(app, 'Enregistrer ce pari WTA en simulation').click().run()
    assert not app.exception
    state = ledger.state(tmp_path/'bets/wta_strategy.sqlite3', '7:test')
    assert state['reserved_cents'] == 250
    assert not (tmp_path/'bets/tennis_strategy.sqlite3').exists()


def test_quote_fetch_does_not_collide_with_widget_state(tmp_path, monkeypatch):
    app = setup(tmp_path, monkeypatch)
    ledger.initialise(tmp_path/'bets/wta_strategy.sqlite3', '7:test', 1000)
    monkeypatch.setattr(page, 'active_sports', lambda _: SimpleNamespace(ok=True, events=[], error=''))
    app.run()
    button(app, 'Consulter les cotes WTA françaises et Pinnacle (quota API)').click().run()
    assert not app.exception
    assert any('0 compétition' in c.value for c in app.caption)


def test_stale_wta_history_leaves_journal_accessible(tmp_path, monkeypatch):
    app = setup(tmp_path, monkeypatch, stale=True)
    ledger.initialise(tmp_path/'bets/wta_strategy.sqlite3', '7:test', 1000)
    app.run()
    assert not app.exception
    assert any('fraîcheur' in e.value for e in app.error)
    assert not any('Calculer selon' in b.label for b in app.button)
    assert app.metric[0].value == '1000.00 €'
