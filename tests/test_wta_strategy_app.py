from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.app import wta_strategy as engine, wta_strategy_ledger as ledger
from src.app import tennis_strategy_ledger as atp_ledger
from src.app.wta_strategy_refresh import merge_current
from src.backtesting.wta_price_residual import rebuild_daily_features
from src.data.tennis_pipeline import DataQualityError

NOW = pd.Timestamp('2026-09-16T10:00:00Z')
ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return {'tour': 'WTA', 'singles_main_draw': True, 'start': '2026-09-16T18:00:00Z',
            'quote_at': NOW.isoformat(), 'bet365_quote_at': NOW.isoformat(), 'pinnacle_quote_at': NOW.isoformat(),
            'player_1': 'Alpha A.', 'player_2': 'Beta B.', 'player_1_rank': 10, 'player_2_rank': 20,
            'odds_1': 2., 'odds_2': 1.85, 'bet365_odds_1': 1.9, 'bet365_odds_2': 1.9,
            'pinnacle_odds_1': 1.95, 'pinnacle_odds_2': 1.95, 'bookmaker': 'winamax_fr',
            'surface': 'Hard', 'tournament': 'Test'}


def history():
    rows = []
    for i, (day, label, status, surface) in enumerate([
        ('2025-01-01', 1, 'completed', 'Clay'), ('2026-09-10', 0, 'completed', 'Hard'),
        ('2026-09-10', 1, 'completed', 'Hard'), ('2026-09-13', 1, 'retired', 'Hard')]):
        rows.append({'_source_row_id': str(i), '_date': pd.Timestamp(day), '_p1': 'Alpha A.', '_p2': 'Beta B.',
                     '_label': label, '_status': status, '_surface': surface, '_series': 'International',
                     '_round': '1st Round', '_tournament': 'Test'})
    return pd.DataFrame(rows)


@pytest.mark.parametrize('field,value', [
    ('tour', 'ATP'), ('singles_main_draw', False), ('surface', None), ('bookmaker', 'pinnacle'),
    ('start', '2026-09-16T09:00:00Z'), ('start', '2026-09-16T18:00:00'),
    ('bet365_quote_at', None), ('pinnacle_quote_at', '2026-09-16T10:01:00Z'),
    ('pinnacle_quote_at', '2026-09-16T09:54:00Z'), ('quote_at', '2026-09-16T09:40:00Z'),
    ('bet365_odds_1', float('nan')), ('pinnacle_odds_2', 0), ('odds_1', 20),
    ('player_2', 'Alpha A.'), ('player_1_rank', float('inf')),
])
def test_incomplete_incoherent_or_stale_inputs_rejected(field, value):
    f = fixture(); f[field] = value
    with pytest.raises(ValueError):
        engine.validate_fixture(f, NOW)


def test_features_match_audited_prior_day_replay_and_ignore_future():
    past, f = history(), fixture()
    actual = engine.fixture_features(past, f)
    target = {**past.iloc[-1].to_dict(), '_date': pd.Timestamp('2026-09-16'), '_status': 'upcoming'}
    expected = rebuild_daily_features(pd.concat([past, pd.DataFrame([target])], ignore_index=True)).iloc[-1]
    for col in ['elo_diff', 'surface_elo_diff', 'form_10_diff', 'rest_diff', 'fatigue_diff']:
        assert actual[col].iloc[0] == pytest.approx(expected[col])
    poisoned = pd.concat([past, pd.DataFrame([{**target, '_status': 'completed', '_label': 0},
                            {**target, '_date': pd.Timestamp('2026-09-17'), '_status': 'completed'}])])
    pd.testing.assert_frame_equal(actual, engine.fixture_features(poisoned, f))
    assert actual.log_rank_diff.iloc[0] == pytest.approx(np.log(11) - np.log(21))


def test_native_bundle_symmetry_and_french_prices_not_model_inputs():
    bundle = engine.load_bundle(ROOT)
    meta, _, protocol, model = bundle
    synthetic = ({**meta, 'history_last_date': '2026-09-15'}, history(), protocol, model)
    f = fixture()
    a = engine.score_fixture(synthetic, f, NOW)
    changed = dict(f, odds_1=1.85, odds_2=2.)
    b = engine.score_fixture(synthetic, changed, NOW)
    assert a['probabilities'] == b['probabilities']
    assert a['expected_returns'] != b['expected_returns']
    for left, right in [('player_1', 'player_2'), ('player_1_rank', 'player_2_rank'),
                        ('odds_1', 'odds_2'), ('bet365_odds_1', 'bet365_odds_2'),
                        ('pinnacle_odds_1', 'pinnacle_odds_2')]:
        f[left], f[right] = f[right], f[left]
    swapped = engine.score_fixture(synthetic, f, NOW)
    np.testing.assert_allclose(a['probabilities'], swapped['probabilities'][::-1], atol=1e-7)
    assert a['event_key'] == swapped['event_key']
    assert not a['real_money_authorised']
    assert meta['train_last_date'] < meta['cutoff_exclusive'] <= '2026-01-01'
    assert not meta['evidence']['research_gate_passed']


def candidate(key='one'):
    return {'strategy_id': engine.STRATEGY_ID, 'eligible': True, 'event_key': key,
            'computed_at': NOW.isoformat(), 'fixture': fixture(), 'selected_side': 0,
            'probability': .6, 'odds': 2., 'pick': 'Alpha A.', 'real_money_authorised': False}


def test_wta_ledger_isolation_backups_and_risk_caps(tmp_path):
    db, atp = tmp_path/'wta.sqlite3', tmp_path/'atp.sqlite3'
    ledger.initialise(db, 'user', 1000, NOW)
    atp_ledger.initialise(atp, 'user', 1000, NOW)
    assert ledger.record(db, 'user', candidate(), NOW) == 2.5
    assert atp_ledger.state(atp, 'user', NOW)['reserved_cents'] == 0
    with pytest.raises(ValueError): ledger.record(db, 'user', candidate(), NOW)
    for i in range(7): ledger.record(db, 'user', candidate(str(i)), NOW)
    with pytest.raises(ValueError): ledger.record(db, 'user', candidate('ninth'), NOW)
    assert ledger.state(db, 'user', NOW)['reserved_cents'] == 2000
    exported = ledger.export_backup(db, 'user', NOW)
    assert json.loads(exported)['format'] == 'wta-paper-v1'
    with pytest.raises(ValueError): atp_ledger.restore_backup(atp, 'other', exported, NOW)
    ledger.restore_backup(tmp_path/'restored.sqlite3', 'other', exported, NOW)
    assert ledger.state(tmp_path/'restored.sqlite3', 'other', NOW)['reserved_cents'] == 2000
    stale = candidate('stale'); stale['fixture']['pinnacle_quote_at'] = '2026-09-16T09:30:00Z'
    with pytest.raises(ValueError): ledger.record(db, 'user', stale, NOW)


def test_refresh_preserves_history_ids_and_rejects_missing_matches():
    previous = history().drop(index=2).reset_index(drop=True)
    mapping = {'_date': 'Date', '_p1': 'Player_1', '_p2': 'Player_2', '_status': 'Status',
               '_surface': 'Surface', '_series': 'Series', '_round': 'Round', '_tournament': 'Tournament'}
    current = previous[previous['_date'].dt.year.eq(2026)].rename(columns=mapping).copy()
    current['Winner'] = np.where(current['_label'].eq(1), current['Player_1'], current['Player_2'])
    merged = merge_current(previous, current, NOW.date())
    assert merged['_source_row_id'].tolist() == previous['_source_row_id'].tolist()
    pd.testing.assert_frame_equal(merged.iloc[:1], previous[engine.HISTORY_COLUMNS].iloc[:1])
    with pytest.raises(DataQualityError): merge_current(previous, current.iloc[:1], NOW.date())


def test_quotes_do_not_invent_missing_prices_or_timestamps():
    event = {'id': 'e', 'sport_key': 'tennis_wta_test', 'commence_time': fixture()['start'],
             'home_team': 'Alice Alpha', 'away_team': 'Betty Beta',
             'bookmakers': [{'key': 'pinnacle', 'last_update': NOW.isoformat(), 'markets': [
                 {'key': 'h2h', 'outcomes': [{'name': 'Alice Alpha', 'price': 2.}, {'name': 'Betty Beta', 'price': 1.85}]}]}]}
    assert len(engine.quotes([event], NOW)) == 1
    f = fixture(); f['pinnacle_api_pair'] = engine.quotes([event], NOW)[0]
    f.update(pinnacle_odds_1=2., pinnacle_odds_2=1.85)
    engine.validate_fixture(f, NOW)
    f['player_1'] = 'Other O.'
    with pytest.raises(ValueError): engine.validate_fixture(f, NOW)
    for edit in ['missing_time', 'atp', 'missing_side']:
        bad = deepcopy(event)
        if edit == 'missing_time': del bad['bookmakers'][0]['last_update']
        if edit == 'atp': bad['sport_key'] = 'tennis_atp_test'
        if edit == 'missing_side': bad['bookmakers'][0]['markets'][0]['outcomes'].pop()
        assert engine.quotes([bad], NOW) == []
