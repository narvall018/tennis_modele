from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from src.app import atp_reference_strategy as engine, atp_reference_ledger as ledger
from src.app import tennis_strategy_ledger as old

NOW = pd.Timestamp('2026-09-17T10:00:00Z')


def event(now=NOW):
    def book(key, prices):
        return {'key': key, 'markets': [{'key': 'h2h', 'last_update': now.isoformat(),
             'outcomes': [{'name': name, 'price': price} for name, price in zip(['Alpha One', 'Beta Two'], prices)]}]}
    return {'id': 'one', 'sport_key': 'tennis_atp_test', 'sport_title': 'ATP Test',
            'home_team': 'Alpha One', 'away_team': 'Beta Two',
            'commence_time': (now+pd.Timedelta(hours=2)).isoformat(),
            'bookmakers': [book('pinnacle', [5/3, 2.5]), book('betclic_fr', [1.9, 1.9])]}


def candidate(now=NOW):
    return engine.analyse_event(event(now), now)['candidate']


def test_score_buffer_and_swap_without_normalising():
    np.testing.assert_allclose(engine.reference_scores([5/3, 2.5]), [.595, .395])
    np.testing.assert_allclose(engine.reference_scores([2.5, 5/3]), [.395, .595])
    assert engine.reference_scores([1.6, 2.6]).sum() < .99


def test_atp_only_no_wta_mma_or_football():
    for sport in ['tennis_wta_test', 'mma_mixed_martial_arts', 'soccer_epl']:
        e = event(); e['sport_key'] = sport
        assert engine.analyse_event(e, NOW)['status'] == 'blocked'
    assert engine.analyse_event(event(), NOW)['status'] == 'candidate'


def test_no_manual_context_or_model_required():
    c = candidate()
    assert c['pick'] == 'Alpha One' and c['odds'] == 1.9
    assert c['probability'] == pytest.approx(.595)
    assert c['expected_returns'][0] == pytest.approx(.595*(1+.9*.98)-1)
    assert c['real_money_authorised'] is False
    assert 'surface' not in c['fixture'] and 'rank' not in c['fixture']
    engine.verify_candidate(c, NOW)


@pytest.mark.parametrize('seconds', [-1, 301])
def test_reference_future_or_old_blocks(seconds):
    e = event(); e['bookmakers'][0]['markets'][0]['last_update'] = (NOW-pd.Timedelta(seconds=seconds)).isoformat()
    assert engine.analyse_event(e, NOW)['status'] == 'blocked'


def test_time_gap_incomplete_pairs_duplicates_and_late_start_block():
    e = event(); e['bookmakers'][0]['markets'][0]['last_update'] = (NOW-pd.Timedelta(minutes=4)).isoformat()
    assert engine.analyse_event(e, NOW)['status'] == 'blocked'
    e = event(); e['bookmakers'][0]['markets'][0]['outcomes'].pop()
    assert engine.analyse_event(e, NOW)['status'] == 'blocked'
    e = event(); e['bookmakers'].append(deepcopy(e['bookmakers'][1]))
    assert engine.analyse_event(e, NOW)['status'] == 'blocked'
    e = event(); e['commence_time'] = (NOW+pd.Timedelta(minutes=9)).isoformat()
    assert engine.analyse_event(e, NOW)['status'] == 'blocked'


def test_missing_reference_is_not_no_signal():
    e = event(); e['bookmakers'].pop(0)
    assert engine.analyse_event(e, NOW)['status'] == 'blocked'
    e = event(); e['bookmakers'][1]['markets'][0]['outcomes'][0]['price'] = 1.7
    e['bookmakers'][1]['markets'][0]['outcomes'][1]['price'] = 2.1
    assert engine.analyse_event(e, NOW)['status'] == 'no_signal'


def test_allocation_deduplicates_matches_and_respects_total_budget(tmp_path):
    db = tmp_path/'new.sqlite3'; ledger.initialise(db, 'u', 1000, NOW)
    c = candidate(); results = []
    for i in range(12):
        copy = deepcopy(c); copy['event_key'] = str(i)
        results.extend([{'candidate': copy}, {'candidate': deepcopy(copy)}])
    plan = engine.allocation(results, ledger.state(db, 'u', NOW))
    assert len(plan) == 12
    assert sum(item['stake_cents'] for item in plan) == 2000
    assert sum(item['stake_cents'] > 0 for item in plan) == 8


def test_record_verifies_calculation_and_protects_against_duplicate_event_ids(tmp_path):
    db = tmp_path/'new.sqlite3'; ledger.initialise(db, 'u', 1000, NOW)
    c = candidate()
    assert ledger.record(db, 'u', c, NOW) == 2.5
    assert engine.allocation([{'candidate': c}], ledger.state(db, 'u', NOW)) == []
    another = event(); another['id'] = 'changed-provider-id'
    with pytest.raises(ValueError, match='déjà'):
        ledger.record(db, 'u', engine.analyse_event(another, NOW)['candidate'], NOW)
    assert not (tmp_path/'tennis_strategy.sqlite3').exists()


@pytest.mark.parametrize('field,value', [('probability', .99), ('odds', 4), ('event_key', 'other'),
                                        ('rule_sha256', 'old'), ('real_money_authorised', True)])
def test_tampered_candidates_rejected(tmp_path, field, value):
    db = tmp_path/'new.sqlite3'; ledger.initialise(db, 'u', 1000, NOW)
    c = candidate(); c[field] = value
    with pytest.raises(ValueError): ledger.record(db, 'u', c, NOW)
    assert not ledger.state(db, 'u', NOW)['bets']


def test_expired_quotes_block_record_even_if_computation_time_is_recent(tmp_path):
    db = tmp_path/'new.sqlite3'; ledger.initialise(db, 'u', 1000, NOW)
    c = candidate(); later = NOW+pd.Timedelta(minutes=6); c['computed_at'] = later.isoformat()
    with pytest.raises(ValueError): ledger.record(db, 'u', c, later)


def test_new_backup_identity_cannot_mix_with_old_atp(tmp_path):
    db = tmp_path/'new.sqlite3'; ledger.initialise(db, 'u', 1000, NOW)
    ledger.record(db, 'u', candidate(), NOW)
    backup = ledger.export_backup(db, 'u', NOW)
    assert json.loads(backup)['format'] == 'atp-reference-paper-v1'
    with pytest.raises(ValueError, match='incompatible'):
        old.restore_backup(tmp_path/'old.sqlite3', 'u', backup, NOW)
    ledger.restore_backup(tmp_path/'restored.sqlite3', 'v', backup, NOW)
    assert ledger.state(tmp_path/'restored.sqlite3', 'v', NOW)['reserved_cents'] == 250


def test_nav_replaces_old_page_and_removes_wta_section():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1]/'unified_app.py').read_text()
    assert '"Stratégie WTA"' not in source
    assert 'from src.app.atp_reference_page import render_atp_reference_page' in source
    assert 'from src.app.tennis_strategy_page import render_tennis_strategy_page' not in source
