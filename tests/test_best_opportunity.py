import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.app import atp_reference_strategy as atp
from src.app import best_opportunity as ranking
from src.app import wta_kernel_strategy as wta

NOW = pd.Timestamp('2026-09-19T10:00:00Z')


def book(key, names, prices, now=NOW):
    stamp = now.isoformat()
    return {'key': key, 'last_update': stamp,
            'markets': [{'key': 'h2h', 'last_update': stamp,
                         'outcomes': [{'name': n, 'price': p} for n, p in zip(names, prices)]}]}


def event(sport, names, books, now=NOW, identifier='one'):
    return {'id': identifier, 'sport_key': sport, 'sport_title': sport,
            'home_team': names[0], 'away_team': names[1],
            'commence_time': (now+pd.Timedelta(hours=2)).isoformat(), 'bookmakers': books}


def atp_model_event(now=NOW):
    """Pinnacle à 6,8 % de marge : le prix français bat le score sans créer d'arbitrage."""
    names = ['Alpha One', 'Beta Two']
    return event('tennis_atp_test', names,
                 [book('pinnacle', names, [1.8, 1.95], now), book('betclic_fr', names, [2.05, 1.9], now)],
                 now, 'atp')


def arbitrage_event(now=NOW, keys=('winamax_fr', 'unibet_fr'), sport='mma_mixed_martial_arts'):
    names = ['Gamma Three', 'Delta Four']
    return event(sport, names,
                 [book(keys[0], names, [2.03, 1.7], now), book(keys[1], names, [1.7, 2.03], now)],
                 now, 'arb')


def test_a_guaranteed_gain_outranks_a_larger_model_expectation():
    events = {'tennis_atp_test': [atp_model_event()], 'mma_mixed_martial_arts': [arbitrage_event()]}
    rows = ranking.arbitrage_rows(events, NOW) + ranking.atp_rows(
        [e for events_of_sport in events.values() for e in events_of_sport], NOW)
    model = [row for row in rows if row['tier'] == ranking.MODEL_TIER]
    arb = [row for row in rows if row['tier'] == ranking.ARBITRAGE_TIER]
    assert len(model) == 1 and len(arb) == 1
    scores = atp.reference_scores([1.8, 1.95])
    assert model[0]['metric'] == pytest.approx(scores[0]*(1+1.05*.98)-1)
    assert model[0]['metric'] > arb[0]['metric'] > 0
    top = ranking.best(rows)
    assert top['tier'] == ranking.ARBITRAGE_TIER and top['pick'] == 'les deux issues'
    assert 'garanti' in top['metric_kind'] and top['real_money_authorised'] is False


def test_ranking_is_deterministic_whatever_the_scan_order():
    events = {'tennis_atp_test': [atp_model_event()], 'mma_mixed_martial_arts': [arbitrage_event()]}
    rows = ranking.arbitrage_rows(events, NOW) + ranking.atp_rows(
        [e for events_of_sport in events.values() for e in events_of_sport], NOW)
    reference = [row['event'] for row in ranking.rank(rows)]
    shuffled = list(rows)
    random.Random(20260919).shuffle(shuffled)
    assert [row['event'] for row in ranking.rank(shuffled)] == reference


@pytest.mark.parametrize('kind,books,minutes', [
    ('exchange', ('betfair_ex_eu', 'unibet_fr'), 0),
    ('accessibilité', ('winamax_fr', 'book_inconnu'), 0),
    ('fraîcheur', ('winamax_fr', 'unibet_fr'), 10),
])
def test_each_guard_excludes_from_the_ranking_but_stays_reported(kind, books, minutes):
    stale = arbitrage_event(NOW-pd.Timedelta(minutes=minutes), books)
    stale['commence_time'] = (NOW+pd.Timedelta(hours=2)).isoformat()
    rows = ranking.arbitrage_rows({'mma_mixed_martial_arts': [stale]}, NOW)
    assert len(rows) == 1 and rows[0]['exploitable'] is False
    assert kind in rows[0]['blocked_reason']
    assert ranking.rank(rows) == [] and ranking.best(rows) is None


def test_football_and_ufc_get_no_model_row_only_the_arithmetic_layer():
    events = [arbitrage_event(sport='soccer_epl'), arbitrage_event(sport='mma_mixed_martial_arts')]
    assert ranking.atp_rows(events, NOW) == []
    rows = ranking.arbitrage_rows({'soccer_epl': events[:1], 'mma_mixed_martial_arts': events[1:]}, NOW)
    assert {row['sport'] for row in rows} == {'soccer_epl', 'mma_mixed_martial_arts'}
    assert all(row['tier'] == ranking.ARBITRAGE_TIER for row in rows)


def test_the_two_models_without_selection_rights_quote_their_own_metadata():
    notes = ranking.models_without_selection_rights(Path('.'))
    assert set(notes) == {'football', 'ufc'}
    assert 'marché' in notes['football'] and 'log-loss' in notes['football']
    assert 'marché' in notes['ufc']


def wta_bundle(root, now=NOW):
    """Paquet minimal : deux joueuses, statistiques antérieures au délai de 28 jours."""
    day = wta.utc(now).tz_convert('Europe/Paris').tz_localize(None).normalize()
    raw = pd.DataFrame([{'tourney_id': 'test', 'match_num': i,
        'tourney_date': int((day-pd.Timedelta(days=40+i)).strftime('%Y%m%d')), 'surface': 'Hard',
        'best_of': 3, 'score': '6-1 6-1', 'winner_id': 1, 'loser_id': 2,
        'winner_name': 'Alice Alpha', 'loser_name': 'Bella Beta', 'w_svpt': 60, 'w_1stIn': 40,
        'w_1stWon': 38, 'w_2ndWon': 19, 'l_svpt': 70, 'l_1stIn': 40, 'l_1stWon': 12,
        'l_2ndWon': 8} for i in range(5)])
    # Un match récent atteste la fraîcheur du paquet ; le délai de 28 jours
    # l'exclut quand même des profils du jour.
    recent = raw.iloc[[0]].copy()
    recent['match_num'] = 10
    recent['tourney_date'] = int((day-pd.Timedelta(days=7)).strftime('%Y%m%d'))
    history = wta.prepare_history(pd.concat([raw, recent]), day)
    folder = Path(root)/wta.FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    history.to_csv(folder/'history.csv.gz', index=False, compression='gzip')
    coefficients = np.zeros((143, 1))
    coefficients[4] = 20
    np.savez_compressed(folder/'model.npz', mean=np.zeros(14), scale=np.ones(14),
        signs=np.r_[-np.ones(9), np.ones(5)], projection=np.zeros((14, 128)),
        phase=np.zeros(128), coefficients=coefficients)
    meta = {'strategy_id': wta.STRATEGY_ID, 'model_year': day.year,
        'history_rows': len(history),
        'history_last_date': str(history.loc[history._valid, '_start'].max().date()),
        'files': {name: wta.digest(folder/name) for name in ['model.npz', 'history.csv.gz']}}
    (folder/'metadata.json').write_text(json.dumps(meta))
    return wta.load_bundle(root)


def wta_event(now=NOW):
    names = ['Alice Alpha', 'Bella Beta']
    return event('tennis_wta_test', names, [book('betclic_fr', names, [1.9, 1.9], now)], now, 'wta')


def test_wta_needs_a_confirmed_surface_and_never_guesses_one(tmp_path):
    bundle = wta_bundle(tmp_path)
    rows, notes = ranking.wta_rows(bundle, [wta_event()], {}, NOW)
    assert rows == [] and notes['surfaces_a_confirmer'] == ['tennis_wta_test']
    rows, notes = ranking.wta_rows(bundle, [wta_event()], {'tennis_wta_test': 'Hard'}, NOW)
    assert notes['surfaces_a_confirmer'] == [] and len(rows) == 1
    assert rows[0]['tier'] == ranking.MODEL_TIER and rows[0]['pick'] == 'Alice Alpha'
    assert rows[0]['detail']['fixture']['surface'] == 'Hard'


def test_a_doubles_pair_is_never_priced_as_a_singles_match(tmp_path):
    bundle = wta_bundle(tmp_path)
    doubles = wta_event()
    doubles['home_team'] = 'Alice Alpha / Carla Gamma'
    doubles['bookmakers'][0]['markets'][0]['outcomes'][0]['name'] = 'Alice Alpha / Carla Gamma'
    rows, notes = ranking.wta_rows(bundle, [doubles], {'tennis_wta_test': 'Hard'}, NOW)
    assert rows == [] and notes == {'surfaces_a_confirmer': [], 'ecartes': []}


def test_stakes_follow_the_frozen_rules_of_the_ledgers():
    rows = ranking.arbitrage_rows({'mma_mixed_martial_arts': [arbitrage_event()]}, NOW)
    plan = ranking.stake_plan(rows[0], 100_000)
    assert plan['cap_cents'] == 2_000 and plan['total_cents'] <= plan['cap_cents']
    for leg in plan['legs']:
        assert leg['stake_cents']*leg['odds'] >= plan['total_cents']
    model = ranking.atp_rows([atp_model_event()], NOW)[0]
    assert ranking.stake_plan(model, 100_000)['total_cents'] == 250
    assert ranking.stake_plan(model, 1_000)['total_cents'] == 2
