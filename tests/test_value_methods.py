from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.app import ufc_markov as markov
from src.app import value_methods as engine
from src.app import value_methods_data as data
from src.app import value_methods_ledger as ledger

NOW = pd.Timestamp('2026-10-03T12:00:00Z')


def event(sport='football', now=NOW, reference=None, prices=None):
    names = ['Alpha One', 'Draw', 'Beta Two'] if sport == 'football' else ['Alpha One', 'Beta Two']
    def book(key, odds):
        return {'key': key, 'markets': [{'key': 'h2h', 'last_update': now.isoformat(),
            'outcomes': [{'name': n, 'price': p} for n, p in zip(names, odds)]}]}
    books = [book('betclic_fr', prices or ([3., 3., 3.] if sport == 'football' else [1.9, 1.9]))]
    if sport == 'football':
        books.insert(0, book('pinnacle', reference or [10/3, 2.5, 10/3]))
    return {'id': 'one', 'sport_key': 'soccer_epl' if sport == 'football' else engine.MMA_KEY,
            'home_team': 'Alpha One', 'away_team': 'Beta Two',
            'commence_time': (now + pd.Timedelta(hours=2)).isoformat(), 'bookmakers': books}


def ufc_inputs(root=None, now=NOW):
    prior = markov._prior(markov._empty())
    fighters = {}
    for identity, name, strong in [('a', 'Alpha One', True), ('b', 'Beta Two', False)]:
        f = {k: v * 100 for k, v in prior.items()}
        f.update(minutes=100., fights=10., name=name)
        f['sig'], f['sig_att'] = (700., 1000.) if strong else (100., 500.)
        f['absorbed'], f['absorbed_att'] = (100., 500.) if strong else (700., 1000.)
        f['ko_for'], f['ko_against'] = (3., 0.) if strong else (0., 3.)
        fighters[identity] = f
    bundle = {'version': markov.VERSION, 'config': markov.CONFIG, 'cutoff': '2026-10-03',
              'checked_at': now.isoformat(), 'history_last_date': '2026-09-26',
              'history_fights_used': 200, 'decision_training_count': 100,
              'decision_coefficients': [3., .5, 1.], 'fighters': fighters,
              'division_priors': {'Lightweight': prior}, 'profitability_validated': False}
    bundle['sha256'] = markov.fingerprint(bundle)
    cards = {'checked_at': now.isoformat(), 'fights': [{'event_name': 'UFC Test', 'event_date': '2026-10-03',
        'fighter_1': 'Alpha One', 'fighter_2': 'Beta Two', 'fighter_1_id': 'a', 'fighter_2_id': 'b', 'weight_class': 'Lightweight'}]}
    if root is not None:
        document = {'bundle': bundle, 'cards': cards}
        document['sha256'] = markov.fingerprint(document)
        data.atomic_json(root / 'models/value_methods/ufc.json', document)
    return bundle, cards


def test_football_draw_score_matches_frozen_method_and_is_not_normalised():
    np.testing.assert_allclose(engine.reference_scores([10/3, 2.5, 10/3]), [.295, .395, .295])
    assert engine.reference_scores([3.05, 2.4, 3.05]).sum() < .985
    result = engine.analyse_event('football', event(), NOW)
    c = result['candidate']
    assert c['selected_side'] == 1 and c['pick'] == 'Match nul'
    assert c['expected_returns'][1] == pytest.approx(.395 * (1 + 2 * .98) - 1)
    assert c['real_money_authorised'] is False


@pytest.mark.parametrize('problem', ['missing_draw', 'duplicate_book', 'future', 'stale', 'time_gap', 'wrong_sport', 'late_start'])
def test_football_market_failures_cannot_become_signals(problem):
    e = event()
    if problem == 'missing_draw': e['bookmakers'][0]['markets'][0]['outcomes'].pop(1)
    if problem == 'duplicate_book': e['bookmakers'].append(deepcopy(e['bookmakers'][0]))
    if problem in {'future', 'stale', 'time_gap'}:
        seconds = {'future': -1, 'stale': 301, 'time_gap': 181}[problem]
        e['bookmakers'][0]['markets'][0]['last_update'] = (NOW - pd.Timedelta(seconds=seconds)).isoformat()
    if problem == 'wrong_sport': e['sport_key'] = 'tennis_atp_test'
    if problem == 'late_start': e['commence_time'] = (NOW + pd.Timedelta(minutes=9)).isoformat()
    assert engine.analyse_event('football', e, NOW)['status'] == 'blocked'


def test_markov_generator_probability_mass_symmetry_and_duration():
    bundle, cards = ufc_inputs()
    prior = bundle['division_priors']['Lightweight']
    a, b = bundle['fighters']['a'], bundle['fighters']['b']
    rates = markov.matchup(a, b, prior)
    q = markov.generator(rates)
    np.testing.assert_allclose(q.sum(axis=1), 0., atol=1e-12)
    np.testing.assert_array_equal(q[3:], np.zeros((4, 7)))
    for rounds in (3, 5):
        forward = markov.probabilities(rates, bundle['decision_coefficients'], rounds)
        reversed_ = markov.probabilities(markov.matchup(b, a, prior), bundle['decision_coefficients'], rounds)
        assert sum(forward['method_probabilities']) == pytest.approx(1.)
        assert min(forward['method_probabilities']) >= 0
        np.testing.assert_allclose(forward['win_probabilities'], reversed_['win_probabilities'][::-1])
        np.testing.assert_allclose(forward['method_probabilities'][:3], reversed_['method_probabilities'][3:])
    neutral = markov.probabilities(markov.matchup(a, a, prior), bundle['decision_coefficients'], 3)
    assert neutral['win_probabilities'] == pytest.approx([.5, .5])
    short = markov.probabilities(rates, bundle['decision_coefficients'], 3)
    long = markov.probabilities(rates, bundle['decision_coefficients'], 5)
    assert long['method_probabilities'][2] + long['method_probabilities'][5] < short['method_probabilities'][2] + short['method_probabilities'][5]


def test_ufc_matches_official_identity_and_does_not_require_market_reference():
    bundle, cards = ufc_inputs()
    e = event('ufc')
    c = engine.analyse_event('ufc', e, NOW, bundle, cards)['candidate']
    assert c and c['pick'] == 'Alpha One'
    assert c['model']['three_rounds']['win_probabilities'][0] >= c['probability']
    assert c['model']['five_rounds']['win_probabilities'][0] >= c['probability']
    assert c['expected_returns'][0] >= .26
    swapped = deepcopy(e)
    swapped['home_team'], swapped['away_team'] = swapped['away_team'], swapped['home_team']
    reversed_ = engine.analyse_event('ufc', swapped, NOW, bundle, cards)['candidate']
    assert reversed_['selected_side'] == 1 and reversed_['pick'] == c['pick']
    assert reversed_['probability'] == pytest.approx(c['probability'])
    assert reversed_['event_key'] == c['event_key']


@pytest.mark.parametrize('problem', ['other_mma', 'mma_draw', 'old_programme', 'old_model', 'changed_model', 'novice', 'future_cutoff'])
def test_ufc_incomplete_or_uncausal_data_blocks(problem):
    bundle, cards = ufc_inputs()
    e = event('ufc')
    if problem == 'other_mma': cards['fights'] = []
    if problem == 'mma_draw': e['bookmakers'][0]['markets'][0]['outcomes'].append({'name': 'Draw', 'price': 33.})
    if problem == 'old_programme': cards['checked_at'] = (NOW - pd.Timedelta(hours=49)).isoformat()
    if problem == 'old_model': bundle['checked_at'] = (NOW - pd.Timedelta(hours=49)).isoformat()
    if problem == 'changed_model': bundle['decision_coefficients'][0] = 10.
    if problem == 'novice': bundle['fighters']['a']['fights'] = 1.
    if problem == 'future_cutoff': bundle['cutoff'] = '2026-10-04'
    if problem != 'changed_model':
        bundle['sha256'] = markov.fingerprint({k: v for k, v in bundle.items() if k != 'sha256'})
    assert engine.analyse_event('ufc', e, NOW, bundle, cards)['status'] == 'blocked'


def test_draw_and_away_records_settle_and_sports_accounts_stay_separate(tmp_path):
    foot = ledger.database(tmp_path, 'football')
    ufc = ledger.database(tmp_path, 'ufc')
    ledger.initialise(foot, 'u', 1000, NOW)
    ledger.initialise(ufc, 'u', 500, NOW)
    draw = engine.analyse_event('football', event(), NOW)['candidate']
    assert ledger.record(tmp_path, 'football', 'u', draw, NOW) == 2.5
    before = ledger.state(foot, 'u', NOW)
    assert before['reserved_cents'] == 250 and before['available_cents'] == 99750
    duplicate = event(); duplicate['id'] = 'changed-api-id'
    with pytest.raises(ValueError, match='déjà'):
        ledger.record(tmp_path, 'football', 'u', engine.analyse_event('football', duplicate, NOW)['candidate'], NOW)
    ledger.settle(foot, 'u', before['bets'][0]['id'], 'won', NOW + pd.Timedelta(hours=3))
    after = ledger.state(foot, 'u', NOW + pd.Timedelta(hours=3))
    assert after['profit_cents'] == 490 and after['reserved_cents'] == 0
    assert after['roi'] == pytest.approx(1.96)
    away_event = event(reference=[10/3, 10/3, 2.5]); away_event['home_team'] = 'Gamma Three'
    for book in away_event['bookmakers']:
        book['markets'][0]['outcomes'][0]['name'] = 'Gamma Three'
    away = engine.analyse_event('football', away_event, NOW)['candidate']
    assert away['selected_side'] == 2
    ledger.record(tmp_path, 'football', 'u', away, NOW)
    assert ledger.state(ufc, 'u', NOW)['balance_cents'] == 50000
    backup = ledger.export_backup(foot, 'u', 'football', NOW + pd.Timedelta(hours=3))
    with pytest.raises(ValueError, match='incompatible'):
        ledger.restore_backup(tmp_path/'other.sqlite3', 'v', backup, 'ufc', NOW + pd.Timedelta(hours=3))


@pytest.mark.parametrize('sport', ['football', 'ufc'])
def test_record_recomputes_candidates_and_rejects_tampering_expiry_and_changed_bundle(tmp_path, sport):
    bundle, cards = ufc_inputs(tmp_path)
    ledger.initialise(ledger.database(tmp_path, sport), 'u', 1000, NOW)
    c = engine.analyse_event(sport, event(sport), NOW, bundle, cards)['candidate']
    for field, value in [('probability', .999), ('event_key', 'forged'), ('real_money_authorised', True), ('rule_sha256', 'old')]:
        tampered = deepcopy(c); tampered[field] = value
        with pytest.raises(ValueError): ledger.record(tmp_path, sport, 'u', tampered, NOW)
    with pytest.raises(ValueError): ledger.record(tmp_path, sport, 'u', c, NOW + pd.Timedelta(minutes=6))
    if sport == 'ufc':
        document = data.read_json(tmp_path/'models/value_methods/ufc.json')
        document['bundle']['decision_coefficients'][0] = 3.1
        document['bundle']['sha256'] = markov.fingerprint({k: v for k, v in document['bundle'].items() if k != 'sha256'})
        document['sha256'] = markov.fingerprint({k: v for k, v in document.items() if k != 'sha256'})
        data.atomic_json(tmp_path/'models/value_methods/ufc.json', document)
        with pytest.raises(ValueError): ledger.record(tmp_path, sport, 'u', c, NOW)
    assert not ledger.state(ledger.database(tmp_path, sport), 'u', NOW)['bets']


def test_automatic_collector_caches_limits_budget_and_keeps_secret_errors_private(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(data, 'resolve_api_key', lambda root: ('PRIVATE', 'test'))
    def request(path, key, params):
        calls.append((path, params))
        if path == 'sports/':
            return SimpleNamespace(ok=True, remaining=100, events=[{'key': k, 'active': True} for k in [*data.DEFAULT_LEAGUES, engine.MMA_KEY]])
        e = event('ufc' if engine.MMA_KEY in path else 'football')
        e['sport_key'] = path.split('/')[1]
        return SimpleNamespace(ok=True, remaining=99, events=[e])
    monkeypatch.setattr(data, '_request', request)
    first = data.collect_live(tmp_path, 'football', now=NOW)
    assert first['requests'] == 5 and len(calls) == 6
    assert data.collect_live(tmp_path, 'football', now=NOW + pd.Timedelta(minutes=30)) == first
    assert len(calls) == 6
    data.collect_live(tmp_path, 'football', now=NOW + pd.Timedelta(hours=1))
    data.collect_live(tmp_path, 'ufc', now=NOW + pd.Timedelta(hours=1))
    last = data.collect_live(tmp_path, 'football', now=NOW + pd.Timedelta(hours=2))
    assert last['requests'] == 1 and last['daily_used'] == 12
    blocked = data.collect_live(tmp_path, 'ufc', now=NOW + pd.Timedelta(hours=2))
    assert blocked['requests'] == 0 and blocked['errors']
    assert all('regions' not in params for path, params in calls if path != 'sports/')
    assert all('pinnacle' in params['bookmakers'] and 'betclic_fr' in params['bookmakers'] for path, params in calls if path != 'sports/')
    def fail(*args): raise ValueError('https://example/?apiKey=PRIVATE')
    monkeypatch.setattr(data, '_request', fail)
    safe = data.collect_live(tmp_path, 'football', now=NOW + pd.Timedelta(days=1))
    assert 'PRIVATE' not in json.dumps(safe) and safe['errors']


@pytest.mark.parametrize('remaining', [None, 20, 0])
def test_unknown_or_reserved_quota_never_triggers_paid_requests(tmp_path, monkeypatch, remaining):
    calls = []
    monkeypatch.setattr(data, 'resolve_api_key', lambda root: ('PRIVATE', 'test'))
    def request(path, key, params):
        calls.append(path)
        return SimpleNamespace(ok=True, remaining=remaining, events=[{'key': 'soccer_epl', 'active': True}])
    monkeypatch.setattr(data, '_request', request)
    result = data.collect_live(tmp_path, 'football', ['soccer_epl'], NOW)
    assert calls == ['sports/'] and result['requests'] == 0 and result['errors']


def history():
    rows = []
    for i in range(220):
        date = pd.Timestamp('2020-01-01') + pd.Timedelta(days=7*i)
        a, b = f'f{i%4}', f'f{(i+1)%4}'
        row = {'fight_id': str(i), 'event_date': date, 'fighter_1_id': a, 'fighter_2_id': b,
            'fighter_1': a, 'fighter_2': b, 'weight_class': 'Lightweight', 'method': 'U-DEC',
            'duration_secs': 900., 'y': float(i%3 != 0)}
        for side in (1, 2):
            row.update({f'p{side}_sig_lnd': 30. + i%8 + side, f'p{side}_sig_att': 70.,
                f'p{side}_td_lnd': float((i+side)%3), f'p{side}_td_att': 5.,
                f'p{side}_sub_att': 1., f'p{side}_rev': 0., f'p{side}_ctrl_secs': 100.+side*5})
        rows.append(row)
    return pd.DataFrame(rows)


def test_bundle_uses_only_past_cards_and_is_independent_of_winner_first_orientation():
    h = history()
    cutoff = '2025-01-01'
    bundle = markov.build_bundle(h, cutoff, NOW.isoformat())
    future = h.iloc[-1:].copy(); future['fight_id'] = 'future'; future['event_date'] = pd.Timestamp(cutoff)
    future['p1_sig_lnd'] = 10000.
    assert markov.build_bundle(pd.concat([h, future]), cutoff, NOW.isoformat()) == bundle
    swapped = h.copy()
    swapped['y'] = 1 - h.y
    for suffix in ['_id', '']:
        swapped[f'fighter_1{suffix}'], swapped[f'fighter_2{suffix}'] = h[f'fighter_2{suffix}'], h[f'fighter_1{suffix}']
    for k in ['sig_lnd', 'sig_att', 'td_lnd', 'td_att', 'sub_att', 'rev', 'ctrl_secs']:
        swapped[f'p1_{k}'], swapped[f'p2_{k}'] = h[f'p2_{k}'], h[f'p1_{k}']
    other = markov.build_bundle(swapped, cutoff, NOW.isoformat())
    np.testing.assert_allclose(other['decision_coefficients'], bundle['decision_coefficients'])
    assert other['fighters'] == bundle['fighters']
    assert bundle['profitability_validated'] is False and bundle['published_model_reproduced'] is False
