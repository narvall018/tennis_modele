import json
from pathlib import Path

import pandas as pd
import pytest

from src.app import football_exchange_strategy as engine, football_exchange_ledger as ledger

ROOT = Path(__file__).resolve().parents[1]


def quote(now, *, home=1.3820, draw=5.528, away=9.950, book='betfair_ex_eu'):
    return {'sport': 'football', 'competition': 'Ligue 1', 'home_team': 'Alpha FC',
            'away_team': 'Beta United', 'bookmaker': book,
            'start': (engine.utc(now)+pd.Timedelta(hours=6)).isoformat(),
            'quote_at': engine.utc(now).isoformat(),
            'odds_home': home, 'odds_draw': draw, 'odds_away': away}


def test_shipped_rule_only_spans_bands_whose_lower_bound_is_proven():
    meta = engine.load_bundle(ROOT)
    bands = meta['rule']['bias_bands']
    assert bands, 'la regle doit porter au moins une bande'
    assert all(b['lower_bound_points'] >= meta['rule']['minimum_lower_bound_points'] for b in bands)
    # Every band the sweep looked at is accounted for, kept or dropped with a reason.
    assert all('reason' in b for b in meta['dropped_bands'])
    assert meta['real_money_authorised'] is False
    assert meta['rule']['commission'] >= .05, 'retenir le bout defavorable de la commission'
    # The rule must never quietly widen to the bands that failed.
    for dropped in meta['dropped_bands']:
        if 'from' not in dropped:
            continue
        assert all(not (b['from'] <= dropped['from'] < b['to']) for b in bands)


def test_a_tampered_rule_is_refused(tmp_path):
    meta = json.loads((ROOT/engine.FOLDER/'metadata.json').read_text())
    folder = tmp_path/engine.FOLDER
    folder.mkdir(parents=True)
    meta['rule']['min_expected_return'] = -1.
    (folder/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='modifiée'):
        engine.load_bundle(tmp_path)
    # A band invented after the fact is refused even with a matching hash.
    meta = json.loads((ROOT/engine.FOLDER/'metadata.json').read_text())
    meta['rule']['bias_bands'].append({'from': .5, 'to': .6, 'lower_bound_points': -2.})
    meta['rule_sha256'] = engine.rule_digest(meta['rule'])
    (folder/'metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='biais prouvé'):
        engine.load_bundle(tmp_path)


def test_bookmakers_are_refused_because_no_reachable_margin_fits():
    now = pd.Timestamp('2026-09-20T10:00:00Z')
    meta = engine.load_bundle(ROOT)
    for book in ['betclic_fr', 'winamax_fr', 'pinnacle']:
        with pytest.raises(ValueError, match='exchange'):
            engine.score_fixture(meta, quote(now, book=book), now)


def test_rule_refuses_outside_its_band_and_on_a_wide_market():
    now = pd.Timestamp('2026-09-20T10:00:00Z')
    meta = engine.load_bundle(ROOT)
    # A near coin-flip sits far below 0.70 and carries no proven bias.
    weak = engine.score_fixture(meta, quote(now, home=1.8018, draw=3.6697, away=5.5249), now)
    assert not weak['eligible'] and 'hors de la bande' in weak['reason']
    # A bookmaker-sized margin cannot masquerade as an exchange price.
    with pytest.raises(ValueError, match='prix d\'exchange'):
        engine.score_fixture(meta, quote(now, home=1.25, draw=4.80, away=9.0), now)
    stale = quote(now)
    stale['quote_at'] = (engine.utc(now)-pd.Timedelta(minutes=6)).isoformat()
    with pytest.raises(ValueError, match='expirées'):
        engine.score_fixture(meta, stale, now)
    started = quote(now)
    started['start'] = (engine.utc(now)-pd.Timedelta(minutes=1)).isoformat()
    with pytest.raises(ValueError, match='commencé'):
        engine.score_fixture(meta, started, now)


def test_the_lower_bound_not_the_point_estimate_drives_the_decision():
    now = pd.Timestamp('2026-09-20T10:00:00Z')
    meta = engine.load_bundle(ROOT)
    band = next(b for b in meta['rule']['bias_bands'] if b['from'] == .70)
    c = engine.score_fixture(meta, quote(now), now)
    assert .70 <= c['reference_probability'] < .75
    assert c['bias_applied'] == pytest.approx(band['lower_bound_points']/100)
    # The applied correction must be the conservative end, never the headline bias.
    assert c['bias_applied'] < band['bias_points']/100
    assert c['corrected_probability'] == pytest.approx(c['reference_probability']+c['bias_applied'])
    expected = c['corrected_probability']*(1+(c['odds']-1)*(1-meta['rule']['commission']))-1
    assert c['expected_return'] == pytest.approx(expected)
    assert c['real_money_authorised'] is False


def test_quotes_keep_three_way_exchange_prices_and_drop_the_rest():
    now = pd.Timestamp('2026-09-20T10:00:00Z')
    event = {'id': 'e1', 'sport_key': 'soccer_france_ligue_one', 'sport_title': 'Ligue 1',
             'home_team': 'Alpha FC', 'away_team': 'Beta United',
             'commence_time': (engine.utc(now)+pd.Timedelta(hours=6)).isoformat(),
             'bookmakers': [{'key': 'betfair_ex_eu', 'last_update': engine.utc(now).isoformat(),
                 'markets': [{'key': 'h2h', 'outcomes': [
                     {'name': 'Alpha FC', 'price': 1.3820}, {'name': 'Draw', 'price': 5.528},
                     {'name': 'Beta United', 'price': 9.950}]}]},
                 {'key': 'betclic_fr', 'last_update': engine.utc(now).isoformat(),
                  'markets': [{'key': 'h2h', 'outcomes': [
                      {'name': 'Alpha FC', 'price': 1.25}, {'name': 'Draw', 'price': 4.8},
                      {'name': 'Beta United', 'price': 9.0}]}]}]}
    rows = engine.quotes([event], now)
    assert len(rows) == 1 and rows[0]['bookmaker'] == 'betfair_ex_eu'
    assert rows[0]['odds_draw'] == 5.528
    assert not engine.quotes([dict(event, sport_key='tennis_wta_x')], now)
    two_way = json.loads(json.dumps(event))
    two_way['bookmakers'][0]['markets'][0]['outcomes'].pop(1)
    assert not [r for r in engine.quotes([two_way], now) if r['bookmaker'] == 'betfair_ex_eu']


def test_ledger_is_isolated_refuses_tampering_and_duplicates(tmp_path):
    now = pd.Timestamp('2026-09-20T10:00:00Z')
    meta = engine.load_bundle(ROOT)
    c = engine.score_fixture(meta, quote(now), now)
    if not c['eligible']:
        pytest.skip(f"la regle ne signale pas ce prix : {c['reason']}")
    db = tmp_path/'bets/football_exchange.sqlite3'
    ledger.initialise(db, 'alice', 1000, now)
    ledger.initialise(db, 'bob', 500, now)
    tampered = dict(c, probability=.99)
    with pytest.raises(ValueError, match='modifié'):
        ledger.record(db, 'alice', tampered, now, root=ROOT)
    assert ledger.record(db, 'alice', c, now, root=ROOT) > 0
    with pytest.raises(ValueError, match='déjà'):
        ledger.record(db, 'alice', c, now, root=ROOT)
    assert ledger.state(db, 'bob', now)['reserved_cents'] == 0
    assert ledger.state(db, 'alice', now)['reserved_cents'] > 0
    assert json.loads(ledger.export_backup(db, 'alice', now))['format'] == 'football-exchange-paper-v1'


def test_best_quotes_keeps_the_better_price_per_match():
    now = pd.Timestamp('2026-09-20T10:00:00Z')
    base = {'event_id': 'e1', 'sport': 'football', 'competition': 'EPL',
            'home_team': 'Alpha FC', 'away_team': 'Beta United',
            'start': (engine.utc(now)+pd.Timedelta(hours=6)).isoformat(),
            'quote_at': engine.utc(now).isoformat(), 'odds_draw': 5.528, 'odds_away': 9.950}
    rows = [dict(base, bookmaker='betfair_ex_eu', odds_home=1.3820),
            dict(base, bookmaker='smarkets', odds_home=1.3950),
            dict(base, bookmaker='matchbook', odds_home=1.3700)]
    kept = engine.best_quotes(rows)
    assert len(kept) == 1 and kept[0]['bookmaker'] == 'smarkets'
    # A different match is never merged into the same slot.
    other = dict(base, event_id='e2', home_team='Gamma', odds_home=1.3820)
    assert len(engine.best_quotes(rows+[other])) == 2
