import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.app import wta_kernel_strategy as engine, wta_kernel_ledger as ledger


def make_bundle(root, now):
    day = engine.utc(now).tz_convert('Europe/Paris').tz_localize(None).normalize()
    raw = pd.DataFrame([{'tourney_id': 'test', 'match_num': i, 'tourney_date': int((day-pd.Timedelta(days=40+i)).strftime('%Y%m%d')),
        'surface': 'Hard', 'best_of': 3, 'score': '6-1 6-1', 'winner_id': 1, 'loser_id': 2,
        'winner_name': 'Alice Alpha', 'loser_name': 'Bella Beta', 'w_svpt': 60,
        'w_1stIn': 40, 'w_1stWon': 38, 'w_2ndWon': 19, 'l_svpt': 70, 'l_1stIn': 40,
        'l_1stWon': 12, 'l_2ndWon': 8} for i in range(5)])
    # A recent identity-only record proves feed currency, but cannot affect profiles.
    recent = raw.iloc[[0]].copy(); recent['match_num'] = 10
    recent['tourney_date'] = int((day-pd.Timedelta(days=7)).strftime('%Y%m%d')); recent['w_svpt'] = np.nan
    history = engine.prepare_history(pd.concat([raw, recent]), day)
    # Add a valid recent match; the28day delay must exclude it from today's profile.
    extra = raw.iloc[[0]].copy(); extra['match_num'] = 11
    extra['tourney_date'] = int((day-pd.Timedelta(days=7)).strftime('%Y%m%d'))
    history = engine.prepare_history(pd.concat([raw, recent, extra]), day)
    folder = Path(root)/engine.FOLDER; folder.mkdir(parents=True, exist_ok=True)
    history.to_csv(folder/'history.csv.gz', index=False, compression='gzip')
    coefficients = np.zeros((143, 1)); coefficients[4] = 20
    np.savez_compressed(folder/'model.npz', mean=np.zeros(14), scale=np.ones(14), signs=np.r_[-np.ones(9), np.ones(5)],
        projection=np.zeros((14, 128)), phase=np.zeros(128), coefficients=coefficients)
    meta = {'strategy_id': engine.STRATEGY_ID, 'model_year': day.year, 'training_max_date': f'{day.year-1}-12-20',
        'history_rows': len(history), 'history_last_date': str((day-pd.Timedelta(days=7)).date()),
        'files': {p: engine.digest(folder/p) for p in ['model.npz', 'history.csv.gz']},
        'evidence': {'roi': {'0.02': .19}, 'settled': 110, 'uncertainty': {'ci95': [-.03, .39]}}}
    (folder/'metadata.json').write_text(json.dumps(meta))
    return engine.load_bundle(root)


def fixture(now):
    return {'tour': 'WTA', 'singles_main_draw': True, 'surface': 'Hard', 'tournament': 'Test',
            'player_1': 'Alice Alpha', 'player_2': 'Bella Beta', 'bookmaker': 'betclic_fr',
            'start': (engine.utc(now)+pd.Timedelta(hours=2)).isoformat(), 'quote_at': engine.utc(now).isoformat(),
            'odds_1': 1.9, 'odds_2': 1.9}


def test_portable_inference_symmetry_and_prior_only_profiles(tmp_path):
    now = pd.Timestamp('2026-09-19T10:00:00Z'); bundle = make_bundle(tmp_path, now); f = fixture(now)
    x, q, ids = engine.fixture_inputs(bundle[1], f, now)
    assert ids == [1, 2] and x[0, -2] == pytest.approx(np.log1p(5))
    candidate = engine.score_fixture(bundle, f, now)
    assert candidate['eligible'] and candidate['pick'] == 'Alice Alpha'
    reverse = dict(f, player_1=f['player_2'], player_2=f['player_1'])
    mirrored = engine.score_fixture(bundle, reverse, now)
    assert mirrored['event_key'] == candidate['event_key']
    np.testing.assert_allclose(mirrored['probabilities'], candidate['probabilities'][::-1])
    damaged = bundle[1].copy(); future = damaged._start.ge(pd.Timestamp('2026-09-01'))
    damaged.loc[future, ['w_won', 'l_won']] = 999
    np.testing.assert_array_equal(engine.fixture_inputs(damaged, f, now)[0], x)


def test_required_context_freshness_unknown_identity_and_quote_binding(tmp_path):
    now = pd.Timestamp('2026-09-19T10:00:00Z'); bundle = make_bundle(tmp_path, now); f = fixture(now)
    for bad in [dict(f, surface=None), dict(f, singles_main_draw=False), dict(f, tour='ATP'),
                dict(f, player_1='Unknown Player'), dict(f, quote_at=(now-pd.Timedelta(minutes=6)).isoformat())]:
        with pytest.raises(ValueError): engine.score_fixture(bundle, bad, now)
    api = dict(f); bad = dict(f, api_pair=api, odds_1=2.)
    with pytest.raises(ValueError, match='API'): engine.validate_fixture(bad, now)
    stale = copy.deepcopy(bundle[0]); stale['history_last_date'] = '2026-01-01'
    assert engine.freshness_reasons(stale, now)
    assert engine.freshness_reasons(bundle[0], pd.Timestamp('2027-01-01T00:00:00Z'))


def test_record_recomputes_duplicates_budget_backup_and_isolation(tmp_path):
    now = pd.Timestamp('2026-09-19T10:00:00Z'); bundle = make_bundle(tmp_path, now)
    c = engine.score_fixture(bundle, fixture(now), now); db = tmp_path/'bets/wta_kernel_strategy.sqlite3'
    ledger.initialise(db, 'alice', 1000, now); ledger.initialise(db, 'bob', 500, now)
    tampered = copy.deepcopy(c); tampered['probability'] = .99
    with pytest.raises(ValueError, match='modifié'): ledger.record(db, 'alice', tampered, now, root=tmp_path)
    assert ledger.record(db, 'alice', c, now, root=tmp_path) == 2.5
    with pytest.raises(ValueError, match='déjà'): ledger.record(db, 'alice', c, now, root=tmp_path)
    assert ledger.state(db, 'bob', now)['reserved_cents'] == 0
    assert ledger.state(db, 'alice', now)['reserved_cents'] == 250
    backup = ledger.export_backup(db, 'alice', now)
    assert json.loads(backup)['format'] == 'wta-kernel-paper-v1'
    ledger.restore_backup(tmp_path/'restored.sqlite3', 'alice', backup, now)
    assert ledger.state(tmp_path/'restored.sqlite3', 'alice', now)['reserved_cents'] == 250
    from src.app import wta_strategy_ledger as old
    with pytest.raises(ValueError, match='incompatible'):
        old.restore_backup(tmp_path/'old.sqlite3', 'alice', backup, now)


def test_quotes_require_no_reference_but_ignore_wrong_sport_and_stale():
    now = pd.Timestamp('2026-09-19T10:00:00Z')
    e = {'id': 'e', 'sport_key': 'tennis_wta_test', 'home_team': 'Alice Alpha', 'away_team': 'Bella Beta',
         'commence_time': (now+pd.Timedelta(hours=2)).isoformat(), 'bookmakers': [
             {'key': 'betclic_fr', 'markets': [{'key': 'h2h', 'last_update': now.isoformat(),
               'outcomes': [{'name': 'Alice Alpha', 'price': 1.9}, {'name': 'Bella Beta', 'price': 1.9}]}]}]}
    assert len(engine.quotes([e], now)) == 1
    assert not engine.quotes([dict(e, sport_key='tennis_atp_test')], now)
    assert not engine.quotes([e], now+pd.Timedelta(minutes=6))


def test_shipped_bundle_loads_without_research_training():
    root = Path(__file__).resolve().parents[1]
    bundle = engine.load_bundle(root)
    assert bundle[0]['model_year'] == 2026
    assert bundle[0]['training_max_date'] < '2025-12-25'
    assert not bundle[0]['evidence']['nominal_historical_gate_passed']


def test_research_feature_parity_on_historical_rows():
    root = Path(__file__).resolve().parents[1]
    cache = root/'models/wta_serve_return_2026_09_18/features.parquet'
    if not cache.exists(): pytest.skip('Optional local research cache not deployed')
    from src.backtesting.kernel_matchups import arrays
    rows = pd.read_parquet(cache).sample(12, random_state=20260919)
    _, history, _ = engine.load_bundle(root)
    for _, row in rows.iterrows():
        day = pd.Timestamp(row['_date']); now = day.tz_localize('Europe/Paris')+pd.Timedelta(hours=10)
        past = history[history._start+pd.Timedelta(days=28) < day]
        names = []
        for identity in [row.id_1, row.id_2]:
            options = []
            for prefix in ['winner', 'loser']:
                options.extend(past.loc[past[prefix+'_id'].eq(identity), prefix+'_name'].tolist())
            names.append(options[-1])
        f = {'player_1': names[0], 'player_2': names[1], 'surface': row.Surface,
             'start': (now+pd.Timedelta(hours=2)).isoformat(), 'odds_1': row.price_0, 'odds_2': row.price_1}
        live_x, live_q, _ = engine.fixture_inputs(history, f, now)
        expected_x, expected_q, _, _ = arrays(pd.DataFrame([row]), 'wta')
        np.testing.assert_allclose(live_x, expected_x, atol=1e-10)
        np.testing.assert_allclose(live_q, expected_q, atol=1e-12)


def feed(rows):
    """Raw-feed columns the identity table reads, with valid service counts."""
    return pd.DataFrame([{'tourney_id': r['t'], 'match_num': i, 'tourney_date': r['date'],
        'surface': 'Hard', 'best_of': 3, 'score': '6-1 6-1',
        'winner_id': r['wid'], 'loser_id': r['lid'], 'winner_name': r['w'], 'loser_name': r['l'],
        'winner_age': r.get('w_age'), 'loser_age': r.get('l_age'),
        'winner_ioc': r.get('w_ioc', 'USA'), 'loser_ioc': r.get('l_ioc', 'USA'),
        'w_svpt': 60, 'w_1stIn': 40, 'w_1stWon': 38, 'w_2ndWon': 19,
        'l_svpt': 70, 'l_1stIn': 40, 'l_1stWon': 12, 'l_2ndWon': 8} for i, r in enumerate(rows)])


RIVALS = ['Ada Rival', 'Bea Rival', 'Cleo Rival', 'Dana Rival', 'Elsa Rival', 'Fay Rival', 'Gia Rival']


def published_age(date, born):
    """The feed publishes age in years to three decimals; invert it as it does."""
    return round((pd.Timestamp(str(date))-pd.Timestamp(born)).days/365.2425, 3)


def test_identity_map_merges_reissued_id_but_never_namesakes_or_co_entrants():
    born = '2008-01-06'
    reissued = feed([{'t': 'a', 'date': 20260105, 'wid': 10, 'lid': 99, 'w': 'Iva Jović',
                      'l': RIVALS[0], 'w_age': published_age(20260105, born)},
                     {'t': 'b', 'date': 20260608, 'wid': 77, 'lid': 98, 'w': 'Iva Jovic',
                      'l': RIVALS[1], 'w_age': published_age(20260608, born)}])
    mapping, merges = engine.identity_map(reissued)
    assert mapping == {77.: 10.} and merges[0]['name'] == 'iva jovic' and merges[0]['merged_rows'] == 1
    # A namesake born five years apart is a different woman.
    namesakes = feed([{'t': 'a', 'date': 20260105, 'wid': 10, 'lid': 99, 'w': 'Anna Namesake',
                       'l': RIVALS[0], 'w_age': published_age(20260105, born)},
                      {'t': 'b', 'date': 20260608, 'wid': 77, 'lid': 98, 'w': 'Anna Namesake',
                       'l': RIVALS[1], 'w_age': published_age(20260608, '2003-01-06')}])
    assert engine.identity_map(namesakes) == ({}, [])
    # Two ids in one draw are two entrants, whatever the ages say.
    co_entrants = feed([{'t': 'a', 'date': 20260105, 'wid': 10, 'lid': 99, 'w': 'Anna Namesake',
                         'l': RIVALS[0], 'w_age': published_age(20260105, born)},
                        {'t': 'a', 'date': 20260105, 'wid': 77, 'lid': 98, 'w': 'Anna Namesake',
                         'l': RIVALS[1], 'w_age': published_age(20260105, born)}])
    assert engine.identity_map(co_entrants) == ({}, [])
    # A corrupt age carries no evidence, so nationality decides; the feed really ships 2808.
    ages = feed([{'t': 'a', 'date': 20260105, 'wid': 10, 'lid': 99, 'w': 'Lea Blank', 'l': RIVALS[0], 'w_ioc': 'FRA'},
                 {'t': 'b', 'date': 20260608, 'wid': 77, 'lid': 98, 'w': 'Lea Blank', 'l': RIVALS[1],
                  'w_age': 2808.0, 'w_ioc': 'FRA'}])
    assert engine.identity_map(ages)[0] == {77.: 10.}
    foreign = ages.copy(); foreign.loc[1, 'winner_ioc'] = 'JPN'
    assert engine.identity_map(foreign) == ({}, [])
    # Neither age nor nationality: an absence of evidence is not evidence of sameness.
    blind = ages.copy(); blind['winner_ioc'] = ''
    assert engine.identity_map(blind) == ({}, [])


def test_identity_table_survives_a_partial_feed_and_repairs_split_profiles():
    # The canonical id sits in an earlier season; this year the re-issued id leads on count.
    born = '2005-01-06'
    old = feed([{'t': f'o{i}', 'date': 20250105+i, 'wid': 10, 'lid': 90, 'w': 'Robin Split',
                 'l': RIVALS[0], 'w_age': published_age(20250105+i, born)} for i in range(7)])
    new = feed([{'t': f'n{i}', 'date': 20260105+i, 'wid': 77, 'lid': 90, 'w': 'Robin Split',
                 'l': RIVALS[0], 'w_age': published_age(20260105+i, born)} for i in range(5)])
    whole = pd.concat([old, new], ignore_index=True)
    assert engine.identity_map(whole)[0] == {77.: 10.}, 'the full feed must elect the long-standing id'
    assert engine.identity_map(new)[0] == {}, 'this year alone cannot see the canonical id'
    # Seeding with the frozen table stops the partial feed electing the re-issued id.
    assert engine.identity_map(new, {'77': 10})[0] == {77.: 10.}
    history = engine.prepare_history(whole, pd.Timestamp('2026-09-19').date(), {77.: 10.})
    assert set(history.winner_id) == {10.}, 'both spells belong to one player'
    # Split across two ids she is unresolvable; merged, both spells feed one profile.
    split = engine.prepare_history(whole, pd.Timestamp('2026-09-19').date())
    for prefix in ['winner', 'loser']:
        for frame in [history, split]: frame['_'+prefix+'_key'] = frame[prefix+'_name'].map(engine.name_key)
    f = {'player_1': 'Robin Split', 'player_2': RIVALS[0], 'surface': 'Hard',
         'start': '2026-09-19T12:00:00+00:00', 'odds_1': 1.9, 'odds_2': 1.9}
    now = pd.Timestamp('2026-09-19T08:00:00Z')
    with pytest.raises(ValueError, match='ambiguë'): engine.fixture_inputs(split, f, now)
    assert engine.fixture_inputs(history, f, now)[2][0] == 10.


def test_shipped_history_leaves_no_player_split_across_two_ids():
    root = Path(__file__).resolve().parents[1]
    meta, history, _ = engine.load_bundle(root)
    seen = {}
    for prefix in ['winner', 'loser']:
        for key, identity in zip(history['_'+prefix+'_key'], history[prefix+'_id']):
            seen.setdefault(key, set()).add(identity)
    assert not {k: v for k, v in seen.items() if len(v) > 1}
    assert meta['identities'], 'the merge table must ship with the bundle'
    merged = {float(k) for k in meta['identities']}
    assert not merged & {i for v in seen.values() for i in v}, 'a merged id must not survive in the history'


def test_refresh_preserves_model_and_refuses_missing_history(tmp_path, monkeypatch):
    from src.app import wta_kernel_refresh as refresh
    now = pd.Timestamp('2026-09-19T10:00:00Z'); bundle = make_bundle(tmp_path, now)
    history = bundle[1]
    raw = history.rename(columns={'_start': 'tourney_date', 'w_points': 'w_svpt', 'l_points': 'l_svpt'}).copy()
    raw['tourney_date'] = raw.tourney_date.dt.strftime('%Y%m%d').astype(int)
    raw['best_of'], raw['score'] = 3, '6-1 6-1'
    for side in ['w', 'l']:
        raw[side+'_1stIn'] = raw[side+'_svpt']
        raw[side+'_1stWon'] = raw[side+'_won']; raw[side+'_2ndWon'] = 0
    source = {'raw': raw}
    monkeypatch.setattr(refresh, 'fetch_wta_matches', lambda *_: (source['raw'], None, {'missing_files': []}))
    before_model = bundle[0]['files']['model.npz']
    refresh.refresh(tmp_path, now.date())
    assert engine.load_bundle(tmp_path)[0]['files']['model.npz'] == before_model
    manifest = tmp_path/engine.FOLDER/'metadata.json'; before_manifest = engine.digest(manifest)
    source['raw'] = raw.iloc[1:]
    with pytest.raises(ValueError, match='incomplet'): refresh.refresh(tmp_path, now.date())
    assert engine.digest(manifest) == before_manifest


def test_refresh_merges_a_newly_reissued_id_and_carries_the_table_forward(tmp_path, monkeypatch):
    from src.app import wta_kernel_refresh as refresh
    now = pd.Timestamp('2026-09-19T10:00:00Z'); day = now.tz_convert('Europe/Paris').tz_localize(None).normalize()
    bundle = make_bundle(tmp_path, now)
    raw = bundle[1].rename(columns={'_start': 'tourney_date', 'w_points': 'w_svpt', 'l_points': 'l_svpt'}).copy()
    raw['tourney_date'] = raw.tourney_date.dt.strftime('%Y%m%d').astype(int)
    raw['best_of'], raw['score'] = 3, '6-1 6-1'
    for side in ['w', 'l']:
        raw[side+'_1stIn'] = raw[side+'_svpt']
        raw[side+'_1stWon'] = raw[side+'_won']; raw[side+'_2ndWon'] = 0
    born = '2000-02-01'
    raw['winner_age'] = [published_age(d, born) for d in raw.tourney_date]
    raw['loser_age'], raw['winner_ioc'], raw['loser_ioc'] = 30., 'USA', 'USA'
    # The provider re-issues Alice under a fresh id for one later tournament.
    reissued = raw.iloc[[0]].copy()
    reissued['tourney_id'], reissued['match_num'], reissued['winner_id'] = 'reissue', 90, 555
    reissued['tourney_date'] = int((day-pd.Timedelta(days=30)).strftime('%Y%m%d'))
    reissued['winner_age'] = published_age(reissued.tourney_date.iloc[0], born)
    monkeypatch.setattr(refresh, 'fetch_wta_matches',
                        lambda *_: (pd.concat([raw, reissued], ignore_index=True), None, {'missing_files': []}))
    refresh.refresh(tmp_path, now.date())
    meta, history, _ = engine.load_bundle(tmp_path)
    assert meta['identities'] == {'555': 1} and meta['identity_merges'][0]['name'] == 'alice alpha'
    assert set(history.winner_id) == {1.}, 'the re-issued id must not survive the refresh'
    assert engine.fixture_inputs(history, fixture(now), now)[2] == [1., 2.]
