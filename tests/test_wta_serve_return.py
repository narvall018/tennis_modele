import numpy as np
import pandas as pd
import pytest

from src.backtesting.wta_serve_return import (prepare_stats, prepare_matches, build_features,
    arrays, profile, predict_year, full_keys, short_key)


@pytest.fixture
def protocol():
    return {'stats_first_year': 2007, 'match_first_year': 2010, 'statistics_delay_days': 28,
            'statistics_lookback_days': 365, 'half_life_days': 180., 'prior_points': 500.,
            'serve_prior': .6, 'return_prior': .4, 'minimum_history_matches': 1,
            'embargo_days': 7, 'penalty': .01, 'min_train': 2}


def raw_stats():
    return pd.DataFrame([{'tourney_date': 20200101, 'tourney_id': '2020-X', 'match_num': 1,
        'surface': 'Hard', 'best_of': 3, 'score': '6-3 6-4', 'winner_id': 1, 'loser_id': 2,
        'winner_name': 'Alice Alpha', 'loser_name': 'Bella Beta',
        'w_svpt': 60, 'w_1stIn': 40, 'w_1stWon': 30, 'w_2ndWon': 10,
        'l_svpt': 70, 'l_1stIn': 40, 'l_1stWon': 25, 'l_2ndWon': 10}])


def raw_matches():
    return pd.DataFrame([{'Date': '2020-01-30', 'Tournament': 'X', 'Series': 'WTA250',
        'Surface': 'Hard', 'Best of': 3, 'Round': '1st Round', 'Player_1': 'Alpha A.',
        'Player_2': 'Beta B.', 'Winner': 'Alpha A.', 'Status': 'completed',
        'B365_1': 1.9, 'B365_2': 1.9}])


def test_dates_delay_and_current_result_invariance(protocol):
    stats = prepare_stats(raw_stats(), protocol)
    matches = prepare_matches(raw_matches(), protocol)
    result, _ = build_features(matches, stats, protocol)
    assert result.stats_max_start.iloc[0]+pd.Timedelta(days=28) < result._date.iloc[0]
    assert result.id_1.iloc[0] == 1
    too_early = matches.copy(); too_early['_date'] = pd.Timestamp('2020-01-29')
    with pytest.raises(ValueError, match='No matches'):
        build_features(too_early, stats, protocol)
    poisoned = matches.copy(); poisoned['_label'] = 1; poisoned['Winner'] = 'Beta B.'
    changed, _ = build_features(poisoned, stats, protocol)
    for candidate in ['price_control', 'serve_global', 'serve_surface']:
        np.testing.assert_array_equal(arrays(result, candidate)[0], arrays(changed, candidate)[0])


def test_future_stats_and_alias_collision_not_read_early(protocol):
    original = raw_stats()
    future = original.copy(); future['tourney_id'] = '2020-Y'; future['tourney_date'] = 20200201
    future['winner_id'] = 3; future['winner_name'] = 'Anna Alpha'
    matches = prepare_matches(raw_matches(), protocol)
    base, _ = build_features(matches, prepare_stats(original, protocol), protocol)
    both = prepare_stats(pd.concat([original, future]), protocol)
    same, _ = build_features(matches, both, protocol)
    np.testing.assert_array_equal(arrays(base, 'serve_surface')[0], arrays(same, 'serve_surface')[0])
    later = matches.copy(); later['_date'] = pd.Timestamp('2020-03-02')
    with pytest.raises(ValueError, match='No matches'):
        build_features(later, both, protocol)


def test_stats_counts_reserve_and_retirement(protocol):
    stats = raw_stats()
    reserve = stats.copy(); reserve['tourney_date'] = 20260101; reserve['w_svpt'] = -100
    out = prepare_stats(pd.concat([stats, reserve]), protocol)
    assert len(out) == 1 and out._valid.all()
    stats['w_2ndWon'] = 40
    assert not prepare_stats(stats, protocol)._valid.any()
    stats = raw_stats(); stats['score'] = '6-3 1-0 RET'
    assert not prepare_stats(stats, protocol)._valid.any()


def test_profile_priors_expiration_and_order_symmetry(protocol):
    np.testing.assert_allclose(profile([], pd.Timestamp('2020-01-30'), 'Grass', protocol),
                               [np.log(.6/.4), np.log(.4/.6), 0])
    stats = prepare_stats(raw_stats(), protocol)
    matches = prepare_matches(raw_matches(), protocol)
    a, _ = build_features(matches, stats, protocol)
    swapped = matches.copy()
    swapped[['_p1', '_p2']] = swapped[['_p2', '_p1']].to_numpy()
    swapped[['price_0', 'price_1']] = swapped[['price_1', 'price_0']].to_numpy()
    b, _ = build_features(swapped, stats, protocol)
    np.testing.assert_allclose(arrays(a, 'serve_surface')[0], -arrays(b, 'serve_surface')[0])
    expired = matches.copy(); expired['_date'] = pd.Timestamp('2021-02-01')
    with pytest.raises(ValueError, match='No matches'):
        build_features(expired, stats, protocol)


def test_keys_and_unpriced_history(protocol):
    assert short_key('Martinez Sanchez M.') in full_keys('Maria Jose Martinez Sanchez')
    assert short_key('García C.') in full_keys('Caroline Garcia')
    matches = prepare_matches(raw_matches(), protocol)
    unpriced = matches.copy(); unpriced['price_0'] = np.nan
    later = matches.copy(); later['_date'] = pd.Timestamp('2020-02-01'); later['_source_row_id'] = 1
    result, quality = build_features(pd.concat([unpriced, later]), prepare_stats(raw_stats(), protocol), protocol)
    assert len(result) == 1 and quality['blocked_price'] == 1 and result.history_count_1.iloc[0] == 1


def test_fit_does_not_use_future_outcomes(protocol):
    rows = []
    for i in range(15):
        row = {'_date': pd.Timestamp('2020-01-01')+pd.Timedelta(days=i*30),
               '_label': i%2, '_status': 'completed', 'price_0': 1.6+.04*i, 'price_1': 2.2-.03*i}
        row.update({scope+'_'+stat: .03*i for scope in ['global', 'surface'] for stat in ['serve', 'return', 'volume']})
        rows.append(row)
    frame = pd.DataFrame(rows)
    first, audit = predict_year(frame, 'serve_surface', 2021, protocol)
    changed = frame.copy()
    changed.loc[changed._date.ge('2020-12-25'), '_label'] = 1-changed.loc[changed._date.ge('2020-12-25'), '_label']
    second, _ = predict_year(changed, 'serve_surface', 2021, protocol)
    np.testing.assert_array_equal(first[['p0', 'p1']], second[['p0', 'p1']])
    np.testing.assert_allclose(first.p0+first.p1, 1)
    assert pd.Timestamp(audit['train_max_date']) < pd.Timestamp('2020-12-25')
