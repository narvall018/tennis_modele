from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.backtesting.wta_single_book import (
    CANDIDATES, LAGGED, SOURCE_COLUMNS, arrays, gate, ledger, predict_fold, prepare, summarise,
)


@pytest.fixture
def protocol():
    return {'training_year_start': 2007, 'evaluation_years': [2023, 2024, 2025],
            'min_train': 20, 'embargo_days': 7, 'half_life_years': 3., 'boost_rounds': 3,
            'booster': {'tree_method': 'hist', 'max_depth': 2, 'eta': .1, 'nthread': 1, 'seed': 4},
            'selection': {'minimum_ev_after_haircut': .02, 'profit_haircut': .02,
                          'odds_min': 1.3, 'odds_max': 5., 'flat_fraction': .0025,
                          'maximum_daily_exposure': .02, 'sensitivity_haircuts': [0., .02, .05]},
            'uncertainty': {'samples': 100, 'seed': 5, 'lower_quantile_per_candidate': .025}}


def raw_frame():
    rng = np.random.default_rng(7)
    rows = []
    for year in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
        for i in range(25):
            q = rng.uniform(.3, .7)
            rows.append({'_source_row_id': f'{year}-{i:03}', '_date': pd.Timestamp(year, 2, 1) + pd.Timedelta(days=i),
                         '_p1': 'Alpha A.', '_p2': 'Beta B.', '_label': int(rng.random() < q),
                         '_status': 'completed', '_surface': 'Hard', '_series': 'WTA250',
                         '_round': '1st Round', '_tournament': 'Test',
                         'Pinnacle_1': 1 / (q * 1.04), 'Pinnacle_2': 1 / ((1-q) * 1.04)})
    return pd.DataFrame(rows)


def predictions(count=12):
    return pd.DataFrame({'_source_row_id': [f'{n:03}' for n in range(count)],
                         '_date': pd.Timestamp('2023-01-02'), '_p1': [f'A{n:03}' for n in range(count)],
                         '_p2': 'Z', '_tournament': 'Test', '_status': 'completed', '_label': 0,
                         'year': 2023, 'p0': .65, 'p1': .35, 'price_0': 2., 'price_1': 1.85,
                         'model_loss': .4, 'market_loss': .7})


def test_preparation_never_needs_bet365_and_excludes_2026(protocol):
    raw = raw_frame()
    a, quality = prepare(raw, protocol)
    b, _ = prepare(raw.assign(B365_1=999, B365_2=-1, log_rank_diff=1e99, elo_diff=1e99), protocol)
    pd.testing.assert_frame_equal(a, b)
    assert a['_date'].dt.year.max() == 2025
    assert not any('B365' in c or 'Pinnacle' in c for c in a)
    assert quality['no_bet365_columns']
    assert not any('B365' in c for c in SOURCE_COLUMNS)


def test_no_same_day_or_future_outcomes_enter_features(protocol):
    raw = raw_frame().iloc[:4].copy()
    raw['_date'] = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-02', '2020-01-03'])
    raw.loc[2, '_p2'] = 'Gamma G.'
    a, _ = prepare(raw, protocol)
    changed = raw.copy()
    mask = changed['_date'] >= pd.Timestamp('2020-01-02')
    changed.loc[mask, '_label'] = 1-changed.loc[mask, '_label']
    changed.loc[mask, '_status'] = 'retired'
    b, _ = prepare(changed, protocol)
    pd.testing.assert_frame_equal(a.loc[a['_date'] <= '2020-01-02', LAGGED],
                                  b.loc[b['_date'] <= '2020-01-02', LAGGED])


def test_invalid_prices_do_not_remove_sporting_state_updates(protocol):
    raw = raw_frame().iloc[:5].copy()
    raw.loc[1, ['Pinnacle_1', 'Pinnacle_2']] = -1
    prepared, quality = prepare(raw, protocol)
    assert len(prepared) == 4 and quality['rebuilt_rows'] == 5
    assert quality['excluded_price_rows'] == 1
    raw.loc[1, '_label'] = 1-raw.loc[1, '_label']
    changed, _ = prepare(raw, protocol)
    assert not np.array_equal(prepared[LAGGED].to_numpy(), changed[LAGGED].to_numpy())


def test_pair_swapping_is_symmetric_and_only_explicit_inputs_used(protocol):
    frame, _ = prepare(raw_frame(), protocol)
    swapped = frame.copy()
    swapped['price_0'], swapped['price_1'] = frame['price_1'], frame['price_0']
    swapped[LAGGED] = -frame[LAGGED]
    for candidate in CANDIDATES:
        x, q, _, _, signs = arrays(frame, candidate)
        sx, sq, _, _, _ = arrays(swapped, candidate)
        np.testing.assert_allclose(sx, x*signs, atol=1e-12)
        np.testing.assert_allclose(sq, q[:, ::-1])
        poisoned = frame.assign(B365_1=1.01, B365_2=999, _label=0, Max_1=500, log_rank_diff=500)
        np.testing.assert_array_equal(x, arrays(poisoned, candidate)[0])
    assert arrays(frame, 'price_only')[0].shape[1] == 3
    assert arrays(frame, 'price_and_history')[0].shape[1] == 11


def test_yearly_training_never_reads_current_or_future_year_outcomes(protocol):
    frame, _ = prepare(raw_frame(), protocol)
    for candidate in CANDIDATES:
        a, fold = predict_fold(frame, candidate, 2023, protocol)
        changed = frame.copy()
        mask = changed['_date'].dt.year >= 2023
        changed.loc[mask, '_label'] = 1-changed.loc[mask, '_label']
        changed.loc[mask, '_status'] = 'retired'
        b, _ = predict_fold(changed, candidate, 2023, protocol)
        np.testing.assert_array_equal(a[['p0', 'p1']], b[['p0', 'p1']])
        assert pd.Timestamp(fold['train_max_date']) < pd.Timestamp(fold['cutoff_exclusive'])


def test_sequential_cap_does_not_use_number_or_results_of_later_bets(protocol):
    rule = protocol['selection']
    a = ledger(predictions(12), rule)
    b = ledger(predictions(12).assign(_label=1), rule)
    earlier = ledger(predictions(3), rule)
    assert len(a) == 8 and a['stake_cash'].sum() == 20
    assert a.attrs['skipped_daily_budget'] == 4
    np.testing.assert_array_equal(a['stake_cash'], b['stake_cash'])
    np.testing.assert_array_equal(a['stake_cash'].iloc[:3], earlier['stake_cash'])
    assert a['bankroll_before'].nunique() == 1


def test_void_bets_reserve_daily_budget_without_profit(protocol):
    rows = predictions(12)
    rows.loc[:7, '_status'] = 'retired'
    bets = ledger(rows, protocol['selection'])
    assert len(bets) == 8
    assert bets['profit_cash'].sum() == 0
    assert bets['stake_cash'].sum() == 20
    assert bets.attrs['final_bankroll'] == 1000
    assert not bets['settled'].any()


def test_next_day_bankroll_changes_only_after_prior_day_settlement(protocol):
    rows = predictions(2)
    rows.loc[1, '_date'] += pd.Timedelta(days=1)
    rows.loc[0, '_label'] = 1
    bets = ledger(rows, protocol['selection'])
    assert bets['bankroll_before'].tolist() == [1000, 997.5]
    assert bets['stake_cash'].tolist() == [2.5, 2.49]


def test_empty_signal_summary_is_valid_and_rejected(protocol):
    rows = predictions(2).assign(p0=.5, p1=.5)
    summary, bets = summarise(rows, protocol)
    assert bets.empty and summary['roi']['0.02'] is None
    assert summary['uncertainty']['ci95'] == [None, None]
    assert not all(gate(summary, False).values())


def test_duplicate_identity_rejected(protocol):
    raw = raw_frame().iloc[:2].copy()
    raw.loc[1, '_date'] = raw.loc[0, '_date']
    with pytest.raises(ValueError, match='identity'):
        prepare(raw, protocol)


def test_gate_cannot_rescue_failed_tuning():
    summary = {'settled': 500, 'roi': {'0.02': .2, '0.05': .1},
               'uncertainty': {'family_lower': .01}, 'roi_without_best_year': .02,
               'yearly': {'2023': {'roi': .1}, '2024': {'roi': .2}}}
    assert all(gate(summary, True).values())
    assert not all(gate(summary, False).values())
