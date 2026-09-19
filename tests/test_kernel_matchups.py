import numpy as np
import pandas as pd
import pytest

from src.backtesting.kernel_matchups import KernelResidual, predict_year, evaluate, arrays


@pytest.fixture
def protocol():
    return {'random_features': 12, 'random_seed': 20260919, 'lookback_years': 6,
            'embargo_days': 7, 'half_life_years': 3.,
            'sports': {'football': {'min_train': 10}},
            'selection': {'minimum_ev_after_haircut': .03, 'profit_haircut': .02,
                          'odds_min': 1.3, 'odds_max': 5., 'flat_fraction': .0025,
                          'maximum_daily_exposure': .02, 'sensitivity_haircuts': [0., .02, .05]},
            'uncertainty': {'samples': 100, 'seed': 4, 'lower_quantile_per_candidate': .05/6}}


@pytest.mark.parametrize('classes', [2, 3])
def test_weighted_analytic_gradient(classes):
    rng = np.random.default_rng(8)
    features = rng.normal(size=(15, 6)); q = np.full((15, classes), 1/classes)
    labels = rng.integers(classes, size=15); weights = rng.uniform(.1, 1., 15)
    coefficient = rng.normal(size=6*(classes-1))*.1
    _, gradient = KernelResidual.objective(coefficient, features, q, labels, weights, .01)
    numeric = []
    for i in range(len(coefficient)):
        delta = np.eye(len(coefficient))[i]*1e-6
        numeric.append((KernelResidual.objective(coefficient+delta, features, q, labels, weights, .01)[0]
                        - KernelResidual.objective(coefficient-delta, features, q, labels, weights, .01)[0])/2e-6)
    np.testing.assert_allclose(gradient, numeric, atol=1e-8)


def test_binary_symmetry_and_train_only_scaler():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 3)); q = np.tile([.6, .4], (80, 1)); y = rng.integers(2, size=80)
    signs = np.array([-1, -1, 1])
    model = KernelResidual(.01, 16, 2, signs).fit(x, q, y, np.ones(80))
    np.testing.assert_allclose(model.mean[:2], 0, atol=1e-15)
    assert model.mean[2] == pytest.approx(x[:, 2].mean())
    initial_mean, initial_scale = model.mean.copy(), model.scale.copy()
    p = model.predict(x, q)
    mirrored = model.predict(x*signs, q[:, ::-1])
    np.testing.assert_allclose(p, mirrored[:, ::-1], atol=1e-12)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    model.predict(x*1e6, q)
    np.testing.assert_array_equal(model.mean, initial_mean)
    np.testing.assert_array_equal(model.scale, initial_scale)
    repeated = KernelResidual(.01, 16, 2, signs).fit(x, q, y, np.ones(80))
    np.testing.assert_array_equal(p, repeated.predict(x, q))


def test_zero_correction_is_market_and_invalid_input_rejected():
    x = np.arange(20.).reshape(10, 2); q = np.tile([.4, .35, .25], (10, 1))
    model = KernelResidual(.01, 8).fit(x, q, np.arange(10)%3, np.ones(10))
    model.coefficients[:] = 0
    np.testing.assert_allclose(model.predict(x, q), q)
    with pytest.raises(ValueError, match='Invalid training'):
        KernelResidual(.01).fit(x, q, np.ones(10)*3, np.ones(10))


def test_future_outcomes_embargo_and_rolling_window(protocol):
    dates = pd.date_range('2014-01-01', '2021-12-01', freq='60D')
    n = len(dates); rng = np.random.default_rng(9)
    frame = pd.DataFrame({'_date': dates, '_label': np.arange(n)%3, '_status': 'completed',
        '_source_row_id': np.arange(n), '_p1': 'A', '_p2': 'B', '_tournament': 'L',
        'price_0': 2.5, 'price_1': 3.2, 'price_2': 3.2})
    x = rng.normal(size=(n, 4)); q = np.tile([.4, .3, .3], (n, 1))
    matrices = (x, q, ['a', 'b', 'c', 'd'], None)
    first, audit = predict_year(frame, matrices, 'football', 'kernel_flexible', 2021, protocol)
    changed = frame.copy(); future = changed._date >= '2020-12-25'
    changed.loc[future, '_label'] = (changed.loc[future, '_label']+1)%3
    second, _ = predict_year(changed, matrices, 'football', 'kernel_flexible', 2021, protocol)
    np.testing.assert_array_equal(first[['p0', 'p1', 'p2']], second[['p0', 'p1', 'p2']])
    assert pd.Timestamp(audit['train_max']) < pd.Timestamp('2020-12-25')
    assert pd.Timestamp(audit['train_min']) >= pd.Timestamp('2014-12-25')


def test_1x2_settlement_void_and_budget(protocol):
    n = 10
    rows = pd.DataFrame({'_source_row_id': np.arange(n), '_date': pd.Timestamp('2023-01-01'),
        '_p1': 'A', '_p2': 'B', '_tournament': 'L', '_label': 1, '_status': 'completed', 'year': 2023,
        'p0': .1, 'p1': .8, 'p2': .1, 'price_0': 2.5, 'price_1': 3.2, 'price_2': 3.2,
        'model_loss': .5, 'market_loss': .6})
    rows.loc[0, '_status'] = 'retired'
    summary, bets = evaluate(rows, protocol)
    assert len(bets) == 8 and summary['settled'] == 7
    assert bets.selected_original_outcome.eq(1).all()
    assert bets.stake_cash.eq(2.5).all() and bets.bankroll_before.eq(1000).all()
    assert bets.iloc[0]['profit_cash'] == 0
    np.testing.assert_allclose(bets.iloc[1:]['return_0.02'], 2.2*.98)
    assert summary['bankroll']['skipped_daily_budget'] == 2


def test_wta_arrays_ignore_current_results():
    row = {'price_0': 1.8, 'price_1': 2.1, 'Surface': 'Clay', 'history_count_1': 10, 'history_count_2': 15}
    row.update({p+'_'+s: .2 for p in ['global', 'surface'] for s in ['serve', 'return', 'volume']})
    frame = pd.DataFrame([row]); before = arrays(frame, 'wta')
    frame['_label'] = 999; frame['Winner'] = 'POISON'; frame['Score'] = 'POISON'; frame['Pinnacle_1'] = 99
    after = arrays(frame, 'wta')
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2][-1] == 'log_mean_history' and len(before[3]) == before[0].shape[1]
