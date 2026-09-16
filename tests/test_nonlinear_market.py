import numpy as np
import pandas as pd
import pytest

from src.backtesting.nonlinear_market import (
    OffsetTrees, blend_market, candidate_predictions, market_margin, recency_weights, repair_atp_context,
)


def parameters():
    return {'tree_method': 'hist', 'max_depth': 2, 'eta': .1,
            'nthread': 1, 'base_score': .5, 'seed': 9}


@pytest.mark.parametrize('classes', [2, 3])
def test_zero_trees_reproduce_market_and_offset_is_used_at_prediction(classes):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(40, 3))
    q = rng.dirichlet(np.ones(classes), size=40)
    model = OffsetTrees(parameters(), 0, [-1, -1, 1])
    model.fit(x, q, np.arange(40) % classes, np.ones(40))
    np.testing.assert_allclose(model.predict(x, q), q, atol=1e-7)
    changed = np.roll(q, 1, axis=1)
    np.testing.assert_allclose(model.predict(x, changed), changed, atol=1e-7)


def test_binary_swap_preserves_context_and_reverses_probabilities():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(100, 4))
    q = rng.dirichlet([3, 3], size=100)
    signs = np.array([-1, -1, 1, 1])
    model = OffsetTrees(parameters(), 12, signs)
    model.fit(x, q, (x[:, 0] < 0).astype(int), np.ones(100))
    p = model.predict(x, q)
    swapped = model.predict(x * signs, q[:, ::-1])
    np.testing.assert_allclose(p, swapped[:, ::-1], atol=1e-12)
    assert (p > 0).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)


def test_recency_is_past_only_and_half_life_is_exact():
    cutoff = pd.Timestamp('2021-01-01')
    dates = [cutoff - pd.Timedelta(days=365.25), cutoff - pd.Timedelta(days=4 * 365.25)]
    w = recency_weights(dates, cutoff, 3)
    assert w[0] == pytest.approx(2 * w[1])
    assert w.mean() == pytest.approx(1)
    np.testing.assert_equal(recency_weights(dates, cutoff, None), [1, 1])
    with pytest.raises(ValueError):
        recency_weights([cutoff], cutoff, 3)
    with pytest.raises(ValueError):
        recency_weights([cutoff + pd.Timedelta(days=1)], cutoff, None)
    with pytest.raises(ValueError):
        recency_weights(dates, cutoff, 0)


def test_shrinkage_is_fixed_convex_mixture():
    q, p = np.array([[.4, .6]]), np.array([[.8, .2]])
    np.testing.assert_allclose(blend_market(p, q, .5), [[.6, .4]])
    np.testing.assert_allclose(blend_market(p, q, 0), q)
    with pytest.raises(ValueError):
        blend_market(p, q, 1.1)


def test_evaluation_labels_and_status_only_change_scoring_not_predictions():
    rows = pd.DataFrame({'p0': [.7, .4], 'p1': [.3, .6], 'q0': [.6, .5], 'q1': [.4, .5],
                         'price_0': [1.6, 2.0], 'price_1': [2.2, 1.9],
                         '_label': [0, 1], '_status': ['completed', 'completed']})
    original = candidate_predictions(rows, 'half', .5)
    rows['_label'] = 1 - rows['_label']
    rows['_status'] = 'void'
    altered = candidate_predictions(rows, 'half', .5)
    np.testing.assert_equal(original[['p0', 'p1']].to_numpy(), altered[['p0', 'p1']].to_numpy())
    assert altered['model_loss'].isna().all()


def test_invalid_market_is_rejected():
    for q in [np.array([[0., 1.]]), np.array([[.8, .8]]), np.array([[np.nan, .5]])]:
        with pytest.raises(ValueError):
            market_margin(q)


def test_atp_context_join_is_exact_and_uses_no_results():
    frame = pd.DataFrame({'_source_row_id': ['b', 'a'], 'is_indoor': [0., 0.],
                          'round_progress': [3 / 7, 3 / 7]})
    meta = pd.DataFrame({'match_id': ['a', 'b'], 'indoor': ['Outdoor', 'Indoor'],
                         'round': ['1st Round', 'The Final']})
    out, audit = repair_atp_context(frame, meta)
    np.testing.assert_equal(out['is_indoor'].to_numpy(), [1, 0])
    np.testing.assert_allclose(out['round_progress'], [1, 1 / 7])
    assert audit['is_indoor_rows_changed'] == 1
    assert audit['round_progress_rows_changed'] == 2
    assert frame['is_indoor'].sum() == 0  # Original is unchanged.
    with pytest.raises(ValueError):
        repair_atp_context(frame, pd.concat([meta, meta.iloc[[0]]]))
    with pytest.raises(ValueError):
        repair_atp_context(frame, meta.iloc[[0]])
    meta.loc[0, 'round'] = 'unknown'
    with pytest.raises(ValueError):
        repair_atp_context(frame, meta)


def test_fold_cannot_train_on_current_year_embargo_or_void_rows(monkeypatch):
    from src.backtesting import nonlinear_market as module
    dates = pd.to_datetime(['2019-01-01', '2020-06-01', '2020-12-24',
                            '2020-12-25', '2020-01-01', '2021-03-01', '2022-01-01'])
    frame = pd.DataFrame({'_date': dates, '_label': [0, 1, 0, 1, 1, 0, 1],
                          '_status': ['completed'] * 4 + ['void', 'completed', 'completed'],
                          '_source_row_id': list('abcdefg'), 'x': np.arange(7, dtype=float)})
    def fake_inputs(f, sport, protocol):
        q = np.tile([.6, .4], (len(f), 1))
        return f[['x']].to_numpy(), q, 1 / q, ['x'], np.array([-1])
    monkeypatch.setattr(module, 'inputs', fake_inputs)
    protocol = {'embargo_days': 7, 'sports': {'atp': {'min_train': 3}},
                'booster': parameters(), 'boost_rounds': 5}
    before, fold = module.predict_fold(frame, 'atp', 2021, 3, protocol)
    assert fold['train_rows'] == 3
    assert fold['train_max'] == '2020-12-24'
    frame.loc[[3, 4, 5, 6], '_label'] = 1 - frame.loc[[3, 4, 5, 6], '_label']
    after, _ = module.predict_fold(frame, 'atp', 2021, 3, protocol)
    np.testing.assert_equal(before[['p0', 'p1']].to_numpy(), after[['p0', 'p1']].to_numpy())
