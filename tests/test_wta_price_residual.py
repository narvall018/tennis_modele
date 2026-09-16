import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtesting.wta_price_residual import (
    evaluate, features, fit_symmetric, ledger, predict_symmetric, prepare, select, walk_forward,
    rebuild_daily_features,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def protocol():
    return json.loads((ROOT / "models/wta_price_residual/protocol.json").read_text())


def sample_frame():
    rng = np.random.default_rng(11)
    rows = []
    for year in [2014, 2015, 2016, 2017, 2018]:
        for index in range(40):
            q = rng.uniform(.25, .75)
            sharp = np.clip(q + rng.normal(0, .03), .15, .85)
            row = {"_source_row_id": f"{year}-{index}", "_date": pd.Timestamp(year, 2, 1) + pd.Timedelta(days=index),
                   "_label": int(rng.random() < q), "_status": "completed", "_p1": "A", "_p2": "B", "_surface": "Hard",
                   "B365_1": 1 / (q * 1.06), "B365_2": 1 / ((1 - q) * 1.06),
                   "Pinnacle_1": 1 / (sharp * 1.03), "Pinnacle_2": 1 / ((1 - sharp) * 1.03)}
            for column in ["elo_diff", "surface_elo_diff", "form_10_diff", "rest_diff", "fatigue_diff", "log_rank_diff"]:
                row[column] = rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


def test_swap_inverts_every_feature_and_probability(protocol):
    frame = sample_frame()
    swapped = frame.copy()
    for first, second in protocol["required_pairs"]:
        swapped[first], swapped[second] = frame[second], frame[first]
    for name in protocol["lagged_form_features"]:
        swapped[name] = -frame[name]
    for candidate in protocol["candidates"]:
        x, q, _, _ = features(frame, candidate, protocol)
        sx, sq, _, _ = features(swapped, candidate, protocol)
        np.testing.assert_allclose(sx, -x, atol=1e-13)
        model = fit_symmetric(x, q, 1-frame["_label"].to_numpy(), .01)
        p = predict_symmetric(model, x, q)
        sp = predict_symmetric(model, sx, sq)
        np.testing.assert_allclose(sp, p[:, ::-1], atol=1e-13)


def test_current_and_future_results_do_not_change_earlier_predictions(protocol):
    protocol.update(test_years=[2017, 2018], minimum_training_completed=100)
    frame, _ = prepare(sample_frame(), protocol)
    original, _ = walk_forward(frame, protocol)
    changed = frame.copy()
    changed.loc[changed["_date"].dt.year >= 2017, "_label"] = 1-changed.loc[changed["_date"].dt.year >= 2017, "_label"]
    changed.loc[changed["_date"].dt.year >= 2017, "_status"] = "retired"
    # The second fold needs a minimum reduced only in this synthetic test.
    newer, _ = walk_forward(changed, protocol)
    a = original[original.year == 2017]
    b = newer[newer.year == 2017]
    np.testing.assert_array_equal(a["p1"], b["p1"])
    np.testing.assert_array_equal(a["selected"], b["selected"])


def test_maximum_and_average_odds_never_enter_inputs(protocol):
    frame = sample_frame()
    changed = frame.assign(Avg_1=1000, Avg_2=1.001, Max_1=100, model_probability_p1=.99, _label=1)
    for candidate in protocol["candidates"]:
        np.testing.assert_array_equal(features(frame, candidate, protocol)[0], features(changed, candidate, protocol)[0])


def test_same_day_stakes_are_unaffected_by_same_day_results(protocol):
    frame = pd.DataFrame({"selected": [0]*12, "_date": [pd.Timestamp("2020-01-01")]*12,
                          "odds": [2.0]*12, "probability": [.65]*12, "_status": ["completed"]*12,
                          "won": [True]*12})
    original = ledger(frame, "quarter_kelly", protocol["selection"])
    changed = ledger(frame.assign(won=False), "quarter_kelly", protocol["selection"])
    np.testing.assert_array_equal(original["stake_cash"], changed["stake_cash"])
    assert original["stake_cash"].sum() == pytest.approx(20)
    assert original["bankroll_before"].nunique() == 1


def test_voids_use_exposure_but_no_profit_or_loss(protocol):
    frame = pd.DataFrame({"selected": [0], "_date": [pd.Timestamp("2020-01-01")],
                          "odds": [2.0], "probability": [.65], "_status": ["retired"], "won": [False]})
    bets = ledger(frame, "flat", protocol["selection"])
    assert bets["stake_cash"].iloc[0] > 0
    assert bets["profit_cash"].iloc[0] == 0
    assert bets.attrs["final_bankroll"] == 1000
    for haircut in [0.0, 0.02, 0.05]:
        assert bets[f"return_{haircut}"].iloc[0] == 0


def test_selection_uses_actual_payout_after_haircut(protocol):
    # A nominal edge of 2.2% becomes less than 2% after the frozen haircut.
    probability = np.array([[.511, .489], [.65, .35]])
    odds = np.array([[2.0, 1.9], [2.0, 1.9]])
    np.testing.assert_array_equal(select(probability, odds, protocol["selection"]), [-1, 0])


def test_source_validation_keeps_retirements_and_rejects_partial_pairs(protocol):
    frame = sample_frame().iloc[:3].copy()
    frame.loc[1, "_status"] = "retired"
    frame.loc[2, "Pinnacle_2"] = np.nan
    good, _ = prepare(frame, protocol)
    assert len(good) == 2
    assert good["_status"].eq("retired").sum() == 1


def test_daily_rebuild_is_independent_of_same_day_outcomes_and_status():
    frame = sample_frame().iloc[:4].copy()
    frame["_round"] = "1st Round"
    frame["_series"] = "International"
    frame["_tournament"] = "Example"
    frame["_date"] = pd.to_datetime(["2014-01-01", "2014-02-01", "2014-02-01", "2014-02-02"])
    # Deliberately repeat both players in two matches on a single day.
    baseline = rebuild_daily_features(frame)
    changed = frame.copy()
    changed.loc[[1, 2], "_label"] = 1-changed.loc[[1, 2], "_label"]
    changed.loc[1, "_status"] = "retired"
    rebuilt = rebuild_daily_features(changed)
    columns = ["elo_diff", "surface_elo_diff", "form_10_diff", "rest_diff", "fatigue_diff"]
    np.testing.assert_array_equal(baseline.loc[[1, 2], columns], rebuilt.loc[[1, 2], columns])
    np.testing.assert_array_equal(baseline.loc[1, columns], baseline.loc[2, columns])


def test_daily_rebuild_does_not_use_future_outcomes():
    frame = sample_frame().iloc[:5].copy()
    frame["_round"] = "1st Round"
    frame["_series"] = "International"
    frame["_tournament"] = "Example"
    a = rebuild_daily_features(frame)
    frame.loc[4, "_label"] = 1-frame.loc[4, "_label"]
    b = rebuild_daily_features(frame)
    np.testing.assert_array_equal(a["elo_diff"], b["elo_diff"])
    np.testing.assert_array_equal(a["form_10_diff"], b["form_10_diff"])
