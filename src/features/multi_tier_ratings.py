"""Ratings built from every match a player actually played, not just the priced ones.

The ATP studies in this repository rate players using main-tour matches alone,
because that is the only table that carries odds. A player's rating therefore
ignores the Challenger circuit and the qualifying draws — which is precisely the
tennis played by qualifiers, by players moving up, and by anyone returning from
injury. Those are the entries a main-tour-only rating knows least about, and they
are over-represented in early rounds.

This module feeds the same audited Elo engine the 121 373 Challenger and 29 739
qualifying matches collected by ``scripts/update_tennis_expansion.py``, then
reads the ratings back for main-tour matches only. Challenger and qualifying
matches can never be bet on here: they carry no market, and they leave this
module as ratings, never as a betting population.

Two decisions are fixed here in advance rather than tuned:

* **Identity comes from TennisMyLife player ids, never from names.** The tiers
  share one id space, so a player crossing between them is the same player by
  construction. Matching abbreviated main-tour names against full Challenger
  names would fail silently on exactly the low-profile players this is meant to
  help.
* **Lower tiers move ratings less.** A Challenger result is real evidence but
  weaker evidence than a main-draw one, so it carries a smaller weight. The
  values below were written before any comparison was run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.elo_system import TOURNAMENT_WEIGHTS, TennisEloEngine


# Series labels used only inside this module, so the tier is visible to the
# engine's K-factor without disturbing the labels the legacy tables use.
CHALLENGER_SERIES = "Challenger"
QUALIFYING_SERIES = "Qualifying"

# Declared before measuring anything. Main-draw weights already live in
# TOURNAMENT_WEIGHTS and range from 0.9 to 1.5.
TIER_WEIGHTS = {
    CHALLENGER_SERIES: 0.7,
    QUALIFYING_SERIES: 0.6,
}

LEVEL_TO_SERIES = {
    "G": "Grand Slam",
    "F": "Masters Cup",
    "M": "Masters 1000",
    "500": "ATP500",
    "250": "ATP250",
    "A": "ATP250",
    "D": "International",
    "O": "International",
    "C": CHALLENGER_SERIES,
}

ROUND_TO_LEGACY = {
    "R128": "1st Round", "R64": "1st Round", "R32": "2nd Round", "R16": "3rd Round",
    "QF": "Quarterfinals", "SF": "Semifinals", "F": "The Final", "RR": "Round Robin",
    "BR": "Bronze Medal", "Q1": "1st Round", "Q2": "2nd Round", "Q3": "3rd Round",
}


def register_tier_weights() -> None:
    """Make the engine aware of the lower tiers.

    ``TOURNAMENT_WEIGHTS.get(series, 1.0)`` would otherwise treat a Challenger
    match as an ATP250, which would overstate what a Challenger win proves.
    """
    TOURNAMENT_WEIGHTS.update(TIER_WEIGHTS)


def _to_engine_frame(frame: pd.DataFrame, segment: str) -> pd.DataFrame:
    """Map one published table onto the columns ``TennisEloEngine.fit`` reads."""
    result = pd.DataFrame(
        {
            "Date": pd.to_datetime(frame["match_date"], errors="coerce"),
            "Player_1": frame["player_1_id"].astype(str),
            "Player_2": frame["player_2_id"].astype(str),
            "Tournament": frame["tourney_name"].astype(str),
            "Surface": frame["surface"].fillna("Hard").astype(str),
            "Round": frame["round"].map(ROUND_TO_LEGACY).fillna("1st Round"),
            "Status": frame["match_status"].astype(str),
            "Best_of": pd.to_numeric(frame.get("best_of"), errors="coerce").fillna(3),
            "Rank_1": pd.to_numeric(frame.get("player_1_rank"), errors="coerce"),
            "Rank_2": pd.to_numeric(frame.get("player_2_rank"), errors="coerce"),
            "Pts_1": pd.to_numeric(frame.get("player_1_rank_points"), errors="coerce"),
            "Pts_2": pd.to_numeric(frame.get("player_2_rank_points"), errors="coerce"),
            "odds_p1": pd.to_numeric(frame.get("player_1_odds"), errors="coerce"),
            "odds_p2": pd.to_numeric(frame.get("player_2_odds"), errors="coerce"),
        }
    )
    series = frame["tourney_level"].astype(str).map(LEVEL_TO_SERIES).fillna("ATP250")
    if segment == "qualifying":
        series = pd.Series(QUALIFYING_SERIES, index=frame.index)
    result["Series"] = series.to_numpy()
    result["segment"] = segment
    result["match_id"] = frame["match_id"].astype(str).to_numpy()
    # Winner as an id, taken from the published winner column rather than from
    # any ordering convention.
    result["Winner"] = frame["winner_id"].astype(str).to_numpy()
    return result.dropna(subset=["Date"])


def build_rating_input(
    main: pd.DataFrame,
    unpriced: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Chronological match stream for the rating pass, keyed by player id."""
    frames = [_to_engine_frame(main, "main")]
    if unpriced is not None and len(unpriced):
        for segment, group in unpriced.groupby("segment"):
            frames.append(_to_engine_frame(group.reset_index(drop=True), str(segment)))
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates("match_id", keep="first")
    combined["source_row_id"] = np.arange(len(combined), dtype=np.int64)
    return combined.sort_values(["Date", "Tournament", "Player_1"], kind="mergesort").reset_index(
        drop=True
    )


def fit_ratings(stream: pd.DataFrame, progress=None) -> pd.DataFrame:
    """Run the audited engine and return its pre-match history for every row."""
    register_tier_weights()
    engine = TennisEloEngine()
    engine.fit(stream, progress_callback=progress)
    history = engine.get_history()
    keys = stream[["source_row_id", "segment", "match_id"]]
    return history.merge(keys, on="source_row_id", how="left", validate="one_to_one")


def elo_probability(history: pd.DataFrame, surface_weight: float = 0.5) -> np.ndarray:
    """Probability that Player_1 wins, from the global and surface ratings.

    The 50/50 mix of global and surface Elo is the engine's existing convention
    and is not tuned here; both variants are compared on identical inputs, so a
    different mix would move the two arms together.
    """
    global_gap = history["elo_p1"].to_numpy(dtype=float) - history["elo_p2"].to_numpy(dtype=float)
    surface_gap = (
        history["surf_elo_p1"].to_numpy(dtype=float) - history["surf_elo_p2"].to_numpy(dtype=float)
    )
    blended = (1.0 - surface_weight) * global_gap + surface_weight * surface_gap
    return 1.0 / (1.0 + 10.0 ** (-blended / 400.0))


RATING_COLUMNS = [
    "elo_p1", "elo_p2", "surf_elo_p1", "surf_elo_p2",
    "momentum_elo_p1", "momentum_elo_p2",
    "p1_form_5", "p2_form_5", "p1_form_10", "p2_form_10", "p1_form_20", "p2_form_20",
    "p1_matches", "p2_matches",
]


def compare_rating_quality(
    main: pd.DataFrame,
    unpriced: pd.DataFrame,
    progress=None,
    thin_record_threshold: int = 20,
    feature_output: Any = None,
) -> dict[str, Any]:
    """Baseline ratings against tier-enriched ratings, on main-tour matches only.

    ``thin_record_threshold`` is declared in advance: the enrichment is supposed
    to help players the main tour has barely seen, so that subgroup is stated
    before the numbers rather than chosen after them.
    """
    from sklearn.metrics import log_loss, roc_auc_score

    baseline_stream = build_rating_input(main, None)
    enriched_stream = build_rating_input(main, unpriced)
    if progress:
        progress(f"Base: {len(baseline_stream):,} matchs | enrichi: {len(enriched_stream):,}")

    baseline = fit_ratings(baseline_stream, progress)
    enriched = fit_ratings(enriched_stream, progress)

    baseline = baseline[baseline["segment"].eq("main")].copy()
    enriched = enriched[enriched["segment"].eq("main")].copy()
    baseline["p_elo"] = elo_probability(baseline)
    enriched["p_elo"] = elo_probability(enriched)

    merged = baseline[["match_id", "p_elo", "label", "Status", "Date", "odds_p1", "odds_p2"]
                      + RATING_COLUMNS].merge(
        enriched[["match_id", "p_elo"] + RATING_COLUMNS],
        on="match_id",
        suffixes=("_base", "_rich"),
        validate="one_to_one",
    )
    if feature_output is not None:
        merged.to_parquet(feature_output, index=False)
    played = merged["Status"].eq("completed")
    merged = merged[played].copy()
    labels = merged["label"].to_numpy(dtype=int)

    def score(column: str, mask: np.ndarray) -> dict[str, float]:
        probability = np.clip(merged.loc[mask, column].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        target = labels[mask.to_numpy() if isinstance(mask, pd.Series) else mask]
        return {
            "n": int(len(target)),
            "log_loss": float(log_loss(target, probability)),
            "auc": float(roc_auc_score(target, probability)),
        }

    everything = pd.Series(True, index=merged.index)
    thin = (merged["p1_matches_base"] < thin_record_threshold) | (
        merged["p2_matches_base"] < thin_record_threshold
    )
    report: dict[str, Any] = {
        "tier_weights": dict(TIER_WEIGHTS),
        "thin_record_threshold": thin_record_threshold,
        "matches_in_rating_pass": {
            "baseline": int(len(baseline_stream)),
            "enriched": int(len(enriched_stream)),
        },
        "groups": {},
    }
    for name, mask in (("all_main_tour", everything), ("thin_main_tour_record", thin)):
        base_metrics = score("p_elo_base", mask)
        rich_metrics = score("p_elo_rich", mask)
        report["groups"][name] = {
            "baseline": base_metrics,
            "enriched": rich_metrics,
            "log_loss_gain": base_metrics["log_loss"] - rich_metrics["log_loss"],
            "auc_gain": rich_metrics["auc"] - base_metrics["auc"],
        }
    extra = merged["p1_matches_rich"] - merged["p1_matches_base"]
    report["extra_matches_known_per_player"] = {
        "median": float(extra.median()),
        "mean": float(extra.mean()),
        "share_of_matches_with_new_information": float((extra > 0).mean()),
    }
    return report


def _residual_matrix(features: pd.DataFrame, arm: str, market_logit: np.ndarray) -> np.ndarray:
    """Signed rating differences plus the market, in that order.

    The signed block comes first so the symmetric fit can negate it: swapping the
    two players must flip every difference and the market logit together.
    """
    columns = []
    for left, right in (
        ("elo_p1", "elo_p2"),
        ("surf_elo_p1", "surf_elo_p2"),
        ("momentum_elo_p1", "momentum_elo_p2"),
        ("p1_form_5", "p2_form_5"),
        ("p1_form_10", "p2_form_10"),
        ("p1_form_20", "p2_form_20"),
    ):
        columns.append(
            features[f"{left}_{arm}"].to_numpy(dtype=float)
            - features[f"{right}_{arm}"].to_numpy(dtype=float)
        )
    experience = np.log1p(features[[f"p1_matches_{arm}", f"p2_matches_{arm}"]].to_numpy(dtype=float))
    columns.append(experience[:, 0] - experience[:, 1])
    columns.append(market_logit)
    return np.column_stack(columns)


def market_residual_comparison(features: pd.DataFrame, first_test_year: int = 2005) -> dict[str, Any]:
    """Does either rating arm add anything on top of the price?

    Walk-forward with the repository's usual fold rule — train through Y-2,
    calibrate on Y-1, test on Y — so the comparison is out of sample even though
    the underlying ATP years are already spent.
    """
    from scipy.special import logit as logit_fn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from src.backtesting.rigorous_strategy import _fit_symmetric, _predict_symmetric

    frame = features[features["Status"].eq("completed")].copy()
    left = pd.to_numeric(frame["odds_p1"], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(frame["odds_p2"], errors="coerce").to_numpy(dtype=float)
    overround = 1.0 / left + 1.0 / right
    valid = (
        np.isfinite(overround) & (left > 1.0) & (right > 1.0)
        & (overround >= 0.95) & (overround <= 1.25)
    )
    frame = frame[valid].copy()
    market = ((1.0 / left) / overround)[valid]
    market_logit = logit_fn(np.clip(market, 1e-6, 1 - 1e-6))
    labels = frame["label"].to_numpy(dtype=int)
    years = pd.to_datetime(frame["Date"]).dt.year.to_numpy()

    def build():
        return Pipeline([
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.5, max_iter=3000)),
        ])

    results: dict[str, Any] = {
        "priced_completed_matches": int(len(frame)),
        "first_test_year": first_test_year,
        "arms": {},
    }
    test_years = [year for year in sorted(set(years)) if year >= first_test_year]
    reference_mask = np.isin(years, test_years)
    results["market_log_loss"] = float(
        log_loss(labels[reference_mask], np.clip(market[reference_mask], 1e-6, 1 - 1e-6))
    )
    for arm in ("base", "rich"):
        matrix = _residual_matrix(frame, arm, market_logit)
        predictions, truth = [], []
        for year in test_years:
            train = years <= year - 2
            test = years == year
            if train.sum() < 5000 or test.sum() < 200:
                continue
            model = _fit_symmetric(build(), matrix[train], labels[train], matrix.shape[1])
            predictions.append(_predict_symmetric(model, matrix[test], matrix.shape[1]))
            truth.append(labels[test])
        predictions = np.concatenate(predictions)
        truth = np.concatenate(truth)
        scored_years = [
            year for year in test_years
            if (years <= year - 2).sum() >= 5000 and (years == year).sum() >= 200
        ]
        scored = np.isin(years, scored_years)
        results["arms"][arm] = {
            "n": int(len(truth)),
            "log_loss": float(log_loss(truth, np.clip(predictions, 1e-6, 1 - 1e-6))),
            "market_log_loss": float(
                log_loss(labels[scored], np.clip(market[scored], 1e-6, 1 - 1e-6))
            ),
        }
        results["arms"][arm]["gain_vs_market"] = (
            results["arms"][arm]["market_log_loss"] - results["arms"][arm]["log_loss"]
        )
    results["enrichment_gain"] = (
        results["arms"]["base"]["log_loss"] - results["arms"]["rich"]["log_loss"]
    )

    # The subgroup the enrichment is supposed to serve, declared before any of
    # these numbers existed: matches where at least one player has a thin
    # main-tour record and the extra tiers therefore carry most of what is known.
    scored_years = [
        year for year in test_years
        if (years <= year - 2).sum() >= 5000 and (years == year).sum() >= 200
    ]
    scored = np.isin(years, scored_years)
    thin = (
        (frame["p1_matches_base"].to_numpy() < 20) | (frame["p2_matches_base"].to_numpy() < 20)
    ) & scored
    subgroup: dict[str, Any] = {"n": int(thin.sum())}
    if thin.sum() >= 500:
        subgroup["market_log_loss"] = float(
            log_loss(labels[thin], np.clip(market[thin], 1e-6, 1 - 1e-6))
        )
        for arm in ("base", "rich"):
            matrix = _residual_matrix(frame, arm, market_logit)
            predictions, truth, keep = [], [], []
            for year in scored_years:
                train = years <= year - 2
                test = years == year
                model = _fit_symmetric(build(), matrix[train], labels[train], matrix.shape[1])
                predictions.append(_predict_symmetric(model, matrix[test], matrix.shape[1]))
                truth.append(labels[test])
                keep.append(thin[test])
            predictions = np.concatenate(predictions)[np.concatenate(keep)]
            truth = np.concatenate(truth)[np.concatenate(keep)]
            subgroup[arm] = {
                "log_loss": float(log_loss(truth, np.clip(predictions, 1e-6, 1 - 1e-6))),
            }
            subgroup[arm]["gain_vs_market"] = (
                subgroup["market_log_loss"] - subgroup[arm]["log_loss"]
            )
        subgroup["enrichment_gain"] = (
            subgroup["base"]["log_loss"] - subgroup["rich"]["log_loss"]
        )
    results["thin_record_subgroup"] = subgroup
    return results
