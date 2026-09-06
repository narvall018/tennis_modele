"""Phase 3: ablation pre-enregistree des classements UFC pre-combat."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .challenger_pipeline import STRUCTURAL_FEATURES, add_engineered_features, bet_summary, probability_metrics
from .data_pipeline import sha256_file
from .model_pipeline import build_features, make_bets


RANK_FEATURES = [
    "division_rank_points_diff",
    "division_rank_known_count",
    "p4p_rank_points_diff",
    "p4p_rank_known_count",
]


def build_phase3_features(base_dir: Path) -> pd.DataFrame:
    features = add_engineered_features(build_features(base_dir))
    rankings = pd.read_parquet(
        base_dir / "data" / "rigorous" / "processed" / "prefight_rankings.parquet"
    )
    keep = ["fight_id", *RANK_FEATURES]
    merged = features.merge(rankings[keep], on="fight_id", how="left", validate="one_to_one")
    merged[RANK_FEATURES] = merged[RANK_FEATURES].fillna(0.0)
    return merged


def _estimator(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=c_value, max_iter=3000, random_state=42)),
        ]
    )


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    c_value: float,
) -> np.ndarray:
    model = _estimator(c_value)
    model.fit(train[columns], train["y"].astype(int))
    return model.predict_proba(test[columns])[:, 1]


def walk_forward_ablation(
    data: pd.DataFrame,
    first_year: int,
    last_year: int,
    minimum_training_rows: int,
    c_value: float,
) -> pd.DataFrame:
    eligible = data.dropna(subset=["y", "market_logit", "odds_1", "odds_2"]).copy()
    rows: list[pd.DataFrame] = []
    rank_columns = [*STRUCTURAL_FEATURES, *RANK_FEATURES]
    for year in range(first_year, last_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train = eligible[
            (eligible["event_date"] >= "2010-01-01") & (eligible["event_date"] < cutoff)
        ]
        test = eligible[
            (eligible["event_date"] >= cutoff)
            & (eligible["event_date"] < cutoff + pd.DateOffset(years=1))
        ]
        if len(train) < minimum_training_rows or test.empty:
            continue
        batch = test.copy()
        batch["p_baseline"] = _fit_predict(train, test, STRUCTURAL_FEATURES, c_value)
        batch["p_rank_challenger"] = _fit_predict(train, test, rank_columns, c_value)
        rows.append(batch)
    return pd.concat(rows, ignore_index=True)


def _yearly_comparison(predictions: pd.DataFrame) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for year, frame in predictions.groupby(predictions["event_date"].dt.year):
        y = frame["y"].astype(int)
        baseline = log_loss(y, frame["p_baseline"].clip(1e-6, 1 - 1e-6))
        challenger = log_loss(y, frame["p_rank_challenger"].clip(1e-6, 1 - 1e-6))
        market = log_loss(y, frame["market_p1"].clip(1e-6, 1 - 1e-6))
        comparison[str(year)] = {
            "n": int(len(frame)),
            "baseline_log_loss": float(baseline),
            "rank_challenger_log_loss": float(challenger),
            "market_log_loss": float(market),
            "challenger_minus_baseline": float(challenger - baseline),
        }
    return comparison


def run_phase3_research(base_dir: Path) -> dict[str, Any]:
    protocol_path = base_dir / "phase3_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    processed = base_dir / "data" / "rigorous" / "processed"
    reports = base_dir / "data" / "rigorous" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    features = build_phase3_features(base_dir)
    features.to_parquet(processed / "phase3_features.parquet", index=False)
    walk = protocol["walk_forward"]
    predictions = walk_forward_ablation(
        features,
        int(walk["first_test_year"]),
        int(walk["last_test_year"]),
        int(walk["minimum_training_rows"]),
        float(protocol["challenger"]["C"]),
    )
    start, end = map(pd.Timestamp, protocol["development_window"])
    predictions = predictions[predictions["event_date"].between(start, end)].copy()
    baseline_metrics = probability_metrics(
        predictions.rename(columns={"p_baseline": "p_model"})
    )
    challenger_metrics = probability_metrics(
        predictions.rename(columns={"p_rank_challenger": "p_model"})
    )
    yearly = _yearly_comparison(predictions)

    economic_predictions = predictions.copy()
    economic_predictions["p_model"] = economic_predictions["p_rank_challenger"]
    rule = protocol["fixed_bet_rule"]
    bets = make_bets(
        economic_predictions,
        float(rule["edge_threshold"]),
        float(rule["minimum_decimal_odds"]),
        float(rule["maximum_decimal_odds"]),
    )
    gate = protocol["gate_before_pristine_holdout"]
    economic = bet_summary(
        bets,
        protocol["staking"],
        samples=int(gate["bootstrap_samples"]),
        confidence=float(gate["cluster_bootstrap_confidence"]),
    )
    years_beating = sum(
        item["rank_challenger_log_loss"] < item["baseline_log_loss"] for item in yearly.values()
    )
    improvement = baseline_metrics["log_loss"] - challenger_metrics["log_loss"]
    ci_key = f"roi_bootstrap_{float(gate['cluster_bootstrap_confidence']):g}"
    checks = {
        "challenger_log_loss_beats_baseline": challenger_metrics["log_loss"] < baseline_metrics["log_loss"],
        "challenger_log_loss_beats_market": challenger_metrics["log_loss"] < challenger_metrics["market_log_loss"],
        "minimum_log_loss_improvement_vs_baseline": improvement >= float(gate["minimum_log_loss_improvement_vs_baseline"]),
        "years_beating_baseline": years_beating >= int(gate["years_beating_baseline_minimum"]),
        "minimum_bets": economic["bets"] >= int(gate["minimum_bets"]),
        "positive_roi": economic["roi"] > 0,
        "positive_years": economic["positive_years"] >= int(gate["positive_years_minimum"]),
        "maximum_drawdown": economic["max_drawdown"] <= float(gate["maximum_drawdown"]),
        "positive_99pct_cluster_bootstrap_lower_bound": economic[ci_key]["low"] > float(gate["roi_lower_bound_must_exceed"]),
    }
    approved = all(checks.values())
    ranking_coverage = {
        "fights": int(len(predictions)),
        "at_least_one_division_rank": int(predictions["division_rank_known_count"].gt(0).sum()),
        "two_division_ranks": int(predictions["division_rank_known_count"].eq(2).sum()),
    }
    report = {
        "protocol": protocol,
        "disclosure": "Development ablation only; 2015-2024 is already research-exposed.",
        "ranking_coverage": ranking_coverage,
        "baseline_probability": baseline_metrics,
        "rank_challenger_probability": challenger_metrics,
        "log_loss_improvement_vs_baseline": float(improvement),
        "yearly_probability_comparison": yearly,
        "years_beating_baseline": int(years_beating),
        "fixed_rule_economic_diagnostic": economic,
        "gate_checks": checks,
        "approved_for_pristine_holdout": approved,
    }
    lock = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol_path),
        "approved_for_pristine_holdout": approved,
        "gate_checks": checks,
        "status": (
            "PHASE3_VALIDATED_PENDING_HOLDOUT"
            if approved else "PHASE3_REJECTED_NO_BET"
        ),
    }
    (reports / "phase3_research_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    (reports / "phase3_locked_strategy.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    predictions.to_parquet(processed / "phase3_oos_predictions_pre_holdout.parquet", index=False)
    bets.to_parquet(reports / "phase3_development_bets.parquet", index=False)
    return report


def run_phase3_holdout(base_dir: Path) -> dict[str, Any]:
    """Garde physique: le holdout reste ferme tant que chaque gate n'est pas vrai."""
    protocol_path = base_dir / "phase3_protocol.json"
    reports = base_dir / "data" / "rigorous" / "reports"
    lock = json.loads((reports / "phase3_locked_strategy.json").read_text())
    if lock["protocol_sha256"] != sha256_file(protocol_path):
        raise RuntimeError("Le protocole phase 3 a change apres verrouillage")
    if lock["approved_for_pristine_holdout"]:
        result = {
            "status": "GATE_PASSED_BUT_HOLDOUT_REQUIRES_SEPARATE_IMPLEMENTATION_REVIEW",
            "reason": "Fail closed: no automatic economic holdout opening in the research command.",
        }
    else:
        result = {"status": "NOT_OPENED_PHASE3_DEVELOPMENT_GATE_FAILED"}
    (reports / "phase3_holdout_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    return result
