"""Phase 2: challengers statistiques/non lineaires contre le marche UFC."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .data_pipeline import sha256_file
from .model_pipeline import (
    FULL_FEATURES,
    bootstrap_roi,
    build_features,
    make_bets,
    simulate_bankroll,
)


STRUCTURAL_FEATURES = [
    "market_logit", "elo_diff", "experience_diff", "experience_total",
    "experience_min", "age_diff", "age_mean", "reach_diff", "height_diff",
    "layoff_days_diff", "market_elo_gap",
]

ENGINEERED_EXTRA = [
    "experience_total", "experience_min", "market_confidence", "market_elo_gap",
    "elo_probability", "abs_elo_diff", "abs_age_diff", "age_mean", "abs_reach_diff",
    "layoff_max", "abs_layoff_diff", "newcomer_count", "stat_missing_count",
    "market_x_elo_gap", "market_x_experience", "age_x_experience",
]
ENGINEERED_FEATURES = FULL_FEATURES + ENGINEERED_EXTRA


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["experience_total"] = df["experience_1"] + df["experience_2"]
    df["experience_min"] = df[["experience_1", "experience_2"]].min(axis=1)
    df["newcomer_count"] = df["experience_1"].eq(0).astype(int) + df["experience_2"].eq(0).astype(int)
    df["market_confidence"] = df["market_logit"].abs()
    df["elo_probability"] = 1.0 / (1.0 + 10 ** (-df["elo_diff"] / 400.0))
    df["market_elo_gap"] = df["elo_probability"] - df["market_p1"]
    df["abs_elo_diff"] = df["elo_diff"].abs()
    df["abs_age_diff"] = df["age_diff"].abs()
    df["age_mean"] = df[["age_1", "age_2"]].mean(axis=1)
    df["abs_reach_diff"] = df["reach_diff"].abs()
    df["layoff_days_1"] = df["layoff_days_1"].clip(0, 1500)
    df["layoff_days_2"] = df["layoff_days_2"].clip(0, 1500)
    df["layoff_max"] = df[["layoff_days_1", "layoff_days_2"]].max(axis=1)
    df["abs_layoff_diff"] = (df["layoff_days_1"] - df["layoff_days_2"]).abs()
    df["stat_missing_count"] = df[FULL_FEATURES].isna().sum(axis=1)
    df["market_x_elo_gap"] = df["market_logit"] * df["market_elo_gap"]
    df["market_x_experience"] = df["market_logit"] * np.log1p(df["experience_total"])
    df["age_x_experience"] = df["age_diff"] * np.log1p(df["experience_min"])
    return df


def _feature_columns(spec: dict[str, Any]) -> list[str]:
    return STRUCTURAL_FEATURES if spec["feature_set"] == "structural" else ENGINEERED_FEATURES


def _make_estimator(spec: dict[str, Any]) -> Any:
    family = spec["family"]
    if family == "logistic":
        estimator = LogisticRegression(C=float(spec["C"]), max_iter=3000, random_state=42)
        return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()), ("model", estimator)])
    if family == "elastic_net":
        estimator = LogisticRegression(
            C=float(spec["C"]), l1_ratio=float(spec["l1_ratio"]),
            solver="saga", max_iter=5000, random_state=42,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()), ("model", estimator)])
    if family == "hist_gbm":
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(spec["learning_rate"]), max_iter=int(spec["max_iter"]),
            max_leaf_nodes=int(spec["max_leaf_nodes"]), min_samples_leaf=int(spec["min_samples_leaf"]),
            l2_regularization=float(spec["l2_regularization"]), random_state=42,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", estimator)])
    if family == "xgboost":
        estimator = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", n_estimators=int(spec["n_estimators"]),
            max_depth=int(spec["max_depth"]), learning_rate=float(spec["learning_rate"]),
            min_child_weight=float(spec["min_child_weight"]), subsample=float(spec["subsample"]),
            colsample_bytree=float(spec["colsample_bytree"]), reg_alpha=float(spec["reg_alpha"]),
            reg_lambda=float(spec["reg_lambda"]), random_state=42, n_jobs=1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", estimator)])
    if family == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=tuple(spec["hidden_layer_sizes"]), alpha=float(spec["alpha"]),
            learning_rate_init=float(spec["learning_rate_init"]), early_stopping=bool(spec["early_stopping"]),
            max_iter=int(spec["max_iter"]), random_state=42,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler()), ("model", estimator)])
    raise ValueError(f"Famille inconnue: {family}")


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, spec: dict[str, Any]) -> np.ndarray:
    columns = _feature_columns(spec)
    family = spec["family"]
    if family in {"logistic", "elastic_net"}:
        model = _make_estimator(spec)
        model.fit(train[columns], train["y"].astype(int))
        return model.predict_proba(test[columns])[:, 1]

    # Calibration chronologique: le modele de base ne voit pas la derniere annee
    # complete du train; cette annee sert uniquement au sigmoid de calibration.
    last_year = int(train["event_date"].dt.year.max())
    calibration = train[train["event_date"].dt.year.eq(last_year)]
    base = train[train["event_date"].dt.year.lt(last_year)]
    if len(base) < 800 or len(calibration) < 100:
        raise RuntimeError("Echantillon insuffisant pour la calibration chronologique")
    model = _make_estimator(spec)
    model.fit(base[columns], base["y"].astype(int))
    raw_calibration = model.predict_proba(calibration[columns])[:, 1]
    calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    calibrator.fit(_logit(raw_calibration), calibration["y"].astype(int))
    raw_test = model.predict_proba(test[columns])[:, 1]
    return calibrator.predict_proba(_logit(raw_test))[:, 1]


def walk_forward_predictions(df: pd.DataFrame, spec: dict[str, Any], start_year: int = 2015, end_year: int = 2024) -> pd.DataFrame:
    eligible = df.dropna(subset=["y", "market_logit", "odds_1", "odds_2"]).copy()
    records: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train = eligible[(eligible["event_date"] >= "2010-01-01") & (eligible["event_date"] < cutoff)]
        test = eligible[(eligible["event_date"] >= cutoff) & (eligible["event_date"] < cutoff + pd.DateOffset(years=1))]
        if len(train) < 1000 or test.empty:
            continue
        batch = test.copy()
        batch["p_model"] = _fit_predict(train, test, spec)
        records.append(batch)
    return pd.concat(records, ignore_index=True)


def monthly_holdout_predictions(df: pd.DataFrame, spec: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    eligible = df.dropna(subset=["y", "market_logit", "odds_1", "odds_2"]).copy()
    records: list[pd.DataFrame] = []
    month = start.replace(day=1)
    while month <= end:
        month_end = month + pd.DateOffset(months=1)
        train = eligible[(eligible["event_date"] >= "2010-01-01") & (eligible["event_date"] < month)]
        test = eligible[(eligible["event_date"] >= max(month, start)) & (eligible["event_date"] < min(month_end, end + pd.Timedelta(days=1)))]
        if len(train) >= 1000 and not test.empty:
            batch = test.copy()
            batch["p_model"] = _fit_predict(train, test, spec)
            records.append(batch)
        month = month_end
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def probability_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y = predictions["y"].astype(int)
    model = predictions["p_model"].clip(1e-6, 1 - 1e-6)
    market = predictions["market_p1"].clip(1e-6, 1 - 1e-6)
    return {
        "n": int(len(predictions)), "log_loss": float(log_loss(y, model)),
        "market_log_loss": float(log_loss(y, market)), "log_loss_improvement": float(log_loss(y, market) - log_loss(y, model)),
        "brier": float(brier_score_loss(y, model)), "market_brier": float(brier_score_loss(y, market)),
        "auc": float(roc_auc_score(y, model)),
    }


def bet_summary(bets: pd.DataFrame, staking: dict[str, Any], samples: int, confidence: float) -> dict[str, Any]:
    if bets.empty:
        return {"bets": 0, "roi": np.nan, "profit_units": 0.0}
    ledger, bankroll = simulate_bankroll(bets, staking)
    ci = bootstrap_roi(bets, samples=samples, confidence=confidence, seed=20260902)
    yearly = bets.assign(year=pd.to_datetime(bets["event_date"]).dt.year).groupby("year").agg(
        bets=("won", "size"), wins=("won", "sum"), profit_units=("unit_profit", "sum"), roi=("unit_profit", "mean")
    )
    return {
        "bets": int(len(bets)), "events": int(bets["event_id"].nunique()), "wins": int(bets["won"].sum()),
        "win_rate": float(bets["won"].mean()), "average_odds": float(bets["odds"].mean()),
        "average_edge": float(bets["edge"].mean()), "profit_units": float(bets["unit_profit"].sum()),
        "roi": float(bets["unit_profit"].mean()), f"roi_bootstrap_{confidence:g}": ci,
        "positive_years": int((yearly["roi"] > 0).sum()), "yearly": {str(k): v for k, v in yearly.to_dict("index").items()},
        **bankroll, "total_staked": float(ledger["stake"].sum()),
    }


def run_challenger_research(base_dir: Path) -> dict[str, Any]:
    protocol_path = base_dir / "challenger_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    processed = base_dir / "data" / "rigorous" / "processed"
    reports = base_dir / "data" / "rigorous" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    features = add_engineered_features(build_features(base_dir))
    features.to_parquet(processed / "challenger_features.parquet", index=False)

    predictions_by_candidate: dict[str, pd.DataFrame] = {}
    selection_metrics: dict[str, Any] = {}
    select_start, select_end = map(pd.Timestamp, protocol["model_selection"])
    for name, spec in protocol["candidates"].items():
        print(f"Challenger {name}...", flush=True)
        predictions = walk_forward_predictions(features, spec)
        predictions_by_candidate[name] = predictions
        window = predictions[predictions["event_date"].between(select_start, select_end)]
        selection_metrics[name] = probability_metrics(window)
    selected_name = min(selection_metrics, key=lambda name: selection_metrics[name]["log_loss"])
    selected_spec = protocol["candidates"][selected_name]
    selected_predictions = predictions_by_candidate[selected_name]

    rule_start, rule_end = map(pd.Timestamp, protocol["rule_selection"])
    rule_predictions = selected_predictions[selected_predictions["event_date"].between(rule_start, rule_end)]
    rule = protocol["bet_rule"]
    threshold_results: dict[str, Any] = {}
    for threshold in rule["edge_threshold_candidates"]:
        bets = make_bets(rule_predictions, threshold, rule["minimum_decimal_odds"], rule["maximum_decimal_odds"])
        threshold_results[str(threshold)] = bet_summary(bets, protocol["staking"], samples=5000, confidence=0.95)
    eligible = [
        threshold for threshold in rule["edge_threshold_candidates"]
        if threshold_results[str(threshold)].get("bets", 0) >= rule["minimum_selection_bets"]
    ]
    selected_threshold = max(
        eligible,
        key=lambda threshold: threshold_results[str(threshold)]["roi_bootstrap_0.95"]["low"],
    ) if eligible else max(rule["edge_threshold_candidates"])

    confirm_start, confirm_end = map(pd.Timestamp, protocol["internal_confirmation"])
    confirmation_predictions = selected_predictions[selected_predictions["event_date"].between(confirm_start, confirm_end)]
    confirmation_probability = probability_metrics(confirmation_predictions)
    confirmation_bets = make_bets(
        confirmation_predictions, selected_threshold, rule["minimum_decimal_odds"], rule["maximum_decimal_odds"]
    )
    gate = protocol["confirmation_gate"]
    confirmation_bets_summary = bet_summary(
        confirmation_bets, protocol["staking"], samples=gate["bootstrap_samples"], confidence=gate["cluster_bootstrap_confidence"]
    )
    ci_key = f"roi_bootstrap_{gate['cluster_bootstrap_confidence']:g}"
    checks = {
        "model_log_loss_beats_market": confirmation_probability["log_loss"] < confirmation_probability["market_log_loss"],
        "minimum_bets": confirmation_bets_summary["bets"] >= gate["minimum_bets"],
        "positive_roi": confirmation_bets_summary["roi"] > 0,
        "positive_years": confirmation_bets_summary["positive_years"] >= gate["positive_years_minimum"],
        "maximum_drawdown": confirmation_bets_summary["max_drawdown"] <= gate["maximum_drawdown"],
        "positive_adjusted_bootstrap_lower_bound": confirmation_bets_summary[ci_key]["low"] > gate["roi_lower_bound_must_exceed"],
    }
    approved = all(checks.values())
    lock = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha256_file(protocol_path),
        "selected_model": selected_name, "selected_model_spec": selected_spec,
        "selected_threshold": selected_threshold, "confirmation_gate_checks": checks,
        "approved_for_pristine_holdout": approved,
        "status": "CHALLENGER_VALIDATED_PENDING_HOLDOUT" if approved else "CHALLENGER_REJECTED_NO_BET",
    }
    report = {
        "protocol": protocol, "model_selection": selection_metrics, "selected_model": selected_name,
        "threshold_selection": threshold_results, "selected_threshold": selected_threshold,
        "confirmation_probability": confirmation_probability, "confirmation_bets": confirmation_bets_summary,
        "confirmation_gate_checks": checks, "approved_for_pristine_holdout": approved,
    }
    (reports / "challenger_research_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n")
    (reports / "challenger_locked_strategy.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False, default=str) + "\n")
    selected_predictions.to_parquet(processed / "challenger_oos_predictions_pre_holdout.parquet", index=False)
    confirmation_bets.to_parquet(reports / "challenger_confirmation_bets.parquet", index=False)
    return report


def run_challenger_holdout(base_dir: Path) -> dict[str, Any]:
    protocol_path = base_dir / "challenger_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    reports = base_dir / "data" / "rigorous" / "reports"
    processed = base_dir / "data" / "rigorous" / "processed"
    lock = json.loads((reports / "challenger_locked_strategy.json").read_text())
    if lock["protocol_sha256"] != sha256_file(protocol_path):
        raise RuntimeError("Le protocole challenger a change apres verrouillage")
    if not lock["approved_for_pristine_holdout"]:
        result = {"status": "NOT_OPENED_CHALLENGER_CONFIRMATION_FAILED"}
        (reports / "challenger_holdout_report.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    features = pd.read_parquet(processed / "challenger_features.parquet")
    start, end = map(pd.Timestamp, protocol["pristine_economic_holdout"])
    predictions = monthly_holdout_predictions(features, lock["selected_model_spec"], start, end)
    predictions = predictions[predictions["temporal_quality"].eq("timestamped_pre_event")]
    rule = protocol["bet_rule"]
    bets = make_bets(predictions, lock["selected_threshold"], rule["minimum_decimal_odds"], rule["maximum_decimal_odds"])
    summary = bet_summary(
        bets, protocol["staking"], samples=protocol["confirmation_gate"]["bootstrap_samples"],
        confidence=protocol["confirmation_gate"]["cluster_bootstrap_confidence"],
    )
    result = {"status": "OPENED_ONCE_BY_VALIDATED_CHALLENGER", "summary": summary}
    predictions.to_parquet(processed / "challenger_holdout_predictions.parquet", index=False)
    bets.to_parquet(reports / "challenger_holdout_bets.parquet", index=False)
    (reports / "challenger_holdout_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    return result
