"""Features pre-combat, selection chronologique et backtests UFC sans fuite."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_pipeline import sha256_file


MARKET_FEATURES = ["market_logit"]
STATS_FEATURES = [
    "elo_diff", "experience_diff", "career_win_rate_diff", "recent_win_rate_diff",
    "sig_landed_pm_diff", "sig_absorbed_pm_diff", "sig_accuracy_diff",
    "td_landed_p15_diff", "td_accuracy_diff", "sub_attempts_p15_diff",
    "ctrl_share_diff", "kd_p15_diff", "layoff_days_diff", "reach_diff",
    "height_diff", "age_diff", "stance_matchup",
]
FULL_FEATURES = MARKET_FEATURES + STATS_FEATURES


@dataclass
class FighterState:
    elo: float = 1500.0
    fights: int = 0
    wins: float = 0.0
    last_date: pd.Timestamp | None = None
    recent: list[dict[str, float]] = field(default_factory=list)


def _mean_recent(state: FighterState, key: str) -> float:
    values = [row[key] for row in state.recent[-5:] if pd.notna(row.get(key))]
    return float(np.mean(values)) if values else np.nan


def _safe_rate(numerator: object, denominator: object, scale: float = 1.0) -> float:
    num = pd.to_numeric(pd.Series([numerator]), errors="coerce").iloc[0]
    den = pd.to_numeric(pd.Series([denominator]), errors="coerce").iloc[0]
    if pd.isna(num) or pd.isna(den) or den <= 0:
        return np.nan
    return float(num / den * scale)


def _profile_map(profiles: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return profiles.drop_duplicates("fighter_id").set_index("fighter_id").to_dict("index")


def _age(profile: dict[str, Any], date: pd.Timestamp) -> float:
    dob = pd.to_datetime(profile.get("dob"), errors="coerce")
    return float((date - dob).days / 365.25) if pd.notna(dob) else np.nan


def _orientation_swap(fight_id: str) -> bool:
    digest = hashlib.sha256(str(fight_id).encode()).digest()
    return bool(digest[0] & 1)


def _state_features(state: FighterState, date: pd.Timestamp) -> dict[str, float]:
    return {
        "elo": state.elo,
        "experience": float(state.fights),
        "career_win_rate": state.wins / state.fights if state.fights else np.nan,
        "recent_win_rate": _mean_recent(state, "result"),
        "sig_landed_pm": _mean_recent(state, "sig_landed_pm"),
        "sig_absorbed_pm": _mean_recent(state, "sig_absorbed_pm"),
        "sig_accuracy": _mean_recent(state, "sig_accuracy"),
        "td_landed_p15": _mean_recent(state, "td_landed_p15"),
        "td_accuracy": _mean_recent(state, "td_accuracy"),
        "sub_attempts_p15": _mean_recent(state, "sub_attempts_p15"),
        "ctrl_share": _mean_recent(state, "ctrl_share"),
        "kd_p15": _mean_recent(state, "kd_p15"),
        "layoff_days": float((date - state.last_date).days) if state.last_date is not None else np.nan,
    }


def _performance(row: pd.Series, side: int, result: float) -> dict[str, float]:
    opponent = 2 if side == 1 else 1
    duration = max(float(row.get("duration_secs") or 0), 1.0)
    minutes = duration / 60.0
    return {
        "result": result,
        "sig_landed_pm": _safe_rate(row.get(f"p{side}_sig_lnd"), minutes),
        "sig_absorbed_pm": _safe_rate(row.get(f"p{opponent}_sig_lnd"), minutes),
        "sig_accuracy": _safe_rate(row.get(f"p{side}_sig_lnd"), row.get(f"p{side}_sig_att")),
        "td_landed_p15": _safe_rate(row.get(f"p{side}_td_lnd"), minutes, 15.0),
        "td_accuracy": _safe_rate(row.get(f"p{side}_td_lnd"), row.get(f"p{side}_td_att")),
        "sub_attempts_p15": _safe_rate(row.get(f"p{side}_sub_att"), minutes, 15.0),
        "ctrl_share": _safe_rate(row.get(f"p{side}_ctrl_secs"), duration),
        "kd_p15": _safe_rate(row.get(f"p{side}_kd"), minutes, 15.0),
    }


def build_features(
    base_dir: Path, return_states: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    """Build the fight feature table.

    ``return_states`` also hands back every fighter's end-of-history state and
    the profile map, so a scheduled bout can be scored with exactly the
    descriptors the model was fitted on instead of a recomputed approximation.
    """
    processed = base_dir / "data" / "rigorous" / "processed"
    fights = pd.read_parquet(processed / "fights.parquet").copy()
    profiles = pd.read_parquet(processed / "fighters.parquet")
    lines = pd.read_parquet(processed / "moneyline_quotes.parquet")
    fights["event_date"] = pd.to_datetime(fights["event_date"])
    quote_columns = [
        "fight_id", "odds_fighter_1", "odds_fighter_2", "market_p1", "overround",
        "source", "collected_at", "temporal_quality", "line_protocol",
    ]
    fights = fights.merge(lines[quote_columns], on="fight_id", how="left")
    profiles_by_id = _profile_map(profiles)
    states: defaultdict[str, FighterState] = defaultdict(FighterState)
    rows: list[dict[str, Any]] = []

    for event_date, event_fights in fights.sort_values(["event_date", "event_id", "fight_id"]).groupby("event_date", sort=True):
        pending_updates: list[tuple[pd.Series, str, str, float, float]] = []
        for _, original in event_fights.iterrows():
            swap = _orientation_swap(str(original["fight_id"]))
            first_side, second_side = (2, 1) if swap else (1, 2)
            first_id = str(original[f"fighter_{first_side}_id"])
            second_id = str(original[f"fighter_{second_side}_id"])
            first_state, second_state = states[first_id], states[second_id]
            first = _state_features(first_state, event_date)
            second = _state_features(second_state, event_date)
            p1_profile = profiles_by_id.get(first_id, {})
            p2_profile = profiles_by_id.get(second_id, {})
            odds_1 = original.get(f"odds_fighter_{first_side}")
            odds_2 = original.get(f"odds_fighter_{second_side}")
            market_p = original.get("market_p1")
            if swap and pd.notna(market_p):
                market_p = 1.0 - float(market_p)
            market_logit = np.log(market_p / (1 - market_p)) if pd.notna(market_p) and 0 < market_p < 1 else np.nan
            y_original = original.get("y")
            y = 1.0 - float(y_original) if swap and pd.notna(y_original) else y_original
            record: dict[str, Any] = {
                "fight_id": original["fight_id"],
                "event_id": original["event_id"],
                "event_date": event_date,
                "event_name": original["event_name"],
                "fighter_1": original[f"fighter_{first_side}"],
                "fighter_2": original[f"fighter_{second_side}"],
                "fighter_1_id": first_id,
                "fighter_2_id": second_id,
                "y": y,
                "odds_1": odds_1,
                "odds_2": odds_2,
                "market_p1": market_p,
                "market_logit": market_logit,
                "overround": original.get("overround"),
                "quote_source": original.get("source"),
                "quote_collected_at": original.get("collected_at"),
                "temporal_quality": original.get("temporal_quality"),
                "line_protocol": original.get("line_protocol"),
                "orientation_swapped": swap,
            }
            for key in (
                "elo", "experience", "career_win_rate", "recent_win_rate", "sig_landed_pm",
                "sig_absorbed_pm", "sig_accuracy", "td_landed_p15", "td_accuracy",
                "sub_attempts_p15", "ctrl_share", "kd_p15", "layoff_days",
            ):
                record[f"{key}_diff"] = first[key] - second[key]
            record["experience_1"] = first["experience"]
            record["experience_2"] = second["experience"]
            record["layoff_days_1"] = first["layoff_days"]
            record["layoff_days_2"] = second["layoff_days"]
            record["reach_diff"] = pd.to_numeric(p1_profile.get("reach_cm"), errors="coerce") - pd.to_numeric(p2_profile.get("reach_cm"), errors="coerce")
            record["height_diff"] = pd.to_numeric(p1_profile.get("height_cm"), errors="coerce") - pd.to_numeric(p2_profile.get("height_cm"), errors="coerce")
            age_1, age_2 = _age(p1_profile, event_date), _age(p2_profile, event_date)
            record["age_1"] = age_1
            record["age_2"] = age_2
            record["age_diff"] = age_1 - age_2
            record["weight_class"] = original.get("weight_class")
            stance_1, stance_2 = str(p1_profile.get("stance") or ""), str(p2_profile.get("stance") or "")
            record["stance_matchup"] = float(bool(stance_1 and stance_2 and stance_1 != stance_2))
            rows.append(record)

            if pd.isna(y_original):
                score_1 = score_2 = 0.5 if str(original.get("result_1")) == "D" else np.nan
            else:
                score_1, score_2 = float(y_original), 1.0 - float(y_original)
            pending_updates.append((original, str(original["fighter_1_id"]), str(original["fighter_2_id"]), score_1, score_2))

        # Toutes les features de la carte sont figees avant d'integrer ses resultats.
        for original, fighter_1_id, fighter_2_id, score_1, score_2 in pending_updates:
            state_1, state_2 = states[fighter_1_id], states[fighter_2_id]
            pre_1, pre_2 = state_1.elo, state_2.elo
            if pd.notna(score_1):
                expected_1 = 1.0 / (1.0 + 10 ** ((pre_2 - pre_1) / 400.0))
                state_1.elo = pre_1 + 24.0 * (score_1 - expected_1)
                state_2.elo = pre_2 + 24.0 * (score_2 - (1.0 - expected_1))
                state_1.wins += score_1
                state_2.wins += score_2
            state_1.fights += 1
            state_2.fights += 1
            state_1.last_date = event_date
            state_2.last_date = event_date
            state_1.recent.append(_performance(original, 1, score_1))
            state_2.recent.append(_performance(original, 2, score_2))

    features = pd.DataFrame(rows).sort_values(["event_date", "event_id", "fight_id"]).reset_index(drop=True)
    path = processed / "features.parquet"
    features.to_parquet(path, index=False)
    if return_states:
        return features, dict(states), profiles_by_id
    return features


def _candidate_features(candidate: dict[str, Any]) -> list[str]:
    return MARKET_FEATURES if candidate["features"] == "market" else FULL_FEATURES


def _fit_model(train: pd.DataFrame, features: list[str], c_value: float) -> Pipeline:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=float(c_value), max_iter=2000, random_state=42)),
        ]
    )
    model.fit(train[features], train["y"].astype(int))
    return model


def _predict_yearly(features: pd.DataFrame, candidate: dict[str, Any], start_year: int, end_year: int) -> pd.DataFrame:
    columns = _candidate_features(candidate)
    records: list[pd.DataFrame] = []
    eligible = features.dropna(subset=["y", "market_logit", "odds_1", "odds_2"]).copy()
    for year in range(start_year, end_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train = eligible[(eligible["event_date"] < cutoff) & (eligible["event_date"] >= "2010-01-01")]
        test = eligible[(eligible["event_date"] >= cutoff) & (eligible["event_date"] < cutoff + pd.DateOffset(years=1))]
        if len(train) < 1000 or test.empty:
            continue
        model = _fit_model(train, columns, candidate["C"])
        batch = test.copy()
        batch["p_model"] = model.predict_proba(test[columns])[:, 1]
        records.append(batch)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _predict_holdout_monthly(features: pd.DataFrame, candidate: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    columns = _candidate_features(candidate)
    eligible = features.dropna(subset=["y", "market_logit", "odds_1", "odds_2"]).copy()
    records: list[pd.DataFrame] = []
    month = start.replace(day=1)
    while month <= end:
        month_end = month + pd.DateOffset(months=1)
        train = eligible[(eligible["event_date"] < month) & (eligible["event_date"] >= "2010-01-01")]
        test = eligible[(eligible["event_date"] >= max(month, start)) & (eligible["event_date"] < min(month_end, end + pd.Timedelta(days=1)))]
        if len(train) >= 1000 and not test.empty:
            model = _fit_model(train, columns, candidate["C"])
            batch = test.copy()
            batch["p_model"] = model.predict_proba(test[columns])[:, 1]
            records.append(batch)
        month = month_end
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y = predictions["y"].astype(int)
    p = predictions["p_model"].clip(1e-6, 1 - 1e-6)
    market = predictions["market_p1"].clip(1e-6, 1 - 1e-6)
    return {
        "n": int(len(predictions)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
        "market_log_loss": float(log_loss(y, market)),
        "market_brier": float(brier_score_loss(y, market)),
    }


def make_bets(predictions: pd.DataFrame, threshold: float, min_odds: float, max_odds: float) -> pd.DataFrame:
    bets: list[dict[str, Any]] = []
    for _, row in predictions.iterrows():
        options = [
            (1, float(row["p_model"]), float(row["odds_1"]), int(row["y"] == 1)),
            (2, 1.0 - float(row["p_model"]), float(row["odds_2"]), int(row["y"] == 0)),
        ]
        options = [option for option in options if min_odds <= option[2] <= max_odds]
        if not options:
            continue
        side, probability, odds, won = max(options, key=lambda option: option[1] * option[2] - 1.0)
        edge = probability * odds - 1.0
        if edge < threshold:
            continue
        bets.append(
            {
                "fight_id": row["fight_id"], "event_id": row["event_id"], "event_date": row["event_date"],
                "fighter_1": row["fighter_1"], "fighter_2": row["fighter_2"], "side": side,
                "selection": row[f"fighter_{side}"], "probability": probability, "odds": odds,
                "edge": edge, "won": won, "unit_profit": (odds - 1.0) if won else -1.0,
                "quote_source": row["quote_source"], "quote_collected_at": row["quote_collected_at"],
            }
        )
    return pd.DataFrame(bets)


def bootstrap_roi(bets: pd.DataFrame, samples: int = 5000, confidence: float = 0.95, seed: int = 42) -> dict[str, float]:
    if bets.empty:
        return {"low": np.nan, "median": np.nan, "high": np.nan}
    events = [group["unit_profit"].to_numpy(float) for _, group in bets.groupby("event_id")]
    rng = np.random.default_rng(seed)
    rois = np.empty(samples)
    for idx in range(samples):
        selected = rng.integers(0, len(events), len(events))
        profits = np.concatenate([events[event_idx] for event_idx in selected])
        rois[idx] = profits.mean()
    alpha = 1.0 - confidence
    return {
        "low": float(np.quantile(rois, alpha / 2)),
        "median": float(np.quantile(rois, 0.5)),
        "high": float(np.quantile(rois, 1 - alpha / 2)),
    }


def simulate_bankroll(bets: pd.DataFrame, staking: dict[str, Any], initial: float = 1000.0) -> tuple[pd.DataFrame, dict[str, float]]:
    bankroll = initial
    peak = initial
    max_drawdown = 0.0
    rows: list[dict[str, Any]] = []
    for event_id, event_bets in bets.sort_values(["event_date", "event_id", "fight_id"]).groupby("event_id", sort=False):
        event_start = bankroll
        fractions = []
        for _, bet in event_bets.iterrows():
            kelly = max(0.0, (bet["probability"] * bet["odds"] - 1.0) / (bet["odds"] - 1.0))
            divisor = float(staking.get("kelly_divisor", 8.0))
            fractions.append(min(float(staking["maximum_fraction_per_bet"]), kelly / divisor))
        total_fraction = sum(fractions)
        scale = min(1.0, float(staking["maximum_event_exposure"]) / total_fraction) if total_fraction > 0 else 0.0
        event_profit = 0.0
        for (_, bet), fraction in zip(event_bets.iterrows(), fractions):
            stake = event_start * fraction * scale
            profit = stake * bet["unit_profit"]
            event_profit += profit
            rows.append({**bet.to_dict(), "stake": stake, "cash_profit": profit, "event_bankroll_start": event_start})
        bankroll += event_profit
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak if peak > 0 else 0.0)
    ledger = pd.DataFrame(rows)
    return ledger, {
        "initial_bankroll": initial, "final_bankroll": float(bankroll),
        "return": float(bankroll / initial - 1.0), "max_drawdown": float(max_drawdown),
    }


def summarise_bets(bets: pd.DataFrame, staking: dict[str, Any], bootstrap_samples: int = 5000) -> dict[str, Any]:
    if bets.empty:
        return {"bets": 0, "profit_units": 0.0, "roi": np.nan, "win_rate": np.nan}
    ledger, bankroll = simulate_bankroll(bets, staking)
    ci = bootstrap_roi(bets, samples=bootstrap_samples)
    return {
        "bets": int(len(bets)), "events": int(bets["event_id"].nunique()),
        "wins": int(bets["won"].sum()), "win_rate": float(bets["won"].mean()),
        "average_odds": float(bets["odds"].mean()), "average_edge": float(bets["edge"].mean()),
        "profit_units": float(bets["unit_profit"].sum()), "roi": float(bets["unit_profit"].mean()),
        "roi_bootstrap_95": ci, **bankroll,
        "total_staked": float(ledger["stake"].sum()) if not ledger.empty else 0.0,
    }


def run_research(base_dir: Path) -> dict[str, Any]:
    protocol_path = base_dir / "rigorous_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    processed = base_dir / "data" / "rigorous" / "processed"
    reports = base_dir / "data" / "rigorous" / "reports"
    models_dir = base_dir / "models" / "rigorous"
    reports.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    features = build_features(base_dir)

    model_start = int(protocol["model_selection"][0][:4])
    model_end = int(protocol["model_selection"][1][:4])
    candidate_results: dict[str, Any] = {}
    candidate_predictions: dict[str, pd.DataFrame] = {}
    for name, candidate in protocol["model_candidates"].items():
        predictions = _predict_yearly(features, candidate, model_start, model_end)
        candidate_predictions[name] = predictions
        candidate_results[name] = _metrics(predictions)
    selected_model = min(candidate_results, key=lambda name: candidate_results[name]["log_loss"])
    selected_candidate = protocol["model_candidates"][selected_model]

    all_oos = _predict_yearly(features, selected_candidate, model_start, 2024)
    strategy_start, strategy_end = map(pd.Timestamp, protocol["strategy_selection"])
    strategy_predictions = all_oos[all_oos["event_date"].between(strategy_start, strategy_end)]
    rule = protocol["bet_rule"]
    threshold_results: dict[str, Any] = {}
    for threshold in rule["edge_threshold_candidates"]:
        bets = make_bets(strategy_predictions, threshold, rule["minimum_decimal_odds"], rule["maximum_decimal_odds"])
        summary = summarise_bets(bets, protocol["staking"], protocol["validation_gate"]["bootstrap_samples"])
        threshold_results[str(threshold)] = summary

    eligible_thresholds = [
        threshold for threshold in rule["edge_threshold_candidates"]
        if threshold_results[str(threshold)].get("bets", 0) >= rule["minimum_selection_bets"]
    ]
    if eligible_thresholds:
        selected_threshold = max(
            eligible_thresholds,
            key=lambda threshold: threshold_results[str(threshold)]["roi_bootstrap_95"]["low"],
        )
    else:
        selected_threshold = max(rule["edge_threshold_candidates"])

    validation_start, validation_end = map(pd.Timestamp, protocol["validation"])
    validation_predictions = all_oos[all_oos["event_date"].between(validation_start, validation_end)]
    validation_bets = make_bets(
        validation_predictions, selected_threshold, rule["minimum_decimal_odds"], rule["maximum_decimal_odds"]
    )
    validation_summary = summarise_bets(
        validation_bets, protocol["staking"], protocol["validation_gate"]["bootstrap_samples"]
    )
    gate = protocol["validation_gate"]
    gate_checks = {
        "minimum_bets": validation_summary.get("bets", 0) >= gate["minimum_bets"],
        "positive_roi": validation_summary.get("roi", -np.inf) > 0,
        "maximum_drawdown": validation_summary.get("max_drawdown", np.inf) <= gate["maximum_drawdown"],
        "positive_95pct_cluster_bootstrap_lower_bound": validation_summary.get("roi_bootstrap_95", {}).get("low", -np.inf) > gate["roi_lower_bound_must_exceed"],
    }
    approved = all(gate_checks.values())
    lock = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol_path),
        "selected_model": selected_model,
        "selected_model_spec": selected_candidate,
        "selected_threshold": selected_threshold,
        "validation_gate_checks": gate_checks,
        "approved_for_holdout": approved,
        "status": "VALIDATED_PENDING_FINAL_HOLDOUT" if approved else "REJECTED_NO_BET",
    }
    research = {
        "protocol": protocol,
        "model_selection": candidate_results,
        "selected_model": selected_model,
        "strategy_selection": threshold_results,
        "selected_threshold": selected_threshold,
        "validation": validation_summary,
        "validation_gate_checks": gate_checks,
        "approved_for_holdout": approved,
    }
    (reports / "research_report.json").write_text(json.dumps(research, indent=2, ensure_ascii=False, default=str) + "\n")
    (reports / "locked_strategy.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False, default=str) + "\n")
    validation_bets.to_parquet(reports / "validation_bets.parquet", index=False)
    all_oos.to_parquet(processed / "oos_predictions_pre_holdout.parquet", index=False)

    pre_holdout = features[(features["event_date"] < pd.Timestamp(protocol["final_holdout"][0])) & features["y"].notna() & features["market_logit"].notna()]
    final_model = _fit_model(pre_holdout, _candidate_features(selected_candidate), selected_candidate["C"])
    joblib.dump(
        {"model": final_model, "features": _candidate_features(selected_candidate), "lock": lock},
        models_dir / "ufc_model_locked.joblib",
    )
    return research


def run_final_holdout(base_dir: Path) -> dict[str, Any]:
    reports = base_dir / "data" / "rigorous" / "reports"
    processed = base_dir / "data" / "rigorous" / "processed"
    protocol = json.loads((base_dir / "rigorous_protocol.json").read_text())
    lock = json.loads((reports / "locked_strategy.json").read_text())
    if lock.get("protocol_sha256") != sha256_file(base_dir / "rigorous_protocol.json"):
        raise RuntimeError("Le protocole a change apres verrouillage; holdout refuse")
    if not lock.get("approved_for_holdout"):
        result = {
            "status": "NOT_OPENED_VALIDATION_FAILED",
            "reason": "La strategie n'a pas franchi la validation 2022-2024.",
        }
        (reports / "final_holdout_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        return result
    features = pd.read_parquet(processed / "features.parquet")
    start, end = map(pd.Timestamp, protocol["final_holdout"])
    candidate = lock["selected_model_spec"]
    predictions = _predict_holdout_monthly(features, candidate, start, end)
    predictions = predictions[predictions["temporal_quality"].eq("timestamped_pre_event")].copy()
    rule = protocol["bet_rule"]
    bets = make_bets(
        predictions, float(lock["selected_threshold"]), rule["minimum_decimal_odds"], rule["maximum_decimal_odds"]
    )
    summary = summarise_bets(bets, protocol["staking"], protocol["validation_gate"]["bootstrap_samples"])
    result = {"status": "OPENED_ONCE", "summary": summary}
    predictions.to_parquet(processed / "final_holdout_predictions.parquet", index=False)
    bets.to_parquet(reports / "final_holdout_bets.parquet", index=False)
    (reports / "final_holdout_report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    return result
