"""Leakage-safe, nested temporal study for ATP match-winner betting.

The module deliberately separates four jobs:

* 2012-2016: choose the probability model and its market blend;
* 2017-2020: choose one betting rule;
* 2021-2023: validate that rule and choose one staking plan;
* 2024-2026: evaluate the frozen combination exactly once.

Only observed, coherent pairs of decimal odds are accepted. Missing odds are
never imputed. Retirements and walkovers remain in the decision population and
can be settled as void, avoiding completed-match survivorship bias.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.elo_system import TennisEloEngine
from src.features.feature_builder import FeatureBuilder

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_XGBOOST = False


MODEL_DEVELOPMENT_YEARS = tuple(range(2012, 2017))
STRATEGY_TUNING_YEARS = tuple(range(2017, 2021))
STRATEGY_VALIDATION_YEARS = tuple(range(2021, 2024))
FINAL_HOLDOUT_YEARS = tuple(range(2024, 2027))


@dataclass(frozen=True)
class ProtocolWindows:
    """The four disjoint périods a study is allowed to use, in order.

    They are a parameter rather than a constant so the same audited engine can
    be pointed at a market it has never been run on.  The windows must be frozen
    before the study starts; changing them after seeing a return is the exact
    failure this whole module exists to prevent.
    """

    development: tuple[int, ...] = MODEL_DEVELOPMENT_YEARS
    tuning: tuple[int, ...] = STRATEGY_TUNING_YEARS
    validation: tuple[int, ...] = STRATEGY_VALIDATION_YEARS
    holdout: tuple[int, ...] = FINAL_HOLDOUT_YEARS
    minimum_tuning_bets: int = 120

    def __post_init__(self) -> None:
        windows = [self.development, self.tuning, self.validation, self.holdout]
        if any(not window for window in windows):
            raise ValueError("Chaque fenêtre du protocole doit contenir au moins une année")
        flat = [year for window in windows for year in window]
        if len(set(flat)) != len(flat):
            raise ValueError("Les fenêtres du protocole se chevauchent")
        if flat != sorted(flat):
            raise ValueError("Les fenêtres du protocole ne sont pas chronologiques")


DEFAULT_WINDOWS = ProtocolWindows()

SIGNED_FEATURES = [
    "elo_diff",
    "surface_elo_diff",
    "momentum_elo_diff",
    "form_3_diff",
    "form_5_diff",
    "form_10_diff",
    "form_20_diff",
    "form_momentum_diff",
    "surface_wr_3m_diff",
    "surface_wr_6m_diff",
    "surface_wr_12m_diff",
    "h2h_signal",
    "h2h_surface_signal",
    "fatigue_diff",
    "rest_diff",
    "log_rank_diff",
    "signed_log_points_diff",
    "surface_specialist_diff",
    "signed_log_slam_experience_diff",
    "elo_x_best_of_5",
    "elo_x_late_round",
]

CONTEXT_FEATURES = [
    "is_hard",
    "is_clay",
    "is_grass",
    "is_indoor",
    "best_of_5",
    "round_num",
    "series_num",
    "month_sin",
    "month_cos",
    "log_h2h_sample",
]

MODEL_FEATURES = SIGNED_FEATURES + CONTEXT_FEATURES
RESIDUAL_FEATURES = SIGNED_FEATURES + ["market_logit"] + CONTEXT_FEATURES

PRICE_COLUMNS = {
    "selected": ("odds_p1", "odds_p2"),
    "average": ("Avg_1", "Avg_2"),
    "bet365": ("B365_1", "B365_2"),
    "pinnacle": ("Pinnacle_1", "Pinnacle_2"),
    "maximum": ("Max_1", "Max_2"),
}


@dataclass(frozen=True)
class BetRule:
    min_edge: float
    min_ev: float
    min_probability: float
    min_odds: float
    max_odds: float


@dataclass(frozen=True)
class StakePlan:
    name: str
    kind: str
    flat_fraction: float = 0.0
    kelly_divisor: float = 1.0
    max_bet_fraction: float = 0.005
    max_daily_fraction: float = 0.02


STAKE_PLANS = (
    StakePlan("flat_0_25pct", "flat", flat_fraction=0.0025, max_bet_fraction=0.0025, max_daily_fraction=0.02),
    StakePlan("flat_0_50pct", "flat", flat_fraction=0.0050, max_bet_fraction=0.0050, max_daily_fraction=0.03),
    StakePlan("flat_1_00pct", "flat", flat_fraction=0.0100, max_bet_fraction=0.0100, max_daily_fraction=0.04),
    StakePlan("kelly_1_24", "kelly", kelly_divisor=24.0, max_bet_fraction=0.0050, max_daily_fraction=0.02),
    StakePlan("kelly_1_16", "kelly", kelly_divisor=16.0, max_bet_fraction=0.0075, max_daily_fraction=0.03),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _signed_log(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.sign(array) * np.log1p(np.abs(array))


def build_feature_table(data_path: str | Path, progress=print) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build pre-match features in one chronological pass and retain exact prices."""
    source_path = Path(data_path).resolve()
    raw = pd.read_csv(source_path, low_memory=False)
    required = {"Date", "Player_1", "Player_2", "Winner", "Odd_1", "Odd_2"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Colonnes requises absentes: {missing}")

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date", "Player_1", "Player_2", "Winner"]).copy()
    raw["Status"] = raw.get("Status", "completed").fillna("completed").astype(str).str.lower()
    if "Best of" in raw and "Best_of" not in raw:
        raw = raw.rename(columns={"Best of": "Best_of"})
    numeric = [
        "Rank_1", "Rank_2", "Pts_1", "Pts_2", "Odd_1", "Odd_2",
        "B365_1", "B365_2", "Pinnacle_1", "Pinnacle_2", "Avg_1", "Avg_2",
        "Max_1", "Max_2", "Betfair_1", "Betfair_2", "Best_of",
    ]
    for column in numeric:
        if column not in raw:
            raw[column] = np.nan
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["source_row_id"] = np.arange(len(raw), dtype=np.int64)
    raw["odds_p1"] = raw["Odd_1"]
    raw["odds_p2"] = raw["Odd_2"]

    progress(f"Features chronologiques: {len(raw):,} matchs, aucun résultat futur utilisé")
    engine = TennisEloEngine()
    engine.fit(raw, progress_callback=progress)
    feature_df = FeatureBuilder().build_dataset(engine.get_history(), progress_callback=progress)

    named_price_columns = sorted(
        {column for source, pair in PRICE_COLUMNS.items() if source != "selected" for column in pair}
    )
    price_meta = raw[["source_row_id", "Status"] + named_price_columns].copy()
    price_meta = price_meta.rename(columns={"Status": "_source_status"})
    feature_df = feature_df.merge(
        price_meta,
        left_on="_source_row_id",
        right_on="source_row_id",
        how="left",
        validate="one_to_one",
    ).drop(columns="source_row_id")
    feature_df["_status"] = feature_df["_source_status"].fillna(feature_df["_status"])
    feature_df = feature_df.drop(columns="_source_status")
    feature_df["_date"] = pd.to_datetime(feature_df["_date"])
    feature_df["_year"] = feature_df["_date"].dt.year.astype(int)

    model = pd.DataFrame(index=feature_df.index)
    model["elo_diff"] = feature_df["elo_diff"]
    model["surface_elo_diff"] = feature_df["surf_elo_diff"]
    model["momentum_elo_diff"] = feature_df["momentum_elo_diff"]
    for horizon in (3, 5, 10, 20):
        model[f"form_{horizon}_diff"] = feature_df[f"p1_form_{horizon}"] - feature_df[f"p2_form_{horizon}"]
    model["form_momentum_diff"] = feature_df["p1_momentum"] - feature_df["p2_momentum"]
    for suffix in ("3m", "6m", "12m"):
        model[f"surface_wr_{suffix}_diff"] = feature_df[f"p1_surf_wr_{suffix}"] - feature_df[f"p2_surf_wr_{suffix}"]
    h2h_weight = np.log1p(feature_df["h2h_total"].clip(lower=0))
    model["h2h_signal"] = (2.0 * feature_df["h2h_p1_wr"] - 1.0) * h2h_weight
    model["h2h_surface_signal"] = (2.0 * feature_df["h2h_surf_p1_wr"] - 1.0) * h2h_weight
    model["fatigue_diff"] = feature_df["p1_fatigue"] - feature_df["p2_fatigue"]
    model["rest_diff"] = feature_df["p1_days_rest"] - feature_df["p2_days_rest"]
    model["log_rank_diff"] = feature_df["log_rank_diff"]
    model["signed_log_points_diff"] = _signed_log(feature_df["pts_diff"])
    model["surface_specialist_diff"] = feature_df["p1_surf_specialist"] - feature_df["p2_surf_specialist"]
    slam_diff = feature_df["p1_slam_experience"] - feature_df["p2_slam_experience"]
    model["signed_log_slam_experience_diff"] = _signed_log(slam_diff)
    model["elo_x_best_of_5"] = feature_df["elo_diff"] * feature_df["best_of_5"]
    model["elo_x_late_round"] = feature_df["elo_diff"] * (feature_df["round_num"] >= 5).astype(float)
    for column in CONTEXT_FEATURES[:-1]:
        model[column] = feature_df[column]
    model["log_h2h_sample"] = h2h_weight

    feature_df[MODEL_FEATURES] = model[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    if feature_df["_source_row_id"].duplicated().any():
        raise AssertionError("Un identifiant source apparaît plusieurs fois dans les features")
    if not set(feature_df["_label"].dropna().unique()).issubset({0, 1}):
        raise AssertionError("Labels invalides")

    audit = {
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "rows": int(len(feature_df)),
        "completed_rows": int(feature_df["_status"].eq("completed").sum()),
        "noncompleted_rows": int(feature_df["_status"].ne("completed").sum()),
        "min_date": feature_df["_date"].min().date().isoformat(),
        "max_date": feature_df["_date"].max().date().isoformat(),
        "label_p1_rate": float(feature_df["_label"].mean()),
    }
    return feature_df, audit


def _make_model(name: str):
    name = name.removeprefix("market_")
    if name == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.1, max_iter=1500, random_state=20260901)),
            ]
        )
    if name == "hist_gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=250,
                        max_leaf_nodes=15,
                        min_samples_leaf=40,
                        l2_regularization=5.0,
                        random_state=20260901,
                    ),
                ),
            ]
        )
    if name == "xgboost" and HAS_XGBOOST:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=350,
                        max_depth=3,
                        learning_rate=0.03,
                        min_child_weight=20,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.2,
                        reg_lambda=5.0,
                        eval_metric="logloss",
                        n_jobs=-1,
                        random_state=20260901,
                    ),
                ),
            ]
        )
    raise ValueError(f"Modèle indisponible: {name}")


def _swap_matrix(matrix: np.ndarray, signed_count: int = len(SIGNED_FEATURES)) -> np.ndarray:
    swapped = np.asarray(matrix, dtype=np.float32).copy()
    swapped[:, :signed_count] *= -1.0
    return swapped


def _fit_symmetric(model, matrix: np.ndarray, labels: np.ndarray, signed_count: int = len(SIGNED_FEATURES)):
    augmented_x = np.vstack([matrix, _swap_matrix(matrix, signed_count)])
    augmented_y = np.concatenate([labels, 1 - labels])
    model.fit(augmented_x, augmented_y)
    return model


def _predict_symmetric(model, matrix: np.ndarray, signed_count: int = len(SIGNED_FEATURES)) -> np.ndarray:
    direct = model.predict_proba(matrix)[:, 1]
    reversed_probability = model.predict_proba(_swap_matrix(matrix, signed_count))[:, 1]
    return np.clip(0.5 * (direct + 1.0 - reversed_probability), 1e-6, 1.0 - 1e-6)


def _fit_temperature(raw_probability: np.ndarray, labels: np.ndarray) -> float:
    z = logit(np.clip(raw_probability, 1e-6, 1.0 - 1e-6)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1000.0, fit_intercept=False, max_iter=1000, random_state=20260901)
    calibrator.fit(np.vstack([z, -z]), np.concatenate([labels, 1 - labels]))
    return float(calibrator.coef_[0, 0])


def _probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return {
        "n": int(len(labels)),
        "log_loss": float(log_loss(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "auc": float(roc_auc_score(labels, probability)),
    }


def walk_forward_predictions(
    features: pd.DataFrame,
    model_name: str,
    test_years: Iterable[int],
    progress=print,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Generate annual predictions with train <= Y-2 and calibration == Y-1."""
    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    completed = features["_status"].eq("completed")
    residual_model = model_name.startswith("market_")
    if residual_model:
        # Before market-average columns became broadly available, ``selected``
        # is a coherent same-book pair (Pinnacle/Bet365 fallback). Development
        # and every betting decision from 2012 onward still use ``average``.
        left, right, valid_market = _valid_market(features, "selected")
        market_probability = (1.0 / left) / (1.0 / left + 1.0 / right)
        working = features.copy()
        working["market_logit"] = logit(np.clip(market_probability, 1e-6, 1.0 - 1e-6))
        feature_columns = RESIDUAL_FEATURES
        signed_count = len(SIGNED_FEATURES) + 1
    else:
        working = features
        valid_market = np.ones(len(features), dtype=bool)
        feature_columns = MODEL_FEATURES
        signed_count = len(SIGNED_FEATURES)

    for year in test_years:
        train = working[completed & valid_market & (working["_year"] <= year - 2)]
        calibration = working[completed & valid_market & (working["_year"] == year - 1)]
        test_mask = working["_year"].eq(year)
        test = working[test_mask]
        if len(train) < 10_000 or len(calibration) < 500 or test.empty:
            raise ValueError(f"Fold {year} insuffisant: train={len(train)}, cal={len(calibration)}, test={len(test)}")
        if not (train["_date"].max() < calibration["_date"].min() <= calibration["_date"].max() < test["_date"].min()):
            raise AssertionError(f"Chevauchement temporel détecté dans le fold {year}")

        x_train = train[feature_columns].to_numpy(dtype=np.float32)
        y_train = train["_label"].to_numpy(dtype=int)
        x_cal = calibration[feature_columns].to_numpy(dtype=np.float32)
        y_cal = calibration["_label"].to_numpy(dtype=int)
        x_test = test[feature_columns].to_numpy(dtype=np.float32)
        progress(f"{model_name} | train <= {year - 2}, calibration {year - 1}, test {year}")
        model = _fit_symmetric(_make_model(model_name), x_train, y_train, signed_count)
        raw_cal = _predict_symmetric(model, x_cal, signed_count)
        temperature = _fit_temperature(raw_cal, y_cal)
        raw_test = _predict_symmetric(model, x_test, signed_count)
        calibrated_test = expit(temperature * logit(raw_test))

        output = test.copy()
        output["model_probability_p1"] = calibrated_test
        output["raw_model_probability_p1"] = raw_test
        predictions.append(output)
        played = output["_status"].eq("completed").to_numpy(copy=True)
        if residual_model:
            _, _, output_valid_market = _valid_market(output, "average")
            played &= output_valid_market
        metrics = _probability_metrics(
            output.loc[played, "_label"].to_numpy(dtype=int),
            output.loc[played, "model_probability_p1"].to_numpy(dtype=float),
        )
        metrics.update(
            {
                "model": model_name,
                "test_year": int(year),
                "train_max_date": train["_date"].max().date().isoformat(),
                "calibration_min_date": calibration["_date"].min().date().isoformat(),
                "calibration_max_date": calibration["_date"].max().date().isoformat(),
                "test_min_date": test["_date"].min().date().isoformat(),
                "temperature": temperature,
            }
        )
        fold_metrics.append(metrics)
    return pd.concat(predictions, ignore_index=True), fold_metrics


def select_probability_model(
    features: pd.DataFrame,
    progress=print,
    windows: ProtocolWindows = DEFAULT_WINDOWS,
) -> tuple[str, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = ["logistic", "hist_gradient_boosting", "market_logistic", "market_hist_gradient_boosting"]
    if HAS_XGBOOST:
        candidates.extend(["xgboost", "market_xgboost"])
    all_predictions: dict[str, pd.DataFrame] = {}
    all_folds: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for name in candidates:
        prediction, folds = walk_forward_predictions(features, name, windows.development, progress)
        all_predictions[name] = prediction
        all_folds.extend(folds)
        _, _, valid_market = _valid_market(prediction, "average")
        played = prediction["_status"].eq("completed").to_numpy() & valid_market
        metrics = _probability_metrics(
            prediction.loc[played, "_label"].to_numpy(dtype=int),
            prediction.loc[played, "model_probability_p1"].to_numpy(dtype=float),
        )
        metrics["model"] = name
        summary.append(metrics)
    chosen = min(summary, key=lambda row: (row["log_loss"], row["brier"]))["model"]
    progress(
        f"Modèle figé sur {windows.development[0]}-{windows.development[-1]}: {chosen}"
    )
    return chosen, all_predictions[chosen], all_folds, summary


def _valid_market(frame: pd.DataFrame, price_source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left_col, right_col = PRICE_COLUMNS[price_source]
    left = pd.to_numeric(frame[left_col], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(frame[right_col], errors="coerce").to_numpy(dtype=float)
    overround = 1.0 / left + 1.0 / right
    low = 0.85 if price_source == "maximum" else 0.95
    high = 1.20 if price_source == "maximum" else 1.25
    valid = np.isfinite(left) & np.isfinite(right) & (left > 1.0) & (right > 1.0) & (overround >= low) & (overround <= high)
    return left, right, valid


def choose_market_blend(development: pd.DataFrame) -> tuple[float, list[dict[str, float]]]:
    left, right, valid = _valid_market(development, "average")
    played = development["_status"].eq("completed").to_numpy()
    use = valid & played
    q1 = (1.0 / left) / (1.0 / left + 1.0 / right)
    model_p = development["model_probability_p1"].to_numpy(dtype=float)
    labels = development["_label"].to_numpy(dtype=int)
    results = []
    for weight in np.linspace(0.0, 1.0, 11):
        blended = weight * model_p + (1.0 - weight) * q1
        metrics = _probability_metrics(labels[use], blended[use])
        metrics["model_weight"] = float(weight)
        results.append(metrics)
    chosen = min(results, key=lambda row: (row["log_loss"], row["brier"]))["model_weight"]
    return float(chosen), results


def prepare_bet_candidates(
    predictions: pd.DataFrame,
    model_weight: float,
    price_source: str = "average",
) -> pd.DataFrame:
    frame = predictions.copy()
    left, right, valid = _valid_market(frame, price_source)
    q1 = (1.0 / left) / (1.0 / left + 1.0 / right)
    model_p1 = frame["model_probability_p1"].to_numpy(dtype=float)
    p1 = model_weight * model_p1 + (1.0 - model_weight) * q1
    p2 = 1.0 - p1
    ev1 = p1 * left - 1.0
    ev2 = p2 * right - 1.0
    choose_p1 = ev1 >= ev2
    frame["price_source"] = price_source
    frame["market_valid"] = valid
    frame["market_probability_p1"] = q1
    frame["bet_side"] = np.where(choose_p1, 1, 2)
    frame["bet_player"] = np.where(choose_p1, frame["_p1"], frame["_p2"])
    frame["bet_probability"] = np.where(choose_p1, p1, p2)
    frame["bet_market_probability"] = np.where(choose_p1, q1, 1.0 - q1)
    frame["bet_odds"] = np.where(choose_p1, left, right)
    frame["edge"] = frame["bet_probability"] - frame["bet_market_probability"]
    frame["expected_roi"] = frame["bet_probability"] * frame["bet_odds"] - 1.0
    frame["won"] = np.where(choose_p1, frame["_label"].eq(1), frame["_label"].eq(0))
    frame["void"] = frame["_status"].ne("completed")
    return frame


def apply_rule(candidates: pd.DataFrame, rule: BetRule) -> pd.DataFrame:
    mask = (
        candidates["market_valid"]
        & candidates["edge"].ge(rule.min_edge)
        & candidates["expected_roi"].ge(rule.min_ev)
        & candidates["bet_probability"].ge(rule.min_probability)
        & candidates["bet_odds"].between(rule.min_odds, rule.max_odds, inclusive="both")
    )
    return candidates.loc[mask].copy()


def _unit_returns(bets: pd.DataFrame, odds_haircut: float = 0.0, settle_retirements_as_void: bool = True) -> tuple[np.ndarray, np.ndarray]:
    effective_odds = 1.0 + (bets["bet_odds"].to_numpy(dtype=float) - 1.0) * (1.0 - odds_haircut)
    if settle_retirements_as_void:
        settled = bets["_status"].eq("completed").to_numpy()
    else:
        settled = ~bets["_status"].isin(["walkover", "awarded", "defaulted", "sched"]).to_numpy()
    returns = np.where(bets["won"].to_numpy(), effective_odds - 1.0, -1.0)
    returns = np.where(settled, returns, 0.0)
    return returns, settled


def flat_metrics(
    bets: pd.DataFrame,
    odds_haircut: float = 0.0,
    settle_retirements_as_void: bool = True,
) -> dict[str, Any]:
    if bets.empty:
        return {"n_bets": 0, "n_settled": 0, "n_void": 0, "roi": 0.0, "profit_units": 0.0, "hit_rate": None, "standard_error": None}
    returns, settled = _unit_returns(bets, odds_haircut, settle_retirements_as_void)
    resolved = returns[settled]
    n_settled = int(settled.sum())
    standard_error = float(np.std(resolved, ddof=1) / math.sqrt(n_settled)) if n_settled > 1 else None
    return {
        "n_bets": int(len(bets)),
        "n_settled": n_settled,
        "n_void": int((~settled).sum()),
        "roi": float(resolved.mean()) if n_settled else 0.0,
        "profit_units": float(resolved.sum()),
        "hit_rate": float(bets.loc[settled, "won"].mean()) if n_settled else None,
        "average_odds": float(bets.loc[settled, "bet_odds"].mean()) if n_settled else None,
        "standard_error": standard_error,
    }


def _rule_grid() -> Iterable[BetRule]:
    odds_bands = ((1.20, 1.75), (1.20, 2.10), (1.20, 2.75), (1.20, 4.00), (1.35, 2.20), (1.50, 3.00), (1.75, 4.00), (2.00, 6.00))
    for edge in (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08):
        for ev in (0.0, 0.01, 0.02, 0.03, 0.05):
            for probability in (0.0, 0.50, 0.55, 0.60, 0.65):
                for min_odds, max_odds in odds_bands:
                    yield BetRule(edge, ev, probability, min_odds, max_odds)


def tune_one_rule(
    candidates: pd.DataFrame, windows: ProtocolWindows = DEFAULT_WINDOWS
) -> tuple[BetRule, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    expected_years = set(windows.tuning)
    for rule in _rule_grid():
        bets = apply_rule(candidates, rule)
        metrics = flat_metrics(bets)
        per_year = {int(year): flat_metrics(group)["roi"] for year, group in bets.groupby("_year")}
        active_years = len(expected_years.intersection(per_year))
        positive_years = sum(per_year.get(year, 0.0) > 0.0 for year in expected_years)
        se = metrics["standard_error"] if metrics["standard_error"] is not None else float("inf")
        eligible = (
            metrics["n_settled"] >= windows.minimum_tuning_bets
            and active_years == len(expected_years)
        )
        conservative_score = metrics["roi"] - se if eligible else -float("inf")
        rows.append(
            {
                **asdict(rule),
                **metrics,
                "active_years": active_years,
                "positive_years": positive_years,
                "conservative_score": conservative_score,
                "yearly_roi": json.dumps(per_year, sort_keys=True),
            }
        )
    grid = pd.DataFrame(rows).sort_values(
        ["conservative_score", "roi", "n_settled"], ascending=[False, False, False]
    ).reset_index(drop=True)
    if not np.isfinite(grid.loc[0, "conservative_score"]):
        raise RuntimeError(
            f"Aucune règle ne satisfait le minimum de {windows.minimum_tuning_bets} paris "
            f"réglés sur les {len(expected_years)} années de réglage"
        )
    selected = BetRule(**{field: grid.loc[0, field] for field in asdict(BetRule(0, 0, 0, 0, 0))})
    return selected, grid


def simulate_bankroll(
    bets: pd.DataFrame,
    plan: StakePlan,
    initial_bankroll: float = 1000.0,
    odds_haircut: float = 0.0,
    settle_retirements_as_void: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    ordered = bets.sort_values(["_date", "_tournament", "_source_row_id"], kind="mergesort").copy()
    if ordered.empty:
        return {"initial_bankroll": initial_bankroll, "final_bankroll": initial_bankroll, "return": 0.0, "max_drawdown": 0.0, "n_bets": 0}, ordered
    bankroll = float(initial_bankroll)
    ledger: list[dict[str, Any]] = []
    daily_equity = [bankroll]
    for day, group in ordered.groupby(ordered["_date"].dt.normalize(), sort=True):
        start_bankroll = bankroll
        effective_odds = 1.0 + (group["bet_odds"].to_numpy(dtype=float) - 1.0) * (1.0 - odds_haircut)
        if plan.kind == "flat":
            fractions = np.full(len(group), plan.flat_fraction)
        else:
            probability = group["bet_probability"].to_numpy(dtype=float)
            full_kelly = np.maximum(0.0, (probability * effective_odds - 1.0) / np.maximum(effective_odds - 1.0, 1e-9))
            fractions = full_kelly / plan.kelly_divisor
        fractions = np.minimum(fractions, plan.max_bet_fraction)
        total_fraction = fractions.sum()
        if total_fraction > plan.max_daily_fraction:
            fractions *= plan.max_daily_fraction / total_fraction
        stakes = start_bankroll * fractions

        if settle_retirements_as_void:
            settled = group["_status"].eq("completed").to_numpy()
        else:
            settled = ~group["_status"].isin(["walkover", "awarded", "defaulted", "sched"]).to_numpy()
        unit_return = np.where(group["won"].to_numpy(), effective_odds - 1.0, -1.0)
        unit_return = np.where(settled, unit_return, 0.0)
        profits = stakes * unit_return
        bankroll += float(profits.sum())
        daily_equity.append(bankroll)
        for position, (_, row) in enumerate(group.iterrows()):
            ledger.append(
                {
                    "date": pd.Timestamp(day),
                    "source_row_id": int(row["_source_row_id"]),
                    "player": row["bet_player"],
                    "status": row["_status"],
                    "odds": float(row["bet_odds"]),
                    "effective_odds": float(effective_odds[position]),
                    "probability": float(row["bet_probability"]),
                    "edge": float(row["edge"]),
                    "expected_roi": float(row["expected_roi"]),
                    "stake": float(stakes[position]),
                    "won": bool(row["won"]),
                    "void": not bool(settled[position]),
                    "profit": float(profits[position]),
                    "bankroll_after_day": bankroll,
                }
            )
    equity = np.asarray(daily_equity)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (peaks - equity) / np.maximum(peaks, 1e-9)
    ledger_df = pd.DataFrame(ledger)
    total_staked = float(ledger_df.loc[~ledger_df["void"], "stake"].sum()) if not ledger_df.empty else 0.0
    metrics = {
        "initial_bankroll": initial_bankroll,
        "final_bankroll": bankroll,
        "return": bankroll / initial_bankroll - 1.0,
        "profit": bankroll - initial_bankroll,
        "max_drawdown": float(drawdowns.max()),
        "n_bets": int(len(ledger_df)),
        "n_settled": int((~ledger_df["void"]).sum()),
        "total_staked": total_staked,
        "roi_on_stakes": float(ledger_df["profit"].sum() / total_staked) if total_staked else 0.0,
    }
    return metrics, ledger_df


def choose_stake_plan(validation_bets: pd.DataFrame) -> tuple[StakePlan, list[dict[str, Any]]]:
    results = []
    for plan in STAKE_PLANS:
        metrics, _ = simulate_bankroll(validation_bets, plan, odds_haircut=0.02)
        admissible = metrics["max_drawdown"] <= 0.20 and metrics["final_bankroll"] > 0
        score = math.log(metrics["final_bankroll"] / metrics["initial_bankroll"]) if admissible else -float("inf")
        results.append({"plan": plan.name, **metrics, "admissible": admissible, "score": score})
    best = max(results, key=lambda row: row["score"])
    selected = next(plan for plan in STAKE_PLANS if plan.name == best["plan"])
    return selected, results


def month_block_bootstrap(
    bets: pd.DataFrame,
    odds_haircut: float = 0.0,
    n_bootstrap: int = 5000,
    seed: int = 20260901,
) -> dict[str, Any]:
    returns, settled = _unit_returns(bets, odds_haircut, settle_retirements_as_void=True)
    resolved = bets.loc[settled, ["_date"]].copy()
    resolved["return"] = returns[settled]
    if resolved.empty:
        return {"samples": 0, "months": 0, "roi_ci_90": [None, None], "probability_roi_positive": None}
    resolved["month"] = resolved["_date"].dt.to_period("M").astype(str)
    blocks = resolved.groupby("month")["return"].agg(["sum", "count"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(blocks), size=(n_bootstrap, len(blocks)))
    profits = blocks["sum"].to_numpy()[draw].sum(axis=1)
    stakes = blocks["count"].to_numpy()[draw].sum(axis=1)
    roi = profits / np.maximum(stakes, 1)
    return {
        "samples": int(n_bootstrap),
        "months": int(len(blocks)),
        "roi_ci_90": [float(np.quantile(roi, 0.05)), float(np.quantile(roi, 0.95))],
        "roi_ci_95": [float(np.quantile(roi, 0.025)), float(np.quantile(roi, 0.975))],
        "probability_roi_positive": float(np.mean(roi > 0.0)),
    }


def market_comparison(
    predictions: pd.DataFrame,
    model_weight: float,
    windows: ProtocolWindows,
    price_source: str = "average",
) -> dict[str, Any]:
    """Does the blend actually forecast better than the price it bets against?

    A betting result on a few hundred wagers is far too noisy to answer that;
    the probability metrics use every priced match in the period, so a real
    forecasting gain shows up here long before it could show up in a ROI. A
    strategy that turns a profit while forecasting *worse* than the market is
    reading price dispersion, not the sport.
    """
    left, right, valid = _valid_market(predictions, price_source)
    market = (1.0 / left) / (1.0 / left + 1.0 / right)
    model = predictions["model_probability_p1"].to_numpy(dtype=float)
    blend = model_weight * model + (1.0 - model_weight) * market
    completed = predictions["_status"].eq("completed").to_numpy()
    labels = predictions["_label"].to_numpy(dtype=int)

    periods = {
        "development": windows.development,
        "tuning": windows.tuning,
        "validation": windows.validation,
        "holdout": windows.holdout,
    }
    result: dict[str, Any] = {"price_source": price_source}
    for name, years in periods.items():
        mask = completed & valid & predictions["_year"].isin(list(years)).to_numpy()
        if mask.sum() < 100:
            result[name] = {"n": int(mask.sum()), "insufficient_sample": True}
            continue
        market_metrics = _probability_metrics(labels[mask], market[mask])
        blend_metrics = _probability_metrics(labels[mask], blend[mask])
        result[name] = {
            "n": int(mask.sum()),
            "market_log_loss": market_metrics["log_loss"],
            "model_log_loss": _probability_metrics(labels[mask], model[mask])["log_loss"],
            "blend_log_loss": blend_metrics["log_loss"],
            "blend_gain_vs_market": market_metrics["log_loss"] - blend_metrics["log_loss"],
            "market_auc": market_metrics["auc"],
            "blend_auc": blend_metrics["auc"],
        }
    return result


def period_evaluation(
    predictions: pd.DataFrame,
    model_weight: float,
    rule: BetRule,
    price_source: str = "average",
    haircut: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = prepare_bet_candidates(predictions, model_weight, price_source)
    bets = apply_rule(candidates, rule)
    metrics = flat_metrics(bets, odds_haircut=haircut)
    metrics["price_source"] = price_source
    metrics["odds_haircut"] = haircut
    metrics["available_markets"] = int(candidates["market_valid"].sum())
    metrics["yearly"] = {
        str(int(year)): flat_metrics(group, odds_haircut=haircut)
        for year, group in bets.groupby("_year")
    }
    return metrics, bets


def _span(years: Iterable[int]) -> str:
    ordered = sorted(int(year) for year in years)
    return f"{ordered[0]}–{ordered[-1]}" if len(ordered) > 1 else str(ordered[0])


def _markdown_report(report: dict[str, Any]) -> str:
    final = report["final_holdout"]["average_haircut_0pct"]
    stressed = report["final_holdout"]["average_haircut_2pct"]
    boot = report["final_holdout"]["bootstrap_average_haircut_2pct"]
    rule = report["frozen_strategy"]["rule"]
    stake = report["frozen_strategy"]["stake_plan"]
    protocol = report["protocol"]
    label = protocol.get("study_label", "ATP")
    decision = "VALIDÉE POUR ESSAI EN PAPER-TRADING" if report["deployment_gate"]["passed"] else "NON VALIDÉE — NE PAS MISER EN ARGENT RÉEL"
    lines = [
        f"# Backtest rigoureux de la stratégie {label}",
        "",
        f"Décision: **{decision}**",
        "",
        "## Protocole verrouillé",
        "",
        f"- Modèle et mélange marché: {_span(protocol['model_development_years'])}.",
        f"- Règle de sélection: {_span(protocol['strategy_tuning_years'])}.",
        f"- Validation et taille des mises: {_span(protocol['strategy_validation_years'])}.",
        f"- Test final jamais utilisé pour les choix: {_span(protocol['final_holdout_years'])}.",
        "- Entraînement d'un fold Y: résultats terminés jusqu'à Y-2; calibration: Y-1; test: Y.",
        "- Aucune cote absente n'est remplacée; prix moyens observés et dévigés pour mesurer l'edge.",
        "- Abandons et walkovers restent dans la population; scénario principal: mises annulées.",
        "",
        "## Stratégie figée",
        "",
        f"- Modèle: `{report['frozen_strategy']['model']}`; poids modèle: {report['frozen_strategy']['model_weight']:.1%}; poids marché: {1-report['frozen_strategy']['model_weight']:.1%}.",
        f"- Règle: edge ≥ {rule['min_edge']:.1%}, EV estimée ≥ {rule['min_ev']:.1%}, probabilité ≥ {rule['min_probability']:.1%}, cote {rule['min_odds']:.2f}–{rule['max_odds']:.2f}.",
        f"- Mise: `{stake['name']}`, plafond par pari {stake['max_bet_fraction']:.2%}, exposition quotidienne {stake['max_daily_fraction']:.2%}.",
        "",
        f"## Test final {_span(protocol['final_holdout_years'])}",
        "",
        f"- Sans décote: {final['n_settled']} paris réglés, ROI {final['roi']:.2%}, profit {final['profit_units']:.2f} unités.",
        f"- Décote de cote 2%: ROI {stressed['roi']:.2%}; IC bootstrap mensuel 90% [{boot['roi_ci_90'][0]:.2%}, {boot['roi_ci_90'][1]:.2%}].",
        f"- Bankroll simulée: {report['final_holdout']['staking']['initial_bankroll']:.2f} → {report['final_holdout']['staking']['final_bankroll']:.2f}; drawdown max {report['final_holdout']['staking']['max_drawdown']:.2%}.",
        "",
        "## Le modèle prévoit-il mieux que le prix qu'il affronte ?",
        "",
        "Sur tous les matchs cotés de chaque période, pas seulement sur les paris pris.",
        "",
        "| Période | Matchs | Log-loss marché | Log-loss mélange | Gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for period in ("development", "tuning", "validation", "holdout"):
        block = report.get("market_comparison", {}).get(period, {})
        if not block or block.get("insufficient_sample"):
            continue
        lines.append(
            f"| {period} | {block['n']} | {block['market_log_loss']:.5f} | "
            f"{block['blend_log_loss']:.5f} | {block['blend_gain_vs_market']:+.5f} |"
        )
    lines += [
        "",
        "Un gain positif et stable indique un vrai pouvoir prédictif supplémentaire. "
        "Un gain nul ou négatif accompagné d'un ROI positif signale au contraire que le "
        "résultat vient de la dispersion des prix, pas d'une meilleure lecture du sport.",
        "",
        "## Limites qui empêchent toute promesse de gain",
        "",
        "- Les cellules par source de prix (Bet365, Pinnacle, maximum) sont des tests de "
        "sensibilité, pas des résultats: l'edge y est recalculé contre chaque prix, donc "
        "chaque source produit une population de paris différente. Choisir après coup la "
        "source la plus flatteuse annulerait le protocole.",
        "- Le prix `maximum` est le meilleur prix trouvé chez un opérateur quelconque. Il "
        "surestime structurellement le rendement et n'est pas exécutable à volume.",
        "- Les cotes sont généralement les dernières avant le match; leur disponibilité exacte et les limites de mise ne sont pas horodatées.",
        "- Le scénario principal suppose que tous les abandons sont annulés; les règles réelles varient selon l'opérateur.",
        "- L'IC par blocs mensuels mesure l'incertitude historique, pas le risque de changement futur du marché.",
        "- Une validation statistique autorise seulement un paper-trading préalable, jamais une garantie de rentabilité.",
        "",
    ]
    return "\n".join(lines)


def run_nested_strategy_study(
    data_path: str | Path,
    output_dir: str | Path,
    progress=print,
    bootstrap_samples: int = 5000,
    reuse_features: bool = False,
    windows: ProtocolWindows = DEFAULT_WINDOWS,
    protocol_notes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "pre_match_features.csv.gz"
    if reuse_features and feature_path.exists():
        progress(f"Réutilisation explicite du cache de features: {feature_path}")
        features = pd.read_csv(feature_path, low_memory=False, parse_dates=["_date"])
        missing = sorted(set(MODEL_FEATURES + ["_status", "_year", "_label"]) - set(features.columns))
        if missing:
            raise ValueError(f"Cache de features incompatible: {missing}")
        source_path = Path(data_path).resolve()
        data_audit = {
            "source_path": str(source_path),
            "source_sha256": _sha256(source_path),
            "rows": int(len(features)),
            "completed_rows": int(features["_status"].eq("completed").sum()),
            "noncompleted_rows": int(features["_status"].ne("completed").sum()),
            "min_date": features["_date"].min().date().isoformat(),
            "max_date": features["_date"].max().date().isoformat(),
            "label_p1_rate": float(features["_label"].mean()),
            "feature_cache_reused_by_explicit_request": True,
        }
    else:
        features, data_audit = build_feature_table(data_path, progress)
        features.to_csv(feature_path, index=False, compression="gzip")

    chosen_model, development_predictions, dev_folds, model_comparison = select_probability_model(
        features, progress, windows
    )
    model_weight, blend_comparison = choose_market_blend(development_predictions)
    later_years = windows.tuning + windows.validation + windows.holdout
    later_predictions, later_folds = walk_forward_predictions(features, chosen_model, later_years, progress)
    all_predictions = pd.concat([development_predictions, later_predictions], ignore_index=True)
    all_predictions.to_csv(output / "oos_predictions.csv.gz", index=False, compression="gzip")

    tuning_predictions = all_predictions[all_predictions["_year"].isin(windows.tuning)]
    tuning_candidates = prepare_bet_candidates(tuning_predictions, model_weight, "average")
    rule, grid = tune_one_rule(tuning_candidates, windows)
    grid.to_csv(
        output / f"strategy_grid_tuning_{windows.tuning[0]}_{windows.tuning[-1]}.csv", index=False
    )
    tuning_metrics, tuning_bets = period_evaluation(tuning_predictions, model_weight, rule)

    validation_predictions = all_predictions[all_predictions["_year"].isin(windows.validation)]
    validation_metrics, validation_bets = period_evaluation(validation_predictions, model_weight, rule)
    validation_stress, _ = period_evaluation(validation_predictions, model_weight, rule, haircut=0.02)
    positive_validation_years = sum(
        year_metrics["roi"] > 0.0 for year_metrics in validation_stress["yearly"].values()
    )
    validation_gate = (
        validation_stress["n_settled"] >= 60
        and validation_stress["roi"] > 0.0
        and positive_validation_years >= 2
    )
    stake_plan, stake_comparison = choose_stake_plan(validation_bets)
    if not validation_gate:
        stake_plan = STAKE_PLANS[0]

    # The final holdout is first referenced only after model, blend, rule, gate,
    # and staking plan have all been frozen above.
    final_predictions = all_predictions[all_predictions["_year"].isin(windows.holdout)]
    final_metrics: dict[str, Any] = {}
    final_bets_for_export = pd.DataFrame()
    for source in ("average", "bet365", "pinnacle", "maximum"):
        for haircut in (0.0, 0.02, 0.05):
            metrics, bets = period_evaluation(final_predictions, model_weight, rule, source, haircut)
            final_metrics[f"{source}_haircut_{int(haircut * 100)}pct"] = metrics
            if source == "average" and haircut == 0.0:
                final_bets_for_export = bets
    final_stressed = final_metrics["average_haircut_2pct"]
    final_metrics["average_haircut_2pct_official_retirement_result"] = flat_metrics(
        final_bets_for_export,
        odds_haircut=0.02,
        settle_retirements_as_void=False,
    )
    bootstrap = month_block_bootstrap(final_bets_for_export, 0.02, bootstrap_samples)
    final_metrics["bootstrap_average_haircut_2pct"] = bootstrap
    staking_metrics, staking_ledger = simulate_bankroll(final_bets_for_export, stake_plan, odds_haircut=0.02)
    final_metrics["staking"] = staking_metrics
    final_bets_for_export.to_csv(output / "final_holdout_bets.csv", index=False)
    staking_ledger.to_csv(output / "final_holdout_staking_ledger.csv", index=False)

    positive_final_years = sum(
        year_metrics["roi"] > 0.0 for year_metrics in final_stressed["yearly"].values()
    )
    final_gate = (
        validation_gate
        and final_stressed["n_settled"] >= 50
        and final_stressed["roi"] > 0.0
        and bootstrap["roi_ci_90"][0] is not None
        and bootstrap["roi_ci_90"][0] > 0.0
        and positive_final_years >= 2
        and staking_metrics["max_drawdown"] <= 0.25
    )

    report = {
        "protocol": {
            "model_development_years": windows.development,
            "strategy_tuning_years": windows.tuning,
            "strategy_validation_years": windows.validation,
            "final_holdout_years": windows.holdout,
            "fold_rule": "train through Y-2, calibrate on Y-1, test on Y",
            "missing_odds": "excluded, never imputed",
            "primary_price": "Tennis-Data market average pre-match decimal odds",
            "primary_settlement": "all non-completed matches void",
            "development_note": (
                "Market-residual candidate families were added during 2012-2016 development after "
                "pure sports models optimally received zero blend weight; no tuning, validation, "
                "or final-holdout return had been inspected at that point."
            ),
            **(protocol_notes or {}),
        },
        "data_audit": data_audit,
        "market_comparison": market_comparison(all_predictions, model_weight, windows),
        "model_comparison_development": model_comparison,
        "fold_metrics": dev_folds + later_folds,
        "blend_comparison_development": blend_comparison,
        "frozen_strategy": {
            "model": chosen_model,
            "model_weight": model_weight,
            "rule": asdict(rule),
            "stake_plan": asdict(stake_plan),
        },
        "tuning": tuning_metrics,
        "validation": {
            "observed_odds": validation_metrics,
            "odds_haircut_2pct": validation_stress,
            "positive_years_after_haircut": positive_validation_years,
            "gate_passed": validation_gate,
            "stake_comparison_after_haircut": stake_comparison,
        },
        "final_holdout": final_metrics,
        "deployment_gate": {
            "passed": final_gate,
            "meaning": "paper-trading only" if final_gate else "no real-money deployment",
            "requirements": {
                "validation_gate": validation_gate,
                "at_least_50_final_settled_bets": final_stressed["n_settled"] >= 50,
                "positive_final_roi_after_2pct_haircut": final_stressed["roi"] > 0.0,
                "positive_90pct_month_block_bootstrap_lower_bound": bootstrap["roi_ci_90"][0] is not None and bootstrap["roi_ci_90"][0] > 0.0,
                "at_least_two_positive_final_years": positive_final_years >= 2,
                "max_drawdown_at_most_25pct": staking_metrics["max_drawdown"] <= 0.25,
            },
        },
    }
    ready = _json_ready(report)
    with (output / "backtest_report.json").open("w", encoding="utf-8") as handle:
        json.dump(ready, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with (output / "frozen_strategy.json").open("w", encoding="utf-8") as handle:
        json.dump(ready["frozen_strategy"], handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    (output / "BACKTEST_REPORT.md").write_text(_markdown_report(ready), encoding="utf-8")
    joblib.dump(
        {
            "model_features": MODEL_FEATURES,
            "signed_features": SIGNED_FEATURES,
            "protocol": ready["protocol"],
            "frozen_strategy": ready["frozen_strategy"],
        },
        output / "strategy_definition.joblib",
    )
    return ready
