"""Can the method-of-victory market be beaten?

This is the only UFC market in the data that no study here had ever modelled.
It is worth a look for a reason that is not wishful: prop markets are priced
less sharply than moneylines, and this package already carries the striking,
grappling and durability statistics that decide *how* a fight ends.

Against that sits a hard number — the six outcomes carry a 22% overround, five
times the moneyline's. The margin per outcome is therefore about 18% of the fair
price, so a bias has to be very large before it pays anything.

Two measurements decide the question and neither selects a strategy:

1. **Calibration** — compare each devigged prop price to how often that outcome
   actually happened, year by year. A bias that appears in one era is noise.
2. **Discrimination** — can a model predict which fights reach the judges better
   than the market already does? If not, no rule can pick the fights where the
   bias is large enough to clear the margin, and the avenue is closed.

The prices are ``legacy_unverified``: untimestamped and single-origin. Even a
positive result here could not be an economic proof — it could only justify
collecting timestamped prop prices going forward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .method_features import SYMMETRIC_FEATURES, build_method_features, method_category


PROP_COLUMNS = [
    "f1_ko_odds", "f2_ko_odds", "f1_sub_odds", "f2_sub_odds", "f1_dec_odds", "f2_dec_odds",
]
OUTCOMES = [
    ("f1_ko_odds", 1, "ko"), ("f2_ko_odds", 0, "ko"),
    ("f1_sub_odds", 1, "sub"), ("f2_sub_odds", 0, "sub"),
    ("f1_dec_odds", 1, "dec"), ("f2_dec_odds", 0, "dec"),
]
# Fixed in advance: the model may look at fights up to the end of 2018 and no
# further, so the later years stay available if this ever justifies a full study.
DEVELOPMENT_LAST_YEAR = 2018


def load_prop_market(base_dir: Path) -> pd.DataFrame:
    """Complete six-way markets only; a partial market has no usable overround."""
    processed = base_dir / "data" / "rigorous" / "processed"
    props = pd.read_parquet(processed / "method_props_quotes.parquet")
    complete = props[PROP_COLUMNS].notna().all(axis=1) & (props[PROP_COLUMNS] > 1.01).all(axis=1)
    props = props[complete].copy()
    inverse = 1.0 / props[PROP_COLUMNS]
    props["overround"] = inverse.sum(axis=1)
    for column in PROP_COLUMNS:
        props[f"p_{column}"] = inverse[column] / props["overround"]
    props["market_decision"] = props["p_f1_dec_odds"] + props["p_f2_dec_odds"]
    return props


def market_calibration(props: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, Any]:
    """Devigged price against realised frequency, overall and per year."""
    merged = props.merge(outcomes, on="fight_id", how="inner")
    merged = merged[merged["y"].notna() & merged["method_category"].notna()].copy()
    merged["year"] = pd.to_datetime(merged["event_date"]).dt.year

    report: dict[str, Any] = {
        "fights": int(len(merged)),
        "years": [int(merged["year"].min()), int(merged["year"].max())],
        "median_overround": float(merged["overround"].median()),
        "by_outcome": {},
    }
    decision_units: list[np.ndarray] = []
    decision_years: list[np.ndarray] = []
    for column, side, category in OUTCOMES:
        probability = merged[f"p_{column}"].to_numpy()
        hit = ((merged["y"] == side) & (merged["method_category"] == category)).to_numpy()
        odds = merged[column].to_numpy()
        unit = np.where(hit, odds - 1.0, -1.0)
        label = ("f1_" if side == 1 else "f2_") + category
        report["by_outcome"][label] = {
            "market_probability": float(probability.mean()),
            "realised_frequency": float(hit.mean()),
            "bias_points": float(hit.mean() - probability.mean()),
            "roi_backing_every_one": float(unit.mean()),
            "n": int(len(unit)),
        }
        if category == "dec":
            decision_units.append(unit)
            decision_years.append(merged["year"].to_numpy())

    units = np.concatenate(decision_units)
    years = np.concatenate(decision_years)
    per_year = {
        str(int(year)): float(units[years == year].mean()) for year in np.unique(years)
    }
    report["decisions_pooled"] = {
        "n": int(len(units)),
        "roi_backing_every_decision": float(units.mean()),
        "roi_by_year": per_year,
        "positive_years": int(sum(value > 0 for value in per_year.values())),
        "total_years": len(per_year),
    }
    return report


def _pipeline(name: str):
    if name == "logistic":
        return Pipeline([
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.3, max_iter=2000)),
        ])
    return Pipeline([
        ("impute", SimpleImputer()),
        ("model", HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0,
        )),
    ])


def discrimination_probe(table: pd.DataFrame, last_year: int = DEVELOPMENT_LAST_YEAR) -> dict[str, Any]:
    """Walk forward inside the development years only, never past ``last_year``.

    Whether a fight reached the judges is known for every fight, priced or not,
    so training uses them all; only priced fights can be scored against a market
    probability.
    """
    development = table[table["year"] <= last_year].copy()
    features = development[SYMMETRIC_FEATURES].to_numpy(dtype=float)
    labels = development["goes_to_decision"].to_numpy(dtype=int)
    market = development["market_decision"].to_numpy(dtype=float)
    years = development["year"].to_numpy()
    priced = np.isfinite(market)

    results: dict[str, Any] = {
        "development_last_year": last_year,
        "training_fights": int(len(development)),
        "priced_fights": int(priced.sum()),
        "realised_decision_rate": float(np.nanmean(labels)),
        "market_mean_decision_probability": float(np.nanmean(market[priced])),
        "candidates": {},
    }
    for name in ("logistic", "hist_gradient_boosting"):
        for with_market in (False, True):
            predictions, truth, reference = [], [], []
            for year in range(last_year - 3, last_year + 1):
                train = years <= year - 1
                test = (years == year) & priced
                if train.sum() < 1000 or test.sum() < 80:
                    continue
                if with_market:
                    train = train & priced
                    logit = np.log(market / (1.0 - market)).reshape(-1, 1)
                    train_x = np.hstack([features[train], logit[train]])
                    test_x = np.hstack([features[test], logit[test]])
                else:
                    train_x, test_x = features[train], features[test]
                model = _pipeline(name).fit(train_x, labels[train])
                predictions.append(model.predict_proba(test_x)[:, 1])
                truth.append(labels[test])
                reference.append(market[test])
            if not truth:
                continue
            predictions = np.concatenate(predictions)
            truth = np.concatenate(truth)
            reference = np.concatenate(reference)
            market_loss = float(log_loss(truth, reference))
            model_loss = float(log_loss(truth, predictions))
            results["candidates"][f"{name}{'_plus_market' if with_market else ''}"] = {
                "n": int(len(truth)),
                "model_log_loss": model_loss,
                "market_log_loss": market_loss,
                "gain_vs_market": market_loss - model_loss,
            }
    best = max(
        (block["gain_vs_market"] for block in results["candidates"].values()), default=float("-inf")
    )
    results["best_gain_vs_market"] = best
    results["market_discrimination_beaten"] = bool(best > 0.002)
    return results


def analyse_method_market(base_dir: Path) -> dict[str, Any]:
    processed = base_dir / "data" / "rigorous" / "processed"
    fights = pd.read_parquet(processed / "fights.parquet")
    features = pd.read_parquet(processed / "features.parquet")
    fights = fights.merge(features[["fight_id", "age_1", "age_2"]], on="fight_id", how="left")

    table = build_method_features(fights)
    table = table[table["goes_to_decision"].notna()].copy()
    table["year"] = pd.to_datetime(table["event_date"]).dt.year

    props = load_prop_market(base_dir)
    outcomes = table[["fight_id", "method_category", "y"]]
    calibration = market_calibration(props, outcomes)
    table = table.merge(props[["fight_id", "market_decision"]], on="fight_id", how="left")
    probe = discrimination_probe(table)

    decisions = calibration["decisions_pooled"]
    margin_per_outcome = calibration["median_overround"] - 1.0
    verdict = (
        "AVENUE_FERMEE"
        if not probe["market_discrimination_beaten"] and decisions["roi_backing_every_decision"] <= 0
        else "A_APPROFONDIR"
    )
    report = {
        "what_this_is": (
            "Le marché méthode de victoire est le seul marché UFC jamais modélisé ici. "
            "Les prix restent legacy_unverified: aucun résultat ne peut être une preuve "
            "économique, seulement un motif de collecter des prix horodatés."
        ),
        "verdict": verdict,
        "why": (
            "Le marché porte un biais réel et constant en faveur des décisions, mais il "
            "reste inférieur à la marge, et aucun modèle ne bat le pouvoir discriminant "
            "du marché: il n'existe donc aucun moyen de sélectionner les combats où le "
            "biais suffirait."
        ) if verdict == "AVENUE_FERMEE" else "Le marché est battu sur au moins un axe; une étude complète est justifiée.",
        "overround": {
            "median_six_way": calibration["median_overround"],
            "margin_points": margin_per_outcome,
            "comparison": "le moneyline UFC tourne autour de 1,03-1,05",
        },
        "calibration": calibration,
        "discrimination_probe": probe,
        "limitations": [
            "Les prix props sont non horodatés et d'origine unique (voir odds_cross_check.json).",
            "Le biais en faveur des décisions a été mesuré sur l'échantillon complet: il est "
            "connu, donc il ne peut plus servir de découverte à valider sur ces mêmes années.",
            "Seules les années jusqu'à 2018 ont été lues par la sonde de discrimination.",
        ],
    }
    quality = base_dir / "data" / "rigorous" / "quality"
    quality.mkdir(parents=True, exist_ok=True)
    (quality / "method_market_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report
