#!/usr/bin/env python3
"""Do football descriptors add anything the price does not already contain?

Five earlier avenues in this repository died at exactly this step, so it is run
first and cheaply, before any strategy machinery exists. Only the development
seasons are read; the tuning, validation and holdout seasons stay closed so that
a full study remains possible if — and only if — this passes.

The gate is stated before the run: the descriptors must improve on the market by
at least 0.001 of log-loss to justify building anything further. That threshold
is what `models/wta_strategy` showed to be worth roughly nothing in ROI terms, so
anything below it cannot possibly pay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.football_audit import devig
from src.features.football_features import FEATURE_COLUMNS, build_football_features

# Frozen before the run. Pinnacle closing prices start in 2015/16.
DEVELOPMENT_SEASONS = tuple(range(2015, 2020))
TUNING_SEASONS = tuple(range(2020, 2022))
VALIDATION_SEASONS = tuple(range(2022, 2024))
HOLDOUT_SEASONS = tuple(range(2024, 2027))
MINIMUM_GAIN_TO_CONTINUE = 0.001

RESULT_ORDER = ["H", "D", "A"]


def _model(name: str):
    if name == "logistic":
        return Pipeline([
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.2, max_iter=3000)),
        ])
    return Pipeline([
        ("impute", SimpleImputer()),
        ("model", HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300,
            min_samples_leaf=80, l2_regularization=1.0, random_state=0,
        )),
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()

    frame = pd.read_csv(root / "data" / "football" / "football_matches.csv.gz", low_memory=False)
    print(f"{len(frame):,} matchs chargés; construction des descripteurs…", flush=True)
    featured = build_football_features(frame, progress=lambda message: print(message, flush=True))

    market, valid = devig(featured, ("PSCH", "PSCD", "PSCA"))
    usable = valid & featured["season_start"].isin(
        DEVELOPMENT_SEASONS + TUNING_SEASONS + VALIDATION_SEASONS + HOLDOUT_SEASONS
    ).to_numpy()
    development = usable & featured["season_start"].isin(DEVELOPMENT_SEASONS).to_numpy()

    data = featured[development].copy()
    market_probability = market[development]
    labels = pd.Categorical(data["result"], categories=RESULT_ORDER).codes
    seasons = data["season_start"].to_numpy()
    features = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    market_logit = np.log(np.clip(market_probability, 1e-6, 1.0))

    print(f"\nDéveloppement: {len(data):,} matchs, saisons "
          f"{DEVELOPMENT_SEASONS[0]}–{DEVELOPMENT_SEASONS[-1]}")
    print(f"Saisons fermées: réglage {TUNING_SEASONS}, validation {VALIDATION_SEASONS}, "
          f"holdout {HOLDOUT_SEASONS}\n")

    report: dict = {
        "development_seasons": list(DEVELOPMENT_SEASONS),
        "development_matches": int(len(data)),
        "minimum_gain_to_continue": MINIMUM_GAIN_TO_CONTINUE,
        "candidates": {},
    }
    test_seasons = [s for s in DEVELOPMENT_SEASONS if (seasons < s).sum() >= 5000]
    scored = np.isin(seasons, test_seasons)
    report["market_log_loss"] = float(
        log_loss(labels[scored], market_probability[scored], labels=[0, 1, 2])
    )
    print(f"{'candidat':28s} {'n':>7s} {'log-loss':>10s} {'marché':>10s} {'gain':>10s}")

    for name in ("logistic", "hist_gradient_boosting"):
        for with_market in (False, True):
            predictions, truth = [], []
            for season in test_seasons:
                train = seasons < season
                test = seasons == season
                if train.sum() < 5000 or test.sum() < 500:
                    continue
                train_x = features[train]
                test_x = features[test]
                if with_market:
                    train_x = np.hstack([train_x, market_logit[train]])
                    test_x = np.hstack([test_x, market_logit[test]])
                model = _model(name).fit(train_x, labels[train])
                predictions.append(model.predict_proba(test_x))
                truth.append(labels[test])
            if not truth:
                continue
            predictions = np.vstack(predictions)
            truth = np.concatenate(truth)
            reference = market_probability[np.isin(seasons, test_seasons)]
            model_loss = float(log_loss(truth, predictions, labels=[0, 1, 2]))
            market_loss = float(log_loss(truth, reference, labels=[0, 1, 2]))
            label = f"{name}{'+marché' if with_market else ''}"
            report["candidates"][label] = {
                "n": int(len(truth)),
                "log_loss": model_loss,
                "market_log_loss": market_loss,
                "gain_vs_market": market_loss - model_loss,
            }
            print(f"{label:28s} {len(truth):7d} {model_loss:10.5f} {market_loss:10.5f} "
                  f"{market_loss - model_loss:+10.5f}")

    best = max((block["gain_vs_market"] for block in report["candidates"].values()), default=-1.0)
    report["best_gain_vs_market"] = best
    report["gate_passed"] = bool(best >= MINIMUM_GAIN_TO_CONTINUE)
    report["decision"] = (
        "Construire l'étude complète: les descripteurs battent le prix."
        if report["gate_passed"] else
        "Arrêt. Les descripteurs n'ajoutent rien au prix; les saisons de réglage, "
        "validation et holdout restent fermées et réutilisables."
    )

    output = root / "models" / "football_conditional_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nMeilleur gain conditionnel: {best:+.5f} (seuil {MINIMUM_GAIN_TO_CONTINUE:+.5f})")
    print(f"Gate: {'PASSÉE' if report['gate_passed'] else 'ÉCHOUÉE'} — {report['decision']}")
    print(f"Rapport: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
