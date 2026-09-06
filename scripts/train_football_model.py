#!/usr/bin/env python3
"""Train and freeze the football probability model used by the app.

Two things are persisted: the fitted model, and every team's end-of-history
state. Predicting a fixture then costs a table lookup instead of a full pass over
191 000 matches.

The model is deliberately trained *without* the market as an input. The app needs
an opinion that can be compared to a price, and a model fed the price would only
be able to agree with it. What the conditional test in
`run_football_conditional_test.py` established still holds and is displayed in
the app: this opinion does not beat the price.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.football_audit import devig
from src.features.football_features import FEATURE_COLUMNS, build_football_features
from src.models.selection import select_best_model

RESULT_ORDER = ["H", "D", "A"]
# The last completed season is held back so the reported quality is measured on
# matches the model never trained on.
EVALUATION_SEASON_COUNT = 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "models" / "football"
    output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(root / "data" / "football" / "football_matches.csv.gz", low_memory=False)
    print(f"{len(frame):,} matchs; construction des descripteurs…", flush=True)
    featured, states = build_football_features(
        frame, progress=lambda message: print(message, flush=True), return_states=True
    )

    labels = pd.Categorical(featured["result"], categories=RESULT_ORDER).codes
    features = featured[FEATURE_COLUMNS].to_numpy(dtype=float)
    seasons = featured["season_start"].to_numpy()
    # A team needs some history before its descriptors mean anything.
    warm = (featured["home_matches_played"] >= 5) & (featured["away_matches_played"] >= 5)

    evaluation_seasons = sorted(set(seasons))[-EVALUATION_SEASON_COUNT:]
    holdout = np.isin(seasons, evaluation_seasons) & warm.to_numpy()
    train = ~np.isin(seasons, evaluation_seasons) & warm.to_numpy()
    print(f"\nentraînement: {train.sum():,} matchs | évaluation: {holdout.sum():,} "
          f"(saisons {evaluation_seasons})")

    development = sorted(set(seasons[train]))
    result = select_best_model(
        features[warm.to_numpy()], labels[warm.to_numpy()], seasons[warm.to_numpy()],
        development, evaluation_seasons,
        progress=lambda message: print(message, flush=True),
    )
    model = result.fitted
    print(f"\n{'modèle':32s} {'n':>6s} {'log-loss':>10s} {'auc':>7s}")
    for row in result.comparison:
        print(f"{row['model']:32s} {row['n']:6d} {row['log_loss']:10.5f} "
              f"{row.get('auc', float('nan')):7.4f}")

    predictions = model.predict_proba(features[holdout])
    truth = labels[holdout]
    model_loss = float(log_loss(truth, predictions, labels=[0, 1, 2]))
    market, market_ok = devig(featured, ("AvgCH", "AvgCD", "AvgCA"))
    both = holdout & market_ok
    comparison = {}
    if both.sum() > 500:
        market_subset = market[both]
        model_subset = model.predict_proba(features[both])
        comparison = {
            "matches_compared": int(both.sum()),
            "model_log_loss": float(log_loss(labels[both], model_subset, labels=[0, 1, 2])),
            "market_log_loss": float(log_loss(labels[both], market_subset, labels=[0, 1, 2])),
        }
        comparison["model_minus_market"] = (
            comparison["market_log_loss"] - comparison["model_log_loss"]
        )

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_matches": int(train.sum()),
        "evaluation_matches": int(holdout.sum()),
        "evaluation_seasons": [int(season) for season in evaluation_seasons],
        "result_order": RESULT_ORDER,
        # A serialised estimator only reloads under the version that
        # produced it; recording it turns a crash into a clear message.
        "sklearn_version": sklearn.__version__,
        "winner": result.winner,
        "comparison": result.comparison,
        "features": FEATURE_COLUMNS,
        "model_log_loss_on_unseen_seasons": model_loss,
        "versus_market": comparison,
        "honest_note": (
            "Le modèle n'utilise pas la cote. Il ne bat pas le marché: le test "
            "conditionnel gelé donne -0,00080 de log-loss. Ses probabilités sont "
            "une opinion calibrée, pas un avantage démontré."
        ),
    }

    joblib.dump(model, output / "football_model.joblib")
    states.to_parquet(output / "team_states.parquet", index=False)
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"\nlog-loss sur saisons jamais vues: {model_loss:.5f}")
    if comparison:
        print(f"marché (moyenne dévigée):        {comparison['market_log_loss']:.5f}")
        print(f"écart modèle - marché:           {comparison['model_minus_market']:+.5f}")
    print(f"\néquipes mémorisées: {len(states):,}")
    print(f"artefacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
