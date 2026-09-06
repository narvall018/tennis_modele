#!/usr/bin/env python3
"""Train the best pure-descriptor UFC model — no odds anywhere in it.

The rigorous pipeline in this package models the *market residual*: it is handed
the price and asked to improve on it. That is the right shape for a betting
study and the wrong shape for an app, which needs an opinion the price can be
compared against. A model given the price can only ever agree with it.

So this trainer sees fighters only: rating, experience, striking volume and
accuracy, absorbed strikes, takedowns, submission attempts, control share,
knockdowns, reach, height, age, layoff and stance. Model families are compared
walk-forward on a development window and the winner is scored on later years it
was never ranked on.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
for path in (str(BASE_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from rigorous.model_pipeline import STATS_FEATURES, build_features  # noqa: E402
from src.models.selection import select_best_model  # noqa: E402


# Frozen split. Early UFC years are too thin and too different to rank on.
DEVELOPMENT_YEARS = tuple(range(2015, 2022))
EVALUATION_YEARS = tuple(range(2022, 2027))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()

    processed = BASE_DIR / "data" / "rigorous" / "processed"
    features_path = processed / "features.parquet"
    output_dir = PROJECT_ROOT / "models" / "ufc"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.rebuild_features or not features_path.exists():
        print("Reconstruction des descripteurs depuis fights.parquet…", flush=True)
        frame, states, profiles = build_features(BASE_DIR, return_states=True)
    else:
        frame = pd.read_parquet(features_path)
        _, states, profiles = build_features(BASE_DIR, return_states=True)
    # Persisted as plain columns so the app can score a scheduled card by lookup,
    # and so the file survives a different pandas or Python than the trainer's.
    from rigorous.upcoming import export_state_table

    export_state_table(states, profiles, pd.Timestamp.today().normalize()).to_parquet(
        output_dir / "fighter_states.parquet", index=False
    )
    print(f"{len(frame):,} combats chargés")

    frame = frame[frame["y"].notna()].copy()
    frame["year"] = pd.to_datetime(frame["event_date"]).dt.year
    # A fighter needs some record before his descriptors mean anything.
    experienced = (frame["experience_1"] >= 2) & (frame["experience_2"] >= 2)
    frame = frame[experienced].copy()

    matrix = frame[STATS_FEATURES].to_numpy(dtype=float)
    labels = frame["y"].to_numpy(dtype=int)
    years = frame["year"].to_numpy()
    print(f"{len(frame):,} combats exploitables, {frame['year'].min()}–{frame['year'].max()}")
    print(f"développement {DEVELOPMENT_YEARS[0]}–{DEVELOPMENT_YEARS[-1]}, "
          f"évaluation {EVALUATION_YEARS[0]}–{EVALUATION_YEARS[-1]}\n")

    result = select_best_model(
        matrix, labels, years, DEVELOPMENT_YEARS, EVALUATION_YEARS,
        progress=lambda message: print(message, flush=True),
    )

    print(f"\n{'modèle':32s} {'n':>6s} {'log-loss':>10s} {'brier':>8s} {'auc':>7s}")
    for row in result.comparison:
        print(f"{row['model']:32s} {row['n']:6d} {row['log_loss']:10.5f} "
              f"{row.get('brier', float('nan')):8.5f} {row.get('auc', float('nan')):7.4f}")

    output = output_dir
    joblib.dump(result.fitted, output / "ufc_descriptor_model.joblib")

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "winner": result.winner,
        "features": STATS_FEATURES,
        "uses_odds": False,
        "development_years": list(DEVELOPMENT_YEARS),
        "evaluation_years": list(EVALUATION_YEARS),
        "comparison": result.comparison,
        "evaluation": result.evaluation,
        "fights_used": int(len(frame)),
        "honest_note": (
            "Modèle de descripteurs purs, sans cote. Il ne prétend pas battre le "
            "marché: les trois phases rigoureuses de ce paquet ont toutes été rejetées. "
            "Ses probabilités sont une opinion calibrée."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    if result.evaluation:
        print(f"\nÉvaluation sur années jamais classées ({EVALUATION_YEARS[0]}–"
              f"{EVALUATION_YEARS[-1]}): n={result.evaluation['n']}, "
              f"log-loss {result.evaluation['log_loss']:.5f}, "
              f"AUC {result.evaluation.get('auc', float('nan')):.4f}")
    print(f"artefacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
