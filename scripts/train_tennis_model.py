#!/usr/bin/env python3
"""Train the best pure-descriptor tennis model, one per tour.

Like the football and UFC trainers, this one never sees a price. The nested
studies in `src/backtesting/rigorous_strategy.py` model the market residual,
which is right for a betting question and wrong for an app: an opinion that was
handed the price cannot disagree with it.

ATP and WTA are trained separately. They are different populations with
different rating scales, and pooling them would let the larger tour dictate the
smaller one's calibration.
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.rigorous_strategy import MODEL_FEATURES, build_feature_table
from src.models.selection import select_best_model

TOURS = {
    "atp": {"data": "data/atp_tennis.csv", "development": range(2012, 2021),
            "evaluation": range(2021, 2027)},
    "wta": {"data": "data/wta_tennis.csv", "development": range(2013, 2021),
            "evaluation": range(2021, 2027)},
}


def train_tour(root: Path, tour: str, config: dict) -> dict:
    print(f"\n{'=' * 70}\n{tour.upper()}\n{'=' * 70}", flush=True)
    features, audit = build_feature_table(
        root / config["data"], progress=lambda message: print(message, flush=True)
    )
    played = features["_status"].eq("completed")
    frame = features[played].copy()
    matrix = frame[MODEL_FEATURES].to_numpy(dtype=np.float32)
    labels = frame["_label"].to_numpy(dtype=int)
    years = frame["_year"].to_numpy()

    development = [year for year in config["development"] if (years == year).sum() > 200]
    evaluation = [year for year in config["evaluation"] if (years == year).sum() > 200]
    print(f"{len(frame):,} matchs joués; développement {development[0]}–{development[-1]}, "
          f"évaluation {evaluation[0]}–{evaluation[-1]}\n")

    result = select_best_model(
        matrix, labels, years, development, evaluation,
        progress=lambda message: print(message, flush=True),
    )
    print(f"\n{'modèle':32s} {'n':>6s} {'log-loss':>10s} {'brier':>8s} {'auc':>7s}")
    for row in result.comparison:
        print(f"{row['model']:32s} {row['n']:6d} {row['log_loss']:10.5f} "
              f"{row.get('brier', float('nan')):8.5f} {row.get('auc', float('nan')):7.4f}")

    output = root / "models" / "tennis"
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.fitted, output / f"{tour}_descriptor_model.joblib")

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "tour": tour,
        "winner": result.winner,
        "uses_odds": False,
        "features": MODEL_FEATURES,
        "matches_used": int(len(frame)),
        "development_years": development,
        "evaluation_years": evaluation,
        "comparison": result.comparison,
        "evaluation": result.evaluation,
        "data_audit": audit,
        "honest_note": (
            "Descripteurs purs, sans cote. Le marché reste meilleur: le mélange "
            "market-résiduel de l'étude gagne au mieux +0,00103 de log-loss contre "
            "Pinnacle, et ce gain ne devient pas du rendement."
        ),
    }
    (output / f"{tour}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if result.evaluation:
        print(f"\nÉvaluation sur années jamais classées: n={result.evaluation['n']}, "
              f"log-loss {result.evaluation['log_loss']:.5f}, "
              f"AUC {result.evaluation.get('auc', float('nan')):.4f}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--tour", choices=sorted(TOURS) + ["both"], default="both")
    args = parser.parse_args()
    root = args.project_root.resolve()

    tours = sorted(TOURS) if args.tour == "both" else [args.tour]
    for tour in tours:
        train_tour(root, tour, TOURS[tour])
    print(f"\nartefacts: {root / 'models' / 'tennis'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
