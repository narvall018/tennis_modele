#!/usr/bin/env python3
"""Lance l'ablation ATP phase 4 pré-enregistrée."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.backtesting.tennis_phase4 import run_phase4  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Réutilise uniquement le cache lié au même hash de données.",
    )
    args = parser.parse_args()
    report = run_phase4(BASE_DIR, reuse_features=args.reuse_features)
    summary = {
        "selected_primary_candidate": report["selected_primary_candidate"],
        "primary_metrics": report["primary_candidate_metrics"][report["selected_primary_candidate"]],
        "deep_learning_diagnostic": {
            key: report["deep_learning_diagnostic"][key]
            for key in ("n", "log_loss", "market_log_loss", "log_loss_improvement_vs_market")
        },
        "economic_haircut_2pct": report["fixed_economic_diagnostic"]["haircut_2pct"],
        "bootstrap_99pct": report["fixed_economic_diagnostic"]["bootstrap_99pct"],
        "decision": report["decision"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
