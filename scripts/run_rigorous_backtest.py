#!/usr/bin/env python3
"""Run the nested temporal ATP betting-strategy study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtesting.rigorous_strategy import run_nested_strategy_study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data" / "atp_tennis.csv"))
    parser.add_argument("--output", default=str(ROOT / "models" / "rigorous_strategy"))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Réutilise explicitement pre_match_features.csv.gz; sans ce drapeau, les features sont recalculées.",
    )
    args = parser.parse_args()
    report = run_nested_strategy_study(
        args.data,
        args.output,
        bootstrap_samples=args.bootstrap_samples,
        reuse_features=args.reuse_features,
    )
    print(json.dumps(report["frozen_strategy"], ensure_ascii=False, indent=2))
    print(json.dumps(report["deployment_gate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
