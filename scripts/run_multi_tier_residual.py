#!/usr/bin/env python3
"""Does the tier-enriched rating add anything the price does not already know?

`run_multi_tier_ratings.py` shows the enriched rating is a better rating. That is
necessary but not sufficient: the market may already contain the same
information. This script puts each rating arm on top of the devigged market price
and asks whether either one improves on the price itself.

The ATP years are spent, so this is a diagnostic. It decides whether the
enrichment is worth carrying into a prospective track, not whether anything
profits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.multi_tier_ratings import market_residual_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--first-test-year", type=int, default=2005)
    args = parser.parse_args()
    root = args.project_root.resolve()

    features = pd.read_parquet(root / "models" / "multi_tier_features.parquet")
    report = market_residual_comparison(features, first_test_year=args.first_test_year)

    output = root / "models" / "multi_tier_residual.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"matchs cotés et terminés: {report['priced_completed_matches']:,}\n")
    print(f"{'bras':28s} {'n':>7s} {'log-loss':>10s} {'marché':>10s} {'gain vs marché':>15s}")
    labels = {"base": "marché + Elo principal", "rich": "marché + Elo enrichi"}
    for arm, block in report["arms"].items():
        print(f"{labels[arm]:28s} {block['n']:7d} {block['log_loss']:10.5f} "
              f"{block['market_log_loss']:10.5f} {block['gain_vs_market']:+15.5f}")
    print(f"\napport propre de l'enrichissement: {report['enrichment_gain']:+.5f} de log-loss")

    thin = report.get("thin_record_subgroup", {})
    if "base" in thin:
        print(f"\nSous-groupe déclaré à l'avance — au moins un joueur à historique mince "
              f"({thin['n']:,} matchs):")
        print(f"  marché seul              log-loss {thin['market_log_loss']:.5f}")
        print(f"  marché + Elo principal   log-loss {thin['base']['log_loss']:.5f}  "
              f"gain {thin['base']['gain_vs_market']:+.5f}")
        print(f"  marché + Elo enrichi     log-loss {thin['rich']['log_loss']:.5f}  "
              f"gain {thin['rich']['gain_vs_market']:+.5f}")
        print(f"  apport de l'enrichissement: {thin['enrichment_gain']:+.5f}")
    print(f"rapport: {output}")
    print("Mesure sur données ATP déjà dépensées: ce n'est pas une preuve de rentabilité.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
