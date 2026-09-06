#!/usr/bin/env python3
"""Does rating players on every tier they played make the rating better?

Baseline: Elo from ATP main-tour matches only, as every study here has used.
Enriched: the same engine, additionally fed the Challenger and qualifying
matches. Both arms are scored on the same main-tour matches.

The ATP tables are already spent, so a gain measured here is not evidence that
any strategy profits. It answers a narrower question that is still worth asking
before deploying anything forward: does the extra data make the rating sharper,
and does it help most where it should — on players the main tour has barely seen?
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

from src.features.multi_tier_ratings import compare_rating_quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()

    main_table = pd.read_csv(root / "data" / "processed" / "atp_matches_enriched.csv.gz", low_memory=False)
    unpriced = pd.read_csv(root / "data" / "processed" / "atp_unpriced_matches.csv.gz", low_memory=False)

    progress = None if args.quiet else (lambda message: print(message, flush=True))
    output = root / "models" / "multi_tier_ratings.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = compare_rating_quality(
        main_table,
        unpriced,
        progress=progress,
        feature_output=root / "models" / "multi_tier_features.parquet",
    )
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nMatchs dans la passe de notation: "
          f"{report['matches_in_rating_pass']['baseline']:,} -> "
          f"{report['matches_in_rating_pass']['enriched']:,}")
    extra = report["extra_matches_known_per_player"]
    print(f"Matchs supplémentaires connus par joueur: médiane {extra['median']:.0f}, "
          f"moyenne {extra['mean']:.1f}; "
          f"{extra['share_of_matches_with_new_information']:.1%} des matchs concernés\n")
    print(f"{'groupe':24s} {'n':>7s} {'log-loss base':>14s} {'enrichi':>10s} {'gain':>10s} {'gain AUC':>10s}")
    for name, block in report["groups"].items():
        print(f"{name:24s} {block['baseline']['n']:7d} "
              f"{block['baseline']['log_loss']:14.5f} {block['enriched']['log_loss']:10.5f} "
              f"{block['log_loss_gain']:+10.5f} {block['auc_gain']:+10.5f}")
    print(f"\nRapport: {output}")
    print("Mesure sur données ATP déjà dépensées: ce n'est pas une preuve de rentabilité.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
