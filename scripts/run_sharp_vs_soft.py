#!/usr/bin/env python3
"""Test the one strategy that uses no sports model: sharp book against soft book."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.sharp_vs_soft import EDGE_THRESHOLDS, analyse, prepare


def _table(name: str, report: dict) -> None:
    print(f"\n=== {name} — {report['matches']:,} matchs, {report['years'][0]}-{report['years'][1]} ===")
    print(f"{'source':16s} {'cotée':>7s} {'overround':>10s} {'arb implicite':>14s} {'pariable':>9s}")
    for source, audit in report["simultaneity_audit"].items():
        print(f"{source:16s} {audit['quoted_matches']:7d} {audit['median_overround']:10.4f} "
              f"{audit['implied_arbitrage_rate']:13.2%} "
              f"{'oui' if audit['executable_as_a_pair'] else 'NON':>9s}")

    print(f"\n{'book':22s} {'seuil':>6s} {'n':>7s} {'taux':>6s} {'ROI':>8s} "
          f"{'IC90':>18s} {'ans+':>7s} {'controle':>9s}")
    for book, block in report["books"].items():
        for threshold in EDGE_THRESHOLDS:
            key = f"{threshold:.2f}"
            cell = block["thresholds"][key]
            control = block["falsification"][key]
            if not cell["n_bets"]:
                continue
            low, high = cell["roi_ci_90"]
            interval = "-" if low is None else f"[{low:+.3f},{high:+.3f}]"
            control_roi = "-" if control["roi"] is None else f"{control['roi']:+.3f}"
            star = " *" if cell["interval_excludes_zero"] else "  "
            label = book if block["executable_as_a_pair"] else f"{book} (NON)"
            print(f"{label:22s} {threshold:6.2f} {cell['n_bets']:7d} {cell['bet_rate']:6.2f} "
                  f"{cell['roi']:+8.4f} {interval:>18s} "
                  f"{str(cell['positive_years']) + '/' + str(cell['total_years']):>7s} "
                  f"{control_roi:>9s}{star}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()

    reports = {}
    for tour, filename in (("ATP", "atp_tennis.csv"), ("WTA", "wta_tennis.csv")):
        frame = prepare(root / "data" / filename, tour)
        reports[tour] = analyse(frame)
        _table(tour, reports[tour])

    output = root / "models" / "sharp_vs_soft.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\n'controle' = même règle, signal inversé. Il doit perdre nettement plus.")
    print("'*' = intervalle 90% strictement positif.")
    print(f"\nRapport: {output}")
    print("Prix non horodatés: une partie d'un écart apparent peut être une cote périmée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
