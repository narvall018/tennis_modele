#!/usr/bin/env python3
"""Audit the tennis moneyline for systematic mispricing. No model involved."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.market_audit import BETTABLE, SHARP, audit, cross_tour_consistency
from src.backtesting.sharp_vs_soft import prepare


def _row(name: str, cell: dict) -> str:
    flag = " *" if cell["significant"] else "  "
    low, high = cell["roi_ci_90"]
    return (f"{name:22s} {cell['n']:7d} {cell['market_probability']:9.4f} "
            f"{cell['realised_rate']:9.4f} {cell['calibration_gap']:+9.4f} "
            f"{cell['roi']:+8.3f} [{low:+.3f},{high:+.3f}]{flag}")


def _print(title: str, report: dict) -> None:
    print(f"\n{'=' * 92}\n{title} — {report['matches_usable']:,} matchs exploitables\n{'=' * 92}")
    header = (f"{'cellule':22s} {'n':>7s} {'p marché':>9s} {'réel':>9s} {'écart':>9s} "
              f"{'ROI':>8s} {'IC90':>16s}")

    print("\n--- Favori, par décile de probabilité (biais favori/outsider) ---")
    print(header)
    for band, cells in report["by_probability_decile"].items():
        print(_row(band, cells["favourite"]))

    print("\n--- Outsider, par décile ---")
    print(header)
    for band, cells in report["by_probability_decile"].items():
        print(_row(band, cells["underdog"]))

    for name, block in report["segments"].items():
        if not block:
            continue
        print(f"\n--- Segment: {name} (favori) ---")
        print(header)
        for value, cells in block.items():
            print(_row(value, cells["favourite"]))

    multiplicity = report["multiplicity"]
    print(f"\nCellules examinées: {multiplicity['cells_examined']} | "
          f"IC excluant zéro: {multiplicity['cells_with_interval_excluding_zero']} | "
          f"rentables: {multiplicity['cells_profitable']} | "
          f"attendues par hasard à 90%: {multiplicity['expected_by_chance_at_90pct']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()

    reports = {}
    for tour, filename in (("ATP", "atp_tennis.csv"), ("WTA", "wta_tennis.csv")):
        frame = prepare(root / "data" / filename, tour)
        for book_name, columns in (("Bet365", BETTABLE), ("Pinnacle", SHARP)):
            key = f"{tour}_{book_name}"
            reports[key] = audit(frame, bettable=columns)
            _print(f"{tour} — mises chez {book_name}", reports[key])

    print(f"\n{'=' * 92}\nContrôle croisé ATP/WTA des cellules rentables\n{'=' * 92}")
    for book in ("Bet365", "Pinnacle"):
        consistency = cross_tour_consistency(reports, book)
        reports[f"consistency_{book}"] = consistency
        cells = consistency.get("cells", {})
        if not cells:
            print(f"{book:10s}: aucune cellule rentable sur l'un ou l'autre circuit.")
            continue
        for name, cell in cells.items():
            values = " | ".join(
                f"{tour}: n={cell[tour]['n']}, ROI={cell[tour]['roi']:+.4f}"
                for tour in cell if tour.endswith(book)
            )
            print(f"{book:10s} {name:18s} {values} | poolé {cell['pooled_roi']:+.4f}")
            print(f"{'':10s} {'':18s} -> {cell['verdict']}")

    output = root / "models" / "market_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\n'*' = intervalle 90% excluant zéro. 'hash_*' est un découpage aléatoire témoin:")
    print("il doit produire autant de cellules marquées que le hasard le prévoit, pas moins.")
    print(f"\nRapport: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
