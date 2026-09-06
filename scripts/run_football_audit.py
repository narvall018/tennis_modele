#!/usr/bin/env python3
"""Audit the football markets: opening against closing, and margin against bias.

Nothing here fits a model. It asks whether the opening price is soft enough to
bet, and whether a calibration bias survives in the markets whose margin is small
enough to let one through.
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

from src.backtesting.football_audit import (
    asian_handicap_audit,
    asian_handicap_by_country,
    calibration_audit,
    cross_league_consistency,
    timing_comparison,
)

LEG_NAMES = {
    "1x2": ("domicile", "nul", "extérieur"),
    "over_under_25": ("plus de 2,5", "moins de 2,5"),
    "asian_handicap": ("handicap dom.", "handicap ext."),
}


def _timing(frame: pd.DataFrame, report: dict) -> None:
    print(f"\n{'=' * 100}\nOUVERTURE CONTRE CLÔTURE — même matchs, même sélection, décote 2%\n{'=' * 100}")
    for market in ("1x2", "over_under_25", "asian_handicap"):
        block = timing_comparison(frame, market, "pinnacle")
        report[f"timing_{market}"] = block
        if not block.get("available"):
            print(f"\n{market}: indisponible ({block.get('n', 0)} matchs exploitables)")
            continue
        print(f"\n--- {market} — {block['matches']:,} matchs avec les deux cotes ---")
        print(f"  log-loss ouverture {block['open_log_loss']:.5f} | clôture {block['close_log_loss']:.5f} "
              f"| la clôture gagne {block['closing_beats_opening_by']:+.5f}")
        print(f"  overround ouverture {block['open_overround']:.4f} | clôture {block['close_overround']:.4f} "
              f"| dérive moyenne {block['mean_absolute_drift']:.4f}")
        names = LEG_NAMES[market]
        print(f"  {'issue':16s} {'ROI ouverture':>14s} {'ROI clôture':>13s} {'écart':>8s} {'IC90 ouverture':>18s}")
        for index, name in enumerate(names):
            leg = block["legs"].get(f"leg_{index}")
            if not leg:
                continue
            low, high = leg["at_open"]["roi_ci_90"]
            interval = "-" if low is None else f"[{low:+.3f},{high:+.3f}]"
            star = " *" if leg["at_open"]["profitable"] else ""
            print(f"  {name:16s} {leg['at_open']['roi']:+14.4f} {leg['at_close']['roi']:+13.4f} "
                  f"{leg['open_minus_close_roi']:+8.4f} {interval:>18s}{star}")


def _calibration(frame: pd.DataFrame, report: dict) -> None:
    print(f"\n{'=' * 100}\nCALIBRATION PAR MARCHÉ — prix Pinnacle de clôture\n{'=' * 100}")
    for market in ("1x2", "over_under_25", "asian_handicap"):
        block = calibration_audit(frame, market, "pinnacle_close")
        report[f"calibration_{market}"] = block
        if not block.get("available") or not block.get("bands"):
            print(f"\n{market}: pas de calibration exploitable (issues non déterminables)")
            continue
        print(f"\n--- {market} — {block['matches']:,} matchs, overround {block['overround_median']:.4f} ---")
        print(f"  {'bande':14s} {'n':>7s} {'p marché':>9s} {'réel':>8s} {'écart':>9s} {'ROI':>8s} {'IC90':>18s}")
        for band, cell in block["bands"].items():
            low, high = cell["roi_ci_90"]
            star = " *" if cell["profitable"] else ""
            print(f"  {band:14s} {cell['n']:7d} {cell['market_probability']:9.4f} "
                  f"{cell['hit_rate']:8.4f} {cell['calibration_gap']:+9.4f} {cell['roi']:+8.4f} "
                  f"[{low:+.3f},{high:+.3f}]{star}")


def _divisions(frame: pd.DataFrame, report: dict) -> None:
    print(f"\n{'=' * 100}\nPROFONDEUR DE DIVISION — le marché est-il plus mou en bas ?\n{'=' * 100}")
    rows = {}
    print(f"  {'division':12s} {'n':>7s} {'overround':>10s} {'ROI favori':>11s} {'IC90':>18s}")
    for rank in sorted(frame["division_rank"].unique()):
        subset = frame[frame["division_rank"] == rank]
        block = calibration_audit(subset, "1x2", "pinnacle_close")
        if not block.get("available") or not block.get("bands"):
            continue
        top_band = max(block["bands"].items(), key=lambda item: float(item[0].split("-")[0]))
        cell = top_band[1]
        low, high = cell["roi_ci_90"]
        rows[f"rang_{rank}"] = {"overround": block["overround_median"], "top_band": top_band[0], **cell}
        print(f"  rang {rank:<7d} {cell['n']:7d} {block['overround_median']:10.4f} "
              f"{cell['roi']:+11.4f} [{low:+.3f},{high:+.3f}]{' *' if cell['profitable'] else ''}")
    report["divisions"] = rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()

    frame = pd.read_csv(root / "data" / "football" / "football_matches.csv.gz", low_memory=False)
    print(f"{len(frame):,} matchs, {frame['league'].nunique()} divisions, "
          f"{frame['match_date'].min()} → {frame['match_date'].max()}")

    report: dict = {}
    _timing(frame, report)
    _calibration(frame, report)
    _divisions(frame, report)

    print(f"\n{'=' * 100}\nHANDICAP ASIATIQUE — la marge la plus faible du jeu de données\n{'=' * 100}")
    ah = {}
    print(f"  {'book/moment':22s} {'n':>7s} {'overround':>10s} {'côté':>6s} "
          f"{'ROI':>9s} {'remb.':>7s} {'IC90':>18s}")
    for book in ("pinnacle", "bet365", "market_average"):
        for timing in ("open", "close"):
            block = asian_handicap_audit(frame, book, timing)
            ah[f"{book}_{timing}"] = block
            if not block.get("available"):
                continue
            for side, cell in block["sides"].items():
                low, high = cell["roi_ci_90"]
                star = " *" if cell["profitable"] else ""
                print(f"  {book + '/' + timing:22s} {cell['n']:7d} "
                      f"{block['overround_median']:10.4f} {side:>6s} {cell['roi']:+9.4f} "
                      f"{cell['push_rate']:7.3f} [{low:+.3f},{high:+.3f}]{star}")
    report["asian_handicap"] = ah

    for key, block in ah.items():
        if not block.get("available"):
            continue
        for side, cell in block["sides"].items():
            if not cell["profitable"]:
                continue
            book, timing = key.rsplit("_", 1)
            check = asian_handicap_by_country(frame, book, timing, side)
            report[f"ah_cross_league_{key}_{side}"] = check
            print(f"\n  Contrôle croisé {key}/{side}: rentable dans "
                  f"{len(check['profitable_countries'])}/{check['countries_examined']} pays "
                  f"-> {'CONFIRMÉ' if check['confirmed_in_majority'] else 'cellule chanceuse'}")

    print(f"\n{'=' * 100}\nCONTRÔLE CROISÉ PAR PAYS des issues rentables\n{'=' * 100}")
    checks = {}
    for market in ("1x2", "over_under_25"):
        for leg in range(3 if market == "1x2" else 2):
            block = report.get(f"calibration_{market}", {})
            if not block.get("available"):
                continue
            consistency = cross_league_consistency(frame, market, "pinnacle_close", leg)
            if not consistency.get("available"):
                continue
            checks[f"{market}_leg{leg}"] = consistency
            name = LEG_NAMES[market][leg]
            print(f"  {market:16s} {name:16s} rentable dans "
                  f"{len(consistency['profitable_countries'])}/{consistency['countries_examined']} pays "
                  f"-> {'CONFIRMÉ' if consistency['confirmed_in_majority'] else 'non confirmé'}")
    report["cross_league"] = checks

    output = root / "models" / "football_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\n'*' = intervalle 90% strictement positif.\nRapport: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
