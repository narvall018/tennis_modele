#!/usr/bin/env python3
"""Measure whether a study's forecasting gain is large enough to be worth money.

This reads data that has already been spent and selects nothing. It exists to
answer one question honestly: a log-loss gain of about 0.001 was measured — can
a gain that size clear a bookmaker's margin, and where in the price range does
it live?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.edge_diagnostics import LONGSHOT_CAP, diagnose_study
from src.backtesting.rigorous_strategy import DEFAULT_WINDOWS, ProtocolWindows

STUDIES = {
    "wta": {
        "directory": PROJECT_ROOT / "models" / "wta_strategy",
        "windows": ProtocolWindows(
            development=tuple(range(2013, 2017)),
            tuning=tuple(range(2017, 2020)),
            validation=tuple(range(2020, 2023)),
            holdout=tuple(range(2023, 2027)),
        ),
    },
    "atp": {
        "directory": PROJECT_ROOT / "models" / "rigorous_strategy",
        "windows": DEFAULT_WINDOWS,
    },
}


def _print_table(report: dict) -> None:
    cap_key = f"odds_below_{LONGSHOT_CAP:g}"
    periods = list(report["periods"])
    for variant, title in (("all_odds", "toutes cotes"), (cap_key, f"cote < {LONGSHOT_CAP:g}")):
        print(f"\n--- Parier tout EV positif, {title}, décote 2% ---")
        header = f"{'source':10s}" + "".join(f"{p[:12]:>18s}" for p in periods)
        print(header + f"{'TOTAL':>26s}")
        for source, cells in report["pooled_across_all_periods"].items():
            row = f"{source:10s}"
            for period in periods:
                cell = report["periods"][period]["sources"][source][variant]
                roi = "-" if cell["roi"] is None else format(cell["roi"], "+.3f")
                row += f"{str(cell['n_bets']) + '/' + roi:>18s}"
            pooled = cells[variant]
            low, high = pooled["roi_ci_90"]
            interval = "-" if low is None else f"[{low:+.3f},{high:+.3f}]"
            row += f"{format(pooled['roi'], '+.4f') + ' ' + interval:>26s}"
            print(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=sorted(STUDIES))
    args = parser.parse_args()
    config = STUDIES[args.study]
    directory = config["directory"]

    report = diagnose_study(
        directory / "oos_predictions.csv.gz",
        directory / "frozen_strategy.json",
        config["windows"],
    )
    output = directory / "edge_diagnostics.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _print_table(report)
    print(f"\nSeuil approximatif de rentabilité (gain de log-loss requis par la marge):")
    for source, block in report["break_even_reference"].items():
        required = block["approximate_log_loss_gain_required"]
        print(
            f"  {source:10s} overround={block['median_overround']:.4f} "
            f"-> requis ~{required:.5f}" if required is not None else f"  {source}: n/a"
        )
    print(f"\nRapport complet: {output}")
    print(report["what_this_is"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
