#!/usr/bin/env python3
"""Étude des descripteurs de marge tirés du score jeu par jeu.

    python3 scripts/run_margin_study.py --freeze-only   # écrit et hashe le protocole
    python3 scripts/run_margin_study.py                 # exécute les deux tests

Le protocole est gelé avant toute mesure. L'étude ne produit aucune conclusion
économique: ATP et WTA sont brûlés sur ce plan, et une gate conditionnelle
franchie ne serait qu'une hypothèse à confirmer sur du budget vierge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.margin_study import freeze_protocol, run_study


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true", help="écrire le protocole et s'arrêter")
    parser.add_argument("--cache", type=Path, default=None, help="répertoire de cache des tables")
    args = parser.parse_args()

    if args.freeze_only:
        protocol = freeze_protocol(PROJECT_ROOT)
        print(f"Protocole gelé: {protocol['protocol_sha256']}")
        return 0

    report = run_study(PROJECT_ROOT, cache_dir=args.cache, progress=lambda m: print(m, flush=True))
    print("\n" + "=" * 72)
    for tour, verdict in report["verdicts"].items():
        print(
            f"{tour.upper():4s} gate A (descripteurs) {verdict['gate_a_gain']:+.5f} "
            f"seuil {verdict['gate_a_threshold']:.3f} → {'FRANCHIE' if verdict['gate_a_passed'] else 'échouée'}"
        )
        gain_b = verdict["gate_b_gain"]
        print(
            f"{tour.upper():4s} gate B (conditionnel) {gain_b:+.5f} "
            f"seuil {verdict['gate_b_threshold']:.3f} → {'FRANCHIE' if verdict['gate_b_passed'] else 'échouée'}"
        )
    print(f"\nStatut: {report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
