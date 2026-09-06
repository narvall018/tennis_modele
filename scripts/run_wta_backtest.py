#!/usr/bin/env python3
"""Run the frozen nested study on the WTA tour.

Why a second tour at all: every walk-forward search this repository has ever run
read the ATP main tour, so an ATP result can no longer distinguish an edge from
the accumulated selection of many passes over the same matches.  The WTA table
published by ``scripts/update_tennis_expansion.py`` has never been read by any
search here, which makes its holdout years genuine out-of-sample evidence — once,
and only once.

The protocol below is written to disk and hashed *before* any return is
computed.  ``--freeze-only`` writes it without running, so the frozen file can
be inspected and committed first.  A later run refuses to start if the protocol
file on disk no longer matches the protocol in this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.rigorous_strategy import ProtocolWindows, run_nested_strategy_study


# Frozen before the first WTA return was computed. The windows mirror the ATP
# study's shape, shifted so that the priced sample starts in 2007 and the
# holdout keeps four full years instead of three.
#
# Amended once, still before any return existed: the first version opened
# development in 2011, and the engine's own data-sufficiency guard refused the
# 2011 and 2012 folds (6 966 and 9 335 usable training rows against a required
# 10 000).  Development therefore starts at 2013, the first year the guard
# accepts.  The decision used only training-set sizes; no ROI, log-loss, or
# per-year result had been computed for any WTA window.  The superseded protocol
# and this reason are kept inside the frozen file.
WTA_WINDOWS = ProtocolWindows(
    development=tuple(range(2013, 2017)),
    tuning=tuple(range(2017, 2020)),
    validation=tuple(range(2020, 2023)),
    holdout=tuple(range(2023, 2027)),
    minimum_tuning_bets=120,
)

PROTOCOL_NOTES = {
    "study_label": "WTA",
    "study_label_long": "WTA main tour match winner",
    "data_source": "data/wta_tennis.csv, built from the official Tennis-Data WTA workbooks",
    "rating_warm_up_years": "2007-2010, used only to build ratings and training folds",
    "primary_price": "Tennis-Data WTA market average pre-match decimal odds",
    "why_this_tour": (
        "Aucune recherche de ce dépôt n'a jamais lu une ligne WTA. Le holdout 2023-2026 est "
        "donc une vraie preuve hors échantillon, utilisable une seule fois."
    ),
    "prior_exposure": (
        "Seul un contrôle d'intégrité agrégé a été lu avant le gel: AUC et log-loss du marché "
        "dévigé sur l'ensemble de la table, plus l'overround médian et le taux de victoire de "
        "Player_1. Aucune performance par année, par segment ou par stratégie."
    ),
    "known_inapplicable_features": (
        "best_of_5 et l'expérience en Grand Chelem à 5 sets sont constants sur le circuit WTA; "
        "les variables correspondantes restent dans la matrice et reçoivent un poids nul."
    ),
    "decision_rule": (
        "Le passage de la gate finale n'autorise qu'un suivi papier prospectif. "
        "Aucun argent réel n'est engagé sur la seule foi de ce backtest."
    ),
}


def protocol_payload() -> dict:
    return {
        "study": "wta_nested_strategy",
        "windows": {key: list(value) if isinstance(value, tuple) else value
                    for key, value in asdict(WTA_WINDOWS).items()},
        "notes": PROTOCOL_NOTES,
        "fold_rule": "train through Y-2, calibrate on Y-1, test on Y",
        "holdout_opens": "exactly once, after model, blend, rule, gate and staking are frozen",
    }


def protocol_sha256(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def freeze(protocol_path: Path, amend_reason: str | None = None) -> dict:
    """Write the protocol once, and record any amendment made before it was used.

    An amendment is only ever legitimate while the holdout is still closed, and
    only for a reason that could be known without looking at a return — a data
    volume, a missing column, a guard the engine refuses. The superseded payload
    and the stated reason are kept in the file so the amendment is auditable
    rather than invisible.
    """
    payload = protocol_payload()
    digest = protocol_sha256(payload)
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") == digest:
            return existing
        if existing.get("holdout_opened"):
            raise SystemExit(
                f"Le holdout de {protocol_path} a déjà été ouvert avec un autre protocole.\n"
                "Le modifier maintenant invaliderait la preuve. Repartez d'une période vierge."
            )
        if not amend_reason:
            raise SystemExit(
                f"Le protocole gelé dans {protocol_path} ne correspond plus au script.\n"
                "Relancez avec --amend-reason \"…\" en expliquant ce qui a imposé le changement, "
                "sachant qu'un amendement n'est admissible qu'avant tout calcul de rendement."
            )
        superseded = list(existing.get("superseded", []))
        superseded.append(
            {
                "amended_at_utc": datetime.now(timezone.utc).isoformat(),
                "reason": amend_reason,
                "previous_protocol_sha256": existing.get("protocol_sha256"),
                "previous_windows": existing.get("windows"),
            }
        )
    else:
        superseded = []

    frozen = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": digest,
        "holdout_opened": False,
        "superseded": superseded,
        **payload,
    }
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="write and hash the protocol without touching any return",
    )
    parser.add_argument(
        "--amend-reason",
        default=None,
        help="required to change a frozen protocol whose holdout is still closed",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    data_path = args.data or root / "data" / "wta_tennis.csv"
    output_dir = args.output or root / "models" / "wta_strategy"
    protocol_path = output_dir / "wta_protocol.json"

    frozen = freeze(protocol_path, args.amend_reason)
    print(f"Protocole gelé: {protocol_path} (sha256 {frozen['protocol_sha256'][:16]}…)")
    for amendment in frozen.get("superseded", []):
        print(f"  amendement {amendment['amended_at_utc']}: {amendment['reason']}")
    if args.freeze_only:
        print(json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    report = run_nested_strategy_study(
        data_path=data_path,
        output_dir=output_dir,
        bootstrap_samples=args.bootstrap_samples,
        reuse_features=args.reuse_features,
        windows=WTA_WINDOWS,
        protocol_notes={**PROTOCOL_NOTES, "protocol_sha256": frozen["protocol_sha256"]},
    )
    frozen["holdout_opened"] = True
    frozen["holdout_opened_at_utc"] = datetime.now(timezone.utc).isoformat()
    frozen["data_sha256"] = report["data_audit"]["source_sha256"]
    protocol_path.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["deployment_gate"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
