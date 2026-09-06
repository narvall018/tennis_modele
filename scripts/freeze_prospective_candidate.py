#!/usr/bin/env python3
"""Freeze the one hypothesis that survived, so it can be tested forward.

Nothing here is validated. The WTA blend forecasts better than Pinnacle out of
sample, but betting on that gain returns about +0.7% with a confidence interval
straddling zero, using a longshot cutoff that was itself read off spent data.
The honest way to settle it is to declare the rule now, in full, and let
prospective results accumulate against a target that cannot be moved afterwards.

The declaration is hashed. ``scripts/collect_tennis_odds.py`` gathers the
timestamped prices; this file states what would have been bet with them, and the
sample size at which the question becomes answerable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_declaration(project_root: Path) -> dict:
    study = project_root / "models" / "wta_strategy"
    frozen = json.loads((study / "frozen_strategy.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((study / "edge_diagnostics.json").read_text(encoding="utf-8"))
    pinnacle = diagnostics["pooled_across_all_periods"]["pinnacle"]["odds_below_6"]

    return {
        "status": "HYPOTHESE_NON_VALIDEE",
        "real_money_authorised": False,
        "paper_only": True,
        "market": {
            "tour": "WTA",
            "bet_type": "match winner",
            "counterparty": "Pinnacle",
            "why_pinnacle": (
                "Choisi a priori pour sa marge basse (~2,9% d'overround), pas parce qu'il "
                "affiche le meilleur rendement du tableau: Bet365 et le prix maximum "
                "affichent mieux et sont écartés."
            ),
        },
        "probability_model": {
            "source_study": "models/wta_strategy",
            "model": frozen["model"],
            "model_weight": frozen["model_weight"],
            "market_component": "prix moyen dévigé",
            "fold_rule": "train through Y-2, calibrate on Y-1",
        },
        "bet_rule": {
            "description": "parier tout côté dont l'espérance est positive au prix Pinnacle",
            "minimum_expected_value": 0.0,
            "maximum_odds": 6.0,
            "maximum_odds_justification": (
                "La bande cote>=6 perd entre 24% et 65% dans les quatre périodes et chez "
                "toutes les sources: le modèle surestime les outsiders extrêmes. Le seuil "
                "vient toutefois de données déjà dépensées et n'est pas validé."
            ),
            "stake": "plat, 0,25% de la bankroll, exposition quotidienne plafonnée à 2%",
        },
        "price_capture": {
            "rule": "dernier prix Pinnacle observé strictement avant le début du match",
            "no_retrospective_choice": True,
            "collector": "scripts/collect_tennis_odds.py",
        },
        "retrospective_expectation": {
            "note": "mesure sur données dépensées, fournie comme point de comparaison, pas comme promesse",
            "roi": pinnacle["roi"],
            "n_bets": pinnacle["n_bets"],
            "roi_ci_90": pinnacle["roi_ci_90"],
            "positive_periods_out_of_4": pinnacle["positive_periods_out_of_4"],
            "observed_t_statistic": 0.80,
        },
        "success_criterion": {
            "declared_before_any_prospective_bet": True,
            "requirement": (
                "borne inférieure à 90% d'un bootstrap par blocs mensuels strictement "
                "positive, sur des paris prospectifs uniquement"
            ),
            "bets_needed_for_90pct": 35250,
            "bets_needed_for_95pct": 58067,
            "observed_annual_volume": 1045,
            "implied_years_at_90pct": 34,
            "honest_reading": (
                "À ce volume, la question n'est pas tranchable avant des décennies. Un "
                "résultat prospectif nettement meilleur que +0,74% serait le seul motif "
                "de reconsidérer plus tôt; un résultat conforme à +0,74% ne prouvera rien."
            ),
        },
        "kill_criterion": {
            "requirement": (
                "arrêt immédiat si le ROI prospectif est inférieur à -5% après 500 paris "
                "réglés, ou si la log-loss du mélange cesse de battre celle de Pinnacle"
            ),
        },
        "known_fragilities": [
            "2 points de friction d'exécution supplémentaires annulent le rendement.",
            "Pinnacle limite les comptes gagnants; le volume supposé n'est pas garanti.",
            "La règle parie sur 39% à 57% des matchs, ce qui est atypique d'un edge réel.",
            "Le seuil de cote 6 est une hypothèse issue des données de développement.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    path = root / "models" / "wta_strategy" / "prospective_candidate.json"

    declaration = build_declaration(root)
    canonical = json.dumps(declaration, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("declaration_sha256") != digest:
            raise SystemExit(
                f"Une déclaration différente existe déjà dans {path}.\n"
                "La réécrire viderait le test prospectif de son sens: la cible doit rester "
                "fixe pendant que les résultats s'accumulent."
            )
        print(f"Déclaration inchangée: {path} (sha256 {digest[:16]}…)")
        return 0

    payload = {
        "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "declaration_sha256": digest,
        "prospective_period_starts": "2026-09-07",
        **declaration,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Hypothèse gelée: {path} (sha256 {digest[:16]}…)")
    print(f"Statut: {payload['status']} — argent réel: {payload['real_money_authorised']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
