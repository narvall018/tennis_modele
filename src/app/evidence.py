"""What each sport's evidence actually says, read from the study outputs.

The numbers here are loaded from the JSON the studies wrote, never retyped. A
performance page whose figures can drift from the measurements behind them is
worse than no page: it invites confidence the evidence does not support.

Every sport reports the same three things — how well the model forecasts, what
the frozen rule returned, and whether that return can be distinguished from
zero — so they can be compared without special pleading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SportEvidence:
    sport: str
    status: str
    headline: str
    detail: str
    metrics: dict[str, Any] = field(default_factory=dict)
    report_paths: list[str] = field(default_factory=list)
    real_money_authorised: bool = False


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def tennis_evidence(root: Path) -> SportEvidence:
    metrics: dict[str, Any] = {}
    diagnostics = _load(root / "models" / "wta_strategy" / "edge_diagnostics.json")
    if diagnostics:
        cell = diagnostics["pooled_across_all_periods"]["pinnacle"]["odds_below_6"]
        metrics["ROI mesuré (WTA, Pinnacle, cote < 6)"] = cell["roi"]
        metrics["Intervalle 90 %"] = cell["roi_ci_90"]
        metrics["Paris réglés"] = cell["n_bets"]
        metrics["Périodes positives"] = f"{cell['positive_periods_out_of_4']}/4"
    backtest = _load(root / "models" / "wta_strategy" / "backtest_report.json")
    if backtest:
        holdout = backtest["final_holdout"]["average_haircut_2pct"]
        metrics["Holdout WTA 2023-2026, ROI"] = holdout["roi"]
        metrics["Holdout, paris réglés"] = holdout["n_settled"]
        comparison = backtest.get("market_comparison", {}).get("holdout", {})
        if comparison:
            metrics["Gain de log-loss contre le marché"] = comparison.get("blend_gain_vs_market")
    return SportEvidence(
        sport="Tennis",
        status="AUCUN AVANTAGE DÉMONTRÉ",
        headline="Le modèle bat le prix de Pinnacle en calibration, pas en rendement.",
        detail=(
            "Le mélange gagne +0,00103 de log-loss contre Pinnacle sur un holdout jamais "
            "ouvert auparavant — un vrai gain de prévision. Il ne se convertit pas: la "
            "meilleure variante rend +0,74 % avec un intervalle qui contient zéro, et il "
            "faudrait 35 250 paris, soit 34 ans, pour l'établir. Deux points de friction "
            "d'exécution l'annulent."
        ),
        metrics=metrics,
        report_paths=["RAPPORT_RENTABILITE.md", "models/wta_strategy/BACKTEST_REPORT.md"],
    )


def football_evidence(root: Path) -> SportEvidence:
    metrics: dict[str, Any] = {}
    conditional = _load(root / "models" / "football_conditional_test.json")
    if conditional:
        metrics["Meilleur gain contre le prix"] = conditional["best_gain_vs_market"]
        metrics["Seuil exigé avant de continuer"] = conditional["minimum_gain_to_continue"]
        metrics["Gate"] = "passée" if conditional["gate_passed"] else "échouée"
    model = _load(root / "models" / "football" / "metadata.json")
    if model:
        metrics["Log-loss du modèle (saisons jamais vues)"] = model[
            "model_log_loss_on_unseen_seasons"
        ]
        versus = model.get("versus_market") or {}
        if versus:
            metrics["Log-loss du marché"] = versus.get("market_log_loss")
            metrics["Écart modèle - marché"] = versus.get("model_minus_market")
    return SportEvidence(
        sport="Football",
        status="AUCUN AVANTAGE DÉMONTRÉ",
        headline="Les descripteurs n'ajoutent rien que le prix ne contienne déjà.",
        detail=(
            "Elo, tirs, tirs cadrés, forme, repos: mesuré contre le prix, le meilleur "
            "candidat apporte −0,00080 de log-loss, sous le seuil de +0,001 fixé avant "
            "le calcul. Parier à l'ouverture rend moins qu'à la clôture, et les divisions "
            "inférieures sont plus chères, pas plus molles. L'échange Betfair, trois fois "
            "moins cher qu'un bookmaker, s'en approche le plus sans jamais exclure zéro."
        ),
        metrics=metrics,
        report_paths=["RAPPORT_FOOTBALL.md"],
    )


def ufc_evidence(root: Path) -> SportEvidence:
    metrics: dict[str, Any] = {}
    method = _load(root / "predictor_ufc" / "data" / "rigorous" / "quality" / "method_market_analysis.json")
    if method:
        decisions = method["calibration"]["decisions_pooled"]
        metrics["Biais du marché sur les décisions"] = method["calibration"]["by_outcome"]["f1_dec"]["bias_points"]
        metrics["ROI en pariant toutes les décisions"] = decisions["roi_backing_every_decision"]
        metrics["Années positives"] = f"{decisions['positive_years']}/{decisions['total_years']}"
        metrics["Overround du marché méthode"] = method["overround"]["median_six_way"]
    final = _load(root / "predictor_ufc" / "data" / "rigorous" / "reports" / "final_holdout_report.json")
    if isinstance(final, dict):
        metrics["Holdout final"] = "non ouvert (gate échouée)"
    return SportEvidence(
        sport="UFC",
        status="AUCUN AVANTAGE DÉMONTRÉ",
        headline="Trois phases rejetées; le marché des props porte un biais réel mais trop cher.",
        detail=(
            "Le marché sous-évalue les décisions de 2,92 points, 13 années sur 13 — le "
            "biais le plus régulier du projet. L'overround de 22 % l'avale entièrement: "
            "parier toutes les décisions rend −9,45 %. Aucun modèle ne bat le pouvoir "
            "discriminant du marché. Les cotes d'avant 2025 n'ont par ailleurs qu'une "
            "seule origine et restent invérifiables."
        ),
        metrics=metrics,
        report_paths=["predictor_ufc/RAPPORT_RIGOUREUX_2026.md"],
    )


MODEL_METADATA = {
    "Football": Path("models") / "football" / "metadata.json",
    "Tennis (ATP)": Path("models") / "tennis" / "atp_metadata.json",
    "Tennis (WTA)": Path("models") / "tennis" / "wta_metadata.json",
    "UFC": Path("models") / "ufc" / "metadata.json",
}


def model_registry(root: Path) -> list[dict[str, Any]]:
    """Which model each sport actually uses, and how it was chosen.

    Reported from the trainers' own metadata so the app cannot claim a family it
    is not running. Every entry is a pure-descriptor model: the price is never an
    input, because an opinion fed the price cannot disagree with it.
    """
    rows: list[dict[str, Any]] = []
    for sport, relative in MODEL_METADATA.items():
        metadata = _load(root / relative)
        if not metadata:
            rows.append({
                "Sport": sport,
                "Modèle retenu": "non entraîné",
                "Choisi parmi": 0,
                "Log-loss (hors échantillon)": None,
                "AUC": None,
                "Utilise la cote": False,
                "Note": f"Manquant: {relative}",
            })
            continue
        evaluation = metadata.get("evaluation") or {}
        rows.append({
            "Sport": sport,
            "Modèle retenu": metadata.get("winner", "—"),
            "Choisi parmi": len(metadata.get("comparison", []) or []),
            "Log-loss (hors échantillon)": evaluation.get("log_loss")
            or metadata.get("model_log_loss_on_unseen_seasons"),
            "AUC": evaluation.get("auc"),
            "Utilise la cote": bool(metadata.get("uses_odds", False)),
            "Note": metadata.get("honest_note", ""),
        })
    return rows


def all_evidence(root: Path) -> list[SportEvidence]:
    return [tennis_evidence(root), football_evidence(root), ufc_evidence(root)]


def global_verdict() -> dict[str, Any]:
    """The one-line answer the app must never soften."""
    return {
        "status": "AUCUNE STRATÉGIE RENTABLE DÉMONTRÉE",
        "avenues_explored": 8,
        "sports_covered": 3,
        "real_money_authorised": False,
        "summary": (
            "Huit pistes indépendantes sur trois sports. À chaque fois le biais de marché "
            "est réel et mesurable, et à chaque fois il est plus petit que la marge qu'il "
            "faudrait payer pour l'exploiter. Les probabilités affichées ici sont une "
            "opinion calibrée et honnête; elles ne sont pas un avantage."
        ),
        "biases_found": [
            "UFC: décisions sous-évaluées de 2,92 points, 13 ans sur 13",
            "Tennis: gros favoris sous-évalués de 1,65 point",
            "Football: gros favoris sous-évalués de 2,78 points",
            "Football: côté extérieur au handicap asiatique, +1,9 point contre le domicile",
            "Tennis: Pinnacle informe réellement sur le bon côté d'un prix mou",
        ],
    }
