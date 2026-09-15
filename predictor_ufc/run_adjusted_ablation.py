#!/usr/bin/env python3
"""Ablation de développement: les descripteurs ajustés améliorent-ils le modèle UFC ?

    python3 predictor_ufc/run_adjusted_ablation.py --freeze-only
    python3 predictor_ufc/run_adjusted_ablation.py

Le découpage reprend celui de la phase 3 déjà gelée dans ce dossier: fenêtre de
développement 2015-2024, holdout économique 2025-09-13 → 2026-08-29 **jamais
ouvert**. La gate est écrite avant la mesure; si elle échoue, le holdout reste
fermé et réutilisable, ce qui est le seul budget de preuve UFC encore intact.

Deux défauts de la table existante sont corrigés ou contournés ici, et signalés
dans le rapport plutôt que tus:

1. `features_v2.parquet` applique une symétrisation qui inverse le signe de
   `diff_*`, `elo_diff`, `reach_diff`, `age_diff`, `market_logit` et de `y`, et
   qui permute `fighter_1`/`fighter_2`, mais **ne permute pas**
   `fighter_1_id`/`fighter_2_id`, ni `elo_1_pre`/`elo_2_pre`, ni les colonnes
   `f1_*`/`f2_*`. L'orientation est donc incohérente d'une famille de colonnes à
   l'autre. Elle est reconstruite ici à partir du signe de `elo_diff`.
2. `height_diff` est intégralement vide et `southpaw_matchup` est constamment
   nul sur les 6 719 combats. Deux des descripteurs annoncés ne portent aucune
   information; ils sont exclus des deux bras.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rigorous.adjusted_features import (  # noqa: E402
    ADJUSTED_FEATURES,
    build_adjusted_features,
    orient_to_pairs,
)

DEAD_FEATURES = ["height_diff", "southpaw_matchup"]
DUPLICATE_OF_EXISTING = ["age_diff_years"]

BASELINE_FEATURES = [
    "diff_sig_lnd_L5", "diff_sig_acc_L5", "diff_td_lnd_L5", "diff_td_acc_L5",
    "diff_sub_att_L5", "diff_ctrl_secs_L5", "diff_kd_L5",
    "diff_def_sig_absorbed_L5", "diff_def_td_absorbed_L5",
    "diff_def_ctrl_secs_L5", "diff_def_kd_L5", "diff_net_sig_L5",
    "diff_result_win_L5", "diff_ufc_fights", "diff_days_off",
    "elo_diff", "reach_diff", "age_diff",
]
NEW_FEATURES = [c for c in ADJUSTED_FEATURES if c not in DUPLICATE_OF_EXISTING]

GATE_MIN_GAIN_VS_BASELINE = 0.001
GATE_MIN_YEARS_BEATING_BASELINE = 7
GATE_TOTAL_YEARS = 10

PROTOCOL = {
    "study": "ufc_adjusted_descriptors",
    "protocol_version": "1.0.0",
    "frozen_before_any_return": True,
    "hypothesis": (
        "Des statistiques ajustées à ce que l'adversaire concède, une usure "
        "cumulée de carrière et une courbe d'âge améliorent le modèle UFC par "
        "rapport aux moyennes brutes sur cinq combats."
    ),
    "development_window": ["2015-01-01", "2024-12-31"],
    "pristine_economic_holdout": ["2025-09-13", "2026-08-29"],
    "holdout_opened": False,
    "holdout_rule": (
        "Le holdout n'est ouvert que si la gate de développement est franchie. "
        "C'est le dernier budget de preuve UFC intact: les trois études "
        "précédentes se sont arrêtées avant de l'ouvrir."
    ),
    "walk_forward": {
        "first_test_year": 2015,
        "last_test_year": 2024,
        "train_uses_only_dates_strictly_before_test_year": True,
        "minimum_training_rows": 1000,
    },
    "baseline_features": BASELINE_FEATURES,
    "challenger_features": BASELINE_FEATURES + NEW_FEATURES,
    "excluded_dead_features": {
        "columns": DEAD_FEATURES,
        "reason": (
            "height_diff est vide sur 6 719 combats sur 6 719; southpaw_matchup "
            "ne prend qu'une seule valeur. Ni l'un ni l'autre ne peut porter "
            "d'information, dans aucun des deux bras."
        ),
    },
    "excluded_duplicates": {
        "columns": DUPLICATE_OF_EXISTING,
        "reason": "age_diff existe déjà dans la table; le doublon fausserait l'ablation.",
    },
    "orientation_repair": (
        "features_v2 inverse le signe des colonnes signées et de y sur 3 671 des "
        "6 719 combats sans permuter fighter_1_id/fighter_2_id. L'orientation "
        "logique est reconstruite par comparaison de elo_diff à elo_1_pre - "
        "elo_2_pre, puis vérifiée sur les noms."
    ),
    "model_families": {
        "logistic": {"C": 0.05, "max_iter": 2000},
        "hist_gradient_boosting": {
            "max_iter": 300, "learning_rate": 0.06, "max_leaf_nodes": 31,
            "min_samples_leaf": 20, "l2_regularization": 1.0, "random_state": 0,
        },
    },
    "gate_before_opening_holdout": {
        "challenger_log_loss_must_beat_baseline_by": GATE_MIN_GAIN_VS_BASELINE,
        "challenger_must_beat_market": True,
        "years_beating_baseline_minimum": GATE_MIN_YEARS_BEATING_BASELINE,
        "years_total": GATE_TOTAL_YEARS,
        "note": "Seuils repris tels quels du protocole phase 3 déjà gelé dans ce dossier.",
    },
    "uncertainty": {
        "method": "bootstrap apparié par grappes d'événement (soirée)",
        "draws": 2000,
        "level": 0.90,
    },
    "economic_claim": (
        "Aucune. Cette ablation mesure de la prévision sur une fenêtre déjà "
        "explorée par les études antérieures. Elle n'autorise aucune mise."
    ),
}


def _canonical_hash(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_protocol() -> dict:
    target = BASE_DIR / "adjusted_protocol.json"
    payload = json.loads(json.dumps(PROTOCOL, ensure_ascii=False))
    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["protocol_sha256"] = _canonical_hash(
        {k: v for k, v in payload.items() if k != "frozen_at_utc"}
    )
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") == payload["protocol_sha256"]:
            return existing
        payload["superseded"] = existing.get("superseded", []) + [
            {"previous_protocol_sha256": existing.get("protocol_sha256")}
        ]
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_oriented_table() -> tuple[pd.DataFrame, dict]:
    """Table v2 + descripteurs ajustés, réorientés de façon cohérente avec y."""
    processed = BASE_DIR / "data" / "processed" / "features_v2.parquet"
    v2 = pd.read_parquet(processed).reset_index(drop=True)

    # La symétrisation de `pipeline/01_build_features.py` tire son masque avec
    # une graine fixe, donc il se reconstruit exactement. Le signe de elo_diff
    # sert de contrôle: il tranche partout sauf sur les combats où les deux Elo
    # valent 1500, où les deux orientations sont numériquement identiques.
    gap = v2["elo_1_pre"] - v2["elo_2_pre"]
    looks_flipped = np.isclose(v2["elo_diff"], -gap, atol=1e-6)
    looks_consistent = np.isclose(v2["elo_diff"], gap, atol=1e-6)
    if not bool((looks_flipped | looks_consistent).all()):
        raise AssertionError("Orientation irrécupérable: elo_diff ne correspond à aucun des deux signes")

    ambiguous = looks_flipped & looks_consistent
    flipped = np.random.default_rng(seed=42).random(len(v2)) < 0.5
    decided = ~ambiguous
    if not bool((flipped[decided] == looks_flipped[decided]).all()):
        raise AssertionError(
            "Le masque de symétrisation reconstruit ne correspond plus au signe "
            "de elo_diff: l'ordre des lignes de features_v2.parquet a changé."
        )

    # Sur une ligne retournée, le combattant auquel y se rapporte est celui que
    # la colonne fighter_2_id désigne.
    true_p1 = np.where(flipped, v2["fighter_2_id"], v2["fighter_1_id"])
    true_p2 = np.where(flipped, v2["fighter_1_id"], v2["fighter_2_id"])
    pairs = pd.DataFrame(
        {"fight_id": v2["fight_id"], "fighter_1_id": true_p1, "fighter_2_id": true_p2}
    )

    appearances = pd.read_parquet(BASE_DIR / "data" / "interim" / "asof_full.parquet")
    bio = pd.read_parquet(BASE_DIR / "data" / "raw" / "fighter_bio.parquet")
    per_fighter = build_adjusted_features(appearances, bio, progress=lambda m: print(m, flush=True))
    adjusted = orient_to_pairs(per_fighter, pairs)

    table = pd.concat([v2, adjusted.drop(columns="fight_id")], axis=1)
    table["_year"] = pd.to_datetime(table["event_date"]).dt.year
    table["_event"] = pd.to_datetime(table["event_date"]).dt.strftime("%Y-%m-%d")

    audit = {
        "fights": int(len(table)),
        "flipped_rows": int(flipped.sum()),
        "consistent_rows": int((~flipped).sum()),
        "ambiguous_rows_resolved_by_seed": int(ambiguous.sum()),
        "age_diff_correlation_after_repair": float(
            table[["age_diff", "age_diff_years"]].corr().iloc[0, 1]
        ),
    }
    return table, audit


def _estimator(family: str):
    spec = PROTOCOL["model_families"][family]
    if family == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=spec["C"], max_iter=spec["max_iter"])),
            ]
        )
    return HistGradientBoostingClassifier(**spec)


def _row_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return -(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped))


def _walk_forward(
    frame: pd.DataFrame, columns: list[str], family: str, test_mask: np.ndarray
) -> pd.DataFrame:
    """Entraîne sur toutes les dates strictement antérieures à l'année testée.

    `frame` porte l'historique complet: les combats antérieurs à la fenêtre de
    développement servent à entraîner, jamais à tester. `test_mask` restreint ce
    qui est évalué à cette fenêtre.
    """
    wf = PROTOCOL["walk_forward"]
    years = frame["_year"].to_numpy()
    labels = frame["y"].to_numpy(dtype=int)
    matrix = frame[columns].to_numpy(dtype=np.float64)
    pieces = []
    for year in range(wf["first_test_year"], wf["last_test_year"] + 1):
        train = years < year
        test = (years == year) & test_mask
        if train.sum() < wf["minimum_training_rows"] or test.sum() == 0:
            continue
        estimator = _estimator(family)
        estimator.fit(matrix[train], labels[train])
        pieces.append(
            pd.DataFrame(
                {
                    "fight_id": frame.loc[test, "fight_id"].to_numpy(),
                    "_year": year,
                    "_event": frame.loc[test, "_event"].to_numpy(),
                    "y": labels[test],
                    "probability": np.clip(estimator.predict_proba(matrix[test])[:, 1], 1e-6, 1 - 1e-6),
                }
            )
        )
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _bootstrap(diff: np.ndarray, clusters: np.ndarray, draws: int = 2000, level: float = 0.90) -> dict:
    if diff.size == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(0)
    unique = np.unique(clusters)
    index = {value: np.flatnonzero(clusters == value) for value in unique}
    means = np.empty(draws)
    for draw in range(draws):
        picked = rng.choice(unique, size=unique.size, replace=True)
        means[draw] = diff[np.concatenate([index[v] for v in picked])].mean()
    tail = (1.0 - level) / 2.0
    return {
        "mean": float(diff.mean()),
        "low": float(np.quantile(means, tail)),
        "high": float(np.quantile(means, 1.0 - tail)),
        "clusters": int(unique.size),
    }


def run() -> dict:
    protocol = freeze_protocol()
    table, audit = load_oriented_table()
    print(f"Orientation: {audit['flipped_rows']:,} lignes retournées sur {audit['fights']:,}; "
          f"corrélation age_diff après réparation = {audit['age_diff_correlation_after_repair']:+.4f}", flush=True)

    window = PROTOCOL["development_window"]
    dates = pd.to_datetime(table["event_date"])
    in_development = ((dates >= window[0]) & (dates <= window[1])).to_numpy()
    development = table.reset_index(drop=True)
    print(f"Développement: {int(in_development.sum()):,} combats testés {window[0]} → {window[1]}; "
          f"{int((dates < window[0]).sum()):,} combats antérieurs servent uniquement à entraîner", flush=True)

    results = {}
    for family in PROTOCOL["model_families"]:
        baseline = _walk_forward(development, BASELINE_FEATURES, family, in_development)
        challenger = _walk_forward(development, BASELINE_FEATURES + NEW_FEATURES, family, in_development)
        merged = baseline.merge(challenger, on=["fight_id", "_year", "_event", "y"], suffixes=("_b", "_c"))
        labels = merged["y"].to_numpy(int)
        loss_b = _row_log_loss(labels, merged["probability_b"].to_numpy())
        loss_c = _row_log_loss(labels, merged["probability_c"].to_numpy())

        per_year = (
            pd.DataFrame({"_year": merged["_year"], "b": loss_b, "c": loss_c})
            .groupby("_year")[["b", "c"]].mean()
        )
        years_better = int((per_year["c"] < per_year["b"]).sum())

        market = development.set_index("fight_id")["proba_market"].reindex(merged["fight_id"])
        has_market = market.notna().to_numpy()
        market_loss = (
            float(_row_log_loss(labels[has_market], market.to_numpy()[has_market]).mean())
            if has_market.any() else float("nan")
        )
        challenger_on_market_rows = float(loss_c[has_market].mean()) if has_market.any() else float("nan")

        results[family] = {
            "baseline_log_loss": float(loss_b.mean()),
            "challenger_log_loss": float(loss_c.mean()),
            "gain": _bootstrap(loss_b - loss_c, merged["_event"].to_numpy()),
            "years_challenger_better": years_better,
            "years_total": int(len(per_year)),
            "n": int(len(merged)),
            "market_log_loss": market_loss,
            "challenger_log_loss_on_market_rows": challenger_on_market_rows,
            "challenger_beats_market": bool(challenger_on_market_rows < market_loss),
            "per_year": {int(y): {"baseline": float(r.b), "challenger": float(r.c)} for y, r in per_year.iterrows()},
        }
        print(
            f"  {family:24s} base={results[family]['baseline_log_loss']:.5f} "
            f"challenger={results[family]['challenger_log_loss']:.5f} "
            f"gain={results[family]['gain']['mean']:+.5f} "
            f"années meilleures={years_better}/{len(per_year)}",
            flush=True,
        )

    # La famille qui porte la décision est choisie sur le bras SANS les nouveaux
    # descripteurs, donc indépendamment de l'effet mesuré.
    decision_family = min(results, key=lambda name: results[name]["baseline_log_loss"])
    chosen = results[decision_family]
    gate_passed = bool(
        chosen["gain"]["mean"] >= GATE_MIN_GAIN_VS_BASELINE
        and chosen["challenger_beats_market"]
        and chosen["years_challenger_better"] >= GATE_MIN_YEARS_BEATING_BASELINE
    )

    report = {
        "protocol_sha256": protocol["protocol_sha256"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "orientation_audit": audit,
        "decision_family": decision_family,
        "results": results,
        "gate_passed": gate_passed,
        "holdout_opened": False,
        "status": "ADJUSTED_GATE_PASSED" if gate_passed else "ADJUSTED_DEVELOPMENT_GATE_FAILED",
        "economic_claim": PROTOCOL["economic_claim"],
    }
    output = BASE_DIR / "data" / "rigorous" / "reports" / "adjusted_ablation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=float), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    if args.freeze_only:
        print(f"Protocole gelé: {freeze_protocol()['protocol_sha256']}")
        return 0

    report = run()
    chosen = report["results"][report["decision_family"]]
    print("\n" + "=" * 72)
    print(f"Famille de décision : {report['decision_family']}")
    print(f"Gain                : {chosen['gain']['mean']:+.5f} "
          f"IC90 [{chosen['gain']['low']:+.5f} ; {chosen['gain']['high']:+.5f}] (seuil {GATE_MIN_GAIN_VS_BASELINE})")
    print(f"Années meilleures   : {chosen['years_challenger_better']}/{chosen['years_total']} "
          f"(minimum {GATE_MIN_YEARS_BEATING_BASELINE})")
    print(f"Bat le marché       : {chosen['challenger_beats_market']} "
          f"({chosen['challenger_log_loss_on_market_rows']:.5f} contre {chosen['market_log_loss']:.5f})")
    print(f"\nStatut : {report['status']}")
    print("Holdout économique 2025-09-13 → 2026-08-29 : NON OUVERT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
