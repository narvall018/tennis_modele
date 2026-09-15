"""Étude des descripteurs de marge: le score jeu par jeu ajoute-t-il quelque chose ?

Deux questions, séparées parce qu'elles n'ont pas la même valeur:

1. **Prévision pure.** La marge en jeux améliore-t-elle la prévision par rapport
   aux descripteurs déjà en place ? C'est la qualité de l'« opinion » que
   l'application affiche à côté du prix.
2. **Conditionnel au prix.** Une fois le prix dévigé donné au modèle, la marge
   apporte-t-elle encore quelque chose ? C'est la seule barre qui a jamais
   décidé d'un pari dans ce dépôt, et aucune étude ne l'a franchie.

Ce que cette étude ne fait pas, et ne peut pas faire: produire une conclusion
économique. Le budget de preuve d'ATP et de WTA est épuisé (voir la section
« Budget de preuve » du README). Une gate franchie ici est une *hypothèse*, pas
un rendement, et elle ne peut être confirmée que sur du budget vierge.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.backtesting.rigorous_strategy import MODEL_FEATURES, _valid_market, build_feature_table
from src.features.score_features import SCORE_FEATURES, build_score_features

# ---------------------------------------------------------------------------
# Protocole — figé avant la moindre mesure
# ---------------------------------------------------------------------------
GATE_DESCRIPTOR_GAIN = 0.002
GATE_CONDITIONAL_GAIN = 0.001
PRIMARY_PRICE = "average"
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_LEVEL = 0.90

TOURS: Dict[str, Dict[str, Any]] = {
    "atp": {"data": "data/atp_tennis.csv", "test_years": list(range(2010, 2027))},
    "wta": {"data": "data/wta_tennis.csv", "test_years": list(range(2013, 2027))},
}

MODEL_FAMILIES = {
    "logistic": {
        "kind": "logistic",
        "C": 0.05,
        "max_iter": 2000,
    },
    "hist_gradient_boosting": {
        "kind": "hist_gradient_boosting",
        "max_iter": 300,
        "learning_rate": 0.06,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
        "random_state": 0,
    },
}

UNCONDITIONAL_ARMS = {
    "base": list(MODEL_FEATURES),
    "base+marge": list(MODEL_FEATURES) + list(SCORE_FEATURES),
    "marge_seule": list(SCORE_FEATURES),
}
CONDITIONAL_ARMS = {
    "marché_seul": ["market_logit"],
    "marché+base": ["market_logit"] + list(MODEL_FEATURES),
    "marché+base+marge": ["market_logit"] + list(MODEL_FEATURES) + list(SCORE_FEATURES),
}

PROTOCOL: Dict[str, Any] = {
    "study": "score_margin_descriptors",
    "protocol_version": "1.0.0",
    "frozen_before_any_return": True,
    "question": (
        "La marge en jeux, lue dans une colonne Score qu'aucun module de ce dépôt "
        "n'avait jamais parsée, ajoute-t-elle de la prévision (a) aux descripteurs "
        "existants, (b) au prix dévigé ?"
    ),
    "evidence_budget": {
        "atp_main": "brûlé pour toute conclusion économique (étude imbriquée + phase 4)",
        "wta_main": "brûlé pour toute conclusion économique (holdout ouvert le 2026-09-06)",
        "consequence": (
            "Cette étude ne produit AUCUNE conclusion économique et n'autorise aucune "
            "mise. Elle mesure de la prévision. Une gate franchie est une hypothèse à "
            "confirmer sur du budget vierge — holdout UFC jamais ouvert, saisons "
            "football jamais ouvertes, ou collecte prospective à partir du 7 septembre "
            "2026 — jamais sur un nouveau passage ATP ou WTA."
        ),
    },
    "populations": {name: cfg["data"] for name, cfg in TOURS.items()},
    "feature_time_rule": (
        "L'état d'un joueur est lu au début de la journée et mis à jour seulement "
        "une fois tous les matchs de cette journée décrits."
    ),
    "settlement_population": "matchs terminés uniquement; abandons et walkovers exclus des métriques",
    "walk_forward": {
        "train_through": "test_year_minus_2",
        "calibration_year": "test_year_minus_1",
        "calibration": "isotone, appliquée identiquement à tous les bras",
        "minimum_train_rows": 10000,
        "minimum_calibration_rows": 500,
        "test_years": {name: cfg["test_years"] for name, cfg in TOURS.items()},
    },
    "model_families": MODEL_FAMILIES,
    "unconditional_arms": {name: len(cols) for name, cols in UNCONDITIONAL_ARMS.items()},
    "conditional_arms": {name: len(cols) for name, cols in CONDITIONAL_ARMS.items()},
    "primary_price": PRIMARY_PRICE,
    "price_validity": "overround dans [0,95 ; 1,25] et deux cotes > 1, convention _valid_market du dépôt",
    "devig": "proportionnel: q1 = (1/o1) / (1/o1 + 1/o2)",
    "gates": {
        "gate_a_descriptor_gain": {
            "threshold": GATE_DESCRIPTOR_GAIN,
            "definition": (
                "log-loss(base) - log-loss(base+marge), poolée sur toutes les années "
                "de test, sur le prix primaire non requis (toute la population jouée)."
            ),
            "meaning": "qualité de l'opinion affichée par l'application, sans portée économique",
        },
        "gate_b_conditional_gain": {
            "threshold": GATE_CONDITIONAL_GAIN,
            "definition": (
                "log-loss(marché_seul) - log-loss(marché+base+marge), poolée, sur les "
                "matchs à prix valide. Seuil repris tel quel du test conditionnel "
                "football déjà appliqué dans ce dépôt."
            ),
            "meaning": "la seule barre qui a jamais décidé d'un pari ici",
        },
    },
    "family_selection": (
        "La famille de modèle qui porte la décision est celle qui minimise la "
        "log-loss du bras SANS marge. Ce choix ne peut donc pas être incliné par "
        "l'effet mesuré. Les deux familles restent publiées dans tous les cas."
    ),
    "reporting_rule": (
        "Toutes les cellules sont publiées ensemble: les deux circuits, les deux "
        "familles de modèle, les trois bras et les quatre sources de prix. Aucune "
        "sélection a posteriori de la cellule la plus flatteuse. La décision lit la "
        "cellule primaire déclarée d'avance: prix moyen, log-loss poolée."
    ),
    "uncertainty": {
        "method": "bootstrap apparié par grappes mensuelles sur la différence de log-loss",
        "draws": BOOTSTRAP_DRAWS,
        "level": BOOTSTRAP_LEVEL,
    },
    "decision_rule": (
        "Gate B échouée → NO BET, aucune conclusion économique, aucun budget vierge "
        "dépensé. Gate A franchie et Gate B échouée → les descripteurs sont retenus "
        "pour l'opinion de l'application et pour rien d'autre."
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def protocol_path(root: Path) -> Path:
    return root / "models" / "margin_study" / "margin_protocol.json"


def freeze_protocol(root: Path) -> Dict[str, Any]:
    """Écrit et hashe le protocole. Refuse d'écraser un protocole déjà gelé."""
    target = protocol_path(root)
    payload = json.loads(json.dumps(PROTOCOL, ensure_ascii=False))
    payload["data_sha256"] = {
        name: _sha256_file(root / cfg["data"]) for name, cfg in TOURS.items()
    }
    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["protocol_sha256"] = _canonical_hash(
        {k: v for k, v in payload.items() if k != "frozen_at_utc"}
    )

    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("protocol_sha256") == payload["protocol_sha256"]:
            return existing
        payload["superseded"] = existing.get("superseded", []) + [
            {
                "amended_at_utc": payload["frozen_at_utc"],
                "previous_protocol_sha256": existing.get("protocol_sha256"),
                "previous_frozen_at_utc": existing.get("frozen_at_utc"),
            }
        ]
        payload["protocol_sha256"] = _canonical_hash(
            {k: v for k, v in payload.items() if k != "frozen_at_utc"}
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Moteur walk-forward
# ---------------------------------------------------------------------------
def _make_estimator(family: str):
    spec = MODEL_FAMILIES[family]
    if spec["kind"] == "logistic":
        from sklearn.impute import SimpleImputer

        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=spec["C"], max_iter=spec["max_iter"])),
            ]
        )
    return HistGradientBoostingClassifier(
        max_iter=spec["max_iter"],
        learning_rate=spec["learning_rate"],
        max_leaf_nodes=spec["max_leaf_nodes"],
        min_samples_leaf=spec["min_samples_leaf"],
        l2_regularization=spec["l2_regularization"],
        random_state=spec["random_state"],
    )


def _row_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return -(labels * np.log(clipped) + (1 - labels) * np.log(1.0 - clipped))


def _walk_forward_arm(
    frame: pd.DataFrame,
    columns: Sequence[str],
    family: str,
    test_years: Sequence[int],
    minimum_train: int,
    minimum_calibration: int,
) -> pd.DataFrame:
    """Une colonne de probabilités par match testé; jamais entraîné sur son année."""
    years = frame["_year"].to_numpy()
    labels = frame["_label"].to_numpy(dtype=int)
    matrix = frame[list(columns)].to_numpy(dtype=np.float64)
    pieces: List[pd.DataFrame] = []

    for year in test_years:
        train_mask = years <= year - 2
        calib_mask = years == year - 1
        test_mask = years == year
        if train_mask.sum() < minimum_train or calib_mask.sum() < minimum_calibration:
            continue
        if test_mask.sum() == 0:
            continue

        estimator = _make_estimator(family)
        estimator.fit(matrix[train_mask], labels[train_mask])
        calib_raw = estimator.predict_proba(matrix[calib_mask])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(calib_raw, labels[calib_mask])
        test_raw = estimator.predict_proba(matrix[test_mask])[:, 1]
        pieces.append(
            pd.DataFrame(
                {
                    "_source_row_id": frame.loc[test_mask, "_source_row_id"].to_numpy(),
                    "_year": year,
                    "_month": frame.loc[test_mask, "_date"].dt.month.to_numpy(),
                    "_label": labels[test_mask],
                    "probability": np.clip(calibrator.predict(test_raw), 1e-6, 1.0 - 1e-6),
                }
            )
        )

    if not pieces:
        return pd.DataFrame(columns=["_source_row_id", "_year", "_month", "_label", "probability"])
    return pd.concat(pieces, ignore_index=True)


def _paired_bootstrap(
    diff: np.ndarray, clusters: np.ndarray, draws: int, level: float, seed: int = 0
) -> Dict[str, float]:
    """IC de la différence moyenne de log-loss, rééchantillonnée par mois.

    Les matchs d'un même mois partagent la forme des joueurs et l'état du
    marché; les traiter comme indépendants rétrécirait l'intervalle à tort.
    """
    if diff.size == 0:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    index = {value: np.flatnonzero(clusters == value) for value in unique}
    means = np.empty(draws, dtype=float)
    for draw in range(draws):
        picked = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([index[value] for value in picked])
        means[draw] = diff[rows].mean()
    tail = (1.0 - level) / 2.0
    return {
        "mean": float(diff.mean()),
        "low": float(np.quantile(means, tail)),
        "high": float(np.quantile(means, 1.0 - tail)),
        "n": int(diff.size),
        "clusters": int(unique.size),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _pooled(predictions: pd.DataFrame) -> Dict[str, float]:
    losses = _row_log_loss(predictions["_label"].to_numpy(int), predictions["probability"].to_numpy(float))
    return {"log_loss": float(losses.mean()), "n": int(len(predictions))}


def _compare(
    reference: pd.DataFrame, candidate: pd.DataFrame, seed: int
) -> Dict[str, float]:
    """Gain de log-loss du candidat sur la référence, sur les matchs communs."""
    merged = reference.merge(
        candidate, on=["_source_row_id", "_year", "_month", "_label"], suffixes=("_ref", "_cand")
    )
    if merged.empty:
        return {"gain": float("nan"), "n": 0}
    labels = merged["_label"].to_numpy(int)
    loss_ref = _row_log_loss(labels, merged["probability_ref"].to_numpy(float))
    loss_cand = _row_log_loss(labels, merged["probability_cand"].to_numpy(float))
    clusters = (merged["_year"].to_numpy() * 100 + merged["_month"].to_numpy())
    stats = _paired_bootstrap(loss_ref - loss_cand, clusters, BOOTSTRAP_DRAWS, BOOTSTRAP_LEVEL, seed)
    return {
        "gain": stats["mean"],
        "ci_low": stats["low"],
        "ci_high": stats["high"],
        "n": stats["n"],
        "clusters": stats.get("clusters", 0),
        "excludes_zero_from_below": bool(stats["low"] > 0.0),
    }


def run_tour(
    frame: pd.DataFrame, tour: str, progress=print
) -> Dict[str, Any]:
    """Les deux tests sur un circuit. Aucune cellule n'est écartée."""
    config = TOURS[tour]
    test_years = config["test_years"]
    wf = PROTOCOL["walk_forward"]
    played = frame[frame["_status"].eq("completed")].reset_index(drop=True)
    progress(f"[{tour}] {len(played):,} matchs terminés")

    result: Dict[str, Any] = {"tour": tour, "matches_played": int(len(played))}

    # --- Test 1: prévision pure, sans jamais montrer un prix ---------------
    unconditional: Dict[str, Dict[str, Any]] = {}
    for family in MODEL_FAMILIES:
        predictions = {}
        for arm, columns in UNCONDITIONAL_ARMS.items():
            predictions[arm] = _walk_forward_arm(
                played, columns, family, test_years,
                wf["minimum_train_rows"], wf["minimum_calibration_rows"],
            )
            progress(f"[{tour}] inconditionnel {family}/{arm}: {len(predictions[arm]):,} matchs testés")
        cells = {arm: _pooled(pred) for arm, pred in predictions.items()}
        unconditional[family] = {
            "cells": cells,
            "gain_base_to_marge": _compare(predictions["base"], predictions["base+marge"], seed=1),
        }
    result["unconditional"] = unconditional

    # --- Test 2: conditionnel au prix, sur les quatre sources -------------
    conditional: Dict[str, Any] = {}
    for source in PRICE_COLUMNS_ORDER:
        left, right, valid = _valid_market(played, source)
        subset = played.loc[valid].copy()
        if len(subset) < 5000:
            conditional[source] = {"skipped": "population insuffisante", "n": int(valid.sum())}
            continue
        q1 = (1.0 / left[valid]) / (1.0 / left[valid] + 1.0 / right[valid])
        subset["market_logit"] = np.log(np.clip(q1, 1e-6, 1 - 1e-6) / np.clip(1 - q1, 1e-6, 1 - 1e-6))

        per_family = {}
        for family in MODEL_FAMILIES:
            predictions = {}
            for arm, columns in CONDITIONAL_ARMS.items():
                predictions[arm] = _walk_forward_arm(
                    subset, columns, family, test_years,
                    wf["minimum_train_rows"], wf["minimum_calibration_rows"],
                )
            per_family[family] = {
                "cells": {arm: _pooled(pred) for arm, pred in predictions.items()},
                "gain_market_to_full": _compare(
                    predictions["marché_seul"], predictions["marché+base+marge"], seed=2
                ),
                "gain_market_to_base": _compare(
                    predictions["marché_seul"], predictions["marché+base"], seed=3
                ),
                "gain_base_to_marge": _compare(
                    predictions["marché+base"], predictions["marché+base+marge"], seed=4
                ),
            }
            progress(
                f"[{tour}] conditionnel {source}/{family}: "
                f"gain marché→complet = {per_family[family]['gain_market_to_full']['gain']:+.5f}"
            )
        conditional[source] = {"n_valid": int(valid.sum()), "families": per_family}
    result["conditional"] = conditional
    return result


PRICE_COLUMNS_ORDER = ["average", "pinnacle", "bet365", "maximum"]


def load_table(root: Path, tour: str, cache_dir: Path | None = None, progress=print) -> pd.DataFrame:
    """Descripteurs existants + descripteurs de marge, joints sur l'identifiant source."""
    if cache_dir is not None:
        cached = cache_dir / f"{tour}_table.parquet"
        if cached.exists():
            progress(f"[{tour}] table lue dans le cache")
            return pd.read_parquet(cached)

    relative = TOURS[tour]["data"]
    features, _ = build_feature_table(root / relative, progress=progress)
    raw = pd.read_csv(root / relative, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date", "Player_1", "Player_2", "Winner"]).copy()
    raw["source_row_id"] = np.arange(len(raw), dtype=np.int64)
    margin, _ = build_score_features(raw, progress=progress)

    if set(features["_source_row_id"]) != set(margin["source_row_id"]):
        raise AssertionError(
            "Identifiants source désalignés entre les deux passes: la jointure "
            "rapprocherait des matchs différents."
        )
    merged = features.merge(
        margin, left_on="_source_row_id", right_on="source_row_id",
        how="left", validate="one_to_one",
    ).drop(columns="source_row_id")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(cache_dir / f"{tour}_table.parquet")
    return merged


def _decide(results: Dict[str, Any]) -> Dict[str, Any]:
    """Applique les gates telles qu'elles ont été écrites, sans les rouvrir."""
    verdicts = {}
    for tour, result in results.items():
        uncond = result["unconditional"]
        # Famille choisie sur le bras SANS marge, comme le protocole l'impose.
        family = min(uncond, key=lambda name: uncond[name]["cells"]["base"]["log_loss"])
        gate_a = uncond[family]["gain_base_to_marge"]

        primary = result["conditional"].get(PRIMARY_PRICE, {})
        gate_b = None
        if "families" in primary:
            cond = primary["families"]
            cond_family = min(
                cond, key=lambda name: cond[name]["cells"]["marché+base"]["log_loss"]
            )
            gate_b = cond[cond_family]["gain_market_to_full"]
            gate_b_family = cond_family
        else:
            gate_b_family = None

        verdicts[tour] = {
            "family_for_decision": family,
            "gate_a_gain": gate_a["gain"],
            "gate_a_threshold": GATE_DESCRIPTOR_GAIN,
            "gate_a_passed": bool(gate_a["gain"] >= GATE_DESCRIPTOR_GAIN),
            "gate_a_ci": [gate_a.get("ci_low"), gate_a.get("ci_high")],
            "conditional_family_for_decision": gate_b_family,
            "gate_b_gain": None if gate_b is None else gate_b["gain"],
            "gate_b_threshold": GATE_CONDITIONAL_GAIN,
            "gate_b_passed": bool(gate_b is not None and gate_b["gain"] >= GATE_CONDITIONAL_GAIN),
            "gate_b_ci": None if gate_b is None else [gate_b.get("ci_low"), gate_b.get("ci_high")],
        }
    return verdicts


def run_study(root: Path, cache_dir: Path | None = None, progress=print) -> Dict[str, Any]:
    protocol = freeze_protocol(root)
    results = {}
    for tour in TOURS:
        frame = load_table(root, tour, cache_dir=cache_dir, progress=progress)
        results[tour] = run_tour(frame, tour, progress=progress)

    verdicts = _decide(results)
    any_economic = any(v["gate_b_passed"] for v in verdicts.values())
    report = {
        "protocol_sha256": protocol["protocol_sha256"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "verdicts": verdicts,
        "status": "MARGIN_CONDITIONAL_GATE_PASSED" if any_economic else "MARGIN_NO_BET",
        "economic_claim": (
            "Aucune. ATP et WTA sont brûlés pour toute conclusion économique; une "
            "gate conditionnelle franchie ici serait une hypothèse à confirmer sur "
            "du budget vierge, jamais un rendement."
        ),
    }
    output = root / "models" / "margin_study"
    output.mkdir(parents=True, exist_ok=True)
    (output / "margin_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=float),
        encoding="utf-8",
    )
    return report
