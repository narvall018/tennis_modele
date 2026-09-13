#!/usr/bin/env python3
"""Un sous-ensemble bat-il le prix là où l'ensemble échoue ? — et à quel prix ?

`three_sport_residual` teste chaque sport en bloc, et les trois ressortent en
NO_ROBUST_CANDIDATE. L'objection naturelle est qu'un avantage réel peut exister
sur une surface, une division ou une catégorie de poids, et se faire diluer dans
la moyenne. Ce script la teste.

Il teste aussi, et c'est le point important, **le coût de la question**. Découper
en N cellules et retenir la meilleure est la machine qui a produit le seul
résultat positif de `favourite_longshot_bias`, mort ensuite au contrôle croisé.
Une cellule gagnante n'est une nouvelle que si elle bat ce que le hasard produit
sur N essais, donc chaque grille est accompagnée de sa simulation sous H0.

Protocole, fixé avant lecture des rendements :

* le modèle de référence est le **prix seul** — logistique L2 sur son logit ;
* le concurrent ajoute les descripteurs du sport, même régularisation ;
* apprentissage sur les années strictement antérieures, test sur l'année en
  cours, cellule par cellule ; aucune cellule ne voit son propre futur ;
* le seuil déclaré est **+0,001 de log-loss** sur le prix, le niveau que
  `edge_too_small_to_prove` chiffre à ~+0,74 % de ROI ;
* les années d'évaluation ont déjà servi aujourd'hui, donc ce balayage est
  **exploratoire** : il génère des hypothèses, il ne prouve rien.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Le seuil sous lequel rien ne peut payer, repris tel quel des études passées.
GATE = 0.001
MINIMUM_TRAIN = 1200
MINIMUM_TEST = 250
# L'UFC tient en 6 000 combats: les mêmes planchers n'y laisseraient aucune
# cellule, et une cellule absente serait lue à tort comme une cellule négative.
SMALL_SPORT_TRAIN = 500
SMALL_SPORT_TEST = 80
EVALUATION_YEARS = {"atp": (2023, 2024, 2025),
                    "football": (2023, 2024, 2025),
                    "ufc": (2022, 2023, 2024)}


# Un descripteur ne doit RIEN contenir du marché. Les colonnes de cotes — et
# surtout celles de clôture, qui datent du coup d'envoi alors que la référence
# n'a que l'ouverture — feraient gagner le modèle en lisant le marché plus tard,
# pas en le battant. C'est ce qui a produit la fausse cellule « division 1 ».
MARKET_TOKENS = ("b365", "ps", "avg", "max", "bfe", "odds", "price", "market",
                 "implied", "vig", "ahh", "aha", "ahc", "ahh", "^ah", "bw", "iw",
                 "wh", "vc", "surprise")


def _is_market_column(name: str) -> bool:
    """Exclure par défaut: une cote oubliée invalide toute la comparaison.

    Le filtre par jetons seul avait laissé passer `P>2.5` et `PC>2.5` — les
    cotes Pinnacle sur les buts, clôture comprise. Toute colonne qui porte un
    seuil de marché, un opérateur de comparaison ou un préfixe d'opérateur est
    désormais traitée comme du marché.
    """
    lowered = name.lower()
    if any(mark in lowered for mark in (">", "<", "2.5")):
        return True
    if lowered.startswith(("ah", "p>", "p<", "pc", "bf", "b3")):
        return True
    return any(token in lowered for token in MARKET_TOKENS)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                 columns: list[str], label: str) -> np.ndarray:
    """Logistique L2 sur `columns`, renvoyée comme probabilités sur `test`."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[columns].to_numpy(dtype=float))
    x_test = scaler.transform(test[columns].to_numpy(dtype=float))
    model = LogisticRegression(C=0.1, max_iter=2000)
    model.fit(x_train, train[label].to_numpy(dtype=int))
    return model.predict_proba(x_test)[:, 1]


def _log_loss_terms(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Contribution de chaque match, pour pouvoir mesurer sa dispersion."""
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))


def evaluate_cell(frame: pd.DataFrame, features: list[str], label: str,
                  years: tuple[int, ...], small: bool = False) -> dict[str, float] | None:
    """Gain de log-loss du modèle sur le prix seul, en avant-marche annuelle."""
    minimum_train = SMALL_SPORT_TRAIN if small else MINIMUM_TRAIN
    minimum_test = SMALL_SPORT_TEST if small else MINIMUM_TEST
    gains: list[np.ndarray] = []
    for year in years:
        train = frame[frame["_year"] < year]
        test = frame[frame["_year"] == year]
        if len(train) < minimum_train or len(test) < minimum_test:
            continue
        labels = test[label].to_numpy(dtype=int)
        market = _fit_predict(train, test, ["market_logit"], label)
        rich = _fit_predict(train, test, ["market_logit"] + features, label)
        gains.append(_log_loss_terms(labels, market) - _log_loss_terms(labels, rich))
    if not gains:
        return None
    pooled = np.concatenate(gains)
    error = float(pooled.std() / np.sqrt(len(pooled)))
    return {
        "n": int(len(pooled)),
        "gain": float(pooled.mean()),
        "erreur_type": error,
        # Le t sert à la simulation sous H0: il est comparable entre cellules.
        "t": float(pooled.mean() / error) if error > 0 else 0.0,
    }


def null_best_cell(cells: list[dict[str, float]], draws: int = 20000) -> float:
    """Sous H0, quelle est la probabilité qu'une des cellules atteigne ce t ?

    Chaque cellule est un test indépendant de moyenne nulle, donc son t est
    approximativement normal centré. La question honnête n'est pas « cette
    cellule est-elle significative » mais « la meilleure de N l'est-elle ».
    """
    if not cells:
        return 1.0
    rng = np.random.default_rng(0)
    best = rng.standard_normal((draws, len(cells))).max(axis=1)
    return float((best >= max(cell["t"] for cell in cells)).mean())


def sweep(name: str, frame: pd.DataFrame, features: list[str], label: str,
          segments: dict[str, pd.Series]) -> list[dict]:
    years = EVALUATION_YEARS[name]
    rows: list[dict] = []
    for segment_name, keys in segments.items():
        for key in [k for k in pd.unique(keys.dropna()) if str(k) != ""]:
            cell = frame[keys == key]
            result = evaluate_cell(cell, features, label, years, small=(name == "ufc"))
            if result is None:
                continue
            rows.append({"sport": name, "découpage": segment_name,
                         "cellule": str(key), **result})
    return rows


def _price_band(frame: pd.DataFrame) -> pd.Series:
    """Le prix est lui-même un découpage, et le plus documenté du projet."""
    probability = 1.0 / (1.0 + np.exp(-frame["market_logit"].to_numpy(dtype=float)))
    return pd.Series(pd.cut(probability, [0, 0.3, 0.45, 0.55, 0.7, 1.0],
                            labels=["outsider", "léger outsider", "pile ou face",
                                    "léger favori", "favori"]).astype(str),
                     index=frame.index)


def load_atp(root: Path) -> tuple[pd.DataFrame, list[str], str, dict]:
    frame = pd.read_parquet(root / "models/three_sport_residual/atp_features.parquet")
    frame = frame[frame["_market_valid"].astype(bool)].copy()
    features = [c for c in frame.columns
                if not c.startswith("_") and not _is_market_column(c)]
    frame[features] = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    segments = {
        "surface": frame["_surface"].astype(str),
        "niveau": frame["level_strength"].round(2).astype(str),
        "prix": _price_band(frame),
    }
    return frame, features, "_label", segments


def load_football(root: Path) -> tuple[pd.DataFrame, list[str], str, dict]:
    frame = pd.read_parquet(root / "models/three_sport_residual/football_features.parquet")
    frame["_year"] = pd.to_datetime(frame["_date"], errors="coerce").dt.year
    # Marché 1X2: on ramène la question au pari « domicile gagne », deux issues.
    frame["market_logit"] = np.log(1.0 / frame["price_0"].astype(float))
    frame = frame[np.isfinite(frame["market_logit"])].copy()
    frame["_label"] = (frame["result"].astype(str) == "H").astype(int)
    drop = {"result", "home_goals", "away_goals", "total_goals", "goal_difference",
            "market_logit", "_label", "_year"}
    features = [c for c in frame.columns
                if not c.startswith("_") and c not in drop
                and pd.api.types.is_numeric_dtype(frame[c])
                and not c.startswith("postmatch_") and not _is_market_column(c)]
    frame[features] = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    segments = {
        "pays": frame["country"].astype(str),
        "rang_division": frame["division_rank"].astype(str),
        "prix": _price_band(frame),
    }
    return frame, features, "_label", segments


def load_ufc(root: Path) -> tuple[pd.DataFrame, list[str], str, dict]:
    frame = pd.read_parquet(root / "models/three_sport_residual/ufc_features.parquet")
    frame["_year"] = pd.to_datetime(frame["_date"], errors="coerce").dt.year
    # `y` est l'étiquette elle-même et `orientation_swapped` un artefact
    # d'augmentation: ni l'un ni l'autre n'est un descripteur d'avant-combat.
    features = [c for c in frame.columns
                if not c.startswith("_") and pd.api.types.is_numeric_dtype(frame[c])
                and not _is_market_column(c)
                and c not in {"y", "orientation_swapped", "overround"}]
    frame[features] = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    segments = {
        "catégorie": frame["weight_class"].astype(str),
        "prix": _price_band(frame),
    }
    return frame, features, "_label", segments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()

    loaders = {"atp": load_atp, "football": load_football, "ufc": load_ufc}
    everything: list[dict] = []
    for name, loader in loaders.items():
        frame, features, label, segments = loader(root)
        print(f"=== {name.upper()} — {len(frame):,} matchs, "
              f"{len(features)} descripteurs ===")
        rows = sweep(name, frame, features, label, segments)
        everything += rows
        if not rows:
            print("  aucune cellule n'atteint la taille minimale\n")
            continue
        table = pd.DataFrame(rows).sort_values("gain", ascending=False)
        for row in table.itertuples(index=False):
            flag = "  <- passe le seuil" if row.gain > GATE else ""
            print(f"  {row.découpage:14s} {row.cellule:22s} n={row.n:>6,} "
                  f"gain {row.gain:+.5f}  t={row.t:+5.2f}{flag}")
        print(f"  meilleure cellule: t={table['t'].max():+.2f}, "
              f"probabilité sous H0 sur {len(rows)} cellules: "
              f"{null_best_cell(rows):.1%}\n")

    frame = pd.DataFrame(everything)
    passing = frame[frame["gain"] > GATE]
    print("=" * 70)
    print(f"{len(frame)} cellules testées, {len(passing)} au-dessus du seuil "
          f"de {GATE:+.3f}")
    print(f"probabilité que le meilleur t global vienne du hasard seul: "
          f"{null_best_cell(everything):.1%}")
    output = root / "models" / "segment_sweep.json"
    output.write_text(json.dumps({
        "gate": GATE,
        "evidence": "EXPLORATORY_REUSED_HISTORY; no independent confirmation claimed",
        "cells": everything,
        "null_probability_best_cell": null_best_cell(everything),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"détail écrit dans {output.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
