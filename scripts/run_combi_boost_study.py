#!/usr/bin/env python3
"""Un combiné boosté peut-il être positif ? La première réponse « oui » du projet.

Onze pistes sont mortes du même diagnostic : un biais réel, plus petit que la
marge à franchir. Un combiné boosté renverse le problème. Le boost n'est pas une
prédiction, c'est une clause contractuelle — le book paie pour acquérir un
client. Il n'y a donc rien à démontrer empiriquement : il suffit de comparer deux
quantités connues, la marge empilée et le boost promis.

Le calcul repose sur une identité, pas sur un modèle. Pour des jambes
indépendantes — des matchs différents — l'espérance d'un combiné est exactement
le produit des espérances par jambe :

    E[combiné] = prod(p_i x O_i) = r^n

Toute la stratégie tient donc dans r, le rendement d'une jambe, et r dépend
massivement de la cote : le biais favori-outsider concentre la marge du book sur
les outsiders. Une jambe à 1,26 rend 0,989 ; une jambe à 12,00 rend 0,68.

Quatre passes :

1. *le biais* — r par tranche de cote, chez un book réel et non sur une moyenne
   de books, qui n'est pas un prix que l'on peut jouer ;
2. *la stabilité* — un biais de vingt ans peut avoir disparu ; découpage par
   période, et intervalles par blocs mensuels ;
3. *le seuil* — le boost qui annule la marge empilée, calculé sur la borne basse
   de r et non sur son point central ;
4. *la fragilité* — de quoi la conclusion dépend-elle ? Cote minimale imposée par
   le barème, marge d'un book français plus chère, boost tiré au sort.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Bet365 plutôt que la moyenne du marché: `Avg` est la moyenne d'opérateurs
# différents, et personne ne parie une moyenne. Même raison qui a invalidé
# `market_maximum`.
BOOK_COLUMNS = {"H": "B365H", "D": "B365D", "A": "B365A"}
BANDS = [(1.15, 1.35), (1.35, 1.55), (1.55, 1.80), (1.80, 2.10)]
LEG_COUNTS = (3, 4, 5, 6, 8, 10, 12, 15)
BOOTSTRAP_DRAWS = 2000


def load_legs(root: Path) -> pd.DataFrame:
    """Une ligne par pari possible: sa cote, et s'il est passé."""
    frame = pd.read_csv(root / "data/football/football_matches.csv.gz", low_memory=False)
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    rows = []
    for side, column in BOOK_COLUMNS.items():
        odds = pd.to_numeric(frame[column], errors="coerce")
        keep = odds.notna() & odds.gt(1.0) & frame["result"].notna()
        rows.append(pd.DataFrame({
            "odds": odds[keep],
            "won": (frame["result"] == side).astype(float)[keep],
            "date": frame.loc[keep, "match_date"],
        }))
    legs = pd.concat(rows, ignore_index=True).dropna(subset=["date"])
    legs["gain"] = legs["won"] * legs["odds"]
    legs["month"] = legs["date"].dt.to_period("M")
    return legs


def month_block_interval(cell: pd.DataFrame, draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float]:
    """Les matchs d'un même mois se ressemblent; rééchantillonner par mois."""
    grouped = [group["gain"].to_numpy() for _, group in cell.groupby("month")]
    rng = np.random.default_rng(0)
    means = [
        np.concatenate([grouped[i] for i in rng.integers(0, len(grouped), len(grouped))]).mean()
        for _ in range(draws)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def required_boost(r: float, mean_odds: float, legs_count: int) -> float:
    """Boost sur le gain net qui ramène un combiné de n jambes à l'équilibre."""
    accumulated = r ** legs_count
    probability = (r / mean_odds) ** legs_count
    if accumulated <= probability:
        return float("inf")
    return (1.0 - probability) / (accumulated - probability) - 1.0


def expected_value(r: float, mean_odds: float, legs_count: int, boost: float) -> tuple[float, float]:
    """(EV, mise Kelly complète) pour un combiné boosté."""
    probability = (r / mean_odds) ** legs_count
    payout = 1.0 + (mean_odds ** legs_count - 1.0) * (1.0 + boost)
    if payout <= 1.0:
        return -1.0, 0.0
    ev = probability * payout - 1.0
    return ev, max(ev / (payout - 1.0), 0.0)


def show_bias(legs: pd.DataFrame) -> dict[tuple[float, float], tuple[float, float, float]]:
    print("=== 1. le rendement d'une jambe dépend de sa cote ===")
    print(f"{'tranche':>12s} {'n':>8s} {'cote moy':>9s} {'r':>8s} {'IC 95% (blocs mensuels)':>26s}")
    measured = {}
    for low, high in BANDS:
        cell = legs[legs["odds"].between(low, high, "left")]
        r = float(cell["gain"].mean())
        mean_odds = float(cell["odds"].mean())
        lo, hi = month_block_interval(cell)
        measured[(low, high)] = (r, lo, mean_odds)
        print(f"{f'{low}-{high}':>12s} {len(cell):>8,} {mean_odds:>9.2f} {r:>8.4f} "
              f"{f'[{lo:.4f}, {hi:.4f}]':>26s}")
    print("\nLa marge du book n'est pas répartie: elle est posée sur les outsiders.")
    return measured


def show_stability(legs: pd.DataFrame) -> None:
    print("\n=== 2. le biais tient-il dans le temps ? ===")
    legs = legs.copy()
    legs["era"] = pd.cut(legs["date"].dt.year, [1999, 2010, 2016, 2021, 2027],
                         labels=["2000-10", "2011-16", "2017-21", "2022-26"])
    print(f"{'période':>12s}" + "".join(f"{f'{a}-{b}':>13s}" for a, b in BANDS))
    for era, block in legs.groupby("era", observed=True):
        line = f"{str(era):>12s}"
        for low, high in BANDS:
            cell = block[block["odds"].between(low, high, "left")]
            line += f"{cell['gain'].mean():>13.4f}" if len(cell) > 400 else f"{'—':>13s}"
        print(line)


def show_threshold(r_low: float, mean_odds: float) -> None:
    print("\n=== 3. quel boost faut-il ? (calculé sur la borne basse de r) ===")
    print(f"{'jambes':>7s} {'cote combi':>11s} {'P(gagne)':>9s} {'boost requis':>13s}")
    for count in LEG_COUNTS:
        print(f"{count:>7d} {mean_odds ** count:>11.2f} "
              f"{(r_low / mean_odds) ** count:>9.1%} "
              f"{required_boost(r_low, mean_odds, count):>13.1%}")
    print("\nLe seuil monte linéairement avec le nombre de jambes. Les barèmes")
    print("annoncés montent plus vite — d'où une fenêtre, au-delà de ~8 jambes.")


def show_fragility(legs: pd.DataFrame, r_base: float, mean_odds: float,
                   boost: float, count: int) -> None:
    print(f"\n=== 4. de quoi la conclusion dépend-elle ? (boost {boost:.0%}, "
          f"{count} jambes) ===")

    print("\na) cote minimale imposée par le barème")
    print(f"{'plancher':>9s} {'cote moy':>9s} {'r':>8s} {'requis':>8s} {'EV':>9s}")
    for floor in (1.15, 1.20, 1.30, 1.40, 1.50):
        cell = legs[legs["odds"].between(floor, floor + 0.20, "left")]
        r, odds = float(cell["gain"].mean()), float(cell["odds"].mean())
        ev, _ = expected_value(r, odds, count, boost)
        print(f"{floor:>9.2f} {odds:>9.2f} {r:>8.4f} "
              f"{required_boost(r, odds, count):>8.1%} {ev:>+9.2%}")

    print("\nb) un book français plus cher que Bet365")
    print(f"{'pénalité':>9s} {'r':>8s} {'requis':>8s} {'EV':>9s}")
    for penalty in (0.0, 0.005, 0.010, 0.015, 0.020, 0.030):
        r = r_base - penalty
        ev, _ = expected_value(r, mean_odds, count, boost)
        print(f"{penalty:>9.1%} {r:>8.4f} "
              f"{required_boost(r, mean_odds, count):>8.1%} {ev:>+9.2%}")

    print("\nc) un boost tiré au sort plutôt qu'affiché")
    for label, values in [("valeur d'affiche seule", [boost]),
                          ("six valeurs, 0 à l'affiche", list(np.linspace(0, boost, 6))),
                          ("six valeurs, 0 à la moitié", list(np.linspace(0, boost / 2, 6)))]:
        evs = [expected_value(r_base, mean_odds, count, value)[0] for value in values]
        print(f"  {label:28s} boost moyen {np.mean(values):>5.1%} -> "
              f"EV {np.mean(evs):>+7.2%}")


def show_supply(legs: pd.DataFrame, low: float, high: float) -> None:
    print(f"\n=== 5. y a-t-il assez de favoris à {low}-{high} ? ===")
    recent = legs[legs["date"].dt.year >= 2022]
    cell = recent[recent["odds"].between(low, high, "left")]
    per_day = cell.groupby(cell["date"].dt.date).size()
    print(f"{len(cell):,} favoris sur {per_day.size:,} jours — "
          f"médiane {per_day.median():.0f}/jour")
    for threshold in (4, 6, 8, 10):
        print(f"  jours offrant au moins {threshold:2d} favoris: "
              f"{float((per_day >= threshold).mean()):5.1%}")
    per_week = cell.groupby(pd.Grouper(key="date", freq="W")).size()
    print(f"  semaines offrant au moins 10 favoris: {(per_week >= 10).mean():.1%}")
    print("\nUn combiné peut s'étaler sur plusieurs jours, donc la semaine est la")
    print("bonne unité. Et ces 22 divisions sous-estiment l'offre réelle d'un")
    print("opérateur qui couvre aussi les autres sports.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--boost", type=float, default=0.50)
    parser.add_argument("--legs", type=int, default=10)
    args = parser.parse_args()

    legs = load_legs(args.project_root.resolve())
    print(f"{len(legs):,} jambes possibles, {legs['date'].dt.year.min():.0f}"
          f"-{legs['date'].dt.year.max():.0f}\n")

    measured = show_bias(legs)
    show_stability(legs)
    band = BANDS[0]
    r_base, r_low, mean_odds = measured[band]
    show_threshold(r_low, mean_odds)
    show_fragility(legs, r_base, mean_odds, args.boost, args.legs)
    show_supply(legs, *band)

    print("\n" + "=" * 72)
    print("Ce que ces chiffres n'établissent pas: les barèmes réels des books")
    print("français, leurs prix sur les favoris, leurs plafonds de mise, et si")
    print("un combiné de gros favoris est seulement éligible au boost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
