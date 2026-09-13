#!/usr/bin/env python3
"""Quel boost rend une sélection unique rentable, et chez quel opérateur ?

Le combiné boosté est mort parce que la marge s'empile géométriquement quand le
boost s'ajoute linéairement. Sur une **sélection unique**, rien ne s'empile : il
n'y a qu'une marge à franchir.

Et cette marge n'est pas la surmarge affichée. Le biais favori-outsider la pose
sur les outsiders : elle vaut ~1,6 % sur un favori à 1,15 et ~24 % sur un
outsider à 10,00. D'où la conclusion contre-intuitive que le même boost de 10 %
rapporte +8 % sur le favori et perd −17 % sur l'outsider.

Deux entrées mesurées :

* le **rendement par cote**, estimé sur 191 458 matchs de football chez Bet365,
  par fenêtre glissante sur le logarithme de la cote ;
* la **surmarge de chaque opérateur français**, relevée en direct sur les grands
  championnats, et convertie en rendement par la pente de l'échelle d'exécution
  (`run_execution_ladder.py`) : un point de surmarge coûte 0,53 point au favori.

Ce que le script ne sait pas : si l'opérateur accepte de booster *cette*
sélection, son plafond de mise, et combien de temps le compte reste ouvert.
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

# Pente de l'échelle d'exécution: le favori ne porte qu'environ la moitié de la
# marge, le reste tombant sur l'outsider.
OVERROUND_SLOPE = 0.53
REFERENCE_OVERROUND = 0.0655          # Bet365, où le rendement est mesuré
# Surmarges relevées en direct sur 95 matchs de grands championnats.
FRENCH_BOOKS = {"PMU": 0.0734, "Netbet": 0.0884, "Betclic": 0.0917,
                "Unibet": 0.1097, "Winamax": 0.1529,
                "meilleur des cinq": 0.0675}


def return_curve(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Rendement par cote chez Bet365, toutes issues du 1X2 confondues."""
    frame = pd.read_csv(root / "data/football/football_matches.csv.gz", low_memory=False)
    frame = frame.dropna(subset=["result"])
    rows = []
    for side, column in (("H", "B365H"), ("D", "B365D"), ("A", "B365A")):
        odds = pd.to_numeric(frame[column], errors="coerce")
        keep = odds.notna() & odds.gt(1.0)
        rows.append(pd.DataFrame({
            "odds": odds[keep],
            "won": (frame["result"] == side).astype(float)[keep],
        }))
    legs = pd.concat(rows, ignore_index=True)
    return (np.log(legs["odds"].to_numpy()),
            (legs["won"] * legs["odds"]).to_numpy())


def return_at(log_odds: np.ndarray, gains: np.ndarray,
              price: float, width: float = 0.20) -> tuple[float, int]:
    """Rendement moyen des paris de cote voisine, fenêtre sur le logarithme."""
    window = np.abs(log_odds - np.log(price)) < width
    return float(gains[window].mean()), int(window.sum())


def adjusted_return(reference: float, overround: float) -> float:
    """Le même pari chez un opérateur plus cher rend mécaniquement moins."""
    return reference - OVERROUND_SLOPE * (overround - REFERENCE_OVERROUND)


def required_boost(rendement: float) -> float:
    """Le boost qui ramène exactement à l'équilibre."""
    return 1.0 / rendement - 1.0 if rendement > 0 else float("inf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--boost", type=float, default=0.10,
                        help="boost proposé, en fraction du prix")
    args = parser.parse_args()
    root = args.project_root.resolve()

    log_odds, gains = return_curve(root)
    prices = (1.15, 1.30, 1.50, 1.80, 2.20, 3.00, 5.00, 10.0)

    print(f"Boost évalué: +{args.boost:.0%} sur le prix, sélection unique.\n")
    print("=== boost minimal pour atteindre l'équilibre ===")
    header = f"{'cote':>6s} {'r Bet365':>9s}" + "".join(
        f"{name:>14s}" for name in FRENCH_BOOKS)
    print(header)
    for price in prices:
        reference, _ = return_at(log_odds, gains, price)
        line = f"{price:>6.2f} {reference:>9.4f}"
        for overround in FRENCH_BOOKS.values():
            line += f"{required_boost(adjusted_return(reference, overround)):>13.2%} "
        print(line)

    print(f"\n=== espérance avec un boost de +{args.boost:.0%} ===")
    print(header)
    for price in prices:
        reference, count = return_at(log_odds, gains, price)
        line = f"{price:>6.2f} {reference:>9.4f}"
        for overround in FRENCH_BOOKS.values():
            value = adjusted_return(reference, overround) * (1.0 + args.boost) - 1.0
            line += f"{value:>+13.2%} "
        print(line)

    print("\nLire ce tableau dans le bon sens: le boost ne crée aucun avantage,")
    print("il franchit une marge. Comme la marge effective est cinq fois plus")
    print("faible sur un favori, un boost donné y devient rentable et reste")
    print("perdant sur un outsider — l'inverse de ce que l'habitude suggère.")
    print("\nNon vérifié ici: que l'opérateur boost cette sélection-là, son")
    print("plafond de mise, et la durée de vie du compte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
