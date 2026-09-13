#!/usr/bin/env python3
"""Situer n'importe quel opérateur sur l'échelle, à partir de ses seules cotes.

L'API ne couvre que 61 opérateurs, et pas les books crypto — Stake, Betify,
Cloudbet et les autres en sont absents. Impossible donc de mesurer leur marge
depuis ce dépôt. Mais la marge se lit sur deux cotes, et le reste est de
l'arithmétique déjà établie ailleurs :

* la **surmarge** est la somme des probabilités implicites moins un ;
* l'**échelle d'exécution** (`run_execution_ladder.py`) donne le rendement d'un
  gros favori en fonction de cette surmarge : un point de surmarge coûte 0,53
  point de rendement, et le seuil de rentabilité est à 4,62 %.

Relevez donc deux ou trois cotes du même match chez l'opérateur à évaluer, et ce
script dit où il se situe — sans que personne n'ait à croire une brochure.

    python3 scripts/run_book_check.py 1.91 1.95
    python3 scripts/run_book_check.py 2.30 3.40 3.10 --nom "Stake"
"""

from __future__ import annotations

import argparse

# Repris de `run_execution_ladder.py`, mesuré sur 191 458 matchs de football.
SLOPE = 0.53
REFERENCE_OVERROUND = 0.0655      # Bet365, où le rendement du favori est mesuré
REFERENCE_RETURN = 0.9852         # rendement d'un favori à ~1,15 chez Bet365
BREAK_EVEN = 0.0462

# Repères mesurés le 2026-09-13, pour situer sans avoir à les rechercher.
LANDMARKS = [
    ("Betfair exchange (tennis)", 0.0091),
    ("Matchbook", 0.0127),
    ("Pinnacle", 0.0302),
    ("handicap asiatique Bet365", 0.0391),
    ("SEUIL DE RENTABILITÉ", BREAK_EVEN),
    ("Bet365 1X2", 0.0655),
    ("meilleur des 5 français", 0.0675),
    ("PMU", 0.0734),
    ("Betclic", 0.0917),
    ("Unibet France", 0.1097),
    ("Winamax France", 0.1529),
]


def overround(prices: list[float]) -> float:
    return sum(1.0 / price for price in prices) - 1.0


def favourite_return(value: float) -> float:
    """Rendement attendu d'un gros favori chez un book de cette surmarge."""
    return REFERENCE_RETURN - SLOPE * (value - REFERENCE_OVERROUND)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prices", nargs="+", type=float,
                        help="toutes les cotes d'un même marché (2 ou 3 issues)")
    parser.add_argument("--nom", default="cet opérateur")
    args = parser.parse_args()

    if any(price <= 1.0 for price in args.prices):
        print("une cote décimale est toujours supérieure à 1")
        return 1
    if len(args.prices) < 2:
        print("il faut toutes les issues du marché, pas une seule cote")
        return 1

    value = overround(args.prices)
    print(f"{args.nom}: {len(args.prices)} issues, cotes "
          f"{', '.join(f'{p:g}' for p in args.prices)}")
    print(f"\n  surmarge                 {value:+.2%}")
    if value <= 0:
        print("  -> somme des inverses sous 1: ce serait un arbitrage à soi seul,")
        print("     donc une erreur de relevé ou des cotes non simultanées.")
        return 0
    print(f"  seuil de rentabilité     {BREAK_EVEN:.2%}")
    print(f"  rendement d'un gros favori attendu  {favourite_return(value):.4f} "
          f"({favourite_return(value) - 1:+.2%} par pari)")

    print("\n  où ça se situe:")
    for label, mark in sorted(LANDMARKS + [(f">>> {args.nom}", value)],
                              key=lambda item: item[1]):
        pointer = "  <<<" if label.startswith(">>>") else ""
        print(f"    {mark:>7.2%}  {label}{pointer}")

    print()
    if value < BREAK_EVEN:
        print("  Sous le seuil: un gros favori y est théoriquement gagnant.")
        print("  Restent le risque de contrepartie et la durée de vie du compte,")
        print("  qu'aucune cote ne mesure — voir RAPPORT_INTERNATIONAL_2026_09_13.md.")
    else:
        gap = value - BREAK_EVEN
        print(f"  Au-dessus du seuil de {gap:.2%}. Il faudrait un modèle battant")
        print(f"  le marché de {gap / SLOPE:.2%} de surmarge équivalente, quand le")
        print("  meilleur jamais mesuré ici perdait déjà à marge nulle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
