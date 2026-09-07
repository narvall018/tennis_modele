#!/usr/bin/env python3
"""Extraire une offre de bienvenue en couvrant les deux jambes chez deux books.

Une offre du type « misez 50 €, recevez 50 € en paris gratuits quoi qu'il
arrive » se décompose en deux paris indépendants, et **ils demandent des cotes
opposées**. C'est le point que l'intuition rate.

*Le pari qualifiant.* Le free bet est acquis dans tous les cas, donc ce pari ne
sert qu'à valider l'offre. Couvert chez un second opérateur, son coût vaut
exactement `mise x cote x surmarge`. Il faut donc la cote **la plus basse**
autorisée — un favori écrasant, pas un match serré.

*Le free bet.* La mise n'est pas rendue : on ne touche que `(O-1)`. Couvert, il
rapporte `mise x (1 - 1/O)` moins la surmarge, ce qui **croît avec la cote**. Il
faut donc la cote **la plus haute** possible, la seule limite étant le capital
nécessaire à la couverture.

Sans couverture, le biais favori-outsider plafonnerait le free bet vers 5,00 —
au-delà, les outsiders sont trop mal payés. La couverture supprime cette limite :
elle rend le rendement réel indifférent, seule compte la surmarge de la paire.
"""

from __future__ import annotations

import argparse


def overround(price_a: float, price_b: float) -> float:
    """Ce que les deux books prélèvent ensemble sur la paire."""
    return 1.0 / price_a + 1.0 / price_b - 1.0


def qualifying_bet(stake: float, price: float, counter_price: float) -> dict[str, float]:
    """Miser `stake` sur une jambe, couvrir l'autre: le coût est verrouillé."""
    hedge = stake * price / counter_price
    return {
        "mise": stake,
        "couverture": hedge,
        "capital": stake + hedge,
        # Le coût vaut mise x cote x surmarge, d'où l'intérêt d'une cote basse.
        "coût": stake + hedge - stake * price,
    }


def free_bet(face: float, price: float, counter_price: float) -> dict[str, float]:
    """Le free bet ne rend pas la mise: on ne couvre que le gain net."""
    payout = face * (price - 1.0)
    hedge = payout / counter_price
    return {
        "valeur_faciale": face,
        "cote": price,
        "couverture": hedge,
        "capital": hedge,
        "garanti": payout - hedge,
        "taux": (payout - hedge) / face,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stake", type=float, default=50.0,
                        help="mise du pari qualifiant")
    parser.add_argument("--face", type=float, default=50.0,
                        help="valeur faciale du free bet")
    parser.add_argument("--margin", type=float, default=0.01,
                        help="surmarge de la paire de books, en fraction")
    args = parser.parse_args()

    print(f"Offre: {args.stake:.0f}€ misés -> {args.face:.0f}€ de paris gratuits, "
          f"acquis quoi qu'il arrive.\n")

    print("=== 1. le pari qualifiant: viser la cote la PLUS BASSE ===")
    print(f"{'cote':>7s} {'contre-cote':>12s} {'couverture':>11s} "
          f"{'capital':>9s} {'coût verrouillé':>16s}")
    for price in (1.10, 1.25, 1.50, 1.80, 2.20, 3.00):
        counter = 1.0 / (1.0 + args.margin - 1.0 / price)
        result = qualifying_bet(args.stake, price, counter)
        print(f"{price:>7.2f} {counter:>12.2f} {result['couverture']:>10.2f}€ "
              f"{result['capital']:>8.2f}€ {result['coût']:>15.2f}€")

    print("\n=== 2. le free bet: viser la cote la PLUS HAUTE ===")
    print(f"{'cote':>7s} {'contre-cote':>12s} {'couverture':>11s} "
          f"{'garanti':>9s} {'extraction':>11s}")
    for price in (2.50, 3.50, 4.50, 6.00, 9.00, 15.00):
        counter = 1.0 / (1.0 + args.margin - 1.0 / price)
        result = free_bet(args.face, price, counter)
        print(f"{price:>7.2f} {counter:>12.2f} {result['couverture']:>10.2f}€ "
              f"{result['garanti']:>8.2f}€ {result['taux']:>10.1%}")

    print("\nLe capital de couverture croît beaucoup plus vite que le gain:")
    print("c'est lui, et non le rendement, qui borne le choix de la cote.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
