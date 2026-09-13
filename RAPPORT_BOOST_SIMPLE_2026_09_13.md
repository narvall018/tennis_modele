# Le boost sur sélection unique — et pourquoi il faut le poser sur un favori

Le combiné boosté est mort parce que la marge s'empile **géométriquement** quand
le boost s'ajoute **linéairement**. Sur une sélection unique, rien ne s'empile :
il n'y a qu'une seule marge à franchir. Et l'échelle d'exécution dit exactement
laquelle.

## Le point qui change tout

La marge à franchir n'est pas la surmarge affichée. Le biais favori-outsider la
pose sur les outsiders :

| cote | rendement mesuré | marge effective |
|---|---:|---:|
| 1,15 | 0,9852 | **1,48 %** |
| 1,50 | 0,9591 | 4,09 % |
| 2,50 | 0,9320 | 6,80 % |
| 5,00 | 0,8819 | 11,81 % |
| 10,00 | 0,7558 | **24,42 %** |

Mesuré sur 191 458 matchs de football chez Bet365, par fenêtre glissante sur le
logarithme de la cote.

**Conséquence, contraire à l'intuition : le même boost de +10 % rapporte
+8,25 % sur un favori à 1,15 et perd −16,98 % sur un outsider à 10,00.** Un
boost ne crée aucun avantage — il franchit une marge — et la marge est cinq fois
plus petite sur les favoris.

## Boost minimal pour atteindre l'équilibre

Surmarges relevées en direct sur 95 matchs de grands championnats, converties en
rendement par la pente de l'échelle d'exécution (0,53 point par point).

| cote | PMU | Netbet | Betclic | Unibet | Winamax | meilleur des 5 |
|---|---:|---:|---:|---:|---:|---:|
| **1,15** | **1,94 %** | 2,77 % | 2,96 % | 3,98 % | 6,51 % | **1,61 %** |
| 1,30 | 3,05 % | 3,90 % | 4,09 % | 5,13 % | 7,72 % | 2,72 % |
| 1,50 | 4,72 % | 5,60 % | 5,80 % | 6,88 % | 9,56 % | 4,38 % |
| 3,00 | 8,12 % | 9,06 % | 9,27 % | 10,42 % | 13,28 % | 7,76 % |
| 10,00 | 33,05 % | 34,48 % | 34,79 % | 36,55 % | 40,96 % | 32,50 % |

## Espérance avec un boost de +10 %

| cote | PMU | Betclic | Unibet | Winamax |
|---|---:|---:|---:|---:|
| **1,15** | **+7,91 %** | +6,84 % | +5,79 % | +3,27 % |
| 1,50 | +5,04 % | +3,97 % | +2,92 % | +0,40 % |
| 2,20 | +2,46 % | +1,39 % | +0,34 % | −2,18 % |
| 5,00 | −3,45 % | −4,51 % | −5,56 % | −8,08 % |
| 10,00 | −17,33 % | −18,39 % | −19,44 % | −21,96 % |

## La règle opérationnelle

1. **Poser le boost sur le favori le plus court disponible**, jamais sur une
   grosse cote. C'est l'inverse de ce que l'habitude suggère, et l'écart entre
   les deux vaut 25 points d'espérance.
2. **Chez PMU de préférence, jamais chez Winamax.** Winamax exige 6,51 % de
   boost là où PMU en demande 1,94 % : plus du triple, parce que sa surmarge est
   de 15,29 % contre 7,34 %.
3. En dessous de **2 % de boost**, même un favori à 1,15 chez PMU ne passe pas.

## Ce que ça n'établit pas

- **Que l'opérateur accepte de booster cette sélection-là.** Les books boostent
  ce sur quoi ils veulent du volume, rarement un favori écrasant. C'est la
  limite pratique principale, et elle n'est pas dans les données.
- Les plafonds de mise, et la durée de vie d'un compte qui ne joue que ça.
- Le rendement par cote vient du football chez Bet365 ; la conversion vers chaque
  opérateur français passe par la pente de l'échelle, estimée sur trois points.
- Les surmarges françaises viennent d'**un seul relevé** de 95 matchs.

## Portée honnête

Ce n'est pas un modèle, et aucun modèle ne marche : voir
`RAPPORT_SEGMENTS_2026_09_13.md`. C'est la même famille que l'extraction des
offres de bienvenue — promotionnel, borné, plafonné. La valeur n'est pas de
prédire quoi que ce soit, mais de savoir **où poser une offre** quand on en
reçoit une.

`scripts/run_boost_calculator.py --boost 0.15` rejoue le tableau pour n'importe
quel boost.
