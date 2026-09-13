# L'exécution, pas la prédiction — et le mur enfin chiffré

Toutes les pistes de modélisation ont échoué pour la même raison : il faut battre
5 à 7 % de marge, et aucun modèle n'y arrive. Cette piste part de l'autre bout.

**Le biais favori-outsider ne répartit pas la marge du book — il la pose sur les
outsiders.** Sur les gros favoris, le prélèvement effectif tombe à 1-2 %. La barre
y est donc cinq fois plus basse, et la question cesse d'être « quel modèle ? »
pour devenir « quel prix ? ».

## Ce qu'un point de surmarge coûte au favori

Trois sources historiques, 191 458 matchs, tranche de cote 1,10-1,30 :

| source | surmarge | n | r | IC 95 % |
|---|---:|---:|---:|---|
| Bet365 | 6,55 % | 7 628 | 0,9899 | [0,9793, 1,0004] |
| Pinnacle | 3,02 % | 4 154 | 1,0080 | [0,9928, 1,0224] |
| Max panel | 1,75 % | 1 986 | 1,0156 | [0,9956, 1,0346] |

La relation est presque exactement linéaire : **un point de surmarge coûte 0,53
point de rendement au favori**. Autrement dit le favori ne porte que la moitié de
la marge ; l'autre moitié tombe sur l'outsider — c'est le biais, mesuré par son
effet plutôt que décrit.

D'où un seuil net : **un gros favori cesse de perdre en dessous de 4,62 % de
surmarge.**

## Où se situe ce qu'on peut atteindre

Surmarges relevées en direct sur 95 matchs de grands championnats :

| accès | surmarge | rendement attendu |
|---|---:|---:|
| Winamax | 15,29 % | −5,67 % |
| Unibet | 10,97 % | −3,38 % |
| Betclic | 9,17 % | −2,42 % |
| Netbet | 8,84 % | −2,24 % |
| PMU | 7,34 % | −1,45 % |
| **meilleur des 5 français** | **6,75 %** | **−1,13 %** |
| Pinnacle *(hors France)* | 5,16 % | −0,29 % |
| *seuil de rentabilité* | *4,62 %* | *0,00 %* |

**Il manque 2,13 points de surmarge**, et ce déficit est entièrement
juridictionnel. C'est la formulation la plus précise du mur que ce projet ait
produite : il ne s'agit plus de dire que l'avantage est hors de portée, mais de
dire de combien.

## Ce que ça donne d'utilisable tout de suite

Même sans rien de rentable, l'échelle est actionnable :

- **Winamax coûte plus du double de PMU** sur le même pari — 15,29 % contre
  7,34 % de surmarge. Sur un favori, c'est 4,2 points de rendement par pari.
- Passer d'un seul book au **meilleur des cinq** vaut 1,3 point si vous partez de
  PMU, et **4,5 points** si vous partez de Winamax.
- Le magasinage rapporte davantage que n'importe quel modèle testé ici. Aucun
  descripteur n'a jamais valu 4 points.

## Ce que ça n'établit pas

- **Aucun intervalle historique n'exclut 1**, pas même celui du Max panel
  ([0,9956, 1,0346]). Rien n'est prouvé rentable, y compris hors de France.
- Le **Max panel implique un arbitrage dans 12 % des matchs** : il contient des
  cotes périmées et surestime donc ce qui est réellement prenable. Le repère
  honnête est Pinnacle, à −0,29 % — l'équilibre, pas le profit.
- L'ajustement repose sur **trois points**. Il colle à 0,001 près, mais trois
  points restent trois points.
- Les surmarges françaises viennent d'**un seul relevé** de 95 matchs, et sont
  croisées avec des rendements mesurés sur l'historique football : deux sources
  différentes.

## Conclusion

Il existe une proposition gagnante — un gros favori pris sous 4,62 % de surmarge
— et elle est inatteignable depuis la France par une marge de 2,13 points. La
bonne question n'a jamais été « quel modèle », mais « quel prix », et la réponse
est que le prix français est structurellement trop cher d'environ deux points.

`scripts/run_execution_ladder.py` rejoue l'échelle entière, `--skip-live` pour
économiser le quota d'API.
