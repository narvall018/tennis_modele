# Chez les bookmakers français : la chaîne complète, mesurée

Question posée : une stratégie rentable chez les opérateurs agréés, sans
arbitrage, juste un modèle. Voici chaque maillon, mesuré plutôt qu'affirmé.

## 1. La barre à franchir

| | |
|---|---:|
| surmarge des cinq opérateurs agréés | **7,3 % à 15,3 %** |
| seuil de rentabilité d'un gros favori | **4,62 %** |
| même marque à l'étranger *(Winamax Allemagne)* | 5,65 % |

**L'écart français est fiscal, pas commercial.** Winamax facture 11,11 % en
France et 5,65 % en Allemagne — même entreprise, même sport, même semaine. C'est
le prélèvement sur les mises répercuté sur la cote, environ cinq points.

Un modèle devrait donc battre le marché de 3 à 8 points selon l'opérateur.

## 2. Aucun modèle n'en trouve un seul

Quatre sports, dix familles d'algorithmes, 31 sous-ensembles (surface, niveau,
pays, division, catégorie de poids, tranche de prix), correction de multiplicité
par simulation sous H0 :

- **la meilleure des 31 cellules a 75,9 % de chances de venir du hasard** ;
- le seul candidat robuste — boosting sur la division 1 de football, gain de
  log-loss **+0,00496, t = +4,75** — donne un **ROI de −4,6 %** ;
- sur les paris qu'il sélectionne, le taux de réussite est de 32,0 % contre
  **33,6 % implicites** : là où le modèle contredit le marché, le marché a
  raison.

Un gain de prévision cinq fois supérieur au seuil déclaré ne produit toujours
pas d'argent. Ce n'est pas un défaut d'algorithme, c'est que les paris se
concentrent là où le modèle est trop confiant.

## 3. Les idées structurelles, testées une par une

| idée | résultat | n |
|---|---:|---:|
| le nul, cote 2,8-3,2 | **−2,14 %** | 24 080 |
| le nul, cote 5,5+ | **−20,33 %** | 7 447 |
| backer le complément d'un outsider à 4-6 | −5,87 % | 39 131 |
| backer le complément d'un outsider à 15+ | **−3,87 %** | 2 596 |
| arbitrage entre books français | **impossible** | 91 marchés |
| valeur contre Pinnacle dévigué, avant-match | **zéro occasion** | — |

**Le nul est le pire pari du football**, pas le maillon faible du book : c'est
là qu'il charge le plus.

**Le complément d'un outsider** s'améliore bien quand l'outsider s'allonge
(−5,87 % → −3,87 %), ce qui confirme le biais favori-outsider — mais on paie la
marge sur deux paris, et ça dépasse toujours ce qu'on évite.

**L'arbitrage entre books français** demanderait une somme des inverses sous
1,0000 ; le meilleur de 91 marchés est à **1,0465**, soit 4,65 points au-dessus.
Parier les deux côtés y fait perdre 4,65 % à 7,41 % de façon garantie.

**La valeur contre Pinnacle** semblait exister : 5,56 % des cotes, +14,98 % de
moyenne. Inspection des trois cas : **les trois portent sur des matchs déjà
commencés**. Ce sont des cotes en direct non encore mises à jour — Betclic en
affichait une vieille de 4,2 minutes en plein match. Aucune n'était pariable.
Avant match, le compte est de **zéro**.

## Verdict

Il n'y a pas de stratégie rentable chez les bookmakers français, avec ou sans
modèle. La raison n'est ni l'efficience du marché ni la qualité des algorithmes :
c'est **un prélèvement d'environ cinq points**, qui porte la marge au-delà de
tout avantage jamais mesuré dans ce dépôt — y compris le meilleur, qui perdait
déjà de l'argent à marge nulle.

Ce qui reste positif chez ces opérateurs relève du **marketing, pas du pari** :
l'extraction d'une offre de bienvenue (~62 % de la valeur faciale, garanti) et
le boost sur un favori court (+8,25 % avec 10 % de boost à cote 1,15). Dans les
deux cas c'est l'opérateur qui rend une part de ce qu'il a prélevé.
