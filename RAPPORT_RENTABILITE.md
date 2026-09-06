# Peut-on trouver une stratégie rentable ?

**Réponse : non, pas de façon démontrable avec les données disponibles.**

Ce document explique ce qui a été cherché, ce qui a été trouvé, et pourquoi ce
qui a été trouvé ne suffit pas. Il ne recommande aucune mise.

Reproduire : `python3 scripts/run_edge_diagnostics.py wta` et `… atp`.

---

## 1. Ce qui a d'abord été vérifié : reste-t-il un marché inexploré ?

Non. Tennis-Data publie 47 fichiers, tous chargés : 27 années ATP et 20 années
WTA, circuits principaux uniquement. Il n'existe aucune source gratuite de cotes
Challenger ou ITF ; la « Huge Tennis Database » (534 Mo, la plus complète de
Kaggle) ne contient aucune cote. Côté UFC, la contre-expertise a montré que les
deux compilations publiques sont la même donnée.

Toute la preuve rétrospective disponible est donc déjà consommée.

## 2. Le signal réel qui a été trouvé

Sur la WTA, le mélange modèle+marché prévoit **mieux que Pinnacle**, un
bookmaker à faible marge, sur des données qu'il n'avait jamais vues :

| Comparaison, holdout WTA 2023–2026, 7 156 matchs | Log-loss |
|---|---:|
| Prix moyen du marché, dévigé | 0,58859 |
| Prix Pinnacle, dévigé | 0,58789 |
| Mélange du modèle | **0,58687** |

Gain contre la moyenne du marché : **+0,00172**. Contre Pinnacle : **+0,00103**.

C'est un vrai résultat, et le premier de ce dépôt à survivre à un holdout. Il
écarte l'explication paresseuse « le modèle a seulement appris que la cote
moyenne est périmée » : il bat aussi le prix d'un opérateur sérieux.

Pour mémoire, l'ATP ne fait pas ça : son gain tombe à +0,00006 hors échantillon.

## 3. Pourquoi ce signal ne devient pas de l'argent

### 3.1 La règle la plus permissive perd

En pariant **tout côté à espérance positive**, sans aucun réglage, décote 2 % :

| Source | Développement | Réglage | Validation | Holdout | Total | IC 90 % |
|---|---:|---:|---:|---:|---:|---|
| moyenne | +1,6 % | −9,2 % | +0,9 % | −2,5 % | **−2,35 %** | [−5,7 % ; +1,0 %] |
| Pinnacle | +0,0 % | −3,5 % | −2,4 % | −1,2 % | **−1,53 %** | [−3,1 % ; +0,0 %] |
| Bet365 | −3,1 % | −9,4 % | +1,4 % | +3,3 % | **−1,34 %** | [−4,3 % ; +1,5 %] |
| maximum | +0,8 % | −2,8 % | −2,0 % | −1,5 % | **−1,15 %** | [−2,4 % ; +0,1 %] |

Le cas `maximum` est le plus parlant : c'est le meilleur prix trouvé chez
n'importe quel opérateur, une marge de 1,2 % seulement, où un gain de 0,00007 de
log-loss suffirait en théorie. Le modèle en a 0,00040, presque six fois plus —
**et il perd quand même.**

### 3.2 Une seule anomalie est structurelle : les gros outsiders

En découpant par bande de cote, une cellule est négative dans **les quatre
périodes et chez toutes les sources** : les cotes ≥ 6, entre −24 % et −65 %.
Le modèle surestime systématiquement les outsiders extrêmes. C'est un défaut
réel du modèle, pas du bruit.

En les excluant, tout remonte au niveau de l'équilibre :

| Source, cote < 6 | Total | IC 90 % | Périodes positives |
|---|---:|---|---:|
| Pinnacle | **+0,74 %** | [−0,7 % ; +2,1 %] | 2 sur 4 |
| maximum | +0,78 % | [−0,3 % ; +1,9 %] | 2 sur 4 |
| Bet365 | +0,63 % | [−1,9 % ; +3,1 %] | 3 sur 4 |
| moyenne | −2,58 % | [−5,7 % ; +0,6 %] | 2 sur 4 |

**Aucune des 16 configurations testées, sur les deux circuits, n'a un intervalle
de confiance à 90 % qui exclut zéro par le haut.** Deux le font par le bas : au
prix moyen, l'ATP perd de façon fiable.

Et le seuil de cote 6 a été lu **sur des données déjà dépensées**. C'est une
hypothèse, pas un résultat validé.

### 3.3 Le +0,74 % est indémontrable, et fragile

Sur les 13 622 paris Pinnacle de la variante la plus favorable :

- écart-type d'un pari : 1,088 ; erreur standard : 0,0093 ; **t = 0,80**.
- pour exclure zéro à 90 % il faudrait **35 250 paris, soit 34 ans** de WTA au
  rythme observé de 1 045 paris par an. À 95 % : 56 ans.
- 2 points de friction d'exécution supplémentaires suffisent à l'annuler :

| Décote totale | 2,0 % | 2,5 % | 3,0 % | 4,0 % |
|---|---:|---:|---:|---:|
| ROI | +0,74 % | +0,50 % | +0,26 % | **−0,23 %** |

Or 2 points de friction, c'est peu : la ligne bouge entre la décision et la mise,
Pinnacle limite les comptes gagnants, et le prix de clôture retenu ici n'est pas
toujours celui qu'on obtient réellement.

Enfin, la règle parie sur **39 à 57 % de tous les matchs**. Un edge authentique
est en général rare. Un edge revendiqué sur la moitié de la carte, pour un
rendement nul, est la signature d'un modèle légèrement mieux calibré en moyenne,
pas d'un modèle qui bat le prix là où il parie.

## 4. Ce que cela laisse

Le gain de prévision est réel mais trop petit : il est réparti sur tous les
matchs, alors que les paris se concentrent là où le modèle s'écarte le plus du
marché — c'est-à-dire précisément là où c'est le plus souvent le modèle qui a
tort.

Une seule chose est honnêtement défendable : **un suivi prospectif sur papier**,
pour voir si le signal WTA existe encore sur des cotes réellement horodatées.
C'est long et ça ne rapporte rien tant que ça dure, mais c'est la seule preuve
qui reste possible. Le protocole candidat est gelé dans
[`models/wta_strategy/prospective_candidate.json`](models/wta_strategy/prospective_candidate.json)
avec le statut `HYPOTHESE_NON_VALIDEE`.

## 5. Inventaire des pistes : ce qui a été exploré, ce qui reste

| Piste | État | Pourquoi |
|---|---|---|
| Familles de modèles (logistique, GBM, XGBoost, MLP, résiduels de marché) | épuisée | Étude imbriquée + phase 4 sur ATP, rejouées sur WTA |
| Recalibrage par surface, serve/return, interactions | épuisée | Phase 4, toutes sous la gate |
| Nouveaux circuits cotés (Challenger, ITF) | **impossible** | Aucune source gratuite ne publie de cotes hors ATP/WTA principaux |
| Autres marchés tennis (handicap, totaux, live) | **impossible** | Tennis-Data ne publie que le vainqueur du match |
| Structure de marché tennis (dispersion entre books) | non testable | Terrain d'évaluation ATP et WTA déjà brûlé |
| **Elo enrichi par les Challengers et qualifications** | **explorée puis fermée** | Meilleur classement (+0,0047, et +0,0223 sur les historiques minces) mais apport nul sur le marché : −0,00002 |
| **UFC méthode de victoire (KO / soum. / décision)** | **explorée puis fermée** | Biais réel de +2,92 pt sur les décisions, 13 ans sur 13, mais overround de 22 % et marché imbattable en discrimination |
| **Book fin contre book mou (Pinnacle vs Bet365)** | **explorée puis fermée** | Mécanisme réel — contrôle inversé à −14,6 % — mais −3,5 % à −14 % : la marge du book mou dépasse l'écart capturé |
| **Biais favori/outsider du marché** | **explorée puis fermée** | Biais réel et monotone (+1,65 pt sur les gros favoris) mais 0 cellule rentable sur 64 ; la seule positive est démentie par l'autre circuit |
| UFC autres marchés (rounds, totaux) | **impossible** | Les données ne contiennent que moneyline et 6 props méthode |
| UFC mouvement de ligne | **impossible** | 199 combats à 2 snapshots, 12 complets |
| UFC moneyline | épuisée | Trois phases, toutes rejetées |

### 5.1 Elo multi-niveaux : une meilleure note que le marché connaît déjà

`python3 scripts/run_multi_tier_ratings.py` puis `… run_multi_tier_residual.py`.

Les 121 373 matchs Challenger et 29 739 qualifications n'avaient jamais servi.
Les ajouter à la passe de notation fait passer celle-ci de 79 705 à 230 817
matchs, et **89,9 % des matchs du circuit principal gagnent de l'information** —
85 matchs supplémentaires connus par joueur, en médiane.

Le classement devient nettement meilleur :

| Population | Log-loss Elo principal | Elo enrichi | Gain |
|---|---:|---:|---:|
| Tout le circuit principal | 0,61614 | 0,61145 | **+0,00469** |
| Au moins un joueur à historique mince | 0,63663 | 0,61433 | **+0,02230** |

Le gain est concentré exactement là où le mécanisme le prédisait : les joueurs
que le circuit principal a peu vus. C'est le plus gros effet mesuré dans tout ce
projet, dix fois l'edge WTA.

**Et il ne sert à rien.** Posé par-dessus le prix, il n'apporte plus rien :

| Modèle | Log-loss | Gain vs marché |
|---|---:|---:|
| Marché dévigé seul | 0,56947 | — |
| Marché + Elo principal | 0,56864 | +0,00083 |
| Marché + Elo enrichi | 0,56867 | +0,00081 |

Apport propre de l'enrichissement : **−0,00002**. Sur le sous-groupe déclaré à
l'avance — 7 669 matchs avec un joueur à historique mince — c'est **−0,00072** :
l'enrichissement est *moins bon* que l'Elo principal là où il devait le plus
aider.

L'explication est simple et aurait dû être anticipée : les bookmakers suivent le
circuit Challenger. Toute la forme de deuxième division est déjà dans le prix, et
y ajouter une note plus bruitée dégrade légèrement sa complémentarité avec le
marché.

Le module reste dans le dépôt (`src/features/multi_tier_ratings.py`) : il produit
un meilleur classement, ce qui a de la valeur pour afficher une estimation, mais
il ne doit pas être présenté comme un avantage de pari.

## 6. Book fin contre book mou — la stratégie sans modèle

`python3 scripts/run_sharp_vs_soft.py`

C'est la conséquence logique du constat « le marché sait déjà » : ne plus essayer
de battre le marché, mais utiliser sa partie la plus fine contre la plus molle.
Pinnacle (overround 2,4 %, limites hautes) sert de vérité ; on parie chez le book
qui s'en écarte. **Aucun modèle sportif, aucun paramètre ajusté** — seulement un
seuil d'écart, et les cinq seuils testés sont tous publiés.

### Le mécanisme est réel

Un bras de falsification tourne en parallèle : la même règle, signal inversé. Il
doit perdre nettement plus si le signal existe.

| Bras | ATP Bet365 | WTA Bet365 |
|---|---:|---:|
| Signal normal | −4,87 % | −5,61 % |
| **Signal inversé (contrôle)** | **−14,6 %** | **−13,8 %** |

Dix points d'écart. Pinnacle porte donc bien une information réelle sur le bon
côté du prix d'un book mou. Ce n'est pas du bruit.

### Il ne rapporte quand même rien

| Book | Seuils testés | ROI | Années positives |
|---|---|---:|---:|
| Bet365 (ATP) | 0 % → 5 % | −3,48 % à −14,07 % | 5 à 9 sur 22 |
| Bet365 (WTA) | 0 % → 5 % | −5,61 % à −10,54 % | 3 à 7 sur 19 |
| Moyenne (ATP/WTA) | 0 % → 5 % | −7,3 % à −23,4 % | pire |

Aucune cellule, sur aucun des deux circuits, n'a d'intervalle à 90 % positif.
La raison est arithmétique : la marge de Bet365 est de 6,7 % d'overround, soit
~3,3 % par côté, et l'écart moyen à Pinnacle est plus petit que ça. On paie plus
cher que ce qu'on capture.

### Le « maximum » n'est pas un prix

Une seule colonne approchait l'équilibre dans toutes mes analyses, y compris
plus haut dans ce rapport : `market_maximum`. L'audit de simultanéité explique
pourquoi, et il est sans appel :

| Source | Overround médian | Arbitrage implicite (overround < 1) |
|---|---:|---:|
| Pinnacle | 1,0241 | **0,03 %** |
| Bet365 | 1,0667 | **0,00 %** |
| Moyenne | 1,0577 | **0,04 %** |
| **Maximum** | 1,0021 | **42,97 %** |

Un overround inférieur à 1 est de l'argent gratuit : ça n'existe pas dans un
marché liquide. En trouver dans **43 % des matchs** prouve que la colonne
`maximum` n'est pas une paire de prix simultanée — c'est le meilleur prix vu
chez un book quelconque à un instant quelconque. Elle n'est pas pariable.

**Conséquence rétroactive : tous les chiffres positifs de ce dépôt impliquant
`maximum` doivent être écartés**, y compris le +0,78 % de la section 3.2 et le
+6,9 % du holdout WTA. Le module marque désormais cette source `NON pariable` et
un test empêche de la présenter autrement.

## 7. Audit de calibration du marché — sans modèle

`python3 scripts/run_market_audit.py`

Question différente de toutes les précédentes : sans rien modéliser, le prix
correspond-il à ce qui arrive ? C'est ce test qui avait trouvé le biais décisions
UFC, il n'avait jamais été fait sur le moneyline tennis.

### Un biais favori/outsider net et monotone

| Probabilité du favori | p marché | Réel | Écart | ROI outsider (Bet365) |
|---|---:|---:|---:|---:|
| 0,5–0,6 | 0,5535 | 0,5519 | −0,16 pt | −6,7 % |
| 0,6–0,7 | 0,6485 | 0,6532 | +0,47 pt | −9,3 % |
| 0,7–0,8 | 0,7469 | 0,7614 | +1,45 pt | −14,4 % |
| 0,8–0,9 | 0,8458 | 0,8589 | +1,31 pt | −22,6 % |
| 0,9–1,0 | 0,9346 | 0,9511 | **+1,65 pt** | **−44,4 %** |

Les gros favoris sont sous-évalués, les outsiders extrêmes ruineux. L'effet est
monotone, présent sur les deux circuits et dans les deux époques : c'est un vrai
biais structurel, pas du bruit.

### Il ne franchit pas la marge

Chez Bet365 (overround 6,7 %), **64 cellules examinées, 0 rentable**. Même la
meilleure — gros favoris — rend −1,2 %.

Chez Pinnacle (overround 2,4 %), le biais arrive presque à passer : ATP −0,3 %,
WTA **+0,8 % avec un IC90 de [+0,1 % ; +1,6 %]**, la seule cellule positive de
toute l'étude.

### Cette cellule positive est du bruit, et voici pourquoi

C'est exactement le résultat qu'une analyse malhonnête mettrait en avant. Trois
contrôles le tuent :

1. **L'autre circuit contredit.** ATP, même cellule, plus de données
   (n = 3 383) : **−0,33 %**, t = −0,87, 11 années positives sur 22.
2. **Le poolé est nul.** ATP + WTA, n = 5 319 : **+0,09 %**, t = +0,30,
   IC90 [−0,40 % ; +0,59 %].
3. **L'effet ne résiste pas au déplacement du seuil.** WTA p ≥ 0,90 : +0,83 % ;
   p ≥ 0,85 : +0,41 % ; p ≥ 0,80 : −0,10 %. Un biais structurel ne s'évapore pas
   quand on élargit la fenêtre de 5 points.

S'ajoute la multiplicité : environ 128 cellules examinées, dont ~13 devraient
exclure zéro par pur hasard. En trouver une positive est le résultat attendu du
hasard, pas une découverte.

Et opérationnellement, c'est absurde : la cote moyenne de ces favoris est
**1,053**. On mise 100 pour gagner 5, en visant 0,8 % d'edge, chez un book qui
limite. Le moindre mouvement de ligne l'efface.

Le contrôle croisé ATP/WTA est désormais automatique dans le module, et un test
vérifie qu'une cellule rentable sur un seul circuit est bien étiquetée
« cellule chanceuse ».

## 8. Ce qui n'est pas honnête et n'a donc pas été fait

- Retenir la cellule Bet365 du holdout WTA (+3,3 %) ou la cellule ATP `maximum`
  (+0,79 %) : ce sont les meilleures d'un tableau de seize, choisies après coup.
- Régler à nouveau la règle de pari sur le holdout maintenant qu'il est ouvert.
- Présenter le prix `maximum` comme atteignable : il n'est ni tenu ni exécutable
  à volume.
- Traiter le seuil de cote 6 comme validé alors qu'il vient des mêmes données.
