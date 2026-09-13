# Découper par surface, championnat, catégorie — et deux fuites trouvées en route

Question posée : puisque chaque sport échoue en bloc, un avantage réel ne
serait-il pas dilué dans la moyenne ? Un sous-ensemble — une surface, une
division, une catégorie de poids — pourrait-il payer ?

Réponse : **non**, et le chemin pour y arriver est plus instructif que la
conclusion.

## Le point de départ

`three_sport_residual` a testé les trois sports en bloc le 13 septembre. Gain de
log-loss **conditionnel au prix**, seuil déclaré à +0,001 :

| sport | modèle retenu | gain sur le prix | ROI évalué | statut |
|---|---|---:|---:|---|
| ATP | base | +0,00050 | −21,9 % à −22,7 % | NO_ROBUST_CANDIDATE |
| Football | enriched_strong | +0,00018 | −0,17 % à −2,1 % | NO_ROBUST_CANDIDATE |
| UFC | base | **+0,00359** | −0,5 % à −4,9 % | NO_ROBUST_CANDIDATE |
| WTA | cross_book_form | — | +5,70 %, borne basse **−7,33 %** | NO_ROBUST_CANDIDATE |

L'UFC passe pourtant le seuil de trois fois et demie, et perd quand même. C'est
le premier signe que le seuil de +0,001 est trop permissif.

## Le balayage : 31 cellules

`scripts/run_segment_sweep.py` découpe chaque sport par surface, niveau de
tournoi, pays, rang de division, catégorie de poids et tranche de prix, puis
compare le prix seul au prix plus descripteurs, en avant-marche annuelle.

Chaque grille est accompagnée de **sa simulation sous H0**, parce que retenir la
meilleure de N cellules est exactement ce qui a produit le faux positif de
`favourite_longshot_bias`.

## Deux fuites, toutes les deux de mon fait

**Première.** L'UFC est ressorti à **+0,62 de log-loss, t = +46**. Un modèle
quasi parfait. La cause : une colonne nommée `y` — l'étiquette elle-même — que
mon filtre de descripteurs ne retirait pas.

**Seconde, plus instructive.** Le football « division 1 » donnait **+0,00434,
t = +3,33, et 0,7 % sous H0** — la première cellule du projet à survivre à la
correction de multiplicité. Elle venait de 64 colonnes de marché présentes dans
les descripteurs, dont les cotes de **clôture** (`B365CH`, `PSC*`, `BFEC*`). Le
modèle ne battait pas le marché : il le lisait au coup d'envoi quand la référence
n'avait que l'ouverture. Un filtre par jetons durci a ensuite encore laissé
passer `P>2.5` et `PC>2.5`, les cotes Pinnacle sur les buts.

**Règle retenue :** un descripteur ne doit contenir aucune colonne de marché, et
l'exclusion doit être par défaut. Une seule cote oubliée invalide la comparaison
entière.

## Après correction

| sport | meilleure cellule | gain | t | P sous H0 |
|---|---|---:|---:|---:|
| ATP | dur | +0,00100 | +1,44 | 61,3 % |
| Football | division 1 | +0,00148 | +1,70 | 54,0 % |
| UFC | léger outsider | −0,00454 | −0,60 | 92,5 % |

**31 cellules, et la meilleure a 75,9 % de chances de venir du hasard seul.**

## D'autres algorithmes ?

Cinq familles sur les trois meilleures cellules. Les modèles plus puissants sont
**systématiquement pires** sur un résidu de marché — ils surajustent ce que le
prix contient déjà :

| cellule | logistique L2 | gradient boosting | boosting profond | forêt |
|---|---:|---:|---:|---:|
| ATP gazon | +0,00248 | −0,00497 | −0,03062 | −0,00925 |
| ATP dur | +0,00100 | −0,00027 | −0,00293 | −0,00612 |
| Foot division 1 | +0,00145 | **+0,00496** | +0,00361 | +0,00170 |

Une exception : le boosting sur la division 1 de football, **t = +4,75**.

## La seule candidate sérieuse, et sa mort

Ce gain est solide. Il survit à la multiplicité (t = 4,75 sur ~49 tests). Il
survit au retrait des descripteurs `new_*` : avec les **19 seuls descripteurs
vérifiés d'avant-match**, il vaut encore +0,00458, t = +4,50. Il ne vient ni
d'une fuite ni d'un artefact.

Puis vient le seul test qui compte :

| seuil d'espérance | paris | ROI | IC 95 % |
|---|---:|---:|---:|
| 0 % | 1 474 | **−4,57 %** | [−12,30 %, +3,16 %] |
| 2 % | 665 | −4,46 % | [−19,12 %, +10,20 %] |
| 5 % | 288 | −10,92 % | [−37,43 %, +15,58 %] |
| 10 % | 109 | −3,41 % | [−56,22 %, +49,39 %] |

**Négatif à tous les seuils.** Et le diagnostic est sans ambiguïté : sur les
matchs sélectionnés, le taux de réussite est de **32,0 %** contre **33,6 %**
implicites. Là où le modèle contredit le marché, c'est le marché qui a raison.

Le gain de log-loss est réel et vit dans la masse des matchs où l'on ne parie
pas ; les paris tombent là où le modèle est trop confiant. C'est
`edge_too_small_to_prove` sous sa forme la plus pure — cette fois un gain **cinq
fois** le seuil, et toujours une perte.

## D'autres stratégies de mise ?

Non, et c'est fermé par l'algèbre : `E[log(1 + f·X)] ≤ log(1 + f·E[X]) < 0` dès
que `E[X] < 0`, pour toute fraction `f > 0`. Sur les paris réellement
sélectionnés (665 paris, cote 3,0, réussite 32,0 %) :

| règle de mise | capital médian | P(ruine) | P(gain) |
|---|---:|---:|---:|
| plat 1 % | 0,74 | 3,4 % | 23,4 % |
| plat 0,25 % | 0,94 | 0,0 % | 23,9 % |
| proportionnel 1 % | 0,72 | 0,0 % | 18,9 % |
| quart de Kelly | 0,87 | 0,0 % | 23,2 % |
| martingale plafonnée | **0,36** | 1,3 % | 5,5 % |
| **Kelly optimal = 0** | **1,00** | 0,0 % | — |

La mise choisit la **forme** de la perte — sa vitesse, sa variance, le risque de
ruine — jamais son **signe**. Kelly appliqué honnêtement à un avantage négatif
recommande de ne pas parier, et c'est la seule règle qui ne perd pas.

## Portée

Les années d'évaluation avaient déjà servi le jour même : ce balayage est
**exploratoire**, il génère des hypothèses et ne prouve rien. Mais il ne fabrique
aucune hypothèse survivante — ce qui rend la question de la preuve sans objet.

Le budget de preuve reste donc intact là où il l'était : le holdout football
2024-2026 et le holdout UFC postérieur au 13 septembre 2025 **n'ont pas été
ouverts**, parce qu'aucune porte ne s'est ouverte pour les justifier.
