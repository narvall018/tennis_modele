# Football — un sport neuf, trois marchés, et la question du timing enfin testable

**Verdict : aucune stratégie rentable. Mais deux hypothèses que j'avais avancées
sont maintenant réfutées par des mesures directes, plus par du raisonnement.**

Reproduire :

```bash
python3 scripts/update_football_data.py
python3 scripts/run_football_audit.py
```

## Pourquoi le football méritait le détour

Football-Data publie, pour 22 divisions européennes, trois choses que ni le
tennis ni l'UFC n'offraient :

| | Tennis / UFC | Football |
|---|---|---|
| Cotes d'ouverture **et** de clôture | non | **oui**, 100 584 matchs |
| Nombre de marchés | 1 (vainqueur) | 3 (1X2, plus/moins 2,5, handicap asiatique) |
| Marge la plus basse disponible | 2,4 % (Pinnacle tennis) | **2,6 %** (handicap asiatique) |
| Échantillons indépendants | 2 circuits | 11 pays, 5 niveaux de division |

Base publiée : **191 458 matchs, 2000–2026, 22 divisions**, 0 doublon, taux de
victoire à domicile 44,5 % (plausible).

## 1. « Parier à l'ouverture » — réfuté

C'est la piste « vitesse » que je vous avais donnée comme la plus prometteuse.
Elle est fausse, et on peut maintenant le montrer.

L'information arrive bien pendant la fenêtre : la cote de clôture prévoit mieux
que celle d'ouverture, de **+0,00310** de log-loss en 1X2 et **+0,00236** en
plus/moins 2,5. Le marché apprend quelque chose entre les deux.

Mais parier à l'ouverture rapporte **moins** que parier à la clôture, sur toutes
les issues :

| Issue (1X2, Pinnacle) | ROI ouverture | ROI clôture | Écart |
|---|---:|---:|---:|
| Domicile | −4,73 % | −4,44 % | **−0,29 pt** |
| Nul | −4,94 % | −4,84 % | −0,10 pt |
| Extérieur | −6,66 % | −6,39 % | −0,27 pt |

La raison est simple et mesurable : l'overround d'ouverture est de **1,0301**
contre **1,0276** à la clôture. Le book se protège de son incertitude en prenant
plus de marge à l'ouverture, et ce surcoût dépasse la valeur de la dérive.

Autrement dit : être rapide ne suffit pas, il faut être rapide **et** avoir
raison. Le prix d'ouverture n'est pas un prix mou, c'est un prix cher.

## 2. « Les divisions inférieures sont plus molles » — réfuté aussi

| Division | n | Overround Pinnacle | ROI gros favoris |
|---|---:|---:|---:|
| Rang 1 (élite) | 4 204 | 1,0260 | +0,10 % |
| Rang 2 | 2 429 | 1,0282 | −2,26 % |
| Rang 3 | 912 | 1,0327 | −1,58 % |
| Rang 4 | 567 | 1,0326 | +1,84 % |
| Rang 5 | 912 | 1,0340 | −1,90 % |

La marge **augmente** quand on descend (1,0260 → 1,0340). Le book compense son
incertitude par le prix, exactement comme à l'ouverture. Aucun IC90 n'exclut
zéro. L'idée que les petites divisions sont exploitables parce que moins suivies
ne résiste pas : elles sont moins suivies *et* plus chères.

## 3. Le même biais favori/outsider qu'au tennis

| Bande (1X2, Pinnacle clôture) | p marché | Réel | Écart | ROI |
|---|---:|---:|---:|---:|
| 0,00–0,15 | 0,1069 | 0,0962 | −1,08 pt | **−16,84 %** |
| 0,15–0,30 | 0,2481 | 0,2445 | −0,35 pt | −5,89 % |
| 0,30–0,45 | 0,3603 | 0,3626 | +0,22 pt | −3,48 % |
| 0,45–0,60 | 0,5140 | 0,5167 | +0,27 pt | −3,32 % |
| 0,60–0,75 | 0,6618 | 0,6807 | +1,90 pt | −0,95 % |
| 0,75–1,00 | 0,8098 | 0,8376 | **+2,78 pt** | **−0,07 %** |

Exactement la forme observée au tennis : favoris sous-évalués, outsiders
ruineux. La meilleure cellule est à l'équilibre (−0,07 %, IC90
[−1,2 % ; +1,0 %]) — jamais au-dessus.

## 4. Le handicap asiatique, marché le moins cher, ne sauve rien

C'était le meilleur espoir : 2,6 % de marge, et un règlement à remboursements et
demi-mises qui rend les lignes quart particulièrement serrées.

| Book / moment | Côté | ROI | IC90 |
|---|---|---:|---|
| Pinnacle / clôture | extérieur | **−2,11 %** | [−2,8 % ; −1,4 %] |
| Pinnacle / clôture | domicile | −4,04 % | [−4,7 % ; −3,3 %] |
| Pinnacle / ouverture | extérieur | −2,67 % | [−3,4 % ; −2,0 %] |
| Bet365 / clôture | extérieur | −3,15 % | [−3,8 % ; −2,5 %] |
| Moyenne / clôture | domicile | −5,55 % | [−6,2 % ; −4,9 %] |

Douze cellules, toutes négatives. Le côté extérieur fait systématiquement mieux
que le domicile (−2,1 % contre −4,0 %) : la ligne sous-corrige légèrement
l'avantage du terrain. C'est un vrai biais de plus — et il reste sous la marge.

## 5. Contrôle croisé

Sur les 11 pays, **0 issue rentable** en 1X2 comme en plus/moins 2,5. Aucune
cellule n'a survécu au contrôle, parce qu'aucune n'était positive au départ.

## 6. Un vrai modèle à descripteurs — construit, testé, arrêté à la gate

```bash
python3 scripts/run_football_conditional_test.py
```

Les audits ci-dessus ne modélisaient rien. Un modèle à descripteurs a donc été
construit pour de bon : Elo avec avantage du terrain fixé d'avance (60 points,
déduit du taux de victoire à domicile publié, jamais ajusté), moyennes glissantes
sur 10 matchs de buts, **tirs, tirs cadrés**, corners et points, séparément à
domicile et à l'extérieur, plus les jours de repos et le niveau de division.
Les tirs cadrés sont le descripteur qui manquait au tennis : ils mesurent la
qualité sous-jacente là où les buts seuls sont trop rares.

Couverture : 127 159 matchs avec les statistiques de tir.

### La gate, déclarée avant le calcul

Les descripteurs doivent améliorer le marché d'au moins **0,001** de log-loss.
Ce seuil n'est pas arbitraire : `models/wta_strategy` a montré qu'un gain de cet
ordre vaut environ +0,74 % de ROI, indémontrable et annulé par 2 points de
friction. En dessous, rien ne peut payer.

### Le résultat

Saisons de développement 2015–2019, 37 552 matchs, walk-forward par saison :

| Candidat | n | Log-loss | Marché | Gain |
|---|---:|---:|---:|---:|
| logistique + marché | 30 118 | 0,99832 | 0,99752 | **−0,00080** |
| gradient boosting + marché | 30 118 | 1,00651 | 0,99752 | −0,00899 |
| logistique seule | 30 118 | 1,01574 | 0,99752 | −0,01822 |
| gradient boosting seul | 30 118 | 1,02239 | 0,99752 | −0,02487 |

**Gate échouée.** Le meilleur candidat n'apporte rien au prix — il le dégrade
légèrement. Les tirs cadrés, la forme, le repos, l'Elo : le marché contient déjà
tout cela.

### Ce qui a été préservé

Le test n'a lu que les saisons 2015–2019. Les saisons de **réglage (2020–2021),
de validation (2022–2023) et le holdout (2024–2026) n'ont jamais été ouvertes**
et restent disponibles pour une hypothèse réellement différente. C'est la
première fois dans ce dépôt qu'une piste est fermée sans dépenser son holdout.

## 7. L'échange Betfair — le seul lieu sans marge de bookmaker

Tous les échecs de ce projet ont la même cause : le biais est réel, la marge le
mange. Il existe un endroit où cette cause disparaît presque — un échange
peer-to-peer, où il n'y a pas de marge intégrée mais une **commission prélevée
sur les gains nets uniquement**.

| Venue (1X2, clôture) | Overround | Arbitrage implicite |
|---|---:|---:|
| Bet365 | 1,0617 | 0,01 % |
| Pinnacle | 1,0277 | 0,00 % |
| **Échange Betfair** | **1,0076** | 0,07 % |

3,6 fois moins cher que Pinnacle, et un taux d'arbitrage quasi nul qui confirme
un marché réellement simultané — contrairement à la colonne `maximum` et ses 32 %.

La commission portant sur les gains, elle coûte d'autant moins que la cote est
courte : sur un favori à 1,20, une commission de 5 % coûte 5 % × 0,20 = **1 % de
la mise**. C'est précisément là que vit le biais favori/outsider.

### Le résultat — le plus proche jamais obtenu, et toujours insuffisant

Favoris regroupés, échange de clôture :

| Seuil | n | ROI commission 2 % | IC 90 % | ROI commission 5 % |
|---|---:|---:|---|---:|
| p ≥ 0,55 | 4 572 | +0,34 % | [−1,2 % ; +1,8 %] | −0,67 % |
| p ≥ 0,60 | 3 092 | +0,30 % | [−1,4 % ; +2,0 %] | −0,59 % |
| p ≥ 0,70 | 1 314 | +0,69 % | [−1,2 % ; +2,7 %] | +0,02 % |
| p ≥ 0,75 | 761 | +0,35 % | [−2,1 % ; +2,9 %] | −0,21 % |

**Aucun intervalle n'exclut zéro**, à aucun seuil. Et 2 % n'est pas le taux
courant : c'est un tarif réservé aux très gros volumes. Au **taux standard de
5 %, tout redevient négatif ou nul**.

Le contrôle par pays donne le même verdict que la cellule WTA : l'Italie (+6,3 %)
et l'Espagne (+6,9 %) ressortent avec un intervalle positif, mais l'Angleterre
(−3,4 %), l'Écosse (−4,1 %) et les Pays-Bas (−3,6 %) sont aussi nettement
négatives. Deux cellules positives sur dix, à 90 % — c'est ce que le hasard
prévoit.

### Comptabilité honnête du budget de preuve

Les cotes d'échange n'existent que de **juillet 2024 à septembre 2026** — soit
exactement la fenêtre de holdout déclarée fermée en §6. Il était impossible de
tester l'échange sans la regarder.

Conséquence, énoncée clairement : **le holdout football n'est plus vierge pour
toute question portant sur l'échange ou sur les bandes de favoris.** Il reste
inutilisé pour le modèle à descripteurs lui-même, qui a été rejeté dès le
développement et n'a jamais été ajusté sur ces saisons. Aucun paramètre n'a été
choisi sur 2024–2026 ; seuls des rendements y ont été lus.

## Ce que le football ajoute vraiment

Le résultat économique est le même que partout ailleurs. Ce qui change, c'est la
qualité de la réfutation : deux idées que je vous avais présentées comme les
seules restantes — parier tôt, et parier les petites divisions — sont maintenant
mesurées et fausses, pour la même raison mécanique dans les deux cas. **Là où le
book est le moins informé, il n'est pas plus mou : il est plus cher.** C'est
ainsi qu'un opérateur se protège, et c'est ce qui ferme ces deux portes.
