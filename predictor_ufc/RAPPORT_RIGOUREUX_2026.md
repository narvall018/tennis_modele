# Rapport UFC rigoureux — mis à jour le 2 septembre 2026

## Verdict

**Aucune stratégie rentable n'est démontrée. Verdict opérationnel : NO BET.**

Le meilleur système figé a produit +2,98 unités sur 380 paris de validation, soit
un ROI de +0,78 %. Son intervalle bootstrap 95 % par événement est
**[-11,30 % ; +14,05 %]** : zéro est largement inclus. Le léger profit observé ne
permet donc pas de distinguer un edge réel du hasard.

Une seconde phase, figée le 2 septembre, a ensuite comparé des approches
statistiques, non linéaires et neuronales. Elle confirme le rejet : le système
retenu perd **−0,60 %** sur 313 paris de confirmation, avec un intervalle ajusté
à 97,5 % de **[-12,84 % ; +11,90 %]**.

Une troisième hypothèse a enfin ajouté les classements UFC connus avant le combat.
Elle échoue elle aussi : le pouvoir prédictif baisse et la règle fixe perd
**−0,51 %** sur 1 297 paris de développement, IC99 % **[-7,79 % ; +6,65 %]**.

## Base reconstruite

| Contrôle | Résultat |
|---|---:|
| Combats | 8 832 |
| Combats décisifs | 8 678 |
| Événements | 787 |
| Dernier événement terminé | 29/08/2026 |
| Profils | 4 589 |
| Couverture frappes / takedowns | 100 % |
| Couverture contrôle au sol | 97,96 % |
| Snapshots de cotes appariés | 105 702 |
| Combats avec au moins une cote | 6 553 |
| Lignes récentes horodatées retenues | 306 |
| Combats avec au moins un rang pré-combat | 2 008 |
| Combats avec deux rangs pré-combat | 1 254 |
| Trajectoires avec ≥2 observations distinctes | 190 |
| Trajectoires complètes J−14 → J−1 | 12 |
| Combats avec props méthode | 5 329, tous non horodatés |

Les 38 combats absents de l'extraction secondaire après le 8 août ont été lus
directement sur UFCStats pour les cartes des 15, 22 et 29 août 2026. Deux profils
récents manquants ont également été complétés.

### Qualité des cotes

- 2010–2024 : 6 040 lignes historiques sans bookmaker ni timestamp pré-combat
  vérifiable. Elles sont explicitement marquées `legacy_unverified` et ne servent
  qu'à la recherche/validation imparfaite.
- Période récente : priorité fixe à Pinnacle, puis BetOnline si Pinnacle manque.
- Cutoff figé : dernière observation disponible au plus tard J−1, âgée de 14 jours
  maximum. Aucun choix rétrospectif de la meilleure cote.
- Les lignes postérieures au combat ou à temporalité inconnue sont exclues.
- Les classements sont joints au dernier snapshot strictement antérieur au combat,
  avec 14 jours d'ancienneté maximum. L'audit compte zéro snapshot du jour/futur.
- Les mouvements de cote emploient le même bookmaker, des horizons J−14/J−7/J−3/J−1
  et au moins deux observations distinctes. L'échantillon est trop petit pour
  entraîner honnêtement un modèle de mouvement.
- Les props de méthode sont préservées, mais les 5 329 anciennes lignes n'ont pas
  d'horodatage vérifiable et sont exclues des conclusions de rentabilité.

## Prévention des fuites

- Toutes les variables d'un combat utilisent uniquement l'état des combattants
  avant la date de l'événement.
- Les combats d'une même carte sont calculés avec le même snapshot, puis leurs
  résultats sont intégrés ensemble.
- Les statistiques roulantes L5, Elo, expérience, forme et délai depuis le dernier
  combat sont décalés avant la cible.
- L'orientation des combattants est permutée par hash déterministe pour empêcher le
  modèle d'exploiter artificiellement la position rouge/bleue.
- Les mises d'un même événement sont simultanées : aucun gain d'un combat n'est
  réinvesti dans le combat suivant de la même carte.

## Découpage figé avant calcul des rendements

| Étape | Période | Usage |
|---|---|---|
| Choix du modèle | 2015–2018 | Comparer six régressions logistiques OOS |
| Choix de la règle | 2019–2021 | Choisir le seuil d'edge |
| Validation | 2022–2024 | Accepter ou rejeter une seule fois |
| Holdout final | 13/09/2025–29/08/2026 | Cotes horodatées, ouvert seulement si validation réussie |

Chaque prédiction 2015–2024 est issue d'un modèle entraîné seulement sur les années
antérieures. Le modèle retenu est une régression logistique régularisée combinant
probabilité de marché, Elo, expérience, forme L5, striking/grappling, layoff, âge,
taille, allonge et stance (`C=0,01`). Sur 2015–2018, son log-loss est 0,61568 contre
0,61693 pour le marché brut : amélioration faible, pas une preuve économique.

## Sélection de la règle, 2019–2021

| Edge minimal | Paris | ROI | IC95 % clusterisé |
|---:|---:|---:|---:|
| 2 % | 686 | +1,99 % | [-6,98 % ; +11,09 %] |
| 3 % | 616 | +2,86 % | [-7,13 % ; +12,99 %] |
| **5 %** | **476** | **+5,58 %** | **[-5,73 % ; +17,27 %]** |
| 8 % | 327 | +6,31 % | [-7,80 % ; +20,46 %] |
| 10 % | 246 | +7,67 % | [-7,74 % ; +23,28 %] |

Le seuil 5 % a été sélectionné parce qu'il maximisait la borne basse bootstrap,
avec au moins 80 paris. La mise pré-engagée était Kelly 1/8, plafonnée à 0,5 % par
pari et 3 % d'exposition par événement.

## Validation indépendante, 2022–2024

| Année | Paris | Profit unités | ROI |
|---:|---:|---:|---:|
| 2022 | 154 | +9,43 | +6,12 % |
| 2023 | 118 | −5,62 | −4,77 % |
| 2024 | 108 | −0,82 | −0,76 % |
| **Total** | **380** | **+2,98** | **+0,78 %** |

Avec une bankroll simulée de 1 000 unités, le Kelly 1/8 aurait terminé à 1 048,82
unités, avec un drawdown maximal de 8,40 %. Ce chiffre ne sauve pas le système : la
borne basse du ROI est −11,30 %, donc le critère statistique pré-engagé échoue.

## Holdout final et décision

Le holdout récent n'a pas été calculé. Aucun fichier de prédictions ou de paris du
holdout n'a été créé. Le verrou porte le statut `REJECTED_NO_BET`, et l'application
désactive les recommandations/mises.

La bonne stratégie de mise aujourd'hui est donc **0 % de bankroll**. La collecte
horodatée peut continuer; une future hypothèse devra être figée avant d'ouvrir ce
holdout ou, mieux, être validée prospectivement sur de nouvelles cartes.

## Phase 2 — autres approches

La phase 2 traite explicitement 2022–2024 comme une confirmation interne et non
comme un holdout vierge, puisque les résultats agrégés de la phase 1 avaient déjà
été observés. Le holdout économique 2025–2026 est, lui, toujours fermé.

### Comparaison prédictive 2015–2018

Le choix du modèle a été fait uniquement sur la log-loss walk-forward, avant de
calculer sa règle de pari.

| Modèle | Log-loss | Écart au marché |
|---|---:|---:|
| **Logit structurel marché + Elo/stats** | **0,61558** | **+0,00135** |
| Elastic-net + interactions | 0,61573 | +0,00120 |
| XGBoost peu profond | 0,62377 | −0,00684 |
| Gradient boosting régularisé | 0,62785 | −0,01092 |
| Petit réseau neuronal MLP | 0,63443 | −0,01751 |
| Marché brut | 0,61693 | — |

Le réseau neuronal est le moins bon challenger. Avec cet effectif et ce bruit, sa
capacité supplémentaire dégrade la généralisation au lieu de créer un edge.

### Règle choisie sur 2019–2021

Le logit structurel a retenu un edge minimal de 3 %, des cotes entre 1,25 et 4,00,
et une mise Kelly 1/10 plafonnée à 0,25 % par pari et 2 % par événement. Sur la
fenêtre de sélection : 375 paris, ROI +3,93 %, mais IC95 %
**[-6,16 % ; +14,43 %]**. La preuve était déjà faible.

### Confirmation interne 2022–2024

| Année | Paris | Profit unités | ROI |
|---:|---:|---:|---:|
| 2022 | 130 | −11,58 | −8,91 % |
| 2023 | 90 | +13,80 | +15,33 % |
| 2024 | 93 | −4,10 | −4,41 % |
| **Total** | **313** | **−1,89** | **−0,60 %** |

Le modèle améliore très légèrement la log-loss du marché (0,59354 contre 0,59423),
mais cela ne se transforme pas en rentabilité. Une seule année sur trois est
positive; la bankroll passe de 1 000 à 993,28 unités et la borne basse bootstrap
ajustée vaut −12,84 %. Le verrou phase 2 porte donc le statut
`CHALLENGER_REJECTED_NO_BET`.

## Phase 3 — classements, mouvements et données pré-combat

Le protocole `phase3_protocol.json` a été écrit avant de calculer les rendements.
Il ne compare que deux modèles logistiques identiques : le logit structurel de la
phase 2, puis ce même logit enrichi de quatre variables de rang pré-combat. Aucun
hyperparamètre, seuil ou famille de modèle n'est recherché après observation.

La fenêtre 2015–2024 est explicitement qualifiée de **développement déjà exposé**,
et non de nouvelle validation indépendante. Chaque année est prédite par un modèle
entraîné uniquement sur les années antérieures.

| Mesure 2015–2024 | Structurel | Structurel + rangs |
|---|---:|---:|
| Combats OOS | 4 397 | 4 397 |
| Log-loss | **0,61211** | 0,61290 |
| Log-loss marché | 0,61285 | 0,61285 |
| AUC | **0,72313** | 0,72213 |

Le challenger ne bat le modèle de base que 5 années sur 10 et dégrade la log-loss
globale de 0,00078. Avec la règle figée à 3 % d'edge et des cotes 1,25–4,00 :

| Diagnostic économique de développement | Résultat |
|---|---:|
| Paris | 1 297 |
| Profit à mise plate | −6,59 unités |
| ROI | −0,51 % |
| Années positives | 6 / 10 |
| IC bootstrap 99 % par événement | [−7,79 % ; +6,65 %] |
| Drawdown Kelly 1/10 plafonné | 7,87 % |

Plusieurs critères essentiels échouent : le challenger ne bat ni le modèle
structurel ni le marché, son ROI est négatif et la borne basse reste négative. Le
verrou porte `PHASE3_REJECTED_NO_BET` et le rapport du holdout indique
`NOT_OPENED_PHASE3_DEVELOPMENT_GATE_FAILED`.

### Ce qui reste réellement testable

Les trajectoires de cote et les informations de pesée/short notice peuvent avoir
une valeur, mais leur historique propre est insuffisant. Le collecteur
`collect-odds` enregistre désormais les snapshots de façon append-only. Les pesées,
changements de camp et blessures doivent comporter une URL source et un timestamp
antérieur au combat selon `prospective_protocol.json`. La première cible est la
log-loss/CLV prospectives; aucun pari réel n'est autorisé avant un échantillon et
une gate pré-enregistrés suffisants.
