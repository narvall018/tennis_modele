# Football, UFC, WTA : interactions non linéaires — 19 septembre 2026

## Conclusion

**La WTA présente un signal historique positif à étudier, mais aucun des six modèles ne passe le filtre fixé avant les calculs.** Ce n’est donc pas une stratégie rentable validée, ni un motif pour remplacer le modèle de l’application.

Le signal WTA provient de la variante flexible : 110 paris réglés, dix supposés remboursés, rendement normalisé de +19,03 % après décote de 2 % des gains. L’intervalle bootstrap 95 % [−2,69 % ; +38,42 %] inclut une perte. La variante n’était pas admissible pendant la sélection 2021–2022 : seulement 44 paris réglés et borne inférieure négative. La signaler maintenant est une **observation exploratoire après évaluation**, pas la sélection initiale d’une stratégie validée.

## Hypothèse fixée avant le calcul

Les interactions entre performances passées, contexte et probabilités du bookmaker pourraient contenir une information que les corrections linéaires ou les arbres peu profonds précédents captent mal.

Implémentation : régression logistique multinomiale avec probabilités du marché en point de départ, descripteurs linéaires et 128 projections de Fourier aléatoires approximant un noyau gaussien. La [documentation officielle sur l’approximation de noyaux](https://scikit-learn.org/1.9/modules/kernel_approximation.html) décrit cette famille de méthodes. Cette référence explique la méthode, pas sa rentabilité sportive ; notre implémentation et ses gradients sont testés séparément.

Deux pénalités fixées : 0,001 (`kernel_flexible`) et 0,01 (`kernel_strong`). Même graine 20260919 et même largeur de noyau pour les deux. Pas de recherche sur les graines, surfaces, championnats ou seuils après les résultats.

- Football : descripteurs antérieurs déjà audités (forme, buts, tirs, fatigue, Elo, surprises par rapport au marché), plus repos total, expérience minimale et niveau de division.
- UFC : descripteurs antérieurs déjà audités (résultats, frappe, lutte, contrôle, Elo), plus expérience minimale, âge moyen et absence maximale. Pas de nouvelles métadonnées physiques supposées historiques.
- WTA : statistiques de service/retour globales et par surface préparées dans l’étude précédente, surface et volume d’historique. Délai de 28 jours après le début du tournoi maintenu pour l’entrée des statistiques ; registre d’identités antérieur uniquement.

## Protocole chronologique

Trente-deux ajustements annuels, sur six années glissantes, avec embargo de sept jours et pondération temporelle de demi-vie trois ans. Mise à l’échelle apprise sur l’entraînement uniquement. Les prédictions WTA/UFC sont symétriques à l’inversion des adversaires. La probabilité du marché est la référence, pas une information de résultat.

Sélection : 2021–2022 pour football/WTA, 2019–2021 pour UFC. Les choix des trois sports ont tous été enregistrés avant les nouveaux diagnostics d’évaluation. Évaluation : 2023–2025 pour football/WTA, 2022–2024 pour UFC. La réserve UFC 2025+ et toutes les observations 2026 restent hors de cette expérience.

Règle fixe : EV estimée ≥ 3 % après décote de 2 % des gains, cotes 1,30–5,00 ; une seule issue par match. Bankroll séparée de 1 000 € pour chaque variante et période ; mise de 0,25 % de la bankroll du début de journée, arrondie au centime inférieur ; plafond quotidien 2 %. Pas de réinvestissement intrajournalier. L’ordre date/compétition/adversaires est déterministe, pas une reconstruction d’horaires de cotation réels.

## Tous les résultats d’évaluation

| Sport et variante | Paris réglés | Rendement normalisé, décote 2 % | Intervalle bootstrap 95 % | Bankroll finale, départ 1 000 € | Filtre |
| --- | ---: | ---: | --- | ---: | --- |
| Football flexible | 1 456 | −4,93 % | [−11,12 % ; +1,72 %] | 831,43 € | Échec |
| Football forte pénalité | 6 | −75,20 % | [−100,00 % ; −25,55 %] | 988,79 € | Échec |
| UFC flexible | 420 | −0,58 % | [−9,04 % ; +9,39 %] | 992,53 € | Échec |
| UFC forte pénalité | 151 | +1,77 % | [−12,38 % ; +16,78 %] | 1 006,21 € | Échec |
| WTA flexible | 110 | +19,03 % | [−2,69 % ; +38,42 %] | 1 053,16 € | Échec |
| WTA forte pénalité | 0 | Non défini | Non défini | 1 000,00 € | Échec |

Les intervalles utilisent 100 000 rééchantillonnages par blocs circulaires de trois mois, sur les mois de la fenêtre de prédictions, y compris ceux sans pari. La borne unilatérale corrigée pour les six variantes est négative pour toutes ; elle vaut −7,71 % pour la WTA flexible. Cette correction limitée ne couvre pas les nombreuses recherches adaptatives précédentes.

### Détail WTA : ce que signifie le résultat positif

Le rendement normalisé du protocole pondère les retours par la fraction de bankroll engagée, sur les paris réglés. Il diffère du ROI monétaire quand la bankroll évolue. Pour cette variante :

- 302,04 € de mises simulées au total, dont 276,87 € sur les 110 paris réglés.
- Bénéfice simulé de 53,16 €, soit +19,20 % des euros misés sur les paris réglés, ou +17,60 % en incluant les mises ensuite remboursées au dénominateur.
- Neuf abandons et un forfait supposés remboursés : ce traitement ne démontre pas l’application des règles d’un opérateur réel.
- Rendement normalisé annuel : 2023 +0,48 % sur 27 paris ; 2024 +6,06 % sur 28 ; 2025 +34,73 % sur 55.
- Sans 2025, rendement normalisé +3,32 % seulement. L’essentiel du gain provient de 2025.
- Avec une décote des gains portée à 5 %, sur les mêmes paris, rendement normalisé +17,08 %.
- Erreur logarithmique sur tous les matchs terminés : 0,598354 contre 0,599229 pour le marché normalisé. L’écart observé est faible ; pas de significativité démontrée.

Il n’est pas légitime d’assimiler le +19,03 % à un rendement futur attendu. La variante n’atteint ni le minimum de 200 paris réglés en évaluation ni une borne d’incertitude positive, et échouait déjà à l’admission initiale.

### Sélection antérieure : pas de gagnant admis

| Sport et variante | Paris réglés en sélection | Rendement normalisé, décote 2 % | Intervalle 95 % |
| --- | ---: | ---: | --- |
| Football flexible | 879 | +3,24 % | [−2,48 % ; +9,88 %] |
| Football forte pénalité | 0 | Non défini | Non défini |
| UFC flexible | 528 | −2,92 % | [−8,97 % ; +3,22 %] |
| UFC forte pénalité | 195 | +3,53 % | [−5,50 % ; +13,16 %] |
| WTA flexible | 44 | +2,18 % | [−35,68 % ; +24,74 %] |
| WTA forte pénalité | 0 | Non défini | Non défini |

La règle identifiait la variante flexible en football et la forte pénalité en UFC comme les moins mauvaises admissibles en volume, mais aucune ne franchissait l’admission statistique. Aucune variante WTA n’avait le volume minimal de sélection. Les résultats d’évaluation sont donc les diagnostics de modèles non admis.

## Automatisation, données et suite utile

Le modèle utilise une paire ou un triplet de cotes d’exécution, pas un couple de bookmakers de référence obligatoires. Une simulation prospective WTA serait techniquement envisageable avec statistiques, surface et identités correctement collectées, mais elle n’a pas été intégrée ou démarrée ici. Les données de cette expérience sont les archives locales, pas une nouvelle collecte actualisée.

Les 6 028 lignes de la source UFC portent `legacy_unverified` pour les cotes : l’heure et les conditions d’exécution ne sont pas démontrées. Les archives football/WTA utilisent Bet365, sans preuve de reproduction aux prix français. Aucun recours à un VPN ni à des comptes de bookmakers n’est effectué.

La prochaine étape scientifiquement utile, si poursuivie, serait de **figer cette hypothèse WTA dans un suivi sur papier à partir de nouvelles observations datées**, sans modifier ses seuils après chaque résultat. Cela testerait une nouvelle expérience prospective, pas une stratégie déjà validée. Ouvrir de nouveau les mêmes années ou choisir seulement 2025 ne fournirait pas cette confirmation.

## Contrôles et traçabilité

- Sept tests spécifiques réussis : gradients binaires et multinomiaux, symétrie, reproductibilité, normalisation apprise sur le passé, embargo et fenêtre glissante, règlements 1X2/voids/plafonds, exclusion des résultats courants des entrées WTA.
- Suite locale : **402 tests réussis**, trois modules UI ignorés car Streamlit n’est pas installé.
- Audit en lecture seule : empreintes des sources/code/protocole, trente-deux fenêtres d’entraînement, prix et résultats de chaque prédiction, sélection des issues, douze carnets de mises, remboursements, plafonds et soldes vérifiés.
- Aucune modification de l’application, aucun push, aucune mise réelle, aucun service de surveillance installé.

Fichiers : `protocol.json`, `registration.json`, `all_selection_locks.json`, `report.json` et les carnets/prédictions Parquet. Code : `src/backtesting/kernel_matchups.py`. Lanceur : `scripts/run_kernel_matchups.py` (refuse d’écraser une expérience). Audit relançable : `python3 scripts/audit_kernel_matchups.py`.
