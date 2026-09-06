# Tennis betting model

Projet de recherche tennis orienté probabilités, calibration et value betting.
Une rentabilité future n'est jamais garantie : toute stratégie doit être évaluée
hors échantillon, avec des cotes réellement disponibles au moment du pari et des
frais réalistes.

## Mise à jour des données

```bash
python3 scripts/update_tennis_data.py       # ATP principal + cotes
python3 scripts/update_tennis_expansion.py  # WTA + Challenger + qualifications
```

La première commande télécharge d'abord toutes les sources en mémoire, normalise
les noms et dates, rapproche les matchs, puis exécute les contrôles qualité. Les
fichiers locaux ne sont remplacés qu'après validation complète.

La seconde publie les circuits que l'ATP principal laisse de côté. Elle sépare
délibérément deux natures de données :

- **WTA principal** : un second marché coté, avec les mêmes colonnes de
  bookmakers depuis 2007. Aucune recherche de ce dépôt ne l'avait jamais lu.
- **Challenger ATP et qualifications** : des matchs sans marché. Ils ne peuvent
  jamais être pariés ici ; ils existent pour qu'un classement Elo connaisse déjà
  les matchs qu'un joueur a réellement disputés avant d'entrer dans un tableau
  principal. Ces tables ne portent aucune colonne de cote.

Pour mettre aussi à jour les Elo utilisés par l'application :

```bash
python3 scripts/run_update_pipeline.py
```

## Tables publiées

| Fichier | Contenu |
|---|---|
| `data/atp_tennis.csv` | Table ATP 2000–2026 : résultats, statuts, classements, points, scores, cote moyenne et cotes Bet365/Pinnacle/maximum/Betfair quand disponibles. |
| `data/raw/tennis_data/atp_odds_2000_current.csv.gz` | Concaténation non altérée des classeurs ATP annuels officiels Tennis-Data. |
| `data/raw/tennis_mylife/atp_matches_2000_current.csv.gz` | Matchs et statistiques brutes, avec fichier source et horodatage. |
| `data/raw/tennis_mylife/atp_rankings_current.csv` | Dernier classement ATP disponible. |
| `data/processed/atp_matches_enriched.csv.gz` | 108 colonnes : identifiants, profils, contexte, statistiques, cotes rapprochées et score de confiance. |
| `data/processed/atp_players_current.csv` | Dernier profil connu de chaque joueur et classement actuel. |
| `data/quality/atp_data_quality.json` | Rapport vérifiable : fraîcheur, doublons, couverture des cotes, valeurs invalides et qualité du rapprochement. |
| `data/data_manifest.json` | Provenance et classification temporelle des colonnes. |
| `data/wta_tennis.csv` | Table WTA 2006–2026 au format exact de `atp_tennis.csv` : mêmes colonnes de cotes, même orientation déterministe. |
| `data/raw/tennis_data/wta_odds_2007_current.csv.gz` | Concaténation non altérée des classeurs WTA annuels Tennis-Data. |
| `data/raw/tennis_mylife/wta_matches_2000_current.csv.gz` | Matchs WTA bruts 2000–2026 avec fichier source et horodatage. |
| `data/processed/wta_matches_enriched.csv.gz` | 110 colonnes : le pendant WTA de la table enrichie ATP, avec `tour` et `segment`. |
| `data/processed/atp_unpriced_matches.csv.gz` | Challenger et qualifications ATP. Colonnes de cote présentes mais **vides par construction**. |
| `data/quality/tour_expansion_quality.json` | Rapport vérifiable des trois ajouts : volumes, fraîcheur, couverture et qualité du rapprochement des cotes WTA. |

Volumes publiés au 6 septembre 2026 :

| Table | Matchs | Période | Dont cotés |
|---|---:|---|---:|
| ATP principal | 79 705 | 2000 → 2026‑09‑05 | 61 935 |
| WTA principal | 72 006 | 2000 → 2026‑08‑30 | 40 953 |
| Challenger ATP | 121 373 | 2000 → 2026‑09‑01 | 0 (pas de marché) |
| Qualifications ATP | 29 739 | 2007 → 2026‑08‑28 | 0 (pas de marché) |

Sources : les [classeurs officiels Tennis-Data](https://www.tennis-data.co.uk/alldata.php)
pour les résultats, statuts et cotes historiques (ATP et WTA) ;
[TennisMyLife](https://stats.tennismylife.org/tennis-match-database) pour les
identifiants, profils et statistiques de match des quatre circuits.

Les cotes et les résultats n'ont pas la même fraîcheur, et l'écart est visible
dans les rapports qualité : Tennis-Data publie ses classeurs par vagues, donc la
dernière date cotée est régulièrement en retard de quelques jours sur la dernière
date jouée. Aucun prix n'est extrapolé pour combler ce retard.

## Backtest de stratégie rigoureux

```bash
python3 scripts/run_rigorous_backtest.py
```

Le protocole est imbriqué et temporel : modèle/mélange sur 2012–2016, règle de
pari sur 2017–2020, validation et taille des mises sur 2021–2023, puis ouverture
unique du holdout 2024–2026. Pour tester une année `Y`, l'entraînement s'arrête à
`Y-2` et la calibration utilise seulement `Y-1`. Les cotes manquantes ne sont
jamais imputées. Les mises d'une même journée sont calculées sur la bankroll du
début de journée, avec plafond d'exposition, puis réglées ensemble.

Le résultat actuel est volontairement bloquant : le meilleur candidat trouvé a
échoué sur 2024–2026 (116 paris réglés, ROI -21,80% avant décote de cote). Toutes
les stratégies argent réel ont donc été désactivées. Le rapport complet est dans
[`models/rigorous_strategy/BACKTEST_REPORT.md`](models/rigorous_strategy/BACKTEST_REPORT.md).

Le diagnostic de calibration ajouté au rapport explique *pourquoi* : le gain de
log-loss du mélange contre le marché dévigé passe de +0,00171 en développement à
+0,00105 en réglage, +0,00127 en validation, puis **+0,00006 sur le holdout**.
Le pouvoir prédictif supplémentaire disparaît hors échantillon, et le ROI négatif
en est la conséquence attendue, pas un accident de variance. C'est le contraste
exact avec l'étude WTA ci-dessous, où ce gain se maintient.

## Étude WTA — un marché jamais exploré

```bash
python3 scripts/run_wta_backtest.py --freeze-only   # écrit et hash le protocole
python3 scripts/run_wta_backtest.py                 # ouvre le holdout une seule fois
```

Pourquoi un second circuit : toutes les recherches précédentes de ce dépôt ont lu
l'ATP masculin, donc un résultat ATP ne distingue plus un edge réel de
l'accumulation de passages sur les mêmes matchs. La table WTA n'avait jamais été
lue par aucune recherche ici : son holdout est une vraie preuve hors échantillon,
utilisable une seule fois.

Le protocole est écrit et hashé **avant** tout calcul de rendement, dans
[`models/wta_strategy/wta_protocol.json`](models/wta_strategy/wta_protocol.json).
Il a été amendé une fois, holdout encore fermé : le garde-fou du moteur exige
10 000 lignes d'entraînement par fold et refusait 2011 et 2012. Le développement
commence donc en 2013, première année admissible. L'amendement, sa raison et le
protocole remplacé sont conservés dans le fichier.

Découpage : modèle et mélange 2013–2016, règle de pari 2017–2019, validation et
mises 2020–2022, holdout 2023–2026. `scripts/run_wta_backtest.py` refuse de
modifier un protocole dont le holdout a déjà été ouvert.

Résultat : **gate non passée, NO BET.** Sur le holdout, la stratégie figée rend
+1,51 % après décote de 2 % sur 93 paris réglés, avec un IC bootstrap mensuel 90 %
de [−13,35 % ; +17,54 %] : indistinguable de zéro. La validation échoue par
ailleurs sur le critère « au moins deux années positives sur trois ».

Le point qui mérite d'être retenu n'est pas le ROI mais la calibration : le
mélange bat le marché dévigé de +0,00168 de log-loss sur les 9 053 matchs cotés
du holdout, après +0,00184 en validation et +0,00054 en réglage. Il y a donc un
petit gain de prévision réel et stable — il ne se convertit simplement pas en
profit démontrable sous la règle de pari figée. Le détail est dans
[`models/wta_strategy/BACKTEST_REPORT.md`](models/wta_strategy/BACKTEST_REPORT.md).

Les cellules par source de prix du rapport (Bet365, Pinnacle, maximum) affichent
des ROI nettement plus élevés. Ce ne sont **pas** des résultats : l'edge y est
recalculé contre chaque prix, donc chaque source sélectionne une population de
paris différente, et le prix `maximum` est le meilleur prix trouvé chez un
opérateur quelconque, ni exécutable à volume ni tenu dans le temps. Retenir après
coup la source la plus flatteuse annulerait le protocole.

## Classements multi-niveaux — meilleurs, mais sans valeur de pari

```bash
python3 scripts/run_multi_tier_ratings.py   # Elo principal seul vs Elo + Challenger + qualifs
python3 scripts/run_multi_tier_residual.py  # le même, posé par-dessus le prix
```

Les matchs Challenger et de qualification ne portent aucun marché, mais ce sont
les matchs que les qualifiés et les joueurs montants ont réellement joués. Les
ajouter à la passe de notation la fait passer de 79 705 à 230 817 matchs et donne
de l'information neuve à 89,9 % des matchs du circuit principal.

L'identité des joueurs vient des identifiants TennisMyLife, jamais des noms : les
deux tables partagent un espace d'identifiants, alors qu'un rapprochement de noms
abrégés échouerait silencieusement sur exactement les joueurs peu connus que
l'opération vise. Les niveaux inférieurs pèsent moins (0,7 pour les Challengers,
0,6 pour les qualifications), valeurs fixées avant toute mesure.

Le classement devient franchement meilleur — **+0,00469** de log-loss en général,
**+0,02230** quand un joueur a moins de 20 matchs de circuit principal. C'est le
plus gros effet du projet.

Il ne se traduit par aucun avantage : posé sur le prix, son apport propre est de
**−0,00002**, et de **−0,00072** sur le sous-groupe déclaré à l'avance. Les
bookmakers suivent déjà le circuit Challenger. Le module sert donc à produire une
estimation de niveau, pas un signal de pari.

## L'application

```bash
python3 scripts/update_football_data.py                    # données football
python3 scripts/train_football_model.py                    # modèle foot + état des équipes
python3 scripts/train_tennis_model.py                      # modèles ATP et WTA
cd predictor_ufc && python3 scripts_train_descriptor_model.py   # modèle UFC
streamlit run unified_app.py
```

### Les modèles

Chaque sport a un modèle de **descripteurs purs** : la cote n'est jamais une
entrée. C'est délibéré — l'app a besoin d'une opinion à comparer au prix, et un
modèle nourri au prix ne peut que l'approuver. Les études de paris du dépôt, qui
modélisent le résiduel de marché, répondent à une autre question.

La sélection est identique pour les trois sports
(`src/models/selection.py`) : dix familles comparées en walk-forward — un fold
n'est entraîné que sur les périodes strictement antérieures —, le gagnant choisi
sur la seule fenêtre de développement par log-loss, puis mesuré sur des périodes
jamais utilisées pour le classement. Les variantes calibrées isotoniquement
concourent aussi, parce que l'app publie des probabilités et qu'un arbre boosté
brut est trop confiant.

| Sport | Descripteurs |
|---|---|
| Football | Elo (avantage terrain fixé à 60 pts), moyennes glissantes 10 matchs de buts, tirs, tirs cadrés, corners, points par lieu, repos, division |
| UFC | Elo, expérience, taux de victoire, frappes portées/encaissées/précision, takedowns, tentatives de soumission, contrôle au sol, knockdowns, allonge, taille, âge, inactivité, garde |
| Tennis | Elo global/surface/momentum, formes 3-5-10-20, taux par surface, H2H, fatigue, repos, classement, points, spécialisation |

Un meilleur modèle ici ne crée aucun avantage : la barre qui compte est le gain
**conditionnel au prix**, et aucun sport ne la franchit.

Trois sections s'ajoutent au carnet de paris existant :

| Section | Contenu |
|---|---|
| **Prédictions** | **Football** : rencontres à venir avec probabilités, prix et écart modèle-marché, classées par recommandation. **UFC** : cartes programmées lues sur UFCStats avec les probabilités du modèle — aucune clé requise, seul le *prix* en demanderait une. **Tennis** : le seul sport sans calendrier gratuit, la page dit précisément ce qu'il faudrait. |
| **Mises** | Kelly fractionné, plafonné par pari et par jour, avec un plancher d'écart. Sous le plancher la mise est **zéro**, avec le motif affiché. |
| **Performances** | Ce que chaque sport a réellement mesuré, lu directement dans les JSON des études — jamais ressaisi, donc impossible à faire diverger. |

Le modèle football n'utilise pas la cote : l'app a besoin d'une opinion
comparable à un prix, et un modèle nourri au prix ne pourrait que l'approuver.
Sa log-loss sur des saisons jamais vues est de 1,01985 contre 1,00025 pour le
marché — il est moins bon, et l'app le dit.

Les trois sections portent le même bandeau : **aucune stratégie rentable
démontrée, argent réel non autorisé**. L'écart affiché sur un match n'est pas un
avantage ; un écart large signale le plus souvent que c'est le modèle qui se
trompe.

## Football — un sport neuf et la question du timing

```bash
python3 scripts/update_football_data.py
python3 scripts/run_football_audit.py
```

Football-Data publie pour 22 divisions européennes ce que le tennis et l'UFC
n'avaient pas : des cotes **d'ouverture et de clôture** pour le même match, trois
marchés au lieu d'un, et onze pays comme échantillons indépendants. Base publiée :
**191 458 matchs (2000–2026), dont 100 584 avec les deux horodatages Pinnacle.**

Deux hypothèses avancées plus haut dans ce dépôt y sont réfutées par la mesure :

- **Parier à l'ouverture ne paie pas.** La clôture prévoit bien mieux
  (+0,00310 de log-loss), mais le ROI à l'ouverture est inférieur de 0,1 à 0,3
  point sur chaque issue, parce que l'overround d'ouverture est plus élevé
  (1,0301 contre 1,0276).
- **Les divisions inférieures ne sont pas plus molles.** L'overround *monte*
  quand on descend : 1,0260 en élite, 1,0340 au cinquième niveau.

Le mécanisme est le même dans les deux cas : là où le book est le moins informé,
il se protège par le prix, pas par une ligne relâchée.

Le handicap asiatique — marché le moins cher du jeu de données, 2,6 % — ne sauve
rien non plus : douze cellules, toutes négatives, la meilleure à −2,11 %.

Un modèle à descripteurs a ensuite été construit pour de bon (Elo, moyennes
glissantes de buts, **tirs, tirs cadrés**, corners, points par lieu, repos) :

```bash
python3 scripts/run_football_conditional_test.py
```

La gate, déclarée avant le calcul, exige +0,001 de log-loss contre le marché.
Meilleur candidat : **−0,00080**. Échec. Les saisons de réglage, de validation et
le holdout **n'ont jamais été ouverts** et restent disponibles. Détail dans
[RAPPORT_FOOTBALL.md](RAPPORT_FOOTBALL.md).

## Audit de calibration du marché — sans aucun modèle

```bash
python3 scripts/run_market_audit.py
```

Question inverse de toutes les autres : sans rien modéliser, le prix
correspond-il à ce qui arrive ? Le même test avait trouvé le biais décisions du
marché UFC.

Il trouve un **vrai biais favori/outsider, monotone et présent sur les deux
circuits** : les favoris à plus de 90 % sont sous-évalués de +1,65 point, et
parier les outsiders extrêmes rend −44 %. Ce n'est pas du bruit.

Il ne franchit pas la marge pour autant : chez Bet365, **0 cellule rentable sur
64**. Chez Pinnacle une seule ressort (WTA, gros favoris, +0,8 %) — et le
contrôle croisé intégré la rejette, parce que l'ATP donne −0,33 % sur la même
cellule avec deux fois plus de données, que le poolé vaut +0,09 % et que l'effet
disparaît si l'on déplace le seuil de 5 points. Sur ~128 cellules examinées, une
cellule positive est ce que le hasard prévoit.

Le module publie toujours toutes les cellules, jamais la meilleure, inclut un
segment témoin découpé par hachage, et rapporte combien de cellules le hasard
devrait marquer.

## Book fin contre book mou — sans aucun modèle

```bash
python3 scripts/run_sharp_vs_soft.py
```

Plutôt que d'essayer de battre le marché, on prend Pinnacle comme vérité
(overround 2,4 %) et on parie chez le book qui s'en écarte. Aucun modèle sportif,
aucun paramètre ajusté ; les cinq seuils testés sont tous publiés.

Un bras de falsification tourne en parallèle — même règle, signal inversé. Il
perd **10 points de plus** (−14,6 % contre −4,9 % chez Bet365 ATP) : le signal de
Pinnacle est donc réel. Il ne suffit pas pour autant. Bet365 rend de −3,5 % à
−14 % selon le seuil, sur les deux circuits, et aucune cellule n'a d'intervalle à
90 % positif. La marge du book mou (6,7 % d'overround) dépasse l'écart capturé.

L'audit de simultanéité intégré disqualifie par ailleurs la colonne
`market_maximum` : **43 % de ses lignes impliquent un arbitrage** (overround
inférieur à 1), contre 0,03 % chez Pinnacle. Ce n'est pas une paire de prix
simultanée mais le meilleur prix vu à un instant quelconque — elle n'est pas
pariable, et tout résultat positif l'impliquant doit être écarté.

## Le signal est-il assez grand pour rapporter ?

```bash
python3 scripts/run_edge_diagnostics.py wta
python3 scripts/run_edge_diagnostics.py atp
```

Un gain de log-loss de 0,001 est facile à mesurer et difficile à interpréter. Ce
diagnostic applique une règle unique, fixée d'avance et volontairement la plus
permissive possible — *parier tout côté à espérance positive* — puis regarde si
elle franchit la marge d'un bookmaker. Il ne sélectionne rien : les seize
cellules sont publiées ensemble, jamais la meilleure.

Verdict : **aucune des seize configurations, sur les deux circuits, n'a un
intervalle de confiance à 90 % excluant zéro par le haut.** Le meilleur cas
honnête (WTA, Pinnacle, cote < 6) rend +0,74 % avec un IC 90 % de
[−0,7 % ; +2,1 %], t = 0,80, et demanderait 35 250 paris — 34 ans — pour être
démontré. Deux points de friction d'exécution supplémentaires l'annulent.

L'analyse complète est dans [RAPPORT_RENTABILITE.md](RAPPORT_RENTABILITE.md).
L'hypothèse qui en sort est gelée, en papier uniquement, par
`python3 scripts/freeze_prospective_candidate.py`.

## Phase 4 — serve/return, surfaces et deep learning

```bash
python3 scripts/run_tennis_phase4.py
```

Le protocole a été figé avant les nouveaux rendements. Il compare en walk-forward
2017–2026 un marché recalibré par surface, un résiduel structurel, le même modèle
avec historiques serve/return, puis une version à interactions de surface et
vitesse de court. Un petit MLP symétrique est testé séparément, sans droit de
sélectionner une stratégie.

Résultat : le résiduel structurel reste le meilleur (log-loss 0,58431 contre
0,58528 pour le marché), mais l'amélioration n'atteint pas la gate figée. Les
variables serve/return font légèrement moins bien (0,58436), les interactions de
surface 0,58443 et le MLP 0,58671. La règle économique fixe donne −0,43 % après
décote de cote sur 563 paris réglés, IC bootstrap 99 % [−11,58 % ; +11,64 %].

Le statut est `PHASE4_REJECTED_NO_BET`. Toutes les données jusqu'au 30 août 2026
sont désormais du développement; une nouvelle preuve doit être prospective à
partir du 3 septembre 2026. Rapport :
[`models/rigorous_strategy/PHASE4_REPORT.md`](models/rigorous_strategy/PHASE4_REPORT.md).

La collecte prospective est append-only et refuse les observations postérieures
au début du match :

```bash
export TENNIS_ODDS_API_KEY="..."
python3 scripts/collect_tennis_odds.py
```

La clé doit être fournie explicitement; le collecteur de recherche ne réutilise
pas le secret encodé de l'interface Streamlit.

## Règles de rigueur

- L'orientation `Player_1` / `Player_2` est déterministe et indépendante du résultat.
- Les années courantes sont remplacées par un snapshot complet afin d'intégrer les corrections de la source, pas seulement les nouvelles dates.
- Les cotes sont des cotes match-winner décimales, généralement les dernières avant le début du match. Les paires moyennes, Bet365, Pinnacle et maximales restent séparées ; aucun côté d'une paire ne vient d'un autre opérateur.
- Les colonnes `postmatch_*`, `score`, `minutes`, vainqueur et résultat sont indisponibles avant le match. Elles ne doivent jamais entrer directement dans un modèle pré-match.
- Les statistiques post-match servent uniquement à construire des agrégats retardés : l'état d'un joueur doit être lu avant d'être mis à jour par le match courant.
- Les abandons et walkovers sont conservés et étiquetés. Leur règlement dépend du bookmaker et doit être traité explicitement dans le backtest.
- La table enrichie contient un `odds_match_confidence` et le décalage de date entre les deux sources ; les rapprochements faibles doivent pouvoir être exclus par seuil.

- Les circuits restent des populations séparées : ATP principal, WTA principal,
  Challenger et qualifications ont des tables distinctes, un préfixe d'identifiant
  distinct et une colonne `tour`/`segment`. Leurs distributions et leur qualité de
  marché ne sont pas interchangeables et ne doivent jamais être empilées en un
  seul échantillon d'entraînement sans raison explicite.
- Les identifiants de joueurs ATP et WTA proviennent de deux espaces de numérotation
  différents ; ils ne doivent jamais être fusionnés sans re-résolution.

## Budget de preuve

Un jeu de données ne sert de preuve qu'une fois. Ce qui a déjà été dépensé :

| Population | État | Conséquence |
|---|---|---|
| ATP principal | brûlé (étude imbriquée + phase 4, toutes deux `NO BET`) | plus aucune conclusion économique nouvelle n'y est admissible |
| WTA principal | holdout 2023–2026 ouvert une fois le 6 septembre 2026, `NO BET` | brûlé à son tour |
| Challenger et qualifications | aucun marché dans les données | ne peuvent porter aucune preuve économique, seulement des classements |
| Période à partir du 7 septembre 2026 | vierge | seule preuve prospective encore disponible |

Toute nouvelle hypothèse doit dire quelles données jamais explorées elle consomme.
Si la réponse est « les mêmes que la dernière fois », la démarche honnête est la
collecte prospective, pas un passage supplémentaire.
