# La marge en jeux — le dernier gisement inexploité, et ce qu'il ne donne pas

Question posée : trouver le meilleur modèle et la meilleure stratégie rentable,
en créant au besoin de nouveaux descripteurs pour le tennis, l'UFC et le football.

Réponse courte : **les descripteurs sont franchement meilleurs, et il n'y a
toujours aucune stratégie rentable.** Le détail de ce qui a été mesuré, et de ce
qui ne pouvait pas l'être, suit.

## Ce qui n'avait jamais été lu

La colonne `Score` des classeurs Tennis-Data n'est parsée nulle part dans ce
dépôt : aucun module de `src/features/` ne la touche. Toutes les variables
existantes — Elo, formes 3/5/10/20, taux par surface, H2H, fatigue — reposent sur
le seul bit « a gagné / a perdu ». Un 6‑0 6‑0 et un 7‑6 6‑7 7‑6 y sont la même
ligne.

Un match porte pourtant une vingtaine de jeux. La marge en jeux est nettement
moins bruitée que l'issue binaire, et elle sépare deux joueurs que le bilan
victoires/défaites confond : celui qui gagne large et celui qui gagne de
justesse. C'est le seul gisement d'information réellement inexploité des tables
locales.

`src/features/score_features.py` en tire dix-sept descripteurs : Elo à
multiplicateur de marge, ratio de jeux glissant (global, par surface, de
carrière), écart entre la marge réalisée et celle qu'un Elo attendait, taux de
tie-breaks, de sets décisifs et de renversements, parts de victoires et de
défaites écrasantes, charge de travail comptée **en jeux** et non en matchs,
fragilité physique lue dans les abandons récents, et un signal d'adversaires
communs pour comparer deux joueurs qui ne se sont jamais rencontrés.

## Protocole, gelé avant la moindre mesure

`models/margin_study/margin_protocol.json`, hashé avant tout calcul de rendement.
Découpage walk-forward repris du dépôt : entraînement jusqu'à `Y‑2`, calibration
isotone sur `Y‑1`, test sur `Y`. Deux gates écrites d'avance :

| Gate | Définition | Seuil |
|---|---|---|
| **A — prévision** | log-loss(base) − log-loss(base+marge) | +0,002 |
| **B — conditionnel au prix** | log-loss(marché) − log-loss(marché+base+marge) | +0,001 |

Le seuil de la gate B est repris tel quel du test conditionnel football déjà
appliqué ici. La famille de modèle qui porte la décision est choisie sur le bras
**sans** marge, donc indépendamment de l'effet mesuré.

Point d'intégrité déclaré dans le protocole : ATP et WTA sont **brûlés** pour
toute conclusion économique (section « Budget de preuve » du README). Cette étude
ne produit donc aucune conclusion économique et n'autorise aucune mise.

## Gate A — franchie, et plus largement que le plus gros effet du dépôt

Log-loss poolée sur toutes les années de test, aucun prix montré au modèle :

| Circuit | Famille | base | base+marge | **marge seule** | gain | IC 90 % |
|---|---|---:|---:|---:|---:|---|
| ATP | gradient boosting | 0,60873 | 0,60606 | **0,60652** | +0,00267 | [−0,00019 ; +0,00544] |
| ATP | logistique | 0,61045 | 0,60378 | **0,60732** | +0,00667 | [+0,00438 ; +0,00902] |
| WTA | gradient boosting | 0,62847 | 0,61913 | **0,62142** | +0,00934 | [+0,00600 ; +0,01271] |
| WTA | logistique | 0,62674 | 0,62021 | **0,62140** | +0,00653 | [+0,00345 ; +0,00973] |

Le résultat qui mérite d'être retenu est la colonne en gras : **dans les quatre
cellules, les dix-sept descripteurs de marge employés seuls battent les trente et
un descripteurs existants.** Le score jeu par jeu contient davantage sur le niveau
d'un joueur que l'ensemble Elo + formes + surfaces + H2H + classement construit
jusqu'ici.

Pour situer l'ordre de grandeur : les classements multi-niveaux, que le README
appelle « le plus gros effet du projet », valaient +0,00469. Trois des quatre
cellules ici font mieux.

## Gate B — échouée, sur les seize cellules

Une fois le prix dévigé donné au modèle, la marge n'ajoute rien :

| Circuit | Source | Famille | marché seul | marché+base | marché+base+marge | gain | IC 90 % |
|---|---|---|---:|---:|---:|---:|---|
| ATP | moyenne | logistique | 0,59428 | 0,59595 | 0,59473 | −0,00044 | [−0,00196 ; +0,00135] |
| ATP | moyenne | gradient boosting | 0,59393 | 0,59807 | 0,59659 | −0,00266 | [−0,00544 ; +0,00006] |
| ATP | Pinnacle | logistique | 0,58081 | 0,58037 | 0,58098 | −0,00016 | [−0,00169 ; +0,00135] |
| ATP | Pinnacle | gradient boosting | 0,58191 | 0,58329 | 0,58201 | −0,00011 | [−0,00257 ; +0,00230] |
| ATP | Bet365 | logistique | 0,58030 | 0,58263 | 0,58251 | −0,00221 | [−0,00386 ; −0,00067] |
| ATP | Bet365 | gradient boosting | 0,58030 | 0,58448 | 0,58336 | −0,00306 | [−0,00567 ; −0,00053] |
| ATP | maximum | logistique | 0,59409 | 0,59470 | 0,59426 | −0,00018 | [−0,00179 ; +0,00168] |
| ATP | maximum | gradient boosting | 0,59422 | 0,59466 | 0,59624 | −0,00201 | [−0,00480 ; +0,00033] |
| WTA | moyenne | logistique | 0,59797 | 0,59878 | 0,59976 | −0,00180 | [−0,00500 ; +0,00129] |
| WTA | moyenne | gradient boosting | 0,59743 | 0,60019 | 0,59976 | −0,00233 | [−0,00571 ; +0,00117] |
| WTA | Pinnacle | logistique | 0,59686 | 0,59880 | 0,59938 | −0,00252 | [−0,00461 ; −0,00052] |
| WTA | Pinnacle | gradient boosting | 0,59689 | 0,60545 | 0,60616 | −0,00927 | [−0,01272 ; −0,00593] |
| WTA | Bet365 | logistique | 0,59402 | 0,59666 | 0,59701 | −0,00298 | [−0,00544 ; −0,00047] |
| WTA | Bet365 | gradient boosting | 0,59676 | 0,59975 | 0,59584 | **+0,00092** | [−0,00177 ; +0,00374] |
| WTA | maximum | logistique | 0,59511 | 0,59837 | 0,59857 | −0,00345 | [−0,00577 ; −0,00141] |
| WTA | maximum | gradient boosting | 0,59884 | 0,60327 | 0,60022 | −0,00139 | [−0,00424 ; +0,00145] |

**Quinze cellules sur seize sont négatives.** La seizième, +0,00092, n'atteint
pas le seuil de 0,001, son intervalle contient zéro, et une cellule positive sur
seize est exactement ce que le hasard prévoit. Le protocole interdit de la
retenir, et elle n'est citée que parce qu'il impose aussi de tout publier.

La lecture qui compte : dans presque toutes les cellules, **le marché seul est
meilleur que le marché plus n'importe quel descripteur**. Ajouter de la
modélisation au prix le dégrade. La marge répare une partie de ce que les
descripteurs existants cassent — +0,00148 et +0,00122 par-dessus marché+base sur
l'ATP — sans jamais repasser devant le prix nu.

Verdict : `MARGIN_NO_BET`.

## UFC — deux défauts trouvés, une gate échouée, un holdout préservé

Les variables UFC existantes sont des moyennes brutes sur cinq combats : elles ne
disent pas contre *qui* les coups ont été portés.
`predictor_ufc/rigorous/adjusted_features.py` ajoute treize descripteurs :
statistiques ajustées à ce que l'adversaire concède habituellement, usure
cumulée de carrière (frappes encaissées, knockdowns subis), courbe d'âge, écart
au pic, activité sur douze mois, longue inactivité, changement de catégorie.

Deux défauts de la table `features_v2.parquet` sont apparus en chemin :

1. **Orientation incohérente.** La symétrisation inverse le signe de `diff_*`,
   `elo_diff`, `reach_diff`, `age_diff`, `market_logit` et de `y`, et permute
   `fighter_1`/`fighter_2`, mais **ne permute pas** `fighter_1_id`/`fighter_2_id`,
   `elo_1_pre`/`elo_2_pre` ni les colonnes `f1_*`/`f2_*`. Sur 3 420 des 6 719
   combats, l'identifiant ne désigne donc plus le combattant auquel `y` se
   rapporte, et `diff_sig_lnd_L5` ne vaut plus `f1_sig_lnd_L5 − f2_sig_lnd_L5`.
   L'orientation est reconstruite ici à partir de la graine fixe du tirage, avec
   un garde-fou qui vérifie l'accord avec le signe de `elo_diff` sur les
   6 209 combats où il tranche. Contrôle de validité : la corrélation entre
   `age_diff` et une différence d'âge recalculée passe de **−0,013 à +1,000**.
2. **Deux descripteurs morts.** `height_diff` est vide sur 6 719 combats sur
   6 719, et `southpaw_matchup` ne prend qu'une seule valeur. Deux des huit
   descripteurs que le README annonce pour l'UFC ne portent aucune information.

Ablation de développement 2015‑2024, protocole gelé dans
`predictor_ufc/adjusted_protocol.json`, gates reprises telles quelles de la
phase 3 :

| Famille | base | challenger | gain | IC 90 % | années meilleures |
|---|---:|---:|---:|---|---:|
| logistique | 0,66380 | 0,66064 | +0,00316 | [−0,00027 ; +0,00676] | 7/10 |
| gradient boosting | 0,72851 | 0,72771 | +0,00081 | — | 4/10 |

Deux critères sur trois sont remplis : le gain dépasse le seuil de 0,001 et
sept années sur dix sont meilleures. Le troisième échoue franchement — le
challenger ne bat pas le marché, **0,66038 contre 0,61306**. La gate est donc
échouée.

Conséquence voulue : le **holdout économique du 13 septembre 2025 au 29 août 2026
n'a pas été ouvert.** C'est le dernier budget de preuve UFC intact, les trois
études précédentes s'étant arrêtées avant lui ; il le reste.

## Football — rien n'a été mesuré, et il faut le dire

Aucune mesure football n'a été produite, parce qu'elle était impossible ici :
`data/football/` est ignoré par git et absent du clone, et l'accès réseau
sortant vers `football-data.co.uk` est fermé dans cet environnement. Il ne reste
que `models/football/team_states.parquet`, 746 états d'équipes courants — un état
final, pas un historique de matchs.

Il n'y avait donc que deux options : livrer du code football non validé, ou ne
rien prétendre. C'est la seconde qui a été retenue. Pour que ce volet devienne
faisable, il faut soit ouvrir la politique réseau de l'environnement, soit
déposer `data/football/` dans le clone ; les saisons de réglage, de validation et
le holdout football n'ont jamais été ouvertes et resteraient disponibles.

## Ce que ce travail change, et ce qu'il ne change pas

**Ce qu'il change.** L'opinion que l'application affiche à côté du prix est
sensiblement meilleure sur les deux circuits de tennis, et la barre franchie l'a
été sur un protocole écrit d'avance. Le modèle UFC gagne un peu, et deux défauts
réels de sa table de features sont documentés et contournés.

**Ce qu'il ne change pas.** Il n'y a toujours aucune stratégie rentable
démontrée, sur aucun des trois sports. Le résultat central de ce dépôt tient : le
marché intègre déjà ce que ces modèles savent, et un meilleur descripteur n'est
pas un avantage. Un écart large affiché sur un match signale le plus souvent que
c'est le modèle qui se trompe.

**Ce qu'il ne pouvait pas établir.** ATP et WTA étant brûlés, même une gate B
franchie n'aurait été qu'une hypothèse. La seule preuve économique encore
possible est prospective, ou passe par les holdouts UFC et football jamais
ouverts — et aucun des deux n'a été dépensé ici.

## Reproduire

```bash
python3 scripts/run_margin_study.py --freeze-only   # écrit et hashe le protocole
python3 scripts/run_margin_study.py                 # les deux tests tennis
python3 predictor_ufc/run_adjusted_ablation.py      # ablation UFC, holdout fermé
python3 -m pytest tests/test_score_features.py predictor_ufc/tests/test_adjusted_features.py
```

Les dix-huit tests ajoutés portent surtout sur la causalité : tronquer l'avenir
ne doit rien changer au passé, deux matchs du même jour doivent voir le même
état, et échanger les deux camps doit exactement inverser le signe des écarts.
