# Arbitrage entre bookmakers — le premier résultat que les statistiques ne tuent pas

**Verdict : les arbitrages sont réels et persistants. Ce qui les rend
inexploitables n'est pas le hasard, c'est la frontière.**

Reproduire : `python3 scripts/run_arbitrage_scan.py`

---

## Pourquoi cette piste est différente des huit autres

Les huit pistes précédentes cherchaient à prévoir mieux qu'un prix, et échouaient
toutes au même endroit : le biais était réel mais plus petit que la marge.

L'arbitrage ne demande rien à un modèle. Si deux bookmakers divergent assez,
parier les deux côtés rapporte quoi qu'il arrive. Il n'y a aucune probabilité à
estimer, donc aucun edge à démontrer.

Cette piste était **impossible à tester avant** : l'historique de ce dépôt
n'avait pas de prix simultanés. La colonne `market_maximum` impliquait un
arbitrage dans 43 % des matchs, ce qui prouvait justement qu'elle mélangeait des
instants différents. L'API des cotes renvoie 40 à 58 bookmakers **dans une même
réponse**, avec l'horodatage de chacun. La simultanéité devient vérifiable.

## Ce qui a été trouvé

Sur un échantillon de 6 marchés tennis et 51 marchés MMA, **7 arbitrages bruts**.
Trois passes espacées de 4 minutes :

| Rencontre | Passe 1 | Passe 2 | Passe 3 | Books |
|---|---:|---:|---:|---|
| Sean King – Jessie Rosas | 1,29 % | 1,29 % | 1,29 % | betonlineag / sportsbet |
| J.J. Aldrich – Regina Tarin | 1,18 % | 1,18 % | 1,18 % | betonlineag / tab |
| Iva Jović – Coco Gauff | 0,51 % | 0,51 % | 0,51 % | smarkets / betfair_ex_uk |
| Tai Tuivasa – R. Despaigne | 0,27 % | 0,27 % | 0,27 % | paddypower / coolbet |
| Andreeva – Potapova | 8,34 % | 2,53 % | **28,31 %** | change à chaque passe |
| Swiatek – Zheng | 5,50 % | 9,43 % | 0,75 % | change à chaque passe |

**6 sur 6 présents aux trois passes.** Une opportunité qui disparaît avant qu'on
ait placé le second pari n'en est pas une ; celles-ci tiennent au moins huit
minutes.

## La lecture honnête : deux populations distinctes

**Les petits arbitrages (0,27 % – 1,29 %) sont crédibles.** Valeur identique et
mêmes bookmakers aux trois passes : ce sont des prix qui n'ont simplement pas
bougé.

**Les gros (2,5 % – 28 %) sont des artefacts.** Le couple de bookmakers change à
chaque passe et l'amplitude est absurde — Pinnacle ne se trompe pas de 28 % sur
un match de l'US Open. Cote périmée, offre d'exchange sans profondeur, ou erreur
de saisie qui sera annulée.

C'est pourquoi le scanner sépare les deux plutôt que d'additionner tout.

## Ce qui les rend inexploitables

Après application des quatre garde-fous : **0 exploitable sur 7**. Et toujours
pour la même raison — `books_accessibles = False`.

| Garde-fou | Ce qu'il écarte |
|---|---|
| Fraîcheur ≤ 5 min | Une cote de 13 minutes contre une de 30 secondes n'est pas une paire simultanée |
| Hors exchange | Un prix Betfair ou Smarkets est une *offre* de profondeur inconnue, avec commission sur les gains |
| Books accessibles | Une marge chez un opérateur où l'on ne peut pas ouvrir de compte n'est pas une marge |
| Persistance | Vérifiée ici : elle tient |

Les bookmakers qui divergent sont **tab** (Australie), **sportsbet** (Australie),
**betrivers** (États-Unis), **betonlineag** (offshore), **tipico_de**
(Allemagne), **paddypower** (Royaume-Uni/Irlande).

Restreint à la seule région `eu`, il reste 2 arbitrages bruts — et ils viennent
d'unibet **Suède** et de mybookie (offshore américain). Le tag `eu` de l'API ne
signifie pas « accessible depuis la France ».

**L'arbitrage existe précisément parce que les marchés sont cloisonnés, et le
cloisonnement est exactement ce qui empêche de l'exploiter.** Un compte unique ne
peut pas enjamber deux juridictions.

## Ce que cela change

C'est la seule piste du projet dont l'obstacle n'est pas statistique. Les huit
autres échouaient sur « l'edge est trop petit pour être démontré ». Celle-ci
échoue sur une contrainte administrative — ce qui veut dire qu'elle **cesserait
d'échouer** pour quelqu'un dont la situation est différente.

Concrètement, elle deviendrait exploitable avec des comptes légalement ouverts
dans plusieurs juridictions. Les ordres de grandeur restent modestes et les
contraintes opérationnelles réelles :

- 0,3 % à 1,3 % par opportunité, sur deux mises, soit environ 0,5 % du capital
  engagé ;
- les opérateurs limitent puis ferment rapidement les comptes d'arbitrage ;
- les plafonds de mise sont bas sur les cotes déséquilibrées ;
- il faut placer les deux jambes avant que le prix ne bouge.

Ce n'est pas un projet de modélisation. C'est de l'exécution et de la logistique
de comptes, et cela ne se démontre pas depuis un carnet Python.

## Ce qui n'a pas été fait

Aucun pari n'est placé, et le scanner n'en propose aucun. Il mesure si
l'opportunité survit à ses propres contraintes — la question que les données
historiques ne pouvaient pas trancher, faute de prix simultanés.

---

## Value betting au meilleur prix accessible — mesuré, pas concluant

Même instrument, question voisine : plutôt que de parier les deux côtés, ne
prendre qu'un côté quand un book accessible le paie **au-dessus de la valeur
juste de Pinnacle**.

La version historique de ce test perdait 3,5 % à 14 %, mais elle comparait la
cote de clôture de Bet365 à celle de Pinnacle — une seule marge contre une autre.
Prendre le meilleur de quarante cotes simultanées est une autre proposition.

| Périmètre | Cotes examinées | Au-dessus de la juste valeur | EV moyenne |
|---|---:|---:|---:|
| Books accessibles depuis la France | 44 | 8 | **−0,62 %** |
| Hors exchange, toutes juridictions | 66 | 21 | +0,43 % |
| Tous books | 70 | 27 | +1,10 % |

**Chez les opérateurs réellement accessibles, l'EV moyenne est négative.** Et les
plus grosses « valeurs » sont invraisemblables — Potapova à 2,74 quand Pinnacle
la juge à 1,93 serait un avantage de 42 %, ce qui n'existe pas sur un marché
liquide. Ce sont les mêmes artefacts que les gros arbitrages : cotes périmées,
offres d'exchange sans profondeur, erreurs de saisie.

Restent des valeurs de 1 à 3 % chez coolbet, nordicbet et onexbet. Elles sont
plausibles, et **invérifiables rétrospectivement** : il n'existe aucun historique
de quarante cotes simultanées à backtester.

## Le vrai obstacle, et l'instrument qui le lève

`RAPPORT_RENTABILITE.md` calcule qu'établir un avantage de +0,74 % demande
**35 250 paris, soit trente-quatre ans**. Aucune stratégie ne se pilote sur un
retour aussi lent, et c'est ce délai — plus que les résultats eux-mêmes — qui a
fermé les neuf pistes.

La valeur de clôture (`src/backtesting/closing_line.py`) est la réponse
professionnelle à ce problème. Elle ne mesure pas un profit mais si le prix pris
valait mieux que celui sur lequel le marché a fini. Deux propriétés :

- **elle converge en centaines de paris, pas en dizaines de milliers**, la cote de
  clôture étant une cible bien moins bruitée qu'une victoire ou une défaite ;
- **un CLV négatif écarte un avantage immédiatement**, sans attendre que le ROI
  devienne significatif.

Elle ne prouve rien dans l'autre sens : battre la clôture et perdre quand même à
cause des commissions et des plafonds reste possible. Mais c'est le test à faire
avant d'engager une année de suivi papier.

## Douzième piste : les middles, et pourquoi ils ne sont pas une piste

Il restait une classe de marché non testée. Un *middle* n'est pas un pari
directionnel : on prend Over sur une ligne basse chez un opérateur et Under sur
une ligne plus haute chez un autre, et tout résultat tombant entre les deux
**gagne les deux jambes**. C'est la seule structure de ce projet où les deux
côtés peuvent gagner, et comme l'arbitrage elle ne demande aucune prévision.

`src/backtesting/middles.py` scanne les marchés de totaux avec les mêmes gardes
que l'arbitrage — fraîcheur, exchange, accessibilité depuis la France — plus une
qui lui est propre : l'écart doit pouvoir contenir un résultat entier. Over 2,5
contre Under 2,75 est un intervalle arithmétique et rien de plus.

Le scan a trouvé **13 middles**, dont 2 chez des opérateurs accessibles : un
38,5 / 39,5 à 1,92 / 1,92 sur Gea–van de Zandschulp, un second sur
Shelton–Alcaraz. Un tel middle gagne double si le match finit sur exactement 39
jeux, et coûte 4 % de la mise sinon. Il vaut donc le coup si et seulement si

    P(exactement 39) × 0,92 > (1 − P) × 0,04    →    **P > 4,17 %**

C'est une propriété du tennis, pas des cotes. Les tables historiques répondent
directement (`scripts/run_middle_study.py`, 17 123 matchs best-of-5) :

| test | P(pile) | verdict |
|---|---|---|
| sans conditionnement, ligne 39 | 3,22 % | sous le seuil |
| conditionnellement à une prévision hors échantillon | **3,63 %** — IC 95 % [3,17 %, 4,16 %] | seuil hors intervalle |

L'objection sérieuse était que **ma prévision est plus bruitée que la ligne d'un
book** (erreur absolue 7,51 jeux), donc que je sous-estime P. Elle est testable :
j'ai dégradé ma propre prévision exprès. Passer d'une erreur de 7,51 à 12,08
jeux — 61 % de dégradation — ne fait perdre que **0,16 point** de P(pile). La
courbe est plate : ce qui limite P n'est pas la finesse de la ligne mais la
dispersion intrinsèque d'un match de tennis, qu'aucun book ne peut réduire.
Affiner au-delà de ma prévision ne rattrape donc pas les 0,6 point manquants.

### Le piège que j'ai failli rapporter

En déplaçant le middle sous la ligne prévue, P(pile) monte : 4,91 % à −8 jeux,
avec une borne basse d'intervalle à 4,52 %, au-dessus du seuil. Résultat
apparemment positif, et faux.

Il tenait les cotes figées à 1,92 / 1,92 en éloignant la ligne de la prévision.
Or un book qui attend 35 jeux ne cote pas Over 26,5 à 1,92 — il le cote 1,18. En
facturant chaque jambe à son vrai prix, l'EV devient **exactement −4,50 % à tous
les décalages**, c'est-à-dire la marge, et cesse complètement de dépendre de la
ligne. C'était le même pari vu de plus loin.

### Ce que ça règle

Un middle **n'a aucun avantage propre**. Toute combinaison de paris justement
cotés rend la marge en négatif, où qu'on place les lignes. Sa seule valeur
possible vient de deux books en désaccord sur la ligne — donc d'un prix meilleur
que le vrai. C'est **exactement l'arbitrage**, avec exactement les mêmes
obstacles : cotes périmées, profondeur inconnue, opérateurs hors juridiction.

Le middle n'est donc pas une douzième piste. C'est la onzième sous un autre
angle, et elle bute sur le même mur. La dernière classe de marché non testée est
fermée, et pour une raison structurelle plutôt que par manque de puissance
statistique — ce qui, pour une fois, est une réponse définitive et non un
« il faudrait trente-quatre ans pour le savoir ».
