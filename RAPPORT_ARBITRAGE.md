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
