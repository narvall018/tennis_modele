# Corners, totaux, handicap, props : les marchés annexes sont-ils battables ?

Question naturelle : un marché moins couru serait-il moins bien coté ? J'ai testé
ceux dont je possède les prix, et je dis clairement pour les autres que je ne
peux pas.

## Ce que je ne peux pas tester

- **Corners** : football-data fournit les corners *joués*, jamais leur prix. Sans
  cote, aucun rendement n'est calculable. Je ne vais pas simuler un marché.
- **Totaux de jeux / sets au tennis** : même problème. L'étude des middles
  (`RAPPORT_ARBITRAGE.md`) a couvert le total de jeux et montré qu'un middle n'a
  aucun avantage propre.
- **Méthode de finition UFC** : déjà clos. Le marché sous-évalue réellement les
  décisions de **+2,92 points, 13 années sur 13** — et un overround de **22 %**
  annule tout. Le biais est vrai, la marge est plus grosse.

## La surmarge, marché par marché

| marché | Bet365 | Pinnacle | Avg | Betfair |
|---|---:|---:|---:|---:|
| 1X2 (3 issues) | 6,55 % | 3,02 % | 6,51 % | 1,90 % |
| totaux 2,5 (2 issues) | 5,79 % | 3,77 % | 6,10 % | 3,00 % |
| **handicap (2 issues)** | **3,91 %** | **2,91 %** | 4,89 % | **1,62 %** |

**Le handicap asiatique est le marché le moins cher**, de loin : 3,91 % chez
Bet365 contre 6,55 % sur le 1X2. C'est le marché des parieurs sérieux, et les
books y rognent leur marge pour rester compétitifs.

## Le value bet par marché

| marché | seuil | paris | ROI | IC 95 % |
|---|---:|---:|---:|---|
| 1X2 | +2 % | 8 558 | +0,29 % | [−4,85 %, +5,15 %] |
| totaux 2,5 | +2 % | 249 | +9,01 % | [−10,02 %, +27,17 %] |
| **handicap** | **+2 %** | **351** | **+11,41 %** | **[+1,13 %, +21,95 %]** |

Le handicap donnait donc le premier intervalle du projet excluant zéro, avec un
gradient de falsification impeccable sur 96 356 paris : **+11,41 % / +2,01 % /
−1,43 % / −3,50 %**, parfaitement monotone. Cinq années positives sur sept. Et
t = +2,41, soit **4,8 %** de probabilité sous H0 après correction pour les six
cellules testées.

## Pourquoi il ne survit pas

Deux tests le tuent, et ce sont les bons.

**Le gradient n'existe qu'à l'ouverture.**

| tranche | ouverture / ouverture | clôture / clôture |
|---|---:|---:|
| > +2 % | **+11,41 %** (351) | **−3,02 %** (872) |
| +0 à +2 % | +2,01 % | −1,61 % |
| −2 % à 0 | −1,43 % | −0,86 % |
| < −2 % | −3,50 % | −3,47 % |

À la clôture le gradient disparaît, et la cellule de tête devient la pire.

**Et surtout le test de la meilleure vérité.** Quand le prix Bet365 d'ouverture
bat l'estimation de **clôture** de Pinnacle — la meilleure estimation disponible
de la probabilité — ces paris rendent **−3,31 % sur 19 969 paris**. Si
l'avantage était réel, cette sélection serait la plus rentable de toutes. C'est
la pire, et sur un échantillon cinquante fois plus grand que les 351 paris du
résultat positif.

La lecture honnête : le +11,41 % est la queue d'une distribution bruitée sur 351
paris, et non un avantage. La seule configuration qui l'exhibe est celle dont la
référence de vérité est la plus faible.

## Ce qui reste vrai

Le handicap asiatique **est** le marché le moins cher accessible — 3,91 % chez
Bet365, 1,62 % sur l'exchange Betfair. Si une piste devait un jour exister, c'est
là qu'il faudrait la chercher, pas sur les corners ou les props.

Mais le bon marché et le prix efficient vont ensemble : un marché serré est
serré *parce que* des parieurs sérieux l'arbitrent. On n'obtient pas les deux.

`scripts/run_market_types.py` rejoue les surmarges et les value bets.
