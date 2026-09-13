# Sans la contrainte française : ce qui marche, et ce qui a cessé de marcher

Les rapports précédents prenaient les cinq opérateurs agréés comme ensemble
atteignable. C'est une contrainte que l'on se donne, pas une donnée du problème.
Sans elle, **Pinnacle est à 3,02 % de surmarge** quand le seuil de rentabilité
d'un gros favori est à 4,62 % : l'analyse change matériellement, et les
conclusions précédentes ne s'appliquent plus.

Deux stratégies, à ne pas confondre. Dans (B) Pinnacle sert de référence de
vérité, il ne peut donc pas être en même temps l'endroit où l'on parie.

## A. Parier les gros favoris — l'avantage a existé, puis a disparu

Favoris cotés 1,20-1,35, au meilleur de Bet365 et Pinnacle :

| période | n | rendement | IC 95 % | ROI |
|---|---:|---:|---|---:|
| **2013-2018** | 1 782 | **1,0356** | **[1,0112, 1,0587]** | **+3,56 %** |
| 2019-2025 | 2 247 | 0,9935 | [0,9723, 1,0143] | −0,65 % |

**Le premier intervalle de tout ce projet à exclure 1 — et il est historique.**
Sur 2013-2018 le biais favori-outsider était réellement exploitable : +3,56 %
par pari, borne basse à +1,12 %.

Sur 2019-2025 il vaut **−0,65 %**, avec zéro dans l'intervalle. Les marchés se
sont resserrés. Un avantage qui a existé six ans et n'existe plus n'est pas une
stratégie : c'est une pièce d'histoire.

## B. Le value bet — stable, jamais prouvé

Parier chez Bet365 quand son prix dépasse le prix Pinnacle dévigué, cotes 2,0-4,0 :

| échantillon | n | ROI | IC 95 % |
|---|---:|---:|---|
| toutes années | 10 568 | +2,84 % | [−0,22 %, +5,92 %] |
| sans 2012 *(année à +29 %)* | 10 415 | +2,46 % | [−0,84 %, +5,54 %] |
| 2015 et après | 8 788 | +1,70 % | [−1,76 %, +5,36 %] |
| 2019 et après | 4 761 | +2,33 % | [−2,80 %, +7,02 %] |

**Le point estimé ne s'érode pas** : entre +1,7 % et +2,8 % sur toutes les
sous-périodes, y compris les plus récentes. Contrairement à (A), il ne dépend
pas d'une époque.

Les contrôles tiennent :

- **simultanéité** — les deux configurations honnêtes sont positives (+2,84 % à
  l'ouverture, +4,43 % à la clôture) ; seules les anachroniques divergent ;
- **falsification** — +1,91 % / +3,33 % au-dessus de zéro de valeur, contre
  −0,28 % / −6,70 % en dessous, sur 164 059 paris de contrôle ;
- **stabilité** — 11 années positives sur 14.

Mais **aucun intervalle n'exclut zéro**, sur aucune sous-période. Après
quatorze ans de données, l'avantage reste compatible avec l'absence d'avantage.

## Ce que ça donne en pratique

Sur 2019 et après : **595 paris par an** à cote moyenne 3,33, ROI +2,33 %.
Kelly complet 1,0 % de bankroll, quart de Kelly **0,25 %**, soit une croissance
attendue de **3,5 % de bankroll par an**. À demi-Kelly, environ 7 %.

## La contrainte qui décide vraiment

Elle n'est pas statistique. **(B) exige un book mou — ici Bet365 — et Bet365
limite agressivement les comptes gagnants.** Quelqu'un qui place 595 paris de
valeur par an, systématiquement du bon côté, correspond exactement au profil
fermé, généralement en quelques mois.

Pinnacle, lui, ne limite pas : c'est son modèle commercial. Mais Pinnacle est la
référence de vérité de (B), pas l'endroit où l'on parie. Et (A), qui se joue
chez Pinnacle, est mort depuis 2019.

C'est la vraie tension : **le book où l'on peut jouer longtemps est celui qui n'a
pas d'erreur à exploiter, et celui qui en a vous ferme.**

## Portée

Historique football déjà exploré à plusieurs reprises ; ces chiffres sont
exploratoires et ne valent pas preuve prospective. La collecte lancée
(`run_prospective_collector.py`) est ce qui permettra un jour de trancher (B)
sur des prix horodatés que personne n'a vus.

`scripts/run_international_strategies.py` rejoue les deux contrôles.
