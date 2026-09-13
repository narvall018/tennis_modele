# Arbitrage avec accès à tous les opérateurs : ce qu'il reste vraiment

L'arbitrage est la seule piste de ce dépôt dont l'obstacle n'était **pas
statistique**. Le rapport du 7 septembre concluait : *« cela ne devient
exploitable que pour quelqu'un détenant des comptes légaux dans plusieurs
juridictions — une question sur sa situation, pas sur la modélisation. »*

La contrainte étant levée, voici la mesure.

## Le relevé

**16 sports, 397 marchés interrogés, 14 arbitrages avant-match** — soit **3,5 %
des marchés**, pour un gain théorique moyen de **0,97 %**.

Test de persistance sur le tennis, deux passes à trois minutes : les petits
écarts persistent **au centième près** (2,3459 % identique, 0,1660 % identique),
les gros **changent de paire d'opérateurs** à chaque passe. C'est la signature
qui sépare un prix figé d'un artefact, et elle est constante avec le relevé du
7 septembre.

## La décomposition qui change la conclusion

| sélection | n | gain moyen |
|---|---:|---:|
| impliquant **1xBet** | 7/14 | **1,65 %** |
| sans 1xBet | 7/14 | **0,29 %** |
| sans exchange | 9/14 | 1,06 % |
| **sans exchange et sans 1xBet** | **4/14** | **0,29 %** |

Les quatre occasions réellement « propres » — deux vrais bookmakers, pas
d'exchange, pas de 1xBet :

| gain | paire |
|---:|---|
| 0,64 % | fanduel / draftkings |
| 0,36 % | betmgm / unibet_nl |
| 0,14 % | draftkings / fanduel |
| 0,02 % | coolbet / marathonbet |

**La moitié rentable de l'arbitrage repose sur un seul opérateur.** 1xBet
apparaît dans 7 des 14 occasions et porte à lui seul l'écart entre 1,65 % et
0,29 %.

## Pourquoi c'est un problème et pas un détail

Un arbitrage est un pari sans risque **de marché**. Il reste intégralement exposé
au risque de **contrepartie** : il faut que les deux jambes soient acceptées,
tenues, et payées.

1xBet fait l'objet d'actions réglementaires dans plusieurs juridictions et d'une
réputation constante d'annulation de paris gagnants et de fermeture de comptes.
Concentrer là-dessus la moitié de son espérance revient à remplacer un risque de
marché — qu'on sait mesurer — par un risque de contrepartie qu'on ne peut ni
couvrir ni diversifier.

Et c'est asymétrique : sur un arbitrage à 1,65 %, si une jambe est annulée, on ne
perd pas 1,65 % mais l'intégralité de l'exposition de l'autre jambe.

## Ce qui reste sans 1xBet

**1 % des marchés, à 0,29 % de gain théorique.** À comparer à ce qu'il faut
absorber :

- l'écart de prix entre le moment du scan et celui des deux exécutions ;
- le refus ou la réduction de mise, courant sur les comptes qui n'envoient que
  des arbitrages ;
- le capital immobilisé chez chaque opérateur jusqu'au règlement ;
- la **limitation de compte**, qui est la contrainte réelle : un compte qui ne
  place que des arbitrages est repéré et bridé en quelques semaines.

0,29 % ne finance pas cette friction.

## Conclusion honnête

L'arbitrage **existe** et devient **visible** avec un accès multi-juridictions :
3,5 % des marchés, ce qui n'est pas rien. Mais sa rentabilité est portée par un
opérateur dont le risque de contrepartie est précisément celui qu'un arbitrage
ne peut pas couvrir.

La question n'est donc plus « y a-t-il un avantage » — il y en a un — mais
« combien de temps un compte survit, et l'opérateur paie-t-il ». Ça ne se mesure
pas dans des cotes, et je ne peux pas y répondre à votre place.

`scripts/run_arbitrage_scan.py --regions eu,uk,us,au` rejoue le relevé. Les
gardes restent affichées colonne par colonne plutôt que d'écarter en silence.
