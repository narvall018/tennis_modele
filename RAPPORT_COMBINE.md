# Le combiné boosté : la première piste que je n'arrive pas à tuer

Onze pistes sont mortes du même diagnostic — un biais réel, systématiquement plus
petit que la marge à franchir. Celle-ci est différente, et il faut dire d'emblée
pourquoi, parce que la différence est structurelle et non statistique.

**Le boost n'est pas une prédiction, c'est une clause contractuelle.** Le book ne
se trompe pas sur la probabilité quand il boost : il paie pour acquérir un
client. La source du gain est un budget marketing, pas une erreur de modèle. Il
n'y a donc rien à démontrer sur 35 000 paris — il suffit de comparer deux
quantités connues, la marge empilée et le boost promis.

## L'identité sur laquelle tout repose

Pour des jambes indépendantes — des matchs différents — l'espérance d'un combiné
est **exactement** le produit des espérances par jambe :

    E[combiné] = ∏ (p_i × O_i) = r^n

Aucun modèle là-dedans, seulement de l'algèbre. Toute la stratégie tient donc
dans une seule quantité mesurable : `r`, le rendement d'une jambe.

## Le biais favori-outsider, mesuré chez Bet365

`r` dépend massivement de la cote, parce que **la marge du book n'est pas
répartie — elle est posée sur les outsiders** (191 458 matchs, 1 019 662 jambes) :

| tranche de cote | n | r | IC 95 % (blocs mensuels) |
|---|---|---|---|
| **1,15 – 1,35** | 9 352 | **0,9890** | [0,9781 – 0,9994] |
| 1,35 – 1,55 | 15 508 | 0,9684 | [0,9571 – 0,9796] |
| 1,55 – 1,80 | 28 161 | 0,9542 | [0,9437 – 0,9645] |
| 1,80 – 2,10 | 42 704 | 0,9392 | [0,9298 – 0,9486] |

Une jambe à 1,26 coûte **1,1 %**, une jambe à 12,00 en coûte 32. C'est
`favourite_longshot_bias.md` — un biais qui, seul, ne franchissait aucune marge.
Combiné à un boost, il devient le levier.

Mesuré chez **Bet365 seul**, jamais sur `Avg` : la moyenne du marché est la
moyenne d'opérateurs différents, et personne ne parie une moyenne. C'est
exactement l'erreur qui a invalidé `market_maximum`.

### Et il ne s'érode pas

| période | 1,15–1,35 | 1,35–1,55 | 1,55–1,80 | 1,80–2,10 |
|---|---|---|---|---|
| 2000–10 | 0,9715 | 0,9607 | 0,9381 | 0,9312 |
| 2011–16 | 1,0154 | 0,9618 | 0,9567 | 0,9513 |
| 2017–21 | 0,9759 | 0,9806 | 0,9531 | 0,9422 |
| 2022–26 | **0,9959** | 0,9756 | 0,9780 | 0,9337 |

Vingt-six ans, et la tranche courte se renforce plutôt qu'elle ne disparaît.

## Le seuil, calculé sur la borne basse de r

| jambes | cote du combiné | P(gagne) | boost requis |
|---|---|---|---|
| 5 | 3,14 | 28,5 % | 17,2 % |
| 8 | 6,22 | 13,5 % | 23,1 % |
| **10** | 9,83 | 8,2 % | **27,6 %** |
| 15 | 30,81 | 2,3 % | 40,8 % |

Le seuil monte **linéairement** avec le nombre de jambes. Les barèmes annoncés
montent plus vite — un Combo Booster passe de 2,5 % à 4 sélections à 100 % et
au-delà sur les longs combinés. D'où une fenêtre, qui s'ouvre vers 8 jambes.

## De quoi ça dépend (boost 50 %, 10 jambes)

| plancher de cote imposé | r | EV |
|---|---|---|
| 1,15 | 0,9890 | **+29,7 %** |
| 1,30 | 0,9774 | +17,9 % |
| 1,40 | 0,9661 | +5,6 % |
| 1,50 | 0,9547 | **−6,0 %** |

| book français plus cher | EV | | boost tiré au sort | EV |
|---|---|---|---|---|
| +0,5 pt | +23,3 % | | affiché 50 % | +29,7 % |
| +1,5 pt | +11,3 % | | 6 valeurs, moy. 25 % | +9,6 % |
| +3,0 pt | **−4,7 %** | | 6 valeurs, moy. 12,5 % | **−0,5 %** |

**La conclusion survit** à un plancher de cote jusqu'à 1,40, à deux points de
marge en plus qu'un book anglais, et à un boost aléatoire de moyenne 25 %. Elle
meurt au-delà. Ce sont les trois chiffres à vérifier avant d'engager un euro.

## L'offre

Sur mes 22 divisions depuis 2022, la médiane est de **2 favoris par jour** à
1,15–1,35, et seuls 0,6 % des jours en offrent dix. Mais un combiné s'étale sur
plusieurs jours : **48,6 % des semaines** en offrent dix. Et ces 22 divisions
sous-estiment largement ce que couvre un opérateur réel, qui prend aussi le
tennis, le basket et le reste du monde.

Rythme réaliste : **un combiné par semaine**, mise Kelly ~1,2 % de bankroll,
soit de l'ordre de **10 % de bankroll par an**. Modeste, et positif.

## Ce que ces chiffres n'établissent pas

Je dois être précis sur la limite, parce qu'elle est sérieuse :

1. **Les barèmes exacts.** J'ai reconstitué la progression boost/jambes à partir
   de copie marketing, pas des CGU. C'est l'hypothèse la plus lourde du rapport.
2. **Les prix des books français sur les favoris.** J'ai mesuré Bet365.
3. **L'éligibilité.** Un combiné de dix gros favoris est exactement ce qu'un book
   ne veut pas voir. Plafonds de mise, exclusions, limitation de compte.
4. **La variance.** À 10 jambes on gagne 8,2 % du temps : des séries de vingt
   pertes sont ordinaires, pas anormales.

`scripts/run_combi_boost_study.py` rejoue les cinq passes et accepte
`--boost` / `--legs` pour tester n'importe quel barème réel.

**Le verdict honnête :** c'est le premier résultat positif du projet, et il est
positif *par arithmétique* plutôt que par prédiction — ce qui est précisément ce
qui le rend crédible. Mais il repose sur un barème que je n'ai pas lu. Tant que
les CGU ne sont pas vérifiées, c'est une piste sérieuse, pas une stratégie.
