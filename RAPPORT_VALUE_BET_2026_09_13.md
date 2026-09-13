# Un système de value bet : le signal existe, et il est hors de France

Question : peut-on ne parier que les paris « à valeur » ? Oui, et le système le
plus honnête ne demande **aucun modèle**. On prend le prix dévigué d'un opérateur
sharp comme estimation de la probabilité, et on parie chez un opérateur mou quand
son prix dépasse cette estimation. Aucun descripteur, aucun paramètre appris,
seulement un seuil d'écart.

`scripts/run_value_bet_system.py`, 101 794 matchs de football.

## D'abord l'audit de simultanéité, car il change tout

| vérité / offre | paris | ROI | IC 95 % |
|---|---:|---:|---|
| sharp pré / mou pré | 19 907 | +1,07 % | [−1,63 %, +3,68 %] |
| sharp clôture / mou clôture | 8 180 | −0,76 % | [−4,87 %, +3,41 %] |
| **sharp CLÔTURE / mou pré** | 64 044 | **+4,62 %** | **[+3,49 %, +5,78 %]** |
| sharp pré / mou clôture | 28 970 | −7,73 % | [−9,44 %, −6,12 %] |

**La seule configuration dont l'intervalle exclut zéro est celle qui lit le
futur** — comparer la clôture du sharp au prix d'ouverture du mou. Son image
miroir perd symétriquement −7,73 %, ce qui confirme le mécanisme : c'est la
clôture qui informe, pas une inefficience exploitable.

Sans cet audit, j'aurais annoncé un système à +4,6 % dont l'intervalle exclut
zéro. C'est la troisième fois aujourd'hui qu'une cote de clôture fabrique un
faux positif.

## Le signal est réel — la falsification est sans ambiguïté

Sur la tranche où les paris tombent réellement (cotes 2,5-8,0) :

| signal | paris | ROI |
|---|---:|---:|
| valeur > +2 % | 6 882 | **+2,20 %** |
| valeur ≈ 0 | 14 872 | **+0,00 %** |
| valeur < −2 % *(contrôle)* | 169 094 | **−8,30 %** |

Gradient parfaitement monotone, et la cellule neutre tombe **exactement** à zéro.
Le prix dévigué de Pinnacle discrimine donc réellement quel côté du prix d'un
book mou est le bon. Ce n'est pas du bruit.

Stabilité : **10 années positives sur 14**, et les cinq dernières entre +0,4 % et
+3,3 %. Le +29,7 % de 2012 est une valeur aberrante qui gonfle la moyenne.

Mais aux instants cohérents, **aucun seuil n'a d'intervalle excluant zéro** :
+1,07 %, +0,29 %, +3,14 %, +1,62 % selon le seuil, tous avec zéro dedans.

## Et voici pourquoi ça ne vous sert à rien

Le système a besoin d'un book **assez bon marché pour passer parfois au-dessus du
prix juste**. Relevé en direct sur 170 événements où Pinnacle est présent :

| opérateur | comparés | valeur > 0 | valeur médiane | meilleure valeur |
|---|---:|---:|---:|---:|
| PMU | 282 | **0,7 %** | −9,84 % | +2,14 % |
| Betclic | 300 | **0,0 %** | −10,68 % | −0,63 % |
| Unibet | 324 | **0,0 %** | −11,03 % | −1,00 % |
| Netbet | 177 | **0,0 %** | −11,23 % | −0,98 % |
| Winamax | 342 | **0,0 %** | −12,87 % | −1,14 % |
| **meilleur des cinq** | 369 | **0,5 %** | −9,43 % | +2,14 % |

**Quatre opérateurs français sur cinq n'offrent jamais de valeur. Pas rarement :
jamais.** Le pari médian chez un book français est à **−9,4 % sous le prix
juste**, et même le meilleur des cinq ne dépasse la vérité que 0,5 % du temps.

Le système fonctionne avec Bet365 (surmarge 6,55 %) comme jambe molle. Les books
français sont à 7,3–15,3 %. Ils sont trop chers pour croiser le prix juste.

## Conclusion

Le value bet est **le bon système** — sans modèle, avec un mécanisme démontré par
falsification, stable sur quatorze ans. Il est aussi **inapplicable depuis la
France**, non par manque de signal mais parce qu'aucun opérateur accessible ne
propose jamais le prix qu'il faudrait.

C'est le même mur que l'arbitrage, le même que l'échelle d'exécution, mesuré
cette fois au niveau du pari individuel : **0,0 % d'occasions chez quatre books
sur cinq.**

L'outil reste dans le dépôt. `run_value_bet_system.py live` le rejoue à tout
moment ; si un opérateur accessible baissait un jour sa marge sous ~6,5 %, il
s'allumerait tout seul.
