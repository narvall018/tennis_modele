# La meilleure opportunité, tous sports confondus

Une seule question posée à tout le dépôt : *parmi tout ce que ces modèles savent
faire, quelle est la meilleure occasion en ce moment, peu importe le sport ?*

```bash
python3 scripts/run_best_opportunity.py
python3 scripts/run_best_opportunity.py --wta-surface "WTA Tokyo=Hard" "WTA Seoul=Hard"
```

Code : `src/app/best_opportunity.py`. Tests : `tests/test_best_opportunity.py`.
Rapport écrit : `models/best_opportunity_scan.json`. Aucune mise n'est envoyée.

## La règle de classement, fixée avant de voir les nombres

Les stratégies de ce dépôt ne répondent pas dans la même unité. Les comparer
suppose de dire d'abord ce qui l'emporte, sinon le classement se choisit après
coup, en regardant les pourcentages.

| Palier | Ce que la mesure vaut | Qui y répond |
| --- | --- | --- |
| 1 — arithmétique | Gain **calculé** : si les deux jambes passent aux prix relevés, le profit existe quel que soit le résultat | scanner d'arbitrage, tous sports |
| 2 — modèles | Espérance **estimée** par une règle figée dont le filtre de validation a échoué | ATP prix de référence, WTA noyau |

**Un palier 1 exploitable passe toujours devant un palier 2**, même quand son
pourcentage est plus petit : 1,5 % garanti et 8 % espérés par un modèle non
validé ne sont pas deux valeurs du même objet. À l'intérieur d'un palier, le
classement suit la mesure, puis l'heure de début, puis les identités : il est
déterministe, jamais tiré au sort.

Seules les lignes exploitables sont classées. Les autres restent affichées avec
le garde-fou qui les écarte — une opportunité écartée n'est pas une opportunité
absente, et les deux ne se lisent pas de la même façon.

## Qui a le droit de sélectionner, et sur quelle preuve

| Sport | Stratégie | Preuve publiée | Droit de sélection |
| --- | --- | --- | --- |
| Tous | Arbitrage entre books | Gain arithmétique, pas de prévision | Oui, sous ses quatre garde-fous |
| ATP | `atp_reference_price_paper_v1` | +24,38 % sur 60 paris réglés 2023–2025, IC 95 % [−7,95 % ; +58,79 %], filtre non franchi | Oui, papier uniquement |
| WTA | `wta_kernel_flexible_paper_v1` | +19,03 % sur 110 paris réglés 2023–2025, IC 95 % [−2,69 % ; +38,42 %], borne famille −7,71 %, filtre non franchi | Oui, papier uniquement, surface confirmée |
| Football | `models/football` | Log-loss inférieure au marché de 0,019754 ; « ne bat pas le marché » dans ses propres métadonnées | Non |
| UFC | `models/ufc` | Trois phases rigoureuses rejetées, modèle sans cote | Non |

Le football et l'UFC ne sont pas exclus du scan : ils restent couverts par le
palier 1, qui ne demande rien à un modèle. Ce qui leur est refusé, c'est
d'émettre une sélection de valeur. Le script relit ces deux verdicts dans les
métadonnées livrées plutôt que de les recopier, pour qu'ils ne puissent pas
vieillir en silence.

La WTA exige une surface confirmée par tournoi (`--wta-surface`). Fournir
l'entrée vaut confirmation du simple, tableau principal. Rien ne devine une
surface, et un tournoi sans confirmation est signalé, jamais supposé. Les doubles
sont écartés. Une joueuse sans historique suffisant, un homonyme ou un paquet
périmé sont dits explicitement : zéro sélection faute de modèle ne se lit pas
comme zéro sélection faute de marché.

## Ce qu'il faut s'attendre à voir

`RAPPORT_ARBITRAGE.md` a déjà mesuré le palier 1 : **0 exploitable sur 7
arbitrages bruts**, tous écartés pour la même raison — les books qui divergent
sont australiens, américains ou offshore. L'obstacle n'est pas statistique, il
est administratif, et un compte unique ne l'enjambe pas. Le palier 1 sera donc
souvent vide depuis la France ; c'est un résultat, pas une panne.

Reste le palier 2, c'est-à-dire une espérance estimée par une règle dont le
filtre a échoué. Le scan peut très bien ne rien rendre du tout : **ne rien miser
est une sortie normale**, pas l'échec du scan.

## Coût et fraîcheur

Chaque sport interrogé coûte une requête du quota mensuel (500 en offre
gratuite), donc la liste des sports est explicite et rien n'est appelé deux fois.
Les cotes ont cinq minutes de validité, les paires trois minutes d'écart maximum,
et le match doit commencer entre dix minutes et 48 heures plus tard. Un scan
relu plus tard ne vaut plus rien : il faut le relancer.

## Ce que ce classement n'est pas

Ce n'est pas une prévision de gain, ni une autorisation de miser de l'argent
réel : `real_money_authorised` vaut `False` sur chaque ligne. Le palier 2 reste
une opinion sur des prix, mesurée sur un passé déjà exploré. Le palier 1 n'est
garanti que si les deux jambes sont réellement prises, aux prix affichés, avant
que l'un des opérateurs ne bouge — ce que ce dépôt ne peut ni faire ni vérifier.
