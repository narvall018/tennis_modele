# Backtest rigoureux de la stratégie WTA

Décision: **NON VALIDÉE — NE PAS MISER EN ARGENT RÉEL**

## Protocole verrouillé

- Modèle et mélange marché: 2013–2016.
- Règle de sélection: 2017–2019.
- Validation et taille des mises: 2020–2022.
- Test final jamais utilisé pour les choix: 2023–2026.
- Entraînement d'un fold Y: résultats terminés jusqu'à Y-2; calibration: Y-1; test: Y.
- Aucune cote absente n'est remplacée; prix moyens observés et dévigés pour mesurer l'edge.
- Abandons et walkovers restent dans la population; scénario principal: mises annulées.

## Stratégie figée

- Modèle: `market_logistic`; poids modèle: 70.0%; poids marché: 30.0%.
- Règle: edge ≥ 0.5%, EV estimée ≥ 0.0%, probabilité ≥ 50.0%, cote 1.75–4.00.
- Mise: `flat_0_25pct`, plafond par pari 0.25%, exposition quotidienne 2.00%.

## Test final 2023–2026

- Sans décote: 93 paris réglés, ROI 2.46%, profit 2.29 unités.
- Décote de cote 2%: ROI 1.51%; IC bootstrap mensuel 90% [-13.35%, 17.54%].
- Bankroll simulée: 1000.00 → 1003.29; drawdown max 2.16%.

## Le modèle prévoit-il mieux que le prix qu'il affronte ?

Sur tous les matchs cotés de chaque période, pas seulement sur les paris pris.

| Période | Matchs | Log-loss marché | Log-loss mélange | Gain |
|---|---:|---:|---:|---:|
| development | 9561 | 0.58163 | 0.58038 | +0.00125 |
| tuning | 7152 | 0.59851 | 0.59796 | +0.00054 |
| validation | 5617 | 0.58184 | 0.58001 | +0.00184 |
| holdout | 9053 | 0.58434 | 0.58266 | +0.00168 |

Un gain positif et stable indique un vrai pouvoir prédictif supplémentaire. Un gain nul ou négatif accompagné d'un ROI positif signale au contraire que le résultat vient de la dispersion des prix, pas d'une meilleure lecture du sport.

## Limites qui empêchent toute promesse de gain

- Les cellules par source de prix (Bet365, Pinnacle, maximum) sont des tests de sensibilité, pas des résultats: l'edge y est recalculé contre chaque prix, donc chaque source produit une population de paris différente. Choisir après coup la source la plus flatteuse annulerait le protocole.
- Le prix `maximum` est le meilleur prix trouvé chez un opérateur quelconque. Il surestime structurellement le rendement et n'est pas exécutable à volume.
- Les cotes sont généralement les dernières avant le match; leur disponibilité exacte et les limites de mise ne sont pas horodatées.
- Le scénario principal suppose que tous les abandons sont annulés; les règles réelles varient selon l'opérateur.
- L'IC par blocs mensuels mesure l'incertitude historique, pas le risque de changement futur du marché.
- Une validation statistique autorise seulement un paper-trading préalable, jamais une garantie de rentabilité.
