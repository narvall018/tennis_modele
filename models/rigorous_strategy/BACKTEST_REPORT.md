# Backtest rigoureux de la stratégie ATP

Décision: **NON VALIDÉE — NE PAS MISER EN ARGENT RÉEL**

## Protocole verrouillé

- Modèle et mélange marché: 2012–2016.
- Règle de sélection: 2017–2020.
- Validation et taille des mises: 2021–2023.
- Test final jamais utilisé pour les choix: 2024–2026.
- Entraînement d'un fold Y: résultats terminés jusqu'à Y-2; calibration: Y-1; test: Y.
- Aucune cote absente n'est remplacée; prix moyens observés et dévigés pour mesurer l'edge.
- Abandons et walkovers restent dans la population; scénario principal: mises annulées.

## Stratégie figée

- Modèle: `market_xgboost`; poids modèle: 70.0%; poids marché: 30.0%.
- Règle: edge ≥ 0.5%, EV estimée ≥ 1.0%, probabilité ≥ 0.0%, cote 1.75–4.00.
- Mise: `flat_1_00pct`, plafond par pari 1.00%, exposition quotidienne 4.00%.

## Test final 2024–2026

- Sans décote: 116 paris réglés, ROI -21.80%, profit -25.29 unités.
- Décote de cote 2%: ROI -22.69%; IC bootstrap mensuel 90% [-35.09%, -9.76%].
- Bankroll simulée: 1000.00 → 763.25; drawdown max 28.96%.

## Le modèle prévoit-il mieux que le prix qu'il affronte ?

Sur tous les matchs cotés de chaque période, pas seulement sur les paris pris.

| Période | Matchs | Log-loss marché | Log-loss mélange | Gain |
|---|---:|---:|---:|---:|
| development | 12555 | 0.55105 | 0.54934 | +0.00171 |
| tuning | 8844 | 0.58641 | 0.58536 | +0.00105 |
| validation | 7562 | 0.58608 | 0.58480 | +0.00127 |
| holdout | 7075 | 0.58914 | 0.58908 | +0.00006 |

Un gain positif et stable indique un vrai pouvoir prédictif supplémentaire. Un gain nul ou négatif accompagné d'un ROI positif signale au contraire que le résultat vient de la dispersion des prix, pas d'une meilleure lecture du sport.

## Limites qui empêchent toute promesse de gain

- Les cellules par source de prix (Bet365, Pinnacle, maximum) sont des tests de sensibilité, pas des résultats: l'edge y est recalculé contre chaque prix, donc chaque source produit une population de paris différente. Choisir après coup la source la plus flatteuse annulerait le protocole.
- Le prix `maximum` est le meilleur prix trouvé chez un opérateur quelconque. Il surestime structurellement le rendement et n'est pas exécutable à volume.
- Les cotes sont généralement les dernières avant le match; leur disponibilité exacte et les limites de mise ne sont pas horodatées.
- Le scénario principal suppose que tous les abandons sont annulés; les règles réelles varient selon l'opérateur.
- L'IC par blocs mensuels mesure l'incertitude historique, pas le risque de changement futur du marché.
- Une validation statistique autorise seulement un paper-trading préalable, jamais une garantie de rentabilité.
