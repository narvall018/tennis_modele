# Tennis phase 4 — serve/return et surfaces

Décision: **PHASE4_REJECTED_NO_BET**

Toutes les données jusqu'au 30 août 2026 sont du développement déjà exposé. 
Aucune performance ci-dessous n'est une nouvelle validation indépendante.

## Modèles walk-forward 2017–2026

| Modèle | Log-loss | Écart au marché | Brier |
|---|---:|---:|---:|
| `market_surface_calibration` | 0.58483 | +0.00045 | 0.20100 |
| `structural_market_residual` | 0.58431 | +0.00097 | 0.20076 |
| `serve_return_residual` | 0.58436 | +0.00092 | 0.20077 |
| `surface_partial_pooling` | 0.58443 | +0.00085 | 0.20081 |
| Marché dévigé | 0.58528 | — | 0.20107 |

Candidat primaire retenu par la règle figée: `structural_market_residual`.

## Deep learning diagnostique

Le petit MLP symétrique obtient une log-loss de 0.58671, soit -0.00143 contre le marché. Il n'était pas autorisé à sélectionner une stratégie ou à ouvrir une gate économique.

## Diagnostic économique fixe, développement uniquement

- Paris réglés après décote 2%: 563.
- ROI: -0.43%; profit: -2.44 unités.
- IC bootstrap mensuel 99%: [-11.58%, 11.64%].
- Bankroll: 1000.00 → 991.85; drawdown 9.37%.

## Gate

- OK — `selected_candidate_is_not_market_only`
- ÉCHEC — `minimum_log_loss_improvement_vs_market`
- OK — `years_beating_market`
- OK — `maximum_surface_log_loss_degradation`
- OK — `minimum_settled_bets`
- ÉCHEC — `positive_roi_after_2pct_haircut`
- ÉCHEC — `positive_years`
- ÉCHEC — `positive_99pct_month_bootstrap_lower_bound`
- OK — `maximum_drawdown`

## Limites

- Les prix historiques sont des paires pré-match cohérentes, mais sans timestamp exact ni garantie d'exécution.
- Les rapprochements de cotes ont une confiance variable; aucun seuil n'a été modifié après observation des résultats.
- L'effet serve/return est fortement redondant avec le marché, Elo et le classement.
- Les résultats 2017–2026 sont tous du développement déjà exposé; ils ne prouvent aucune rentabilité future.

Même si la gate de développement passait, elle autoriserait seulement un suivi prospectif papier à partir du 3 septembre 2026. L'argent réel reste bloqué.
