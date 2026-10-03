Deux nouvelles entrées sont disponibles dans `unified_app.py` : **Stratégie Football** et **Stratégie UFC**. Les calculs se lancent automatiquement, indépendamment de la création d’un carnet. Chaque sport conserve une bankroll de simulation privée par compte.

- Football : marché 1X2 complet, référence Pinnacle corrigée de la marge, minimum des scores proportionnel et puissance moins 0,5 point, EV estimée après décote de 2 % ≥ 3 %. Une seule sélection par rencontre, nul compris.
- UFC : adaptation expérimentale d’un modèle Markov à sept états, avec taux de frappe, défense, amenées au sol, contrôle, KO et soumissions calculés à partir des résultats antérieurs. Le modèle de jugement est ajusté sur les caractéristiques antérieures à chaque carte. Le même favori doit dépasser 26 % d’EV estimée après décote dans les scénarios de trois **et** cinq rounds, la durée n’étant pas fournie dans les cotes. Au moins deux combats statistiques par combattant. Les autres organisations MMA sont exclues par rapprochement avec le programme officiel UFC.

Le rendement historique exploratoire du football (+14,50 %, 499 paris en 2021–2024) utilise des meilleurs prix anonymes dont l’accessibilité n’est pas prouvée. Le rendement UFC publié (+15,27 %, 144 paris en 2018) concerne le modèle bayésien original de [Holmes, McHale et Zychaluk](https://doi.org/10.1016/j.ijforecast.2022.01.007), qui n’est pas reproduit ici. Aucun des deux chiffres ne valide la rentabilité actuelle de ces pages.

L’actualisation des prix se fait à l’ouverture et une fois par heure quand la page est active. La fraîcheur est revérifiée toutes les 60 secondes et à l’enregistrement. Les cotes expirent après cinq minutes ; un scan manuel relance la collecte immédiatement. Les deux sections partagent un cache et un plafond de 12 appels de cotes par jour, avec une réserve de 20 crédits sur le quota du fournisseur. Les identifiants précis des bookmakers français et de la référence sont envoyés au fournisseur : aucune publicité, offre ou lien commercial n’est ajouté. Les matchs sont limités aux 14 prochains jours et ne sont plus sélectionnables dix minutes avant le début.

Les résultats UFC terminés et les trois prochaines cartes sont vérifiés gratuitement sur UFCStats une fois par jour, ou avec le bouton d’actualisation. Un échec de vérification conserve les données précédentes sans prolonger leur fraîcheur. Les calculs UFC refusent une vérification de plus de 48 heures. Le paquet JSON est portable et vérifié par empreinte SHA-256 ; il ne dépend pas de fichiers scikit-learn sérialisés.

Le workflow `.github/workflows/value_methods.yml` actualise quotidiennement les données publiques à 06:40 UTC et effectue un scan des deux méthodes. Une exécution manuelle est également disponible dans GitHub Actions. La clé peut être fournie par le secret GitHub `ODDS_API_KEY`. Dans l’application, la configuration existante de cette clé est conservée. La commande locale est :

```bash
python scripts/refresh_value_methods.py
```

Les carnets `bets/value_football.sqlite3` et `bets/value_ufc.sqlite3`, les caches et les verrous sont exclus de Git. Les simulations réservent réellement leur mise dans la bankroll fictive, interdisent les doublons et respectent 0,25 % du capital de début de journée par sélection et 2 % par jour. Les résultats sont saisis après confirmation et les sauvegardes JSON sont propres à chaque sport. Les anciens carnets restent accessibles dans leurs sections. Sur un serveur éphémère, télécharger les sauvegardes avant un redéploiement.
