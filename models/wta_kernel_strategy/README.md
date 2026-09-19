# WTA kernel — simulation prospective

Section `Stratégie WTA` de `unified_app.py`, indépendante de l’ATP et de l’ancien modèle WTA à deux bookmakers de référence.

Modèle : variante `kernel_flexible` de l’étude `kernel_matchups_2026_09_19`, paramètres inchangés. Le choix de déployer cette hypothèse en simulation intervient à la demande de l’utilisateur **après examen du diagnostic**, malgré l’échec du filtre de sélection. La rentabilité n’est pas démontrée.

Le paquet2026 est entraîné sur9001 matchs terminés antérieurs au25décembre2025, selon la fenêtre de six ans, l’embargo et la pondération de l’étude. `model.npz` contient uniquement des tableaux NumPy, pas un pickle ; chargement avec `allow_pickle=False`. `metadata.json` porte les empreintes, le diagnostic et les dates. `history.csv.gz` contient des statistiques publiques, jamais les carnets utilisateurs.

## Utilisation

1. Ouvrir `Stratégie WTA` et initialiser une bankroll fictive ou restaurer une sauvegarde de cette nouvelle stratégie.
2. Consulter les cotes françaises : appel explicite à l’API actuelle, régions `fr`, maximum huit compétitions. Aucun appel API en arrière-plan.
3. Confirmer la surface et le simple tableau principal pour chaque compétition. L’API ne permet pas de certifier ces champs ; ils ne sont pas devinés.
4. Lancer l’analyse automatique des matchs. La page distingue absence de signal et analyse impossible, donne les raisons de blocage et propose des mises respectant le budget.
5. Enregistrer explicitement une simulation, puis régler son résultat dans le carnet et télécharger une sauvegarde JSON.

Ni référence Bet365/Pinnacle, ni classement manuel requis. Les identités complètes sont rapprochées uniquement par normalisation de caractères et correspondance unique dans l’historique antérieur. Aucun rapprochement flou ; un nom non reconnu bloque le match.

Le fournisseur réémet parfois un identifiant neuf pour une joueuse déjà présente, ce qui scinde son historique et la rend non analysable. `identity_map` fusionne deux identifiants sur preuve positive seulement : même nom normalisé et même date de naissance impliquée par l’âge publié, à trois jours près ; à défaut d’âge exploitable, il faut une nationalité commune. Deux identifiants inscrits au même tableau sont deux joueuses et ne sont jamais fusionnés, ce qui couvre les homonymes. La table obtenue est figée dans `metadata.json` (`identities`, `identity_merges`) et sert d’amorce à chaque actualisation, afin qu’un fichier annuel partiel ne puisse pas élire l’identifiant réémis. Sur le flux du 19 septembre 2026, 55 identifiants sont ainsi fusionnés et plus aucune joueuse n’est scindée. Les références de prix historiques du backtest étaient Bet365 : les prix français et la comparaison entre bookmakers constituent une nouvelle expérience, pas sa reproduction prouvée.

Règle : EV estimée≥3 % après décote de2 % des gains, cotes1,30–5,00, mise0,25 % de la bankroll du début de journée, plafond2 % quotidien. Une sélection par match. Fraîcheur des cotes≤5minutes, match à plus de10minutes et au plus à7jours. Le calcul complet et les empreintes du paquet sont revérifiés à l’enregistrement.

Les statistiques entrent strictement après début de tournoi+28jours et jamais avant leur disponibilité théorique à l’instant du calcul, même pour un match futur. Cinq matchs de statistiques sur les365derniers jours sont requis par joueuse ; demi-vie180jours. Le début de tournoi le plus récent doit avoir au plus35jours : retard structurel28jours plus tolérance7jours. Ce contrôle ne prouve pas l’exhaustivité ni l’heure de publication de la source.

## Actualisation et persistance

Le bouton `Actualiser les statistiques WTA` ou `python3 scripts/refresh_wta_kernel.py` met à jour les statistiques publiques de l’année sans toucher au modèle ni au carnet. Source incomplète, identité contradictoire, perte de statistiques valides ou régression de date : refus. Une nouvelle année nécessite un audit et une reconstruction, pas un réentraînement silencieux.

`python3 scripts/prepare_wta_kernel.py` construit le paquet une première fois à partir des caches de recherche locaux gelés et des données brutes publiques. Ces gros caches ne sont pas tous déployés. Le script refuse d’écraser un paquet existant ; l’application déployée n’en a pas besoin pour calculer ou actualiser les statistiques.

Carnet : `bets/wta_kernel_strategy.sqlite3`, propriétaire par compte, ignoré par Git, format de sauvegarde `wta-kernel-paper-v1`. Les anciens carnets WTA et ATP restent accessibles dans les archives de la section ATP. Les bankrolls sont indépendantes : pas de plafond combiné entre sports. Sur Streamlit Cloud, le disque local n’est pas une sauvegarde durable.

Tests : parité des probabilités exportées, parité des profils avec l’étude historique, absence d’utilisation des données futures, dates et contextes bloquants, cotes périmées, calcul altéré, doublons, isolation des comptes, export/restauration, actualisation sans réentraînement et parcours d’interface complet avec API simulée.
