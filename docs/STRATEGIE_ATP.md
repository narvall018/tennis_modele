# Section « Stratégie ATP »

Cette section de `unified_app.py` est indépendante de l'ancien Elo affiché dans
« Prédictions ». Elle applique la famille **arbres récents** du dernier essai
ATP : correction non linéaire des probabilités du marché, poids temporels de
demi-vie trois ans, seuil d'EV de 2 % après décote de 2 % des gains, cotes de
1,30 à 5,00. Le modèle annuel 2026 est ajusté sur 58 271 matchs terminés avant
le 25 décembre 2025, sans ajustement des paramètres sur les résultats 2026.

## Ce que la page permet

1. Initialiser une bankroll fictive, propre au compte connecté.
2. Consulter volontairement les cotes ATP françaises via la clé Odds API déjà
   configurée, ou saisir une paire de cotes relevées chez un même opérateur.
3. Confirmer les identités, les classements et points actuels, le tournoi, la
   surface, le tour, les conditions indoor/outdoor et le format. Ces informations
   ne sont pas toutes fournies par l'API : elles ne sont pas devinées.
4. Calculer les probabilités avec le modèle étudié et voir une sélection ou le
   motif « aucun pari ». Les cotes d'un relevé API ne sont pas modifiables et le
   rapprochement des noms contrôle le nom de famille et l'initiale ; l'utilisateur
   confirme toujours l'identité exacte, ce contrôle ne résolvant pas tous les homonymes.
5. Enregistrer une simulation avec la cote, la probabilité, les horodatages et
   les empreintes du modèle et des données figés. Aucun pari n'est transmis.
6. Saisir le résultat final : gagné, perdu ou annulé. Les abandons sont annulés
   dans ce scénario. Le résultat est définitif ; ne pas saisir avant vérification.
7. Consulter bankroll, capital disponible, mises ouvertes, profit, ROI et courbe.
   Exporter un CSV ou une sauvegarde complète JSON ; restaurer cette dernière
   uniquement dans un compte sans bankroll, sans écraser ni fusionner un carnet.

La page reste une **simulation**, pas un système d'argent réel. Le diagnostic
historique +14,97 % porte sur seulement 54 paris de 2023–2025 ; son intervalle
à 95 % [−11,88 % ; +47,46 %] inclut des pertes. Le filtre de validation a échoué.
Ces données sont affichées depuis les métadonnées du calcul, et non utilisées
pour promettre un rendement. Passer des cotes Bet365 historiques aux cotes
françaises est une nouvelle expérience, pas une rentabilité déjà validée.

## Mises et comptabilité

Mise théorique : 0,25 % du capital de début de journée, plafonnée par le budget
journalier restant (2 %) et les fonds non réservés. Journée en Europe/Paris.
Les sommes sont stockées en centimes et les mises arrondies vers le bas.
Les gains du jour ne font pas augmenter la mise unitaire. Une transaction SQLite
revérifie les plafonds à chaque enregistrement, même en cas de clics simultanés.
Une paire de joueurs ne peut être enregistrée deux fois pour une même date,
quel que soit le bookmaker ou le côté sélectionné.

Le budget prospectif est consommé **successivement au moment de la décision**.
Ce n'est pas la répartition proportionnelle rétrospective de tous les paris
d'une journée du backtest : les futures sélections ne sont pas encore connues.
Le suivi ne doit donc pas être présenté comme une reproduction exacte de sa
courbe historique. Annuler un pari libère la réservation, mais ne recrée pas du
budget de décision le même jour. Les gains simulés appliquent la décote de 2 %,
ils ne représentent pas un relevé bancaire ou un paiement réel du bookmaker.

## Contrôles bloquants

- ATP masculin en simple, tableau principal uniquement : pas de WTA, doubles,
  Challenger, qualifications ou in-play.
- Match à venir dans les sept jours ; horodatages explicites, jamais dans le
  futur pour une cote et jamais de plus de 15 minutes à l'enregistrement.
- Une paire complète chez le même opérateur français pris en charge, sans
  mélange de maxima entre bookmakers ; overround entre 100 % et 120 %.
- Modèle de l'année courante, fichiers conformes aux empreintes du manifeste.
- Dernière date de l'historique au maximum sept jours avant aujourd'hui.
  Ce seuil de fraîcheur est une précaution prospective, pas une garantie que
  chaque rencontre récente ou chaque profil est complet.
- Aucun substitut Elo, aucune probabilité ou cote fabriquée en cas d'absence.

Le paquet actualisé le **16 septembre 2026** contient **71 496 matchs jusqu'au
13 septembre 2026**, soit 127 matchs supplémentaires par rapport au paquet
initial du 29 août. Le contrôle de fraîcheur est satisfait à cette date.
Le modèle annuel et les seuils n'ont pas changé. Quand les données deviennent
trop anciennes, le carnet reste accessible mais les sélections sont bloquées.
La page propose désormais **« Actualiser les données de la stratégie ATP »**.
Ce bouton télécharge uniquement la saison courante chez Tennis-Data et
TennisMyLife, contrôle les identités et les dates, conserve les années passées
et le modèle annuel, puis recharge les caches de la page. Il fonctionne avec
le paquet livré, sans dépendre des archives de recherche locales. Il ne touche
pas à la bankroll. Équivalent local :

```bash
python3 scripts/refresh_tennis_strategy.py
```

Une panne fournisseur, des données encore anciennes, des matchs disparus ou
des dates modifiées empêchent la publication. Le carnet et l'ancien paquet
sont conservés. Le 16 septembre, après des erreurs HTTP 503 puis 404 sur les
anciennes URL, l'index officiel a révélé un nouveau préfixe de dossier. Le
téléchargement suit désormais les liens réellement publiés sur cet index,
pour l'ATP et la WTA, aux formats XLS/XLSX. Aucun préfixe opaque n'est codé en dur.
Une réponse 404 ne prouve pas qu'une saison n'est pas publiée ; les erreurs
403/429 sont distinguées des liens introuvables. Les dates TennisMyLife seules
ne remplacent pas les dates de matchs Tennis-Data du protocole.

Si un environnement Conda `base` affiche l'avertissement SciPy « binary
incompatibility », il s'agit d'un problème distinct de l'adresse de téléchargement.
Les imports scientifiques et cette actualisation ont été vérifiés ici avec
`/usr/bin/python3` (Python 3.12), sans cet avertissement. Ne pas masquer le warning
ni réinstaller à l'aveugle dans `base` ; utiliser un interpréteur vérifié ou un
environnement dédié avec les dépendances du projet.

L'actualisation complète des tables locales reste disponible :

```bash
python3 scripts/update_tennis_data.py
python3 scripts/prepare_tennis_strategy.py
```

Ces actions existent également dans « Mise à jour ». La seconde conserve le
modèle annuel déjà exporté et reconstruit seulement l'historique de service,
forme et contexte depuis les données locales. Elle ne télécharge rien et ne
réentraîne pas sur les nouveaux résultats. Si le fournisseur reste en retard,
la fraîcheur reste bloquante : ne pas modifier la date pour forcer le calcul.
Le calcul des descripteurs rejoue les formules historiques en ne conservant
que les statistiques glissantes des deux joueurs concernés. La parité avec le
constructeur historique est testée. Le calcul est explicite et mis en cache,
sans nouvel entraînement à chaque affichage.

Les désaccords entre sources sur des matchs passés sont conservés et signalés
dans les métadonnées. Pour l'historique prospectif, les lignes en conflit
n'alimentent ni les résultats, ni les statistiques de service, ni la charge de
jeu. Aucun vainqueur n'est choisi pour améliorer une performance. Les sources
et le cache d'entraînement des recherches précédentes ne sont pas modifiés.

## Stockage et déploiement

- Modèle natif XGBoost : `models/tennis_strategy/booster.ubj`, sans pickle.
- Historique sportif public et protocole : même dossier, avec empreintes dans
  `metadata.json`. Le paquet est livré pour que Cloud n'ait pas à retrouver les
  gros caches de recherche lors du chargement.
- Carnet privé : `bets/tennis_strategy.sqlite3`, exclu de Git avec ses fichiers
  auxiliaires. Aucune clé API, authentification ou bankroll n'est ajoutée au dépôt.
- Isolation des comptes, dédoublonnage, transaction de budget et règlements
  définitifs ; aucune modification des anciens carnets.

**Attention Streamlit Cloud : le disque n'est pas une sauvegarde durable.**
Un redéploiement peut faire disparaître la base locale. Télécharger régulièrement
le JSON et le restaurer au besoin dans un compte vide. Cette version n'envoie
pas de données personnelles sur la branche GitHub de suivi et ne configure pas
un service de stockage externe. Une sauvegarde restaurée reste une saisie
personnelle, pas une preuve indépendante de performance.

Le modèle utilise le format natif de [sauvegarde XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html).
Une reconstruction initiale complète nécessite les caches archivés de recherche ;
le rafraîchissement d'un paquet déjà livré n'en a pas besoin. Le passage à 2027
est volontairement bloqué jusqu'à un nouvel export annuel audité, sans changement
opportuniste des règles selon les gains du carnet.

## Vérification

```bash
PYTHONPATH=predictor_ufc OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q
```

La suite complète suppose les archives de recherche locales (non publiées dans
Git). Dans une copie de déploiement sans ces archives, les tests d'intégration
historiques UFC/phase 4 ne peuvent pas lire leurs fichiers. Les contrôles de la
nouvelle section n'en dépendent pas :

```bash
python3 -m pytest -q tests/test_tennis_strategy_app.py tests/test_tennis_strategy_ui.py
```

Les tests couvrent les gardes de données et de prix, les statistiques strictement
antérieures, le modèle natif livré, la confidentialité entre comptes, les clics
concurrents, le non-réinvestissement intrajournalier, les doublons, les règlements
et la sauvegarde/restauration. Avec Streamlit installé, deux tests supplémentaires
parcourent réellement la création de bankroll, le calcul, l'enregistrement, et
le blocage des données périmées. Aucune clé ni appel API réel n'est nécessaire.
