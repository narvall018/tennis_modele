# Stratégie WTA — simulation

Dans la navigation de l’application, ouvrir **Stratégie WTA** et initialiser
une bankroll fictive. Le carnet WTA est indépendant de l’ATP : aucun capital,
pari ou plafond quotidien n’est partagé. Aucun pari réel n’est envoyé.

La présentation reprend celle de l’ATP : mêmes cinq compteurs, budget quotidien,
onglets **Analyser un match** et **Carnet et sauvegarde**, formulaire à deux côtés,
boutons **Calculer selon la stratégie figée** et **Enregistrer ce pari en simulation**.
Le tournoi peut être choisi dans l’historique ou saisi. Les quatre cotes de référence
propres au modèle WTA sont regroupées dans un bloc dédié obligatoire ; les champs
spécifiques au modèle ATP qui ne servent pas au WTA ne sont pas demandés.

## Modèle et limites

Le candidat `trees_recent` de la recherche du 16 septembre 2026 utilise des
arbres peu profonds, des statistiques des jours précédents, la surface et
deux paires de cotes de référence : Bet365 et Pinnacle. Le modèle annuel 2026
est entraîné sur les matchs terminés strictement avant le 25 décembre 2025,
avec pondération de récence de trois ans. Il n’est pas réentraîné lors des
actualisations de résultats. Un nouvel audit est nécessaire au changement d’année.

Diagnostic exploratoire 2023–2025 : 114 paris réglés, ROI après décote de 2 %
de +14,61 %, intervalle 95 % [−14,79 % ; +36,91 %]. Le filtre de validation
échoue, comme celui de réglage. Cet historique a déjà servi à la recherche :
ce n’est pas une validation indépendante. Le transfert aux cotes françaises
est une nouvelle expérience, pas une rentabilité démontrée.

## Analyse d’un match

- Simple WTA tableau principal uniquement : ni qualifications, ni doubles,
  ni WTA 125. Confirmer les identités, leur ordre, la surface et les classements.
- Six cotes authentiques sont nécessaires : les deux côtés chez Bet365,
  Pinnacle et le bookmaker français choisi. Les références ne sont pas des
  lieux où l’application invite à parier. Aucune substitution n’est permise.
- Le bouton de consultation utilise explicitement le quota Odds API pour les
  régions `eu,fr`. Il peut récupérer les paires françaises et Pinnacle.
  Bet365 reste à saisir manuellement. Sans ces références, aucun calcul.
- Les trois relevés doivent dater de moins de quinze minutes, être séparés
  d’au plus cinq minutes, et précéder le début du match. La saisie manuelle
  exige de confirmer que les prix viennent d’être observés.
- Sélection théorique si EV estimée ≥ 2 % après une décote de 2 % sur le gain,
  cote comprise entre 1,30 et 5. Cette EV dépend du modèle et n’est pas garantie.

## Carnet

Mise maximale de 0,25 % de la bankroll au début de la journée Europe/Paris,
plafond quotidien de 2 %, sans réinvestir les gains du jour. Les contraintes
sont revérifiées à l’enregistrement. Les mises ouvertes sont réservées ; les
résultats sont saisis manuellement après le début prévu. Annuler en cas d’abandon
ou walkover, conformément au scénario historique. Les gains simulés sont décotés.

Exporter régulièrement le JSON : le stockage local Streamlit Cloud peut être
perdu au redéploiement. Une restauration WTA n’est autorisée que dans un compte
vide ; une sauvegarde ATP est refusée. Le fichier `bets/wta_strategy.sqlite3`
est privé et exclu de Git.

## Actualisation

Le paquet portable `models/wta_live_strategy` comprend le modèle natif, les
paramètres, les preuves et l’historique public. Aucun cache de recherche n’est
nécessaire pour l’utiliser ou actualiser l’année courante :

```bash
python3 scripts/refresh_wta_strategy.py
```

La même opération est disponible depuis la page lorsque les données sont
trop anciennes, et depuis **Mise à jour**. Elle consulte l’index officiel
Tennis-Data, préserve les années antérieures et le modèle, refuse les pertes
de matchs et revient à la génération précédente si la publication échoue.
Les données du jour et du futur sont exclues. Un historique de plus de sept
jours bloque les nouveaux calculs, sans bloquer l’accès au carnet.

`scripts/prepare_wta_strategy.py` est uniquement l’export initial depuis les
archives locales de recherche vérifiées ; il refuse d’écraser un paquet
existant. Il ne constitue pas une commande d’actualisation quotidienne.
