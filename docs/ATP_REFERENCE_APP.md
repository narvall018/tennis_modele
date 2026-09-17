# Section Stratégie ATP : comparaison de cotes

La navigation « Stratégie ATP » ouvre désormais `src/app/atp_reference_page.py`.
L'ancien formulaire de modèle ATP n'est plus proposé et la section « Stratégie
WTA » est retirée de la navigation. Les modules historiques restent disponibles
pour lire les archives et exécuter leurs tests, sans les réactiver dans l'app.

## Parcours

1. Créer une nouvelle bankroll fictive ou restaurer une sauvegarde de cette
   nouvelle stratégie. Aucun transfert implicite d'un ancien solde.
2. Cliquer « Scanner les opportunités ATP ». Ce clic appelle le catalogue puis
   les cotes ATP, régions France et Europe, au plus huit compétitions.
3. Lire le tableau : sélection, opérateur français, cote, score de référence,
   EV estimée et mise simulée. Les matchs non analysables sont comptés à part.
4. Enregistrer explicitement la sélection prioritaire en simulation, si désiré.
   Cela ne place aucun pari auprès d'un opérateur.
5. Régler les simulations dans le carnet après vérification du résultat final
   et télécharger régulièrement la sauvegarde JSON.

Pas de classement, de surface, de modèle annuel à reconstruire ni de cotes
Bet365 à renseigner. Pas d'appel payant au chargement, aux changements d'onglet
ou à chaque rafraîchissement. La zone de résultat revérifie sa fraîcheur toutes
les trente secondes lorsque la page est ouverte, sans rappeler l'API. Ce n'est
pas un service de surveillance lorsque l'application est fermée.

## Règle et blocages

Référence : paire Pinnacle du même match. Score = minimum des méthodes de
retrait de marge proportionnelle et puissance, moins 0,5 point de pourcentage.
Il ne s'agit ni d'une probabilité validée, ni d'une borne de confiance ; aucune
renormalisation de ce minimum. Cote d'exécution française entre 1,30 et 5,00,
EV estimée au moins 3 % après décote de 2 % sur le gain `cote − 1`.

Deux issues exactes, marges 0–8 % pour Pinnacle et 0–20 % pour l'opérateur,
cotes âgées de cinq minutes au maximum et espacées de trois minutes au maximum.
Match prévu dans dix minutes à 48 heures. Référence manquante ou trop ancienne,
marché incomplet, ambigu ou hors circuit ATP : pas de calcul autorisé.

Un seul candidat par match, au meilleur avantage théorique parmi les paires
françaises complètes. Allocation par début de match, 0,25 % du capital de début
de journée, plafond 2 % par jour Europe/Paris, arrondi inférieur en centimes.
Le total proposé respecte le disponible ; les matchs déjà enregistrés sont
exclus. Chaque enregistrement recalcule la sélection et les contrôles, vérifie
la version de la règle et réserve la mise transactionnellement. Pas de
réinvestissement des gains du jour. Résultats et règles d'abandon à vérifier
manuellement ; les simulations d'abandon sont annulées selon le scénario étudié.

## Comptes et archives

- Nouveau fichier privé : `bets/atp_reference_price.sqlite3` (ignoré par Git).
- Sauvegarde : `atp-reference-paper-v1`, stratégie `atp_reference_price_paper_v1`.
- Anciens fichiers ATP et WTA conservés, jamais fusionnés ni supprimés.
- Onglet Archives : consultation, règlement des anciennes mises, export JSON
  et restauration d'un ancien carnet dans un compte archivé vide seulement.
- Les sauvegardes anciennes et nouvelles ne sont pas interchangeables.

Sur Streamlit Cloud, le stockage local peut être perdu au redéploiement.
Conserver les exports JSON hors du serveur ; ce changement ne constitue pas
une migration vers un stockage persistant.

## Limite scientifique

Diagnostic ATP 2023–2025 : +24,38 % sur 60 paris réglés, intervalle 95 %
[−7,95 % ; +58,79 %]. Filtre de validation non franchi, passé déjà exploré,
exécution historique Bet365 non reproduite aux prix français. Les horodatages
historiques ne démontrent pas la simultanéité. Pinnacle peut être retardé ou
erroné. **Simulation uniquement, rentabilité non démontrée.**

La mise à disposition dans l'application est une expérience prospective
demandée par l'utilisateur, pas une admission scientifique rétroactive.
Les résultats ne doivent pas être additionnés à ceux de l'ancien modèle ATP.
