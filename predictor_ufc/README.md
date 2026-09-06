# UFC — pipeline de recherche rigoureux

La base est à jour jusqu'à **UFC Fight Night: Hooker vs. Parnasse (5 septembre 2026)**.
Le verdict actuel est **NO BET** : aucun système de mise rentable n'est démontré.

## Résultat honnête

- Phase 1, validation 2022–2024 : 380 paris, ROI **+0,78 %**, mais IC95 % **[-11,30 % ; +14,05 %]**.
- Phase 2 (stats, boosting, XGBoost et petit réseau neuronal) : le meilleur système fait **−0,60 %** sur 313 paris de confirmation.
- Son intervalle bootstrap ajusté à 97,5 % est **[-12,84 % ; +11,90 %]**.
- Phase 3 (classements UFC strictement pré-combat) : **−0,51 %** sur 1 297 paris de développement, IC99 % **[-7,79 % ; +6,65 %]**.
- Le modèle avec classements dégrade la log-loss : 0,61290 contre 0,61211 pour le modèle structurel sans classements.
- Le résultat est donc compatible avec le hasard.
- Le holdout horodaté 13/09/2025–29/08/2026 n'a pas été ouvert et reste disponible pour une future hypothèse réellement validée.
- L'application bloque les recommandations et l'enregistrement de nouvelles mises.

Le détail est dans [RAPPORT_RIGOUREUX_2026.md](RAPPORT_RIGOUREUX_2026.md).

## Données

- 8 846 combats, dont 8 692 avec vainqueur exploitable.
- 788 événements et 4 592 profils de combattants.
- Statistiques historiques détaillées UFCStats, complétées directement jusqu'au dernier événement terminé.
- 106 807 snapshots de cotes appariés à 6 563 combats.
- 316 combats récents disposent d'une ligne sélectionnée avec bookmaker et timestamp pré-combat.
- Historique hebdomadaire de classements depuis 2013 : 2 008 combats avec au moins un combattant classé et 1 254 avec les deux.
- 199 combats seulement ont au moins deux snapshots distincts utilisables pour un mouvement de cote; 12 seulement couvrent J−14 à J−1.
- 5 329 combats ont des cotes par méthode, toutes `legacy_unverified`; elles ne sont pas admissibles comme preuve économique.
- Les 6 040 anciennes lignes 2010–2024 restent marquées `legacy_unverified` : bookmaker et instant de capture inconnus.

Les hashes, versions et limites sont enregistrés dans `data/rigorous/quality/data_manifest.json`.

## Contre-expertise des cotes historiques

```bash
python3 run_rigorous_pipeline.py cross-check-odds
```

La commande télécharge une **seconde** compilation publique du même marché
historique (Ultimate UFC Dataset, d'origine BestFightOdds), la rapproche des
mêmes combats et compare les deux prix combat par combat. Elle n'améliore aucune
qualité temporelle : la seconde source n'est pas horodatée non plus.

Le résultat est net et il compte : sur les **5 957** combats `legacy_unverified`
comparables, les deux compilations donnent des probabilités identiques au bit
près (écart médian 0, verdict `sources_are_not_independent`). Les 6 040 lignes
qui portent tout l'échantillon économique d'avant 2025 n'ont donc **qu'une seule
origine** et restent invérifiables ; elles ne sont pas confirmées par une source
indépendante, contrairement à ce qu'un second jeu de données pouvait laisser
croire. Sur les 179 combats où la ligne primaire est horodatée (Pinnacle /
BetOnline), l'écart médian est de 2,2 points de probabilité, c'est-à-dire l'écart
normal entre deux bookmakers — la comparaison distingue donc bien deux sources
réellement différentes quand elles le sont.

Rapport : `data/rigorous/quality/odds_cross_check.json`. La contre-expertise
identifie aussi 518 combats cotés uniquement par la source secondaire ; ils sont
publiés séparément et conservent le même grade non vérifié.

## Marché méthode de victoire — exploré, puis fermé

```bash
python3 run_rigorous_pipeline.py method-market
```

C'était le seul marché UFC des données que ce dépôt n'avait jamais modélisé :
4 826 combats de 2012 à 2024 avec les six issues cotées (KO, soumission,
décision, pour chaque combattant). Le marché est cohérent — la probabilité de
victoire reconstruite depuis les props corrèle à 0,979 avec le moneyline.

**Le marché porte un vrai biais.** Il sous-évalue les décisions et surévalue les
finitions, de façon constante :

| Issue | Prob. marché | Fréquence réelle | Biais |
|---|---:|---:|---:|
| KO | 17,35 % | 16,37 % | −0,98 pt |
| Soumission | 10,82 % | 8,88 % | −1,94 pt |
| **Décision** | **21,83 %** | **24,75 %** | **+2,92 pt** |

Le biais sur les décisions est positif **13 années sur 13**. C'est le phénomène
le plus régulier trouvé dans tout ce projet.

**Il ne suffit pourtant pas.** L'overround des six issues est de **1,2228**, cinq
fois celui du moneyline. Parier toutes les décisions rend **−9,45 %** sur 9 652
paris, avec une seule année positive sur treize : le biais couvre à peine plus de
la moitié de la marge.

**Et rien ne permet de choisir où il suffirait.** Une sonde de discrimination,
limitée aux années ≤ 2018, entraîne des modèles sur les 4 856 combats dont on
connaît l'issue et les compare à la probabilité de décision du marché :

| Modèle | Gain de log-loss vs marché |
|---|---:|
| logistique + marché | −0,00099 |
| logistique | −0,01905 |
| gradient boosting | −0,03102 |
| gradient boosting + marché | −0,03398 |

Aucun ne bat le marché ; le meilleur n'apporte rigoureusement rien. Le marché
sait déjà tout ce que codent les statistiques de frappe, de lutte et de
résistance de ce dépôt. Verdict : `AVENUE_FERMEE`.

Rapport : `data/rigorous/quality/method_market_analysis.json`. Les prix props
restent `legacy_unverified` : même un résultat positif n'aurait pas été une
preuve économique.

## Reproduire

Depuis ce dossier :

```bash
python3 run_rigorous_pipeline.py update-data
python3 run_rigorous_pipeline.py cross-check-odds
python3 run_rigorous_pipeline.py method-market
python3 run_rigorous_pipeline.py research
python3 run_rigorous_pipeline.py final-holdout
python3 run_rigorous_pipeline.py challengers
python3 run_rigorous_pipeline.py challenger-holdout
python3 run_rigorous_pipeline.py phase3
python3 run_rigorous_pipeline.py phase3-holdout
```

Les commandes `*-holdout` refusent automatiquement d'ouvrir le holdout si la gate
précédente échoue. Les protocoles sont dans `rigorous_protocol.json`,
`challenger_protocol.json` et `phase3_protocol.json`.

## Collecte prospective

Le collecteur append-only conserve le flux MMA brut sans prétendre que chaque
événement est UFC. Après avoir fourni sa propre clé The Odds API :

```bash
export UFC_ODDS_API_KEY="..."
python3 run_rigorous_pipeline.py collect-odds
```

Les événements restent `UNVERIFIED_MMA_EVENT` jusqu'à un appariement UFC explicite.
Le calendrier des snapshots et les champs manuels sourcés (pesée, short notice,
changement de camp, blessure déclarée) sont figés dans `prospective_protocol.json`.

## Règle de mise étudiée, non autorisée

- Une seule sélection maximum par combat.
- Edge minimal retenu sur développement : 5 %.
- Cotes décimales entre 1,25 et 5,00.
- Kelly 1/8, plafonné à 0,5 % de bankroll par pari et 3 % par événement.
- Toutes les mises d'un même événement utilisent la bankroll au début de l'événement.
- Aucun minimum de mise forcé.

Cette règle est documentée à des fins de recherche. Elle ne doit pas être utilisée avec de l'argent réel tant que le verrou indique `REJECTED_NO_BET`.

## Application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Les probabilités peuvent servir au paper trading. Elles ne constituent pas une garantie de gain ni une stratégie validée.
