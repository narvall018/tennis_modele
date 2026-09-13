# Collecte prospective, et le coût de la France mesuré à marque constante

Tout ce dépôt rejoue le même historique, et chaque conclusion porte la même
réserve : les années ont déjà servi. La seule sortie est de constituer un jeu de
données que personne n'a vu — des prix horodatés relevés avant les matchs.
`scripts/run_prospective_collector.py` fait exactement ça, sous budget de quota.

Premier relevé : **900 lignes, 13 événements, 24 opérateurs**, sur les trois
marchés tennis — résultat sec, totaux de jeux et handicaps. C'est la première
donnée de ce dépôt qui ne soit pas un ré-usage.

## Surmarge par opérateur, tennis

| opérateur | résultat sec | totaux |
|---|---:|---:|
| Betfair (exchange) | **0,91 %** | — |
| Matchbook | 1,27 % | 7,39 % |
| Pinnacle | 3,43 % | **3,90 %** |
| BetOnline | 3,75 % | 4,74 % |
| *médiane internationale* | *5,36 %* | *7,25 %* |
| **Betclic** | **8,02 %** | — |
| **PMU** | 8,96 % | **11,86 %** |
| **Unibet France** | 9,89 % | — |
| **Netbet** | 9,94 % | **17,34 %** |
| **Winamax** | **11,11 %** | — |

Les cinq opérateurs agréés occupent les cinq dernières places sur vingt-trois.

## Le chiffre qui clôt le sujet

Deux marques présentes des deux côtés de la frontière, **même technologie, même
sport, même semaine** :

| opérateur | France | ailleurs | écart |
|---|---:|---:|---:|
| Winamax | 11,11 % | 5,65 % *(Allemagne)* | **+5,46 pt** |
| Unibet | 9,89 % | 5,37 % *(Suède, Pays-Bas)* | **+4,52 pt** |

**La même entreprise facture deux fois plus cher en France.** Ce n'est pas un
choix commercial local ni une incompétence de pricing : c'est le prélèvement sur
les mises, répercuté sur la cote.

C'est la mesure directe de ce que les rapports précédents appelaient « le mur
juridictionnel ». Il ne s'agissait pas d'une impossibilité technique ni d'un
marché plus efficient : il s'agit d'environ **cinq points de fiscalité** qui
s'ajoutent à chaque pari, et qui dépassent à eux seuls tout avantage jamais
mesuré dans ce dépôt.

Pour mémoire, l'échelle d'exécution chiffrait le seuil de rentabilité d'un gros
favori à **4,62 % de surmarge**. Les cinq opérateurs français sont entre 8,02 %
et 11,11 %. L'écart n'est pas rattrapable par un modèle : il est plus grand que
la marge totale d'un book international.

## Les totaux tennis, enfin observables

Aucune archive de cotes de totaux tennis n'existe dans ce dépôt : c'était la
raison pour laquelle ce marché n'avait jamais été testé. La collecte est
lancée. Les premiers relevés donnent Pinnacle à **3,90 %** et PMU à **11,86 %**,
soit un rapport de trois. Il faudra plusieurs mois de relevés, puis les
résultats, pour dire si la ligne est battable — et ce sera alors une preuve, pas
un ré-usage.

## Exploitation

```bash
python3 scripts/run_prospective_collector.py collect --markets h2h totals spreads
python3 scripts/run_prospective_collector.py summary
```

Le script tient un registre de quota (`models/prospective/quota_ledger.json`) et
refuse de dépasser le budget quotidien fixé, le palier gratuit donnant 500
requêtes par mois. La clé se lit depuis un `.env` local, que `.gitignore`
couvre — le dépôt étant public, rien de sensible n'y est écrit.
