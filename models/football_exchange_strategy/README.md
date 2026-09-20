# Stratégie football — biais du favori sur exchange (papier)

`real_money_authorised: false`. Ce paquet accumule des preuves prospectives ; il n'en
apporte aucune.

## Ce que la règle exploite

Un seul fait mesuré : sur les 90 824 matchs des saisons **2012-2023** (22 ligues, prix de
clôture Pinnacle, devig proportionnel, overround médian 2,67 %), le prix sous-estime un
favori marqué. Le biais est monotone sur 270 000 issues.

Mais la règle n'utilise pas le biais mesuré — elle utilise sa **borne basse à 5 %** par
bootstrap hebdomadaire (les matchs d'un même week-end se dénouent ensemble) :

| Bande | n | Biais | Borne basse | Retenue |
|---|---|---|---|---|
| 0,50–0,55 | 10 867 | +0,52 pt | **−0,31 pt** | non |
| 0,55–0,60 | 8 301 | +0,29 pt | **−0,61 pt** | non |
| 0,60–0,65 | 5 690 | +1,21 pt | +0,14 pt | non |
| 0,65–0,70 | 4 021 | +1,38 pt | +0,15 pt | non |
| **0,70–0,75** | 2 814 | +3,85 pts | **+2,42 pts** | **oui** |
| **0,75–0,85** | 3 308 | +3,07 pts | **+1,99 pts** | **oui** |
| 0,85–1,01 | 807 | +2,15 pts | +0,45 pt | non |

Une bande dont la borne basse n'atteint pas **1 point entier** est écartée : +0,14 pt ne se
distingue pas de l'absence de biais. C'est pourquoi la règle ne couvre que 0,70–0,85.

## Pourquoi l'exchange, et seulement lui

Le biais tolère entre 2,9 et 4,1 points de marge. Mesuré le 20/09/2026 sur la Ligue 1 :

| Lieu | Marge médiane |
|---|---|
| Betfair exchange | **0,99 %** |
| Books français | 7,87 % |

Aucun bookmaker joignable depuis la France ne facture assez peu. `validate_fixture` refuse
donc explicitement tout opérateur hors exchange, et refuse aussi un prix dont l'overround
dépasse 2 % — à ce niveau ce n'est plus un prix d'exchange.

## La règle figée

Hachée dans `rule_sha256` ; toute modification fait échouer `load_bundle`.

- Issue : uniquement le favori des trois. Le nul est refusé explicitement.
- Probabilité de référence dans [0,70 ; 0,85], devig proportionnel du prix exchange.
- Correction : borne basse de la bande, jamais l'estimation ponctuelle.
- Commission retenue : **5 %**, le bout défavorable de la fourchette 2–5 %.
- Espérance minimale après commission : **+1 %**.
- Cote entre 1,15 et 1,60. Cotes de moins de 5 minutes. Match à plus de 10 minutes et au
  plus à 7 jours.

En pratique la règle est serrée : au relevé du 20/09/2026, sur 219 marchés exchange, deux
signaux seulement, et le même marché à 1,35 au lieu de 1,36 était déjà refusé.

## Limites, explicitement

1. Le biais est mesuré sur des prix de **clôture Pinnacle** ; le scanner lit des prix
   **exchange avant clôture**. En développement la ligne ne bouge pas sur les favoris
   (0,6961 → 0,6965), mais ce report reste une hypothèse non vérifiée.
2. Un prix d'exchange est une **offre de profondeur inconnue** : la cote affichée peut ne
   pas être disponible pour la mise proposée.
3. Les saisons 2012-2023 ont **déjà servi à chercher**. Ce ne sont pas des preuves.
   Le holdout football 2024-2026 n'a pas été ouvert.
4. La meilleure cellule ROI du balayage (favoris ≥ 0,70 chez Pinnacle, +1,22 %) était un
   **pic local** dont l'intervalle touchait zéro. C'est le biais qui porte cette règle,
   pas cette cellule.

## Usage

```bash
python3 scripts/prepare_football_exchange.py   # refige la règle depuis les données
```

Le carnet `src/app/football_exchange_ledger.py` est isolé dans son propre fichier SQLite
et recalcule chaque sélection avant de l'accepter.
