# UFC et football — pistes de microstructure, 20 septembre 2026

**Verdict : aucune stratégie exécutable. Mais le football livre le biais le mieux
quantifié du projet, et on sait maintenant exactement de combien il manque.**

Suite de `RAPPORT_UFC_RECHERCHE_2026_09_19.md`. Les pistes retenues ici sont
*structurellement* nouvelles : d'autres marchés et la structure du prix lui-même,
pas d'autres découpes des mêmes données.

Développement football : saisons 2012-2023, 89 987 matchs, 22 ligues.
**Le holdout football 2024-2026 n'a pas été touché.**

## UFC — deux pistes neuves, deux échecs

### Variables de série (source secondaire, 5 940 combats appariés)

Séries de victoires/défaites en cours, plus longue série, rounds disputés, combats
de titre, distribution des méthodes de victoire, combat en 5 rounds, genre.
Aucune n'était dans le jeu de descripteurs des phases 1-3.

Contrôle anti-fuite : corrélation maximale avec le résultat +0,136
(`current_win_streak`), plausible et non suspecte.

| Modèle | Gain log-loss | t |
|---|---|---|
| Prix + descripteurs | −0,00002 | −0,01 |
| Prix + descripteurs + **nouvelles** | **+0,00007** | **+0,04** |
| Prix + nouvelles seules | −0,00135 | −1,19 |

Les séries corrèlent bien avec le résultat, mais **l'information est déjà dans le
prix**. Rien à ajouter.

### Microstructure UFC : impossible à développer

Les 70 294 relevés multi-books horodatés commencent le 02/08/2025 — entièrement
dans le holdout. Aucun échantillon de développement n'existe pour le line shopping
ou le mouvement de ligne.

## Football — sept pistes de microstructure

### M1. La clôture résume-t-elle l'ouverture ?

| Issue | coef clôture | coef ouverture |
|---|---|---|
| Domicile | +1,051 (t=+20,6) | +0,013 (t=+0,3) |
| **Nul** | +0,920 (t=+8,5) | **+0,288 (t=+2,5)** |
| Extérieur | +1,098 (t=+20,4) | −0,025 (t=−0,5) |

La clôture est une statistique exhaustive pour domicile et extérieur. **Seul le nul
garde un résidu d'information dans l'ouverture** — modeste, et sur un marché où la
marge est la plus lourde.

### M2. Sur-réaction au mouvement de ligne : non

Résidus par quintile de mouvement : −0,76 / +0,26 / +1,21 / +0,26 / −0,36 points.
Non monotone — pas de signal directionnel exploitable.

### M3. Qui est le plus affûté ?

Log-loss sur 37 771 matchs communs : **Pinnacle 0,99861** < moyenne marché 0,99928
< B365 0,99970. Pinnacle bat B365 de 0,0011, uniformément dans les quatre quartiles
de dispersion. Le désaccord entre books ne prédit rien.

### M5. Le nul

Gradient monotone : les nuls peu probables sont surcotés (−1,18 pt), les nuls
probables sous-cotés (+1,02 pt). Même forme que le biais favori-outsider.

### M7. Le biais favori-outsider — le résultat principal

| Probabilité | n | Implicite | Réalisé | Écart | ROI Pinnacle |
|---|---|---|---|---|---|
| 0,00–0,10 | 6 147 | 0,0719 | 0,0563 | **−1,56 pt** | −25,61 % |
| 0,10–0,20 | 29 097 | 0,1591 | 0,1537 | −0,55 pt | −6,42 % |
| 0,20–0,30 | 105 043 | 0,2609 | 0,2573 | −0,37 pt | −4,33 % |
| 0,30–0,45 | 78 945 | 0,3603 | 0,3625 | +0,22 pt | −2,16 % |
| 0,45–0,60 | 34 388 | 0,5138 | 0,5174 | +0,36 pt | −2,12 % |
| **0,60–1,00** | 16 341 | 0,6974 | 0,7182 | **+2,08 pt** | **−0,13 %** |

Parfaitement monotone sur 270 000 issues. Le biais est réel et il **grandit avec le
statut de favori**.

## Une erreur que j'ai commise et corrigée

J'ai d'abord mesuré « parier les favoris au prix d'ouverture rapporte +1,39 % contre
−0,13 % à la clôture ». C'était un **look-ahead** : je sélectionnais les favoris avec
la probabilité de *clôture* puis je payais au prix d'*ouverture*.

Corrigé — sélection et prix au même instant — l'effet disparaît entièrement :

| Seuil | ROI ouverture | ROI clôture |
|---|---|---|
| 0,60 | −0,66 % | −0,13 % |
| 0,70 | +0,37 % | +1,22 % |

La ligne ne bouge pas sur les favoris : probabilité moyenne 0,6961 à l'ouverture,
0,6965 à la clôture. **Il n'y a pas de valeur de ligne de clôture à capter ici.**

## La meilleure cellule, et les trois défenses

Favoris p ≥ 0,70 au prix de clôture Pinnacle : **ROI +1,22 %, t = 1,99, n = 6 733**,
IC95 par semaine **[−0,06 %, +2,49 %]**.

1. **Déplacer le seuil** : 0,66 → +0,43 % · 0,68 → +0,75 % · **0,70 → +1,22 %** ·
   0,72 → +0,97 % · 0,74 → +0,69 %. Pic local exactement au seuil retenu — signature
   d'une sélection, pas d'un effet.
2. **Par saison** : 9/12 positives, mais 2020 à −3,80 %.
3. **Par ligue** : 9/13 positives, de +9,73 % (Turquie) à −3,72 % (Pays-Bas).

L'intervalle touche zéro et le seuil est instable. Même forme que la cellule tennis
morte en septembre.

## Le chiffre qui conclut : l'échelle de marge

| Seuil | Biais | **Marge maximale tolérable** |
|---|---|---|
| 0,60 | +2,08 pts | **2,89 %** |
| 0,65 | +2,56 pts | **3,36 %** |
| 0,70 | +3,30 pts | **4,10 %** |

| Lieu | Overround | Accessible depuis la France |
|---|---|---|
| Pinnacle clôture | **2,67 %** | non (pas de licence ANJ) |
| Betfair exchange | ~1–2 % | données inexistantes avant 2024 |
| Books français | **6–11 %** | oui |

**Le biais vaut entre 2,9 et 4,1 points de marge. Pinnacle en prend 2,67 — d'où un
résultat qui oscille autour de zéro. Tout ce qui est joignable depuis la France en
prend 6 à 11.**

C'est la même conclusion que le tennis, l'UFC et l'arbitrage, mais pour la première
fois avec le chiffre exact : il ne manque pas un meilleur modèle, il manque
**3 à 7 points de marge**.

## Conclusion

Le biais favori-outsider du football est réel, monotone sur 270 000 issues, et
robuste. C'est le résultat le plus solide produit par ce projet. Il reste
inexploitable pour une raison qui n'est pas statistique : aucun lieu d'exécution
légalement accessible ne coûte assez peu.

Les deux seules ouvertures restantes sont des problèmes de **données**, pas de
modèle : une série longue de prix d'exchange (inexistante avant 2024) et un accès
à un opérateur sous 4 % de marge.
