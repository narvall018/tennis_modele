# Recherche UFC — 19 septembre 2026

**Verdict : aucune stratégie rentable démontrable. Le holdout intact reste fermé.**

Quatrième passage sur l'UFC après les trois phases rejetées de `data/rigorous/reports/`.
Environ 95 tests, pré-enregistrés avant inspection des résultats, sur les données de
développement uniquement.

## Ce qui plafonne l'exercice avant tout calcul

**Le holdout intact ne contient que 306 combats** (2025-09-13 → 2026-08-29). Écart-type du
ROI sur 306 paris à cote 2,0 : 5,7 points. Il faudrait donc un ROI réel de **+11,2 %** pour
le distinguer de zéro à 95 % — et encore, en pariant sur la totalité des combats. Aucun edge
réaliste n'est mesurable là.

**Les cotes historiques ont une seule origine.** Sur les 6 136 combats recoupés entre source
primaire et secondaire, **5 957 ont des probabilités identiques à 1e-9 près**, écart médian
exactement 0,00000. Ce ne sont pas deux sources indépendantes qui concordent, c'est la même
donnée comptée deux fois. Seules **306 cotes sur 6 346** sont horodatées avant l'événement,
toutes postérieures au 02/08/2025.

**Le line shopping est indéveloppable.** Les 70 294 relevés multi-books horodatés commencent
au 02/08/2025 : ils sont entièrement dans le holdout. Aucun échantillon de développement
n'existe pour cette piste.

## A. Efficience du marché, sans modèle

| Bande de prix | n | Implicite | Réalisé | Écart | ROI |
|---|---|---|---|---|---|
| 0,00–0,20 | 251 | 0,1607 | 0,1036 | **−5,71 pts** | −37,24 % |
| 0,20–0,35 | 1202 | 0,2842 | 0,2546 | −2,97 pts | −14,83 % |
| 0,35–0,50 | 1513 | 0,4203 | 0,4243 | +0,41 pt | −2,10 % |
| 0,50–0,65 | 1610 | 0,5757 | 0,5627 | −1,29 pt | −5,71 % |
| 0,65–0,80 | 1208 | 0,7157 | 0,7310 | +1,52 pt | −1,29 % |
| 0,80–1,00 | 256 | 0,8382 | 0,8555 | +1,72 pt | −1,36 % |

Le biais favori-outsider est net et va dans le sens attendu. Les favoris sont réellement
sous-cotés de 1,5 à 1,7 point — mais à −1,3 % de ROI, **le biais ne franchit pas la marge**.
Pari en aveugle : tout outsider −7,47 %, tout favori −2,83 %.

## B. Un modèle bat-il le prix ? Walk-forward annuel 2015-2024, n=4 397

| Modèle | Gain log-loss | t |
|---|---|---|
| Recalibration du prix seul | −0,00125 | −2,35 |
| Prix + descripteurs (logit C=0,01) | **+0,00029** | **+0,15** |
| Prix + descripteurs (logit C=1) | −0,00035 | −0,15 |
| Prix + descripteurs (HistGBM) | −0,00507 | −2,07 |
| Prix + descripteurs + classements UFC | −0,00030 | −0,15 |
| Descripteurs seuls | −0,05093 | −11,22 |
| Descripteurs seuls (HistGBM) | −0,05400 | −11,74 |

**Rien ne bat le prix.** Le meilleur candidat est du bruit. Recalibrer le prix seul le dégrade
significativement : il est déjà bien calibré. Les classements officiels n'ajoutent rien.

## C. Le seul survivant, et sa mort

Balayage de 72 cellules (24 segments × 3 seuils). Meilleure cellule : **combats serrés
(prix 0,40–0,60), seuil 0,03, n=841, ROI +8,81 %, t=+2,57**.

Probabilité family-wise sous H0 (2 000 simulations) : **3,4 %**. Premier résultat du projet
à franchir un test de multiplicité.

Il résiste à plusieurs contrôles :

- trois familles de modèles : +8,81 %, +6,67 %, +6,90 %
- 8 années positives sur 10
- le pari en aveugle dans la même bande perd (−5,48 % côté 1, −0,78 % côté 2)
- les paris sont équilibrés entre les deux côtés (408/841)

### Le signe qui trahit

**Le gain de log-loss dans cette bande n'est que +0,00269, t = +0,86.** Le ROI y est
significatif (t = +2,57) alors que la prévision ne l'est pas. C'est à l'envers : le ROI est
une fonction plus bruitée et seuillée du même signal, il ne peut pas être *plus* significatif
que la prévision — sauf si la règle de pari sélectionne quelque chose que la prévision ne
contient pas.

### La cause : un artefact d'orientation

En conditionnant sur `orientation_swapped`, les deux groupes montrent le même biais signé :
**le combattant listé en premier gagne environ 2,3 points de plus que son prix ne l'implique**
(+1,57 pt non permuté, +2,96 pts permuté). La permutation déterministe équilibre `P(y=1)` à
0,4937, ce qui rend le biais invisible en agrégat.

À cote 2,0, 2,3 points de probabilité valent environ 4,6 points de ROI. Test décisif —
entraînement sur les deux orientations, prédiction moyennée avec son miroir :

| | ROI cellule | t | IC95 événementiel | Gain log-loss bande |
|---|---|---|---|---|
| Asymétrique | +7,77 % | +2,25 | [+0,22 %, +15,23 %] | +0,00262 (t=0,86) |
| **Symétrisé** | **+5,86 %** | **+1,62** | **[−1,86 %, +13,47 %]** | **+0,00032 (t=0,11)** |

L'intervalle recouvre zéro et le gain de prévision s'effondre. La cellule ne survit pas.

À cela s'ajoute que la multiplicité de 3,4 % ne comptait que mes 72 cellules : les trois
phases antérieures ont déjà dépensé une vingtaine de combinaisons modèle × seuil sur les
mêmes données. Famille réelle ≈ 90+, donc p > 5 %.

**Soupçon à porter au dossier :** le gain de +0,00125 de la phase 1 (`market_plus_c0.01`,
2015-2018) n'a jamais été symétrisé et vient probablement en partie du même artefact.

## D. Marchés annexes et mise

**Méthode de victoire** : overround médian **22,3 %** contre 3,7 % sur le vainqueur, soit
18,6 points de surcoût. Le marché est fermé par sa marge, quelle que soit la qualité du modèle.

**La mise ne change pas un signe** (simulation en espace log, 800 paris à cote 2,0) :

| ROI réel | f=0,02 | f=0,05 | f=0,125 | f=0,25 |
|---|---|---|---|---|
| −3,00 % | décroît | décroît | décroît | décroît |
| 0,00 % | décroît | décroît | décroît | décroît |
| +5,86 % | croît | croît | **décroît** | décroît |

Même en supposant vrai le +5,86 % non prouvé, un huitième de Kelly fait déjà décroître le
capital : à cote 2,0 la variance mange la moyenne.

## Conclusion

Les quatre passages sur l'UFC disent la même chose de quatre façons. Le prix contient
l'information ; les descripteurs n'ajoutent rien au-dessus de lui ; le seul biais réel
(favori-outsider) ne franchit pas la marge ; le seul segment prometteur était un défaut de
pipeline.

**Aucune porte ne s'est ouverte. Le holdout 2025-09-13 → 2026-08-29 n'a pas été touché et
reste la seule preuve possible — pour une stratégie qui n'existe pas encore.**

Ce qu'il faudrait pour rouvrir le dossier honnêtement : une source de prix horodatée et
indépendante sur plusieurs années, pas un modèle de plus sur les mêmes 6 040 lignes.
