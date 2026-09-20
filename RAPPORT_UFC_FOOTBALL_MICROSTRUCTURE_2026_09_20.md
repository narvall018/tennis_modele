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

---

# Addendum du 20 septembre 2026 — la piste exchange est fermée

J'ai construit un scanner football sur prix exchange à partir du biais mesuré ci-dessus,
puis je l'ai retiré. Voici pourquoi, parce que l'erreur est instructive.

## Le raisonnement initial

Le biais favori mesuré sur prix **Pinnacle** tolère 2,9 à 4,1 points de marge. Betfair
exchange facture 0,99 % sur la Ligue 1, contre 7,87 % pour les books français. Première
fois qu'un lieu passe sous le biais — la règle semblait fondée.

## L'erreur

Le biais est une propriété du **prix du bookmaker**. Un exchange n'est pas un bookmaker.
Sur les 751 matchs où les deux prix existent :

| | Probabilité du favori |
|---|---|
| Implicite Pinnacle | 0,7605 |
| **Implicite exchange** | **0,7738** |
| Réalisé | 0,7963 |

**L'exchange a déjà absorbé environ la moitié du biais.** Appliquer la correction mesurée
chez Pinnacle à un prix d'exchange revient à la compter deux fois.

## Le test direct

Biais recalculé aux prix exchange de clôture (15 537 matchs, 2024-07 → 2026-09), même
méthode de borne basse par bootstrap hebdomadaire :

| Bande | n | Biais | **Borne basse 5 %** | ROI comm. 5 % |
|---|---|---|---|---|
| 0,60–0,65 | 1 052 | +1,04 pt | **−1,46 pt** | — |
| 0,65–0,70 | 727 | +1,04 pt | **−1,91 pt** | −1,00 % |
| 0,70–0,75 | 553 | +1,80 pt | **−1,42 pt** | +0,32 % |
| 0,75–0,85 | 598 | +1,27 pt | **−1,57 pt** | −0,15 % |
| 0,85–1,01 | 163 | +0,64 pt | **−3,45 pt** | −0,40 % |

**Aucune bande ne franchit zéro.** Le biais résiduel à l'exchange est à 0,8–1,0 écart-type,
et le rendement réalisé à commission standard oscille autour de zéro.

Le garde-fou du script de préparation — « aucune bande ne prouve un biais : ne pas figer de
règle » — est exactement ce qui doit se déclencher. Le scanner a donc été supprimé plutôt
que publié avec une règle que les données ne portent pas.

## Ce qui reste vrai

Le biais favori-outsider **chez les bookmakers** est réel et monotone sur 270 000 issues.
Mais il ne survit pas au seul lieu dont la marge serait assez faible pour l'exploiter,
parce que ce lieu le price déjà.

C'est une réponse plus forte que « la marge est trop élevée » : même en supprimant la
marge, **il n'y a rien à prendre**.

---

# Addendum 2 — quatre approches structurellement différentes

Après l'échec de la piste exchange, quatre questions que je n'avais jamais posées. Toutes
sur données de développement 2012-2023 ; le holdout reste fermé.

## 1. Sharp contre soft : parier B365 quand Pinnacle dit que le prix est généreux

| Filtre | n | ROI | IC95 hebdo |
|---|---|---|---|
| Aucun (contrôle) | 113 763 | **−7,13 %** | — |
| EV ≥ 0 % | 6 687 | +0,10 % | [−4,34 %, +4,49 %] |
| EV ≥ 1 % | 4 437 | +0,75 % | [−4,71 %, +6,42 %] |
| EV ≥ 5 % | 1 048 | +2,87 % | [−11,60 %, +18,92 %] |

Le filtre fonctionne : il enlève les 7 points de marge. **Mais il s'arrête à zéro.**

Et il ne détecte jamais de valeur sur cote courte : **1,0 %** seulement des paris retenus
sont sous 2,00. La valeur est entièrement sur cote 3 à 7, donc variance énorme et
intervalles inutilisables. Par bande, le meilleur est cote 3,2–5,0 : +1,94 %,
IC95 [−3,44 %, +7,23 %], sur n=3 964.

## 2. Le marché asiatique comme modèle du 1X2

Jamais tenté : l'étude cross-market utilisait le handicap comme **covariable**, sans
interpréter ses prix comme des probabilités. Ici je les inverse structurellement.

Totaux 2,5 → μ par inversion de Poisson. Handicap (demi-lignes seules, sémantique exacte)
→ suprématie par inversion de Skellam. Puis 1X2 par Skellam. 8 618 matchs, 100 %
d'inversions réussies.

| Source du 1X2 | log-loss |
|---|---|
| Dérivé du marché asiatique | 0,98317 |
| Coté B365 | 0,98264 |
| Coté Pinnacle | 0,98094 |

| Comparaison appariée | gain | t |
|---|---|---|
| vs B365 | −0,00054 | **−0,92** |
| vs Pinnacle | −0,00223 | −3,75 |

Le dérivé est **indiscernable du 1X2 coté de B365** et significativement moins bon que
Pinnacle. B365 price son 1X2 de façon cohérente avec son handicap et ses totaux : aucune
incohérence inter-marchés à exploiter.

## 3. Fatigue et calendrier

Un descripteur de *programme*, pas de qualité d'équipe — classe de variable jamais testée
ici. Jours depuis le match précédent de chaque équipe, 87 974 matchs.

Résidus par quintile d'écart de repos : +0,29 / +0,08 / −0,17 / +0,73 points. Non monotone.
Régression sur le logit du prix : coefficient du prix **+1,0607**, coefficient de l'écart de
repos **−0,0002**. Le marché price déjà le calendrier.

## 4. Le nul

Calibration globale : implicite **0,2641**, réalisé **0,2641**, écart **+0,00 point** sur
89 987 matchs. Le nul est l'issue la mieux cotée du marché.

Par bande, aucune ne franchit la marge ; la seule positive est p ∈ [0,30 ; 0,33] à
+0,62 % ± 2,4 %.

## La mesure qui ferme le dossier

Charge de marge par issue : (1/cote) / p_sharp − 1. Zéro signifie prix équitable.

| Book | Favori | 2ᵉ issue | 3ᵉ issue |
|---|---|---|---|
| B365 | +6,19 % | +5,37 % | +6,77 % |
| Moyenne marché | +5,12 % | +6,10 % | +8,00 % |

Par bande de probabilité, le minimum atteint **nulle part** moins de **+4,16 %** :

| Bande | Charge médiane (moyenne marché) | ROI réel |
|---|---|---|
| p < 0,10 | +13,26 % | −32,82 % |
| p 0,10–0,20 | +9,26 % | −11,53 % |
| p 0,20–0,35 | +6,50 % | −7,63 % |
| p 0,35–0,50 | +5,59 % | −4,66 % |
| p 0,50–0,70 | +4,67 % | −2,63 % |
| **p > 0,70** | **+4,16 %** | **−0,94 %** |

Le biais favori tolère au mieux **4,10 points**. La charge minimale existante est de
**4,16 points**. Ils se ratent d'un dixième de point, et le ROI réel de la meilleure cellule
accessible est **−0,94 %**.

## Conclusion

Neuf approches structurellement distinctes en deux jours, sur deux sports, plus la centaine
de tests UFC. Le motif ne varie pas : chaque biais réel est plus petit que la charge de
marge la plus faible qui existe, et le seul lieu sans marge (l'exchange) n'a pas le biais.

Ce n'est pas un problème de modèle. Les neuf approches n'ont pas échoué à modéliser — elles
ont toutes mesuré correctement, et ce qu'elles mesurent est un marché dont le prix contient
l'information et dont la marge dépasse ce qui reste.

---

# Addendum 3 — layer les outsiders, et le mirage du line shopping

Deux angles que j'avais manqués : je n'avais testé l'exchange **que sur les favoris**, et
je n'avais jamais envisagé de **layer** — or à l'exchange on peut parier contre.

## Le biais de l'outsider survit à l'exchange

L'exchange a absorbé le biais du favori (addendum 1). Il n'a **pas** absorbé celui de
l'outsider. Gamme complète, 15 537 matchs, prix de clôture Betfair :

| Probabilité | n | Implicite | Réalisé | Écart | BACK | **LAY** |
|---|---|---|---|---|---|---|
| 0,00–0,05 | 229 | 0,0382 | 0,0393 | +0,11 pt | −9,37 % | −0,29 % |
| **0,05–0,10** | **997** | **0,0797** | **0,0612** | **−1,85 pt** | −26,99 % | **+1,67 %** |
| 0,10–0,15 | 1 838 | 0,1275 | 0,1197 | −0,78 pt | −11,94 % | +0,34 % |
| 0,15–0,20 | 3 565 | 0,1773 | 0,1840 | +0,67 pt | −1,12 % | −1,50 % |
| 0,60–0,75 | 2 332 | 0,6627 | 0,6750 | +1,22 pt | −0,68 % | −5,41 % |

Rendement du lay exprimé **par unité de liabilité**, commission 5 %.

## Les défenses

**Déplacer les bornes** — ce n'est pas un pic isolé mais un **plateau** :

| Fenêtre | n | LAY | IC95 hebdomadaire |
|---|---|---|---|
| 0,040–0,090 | 761 | +0,98 % | [−0,86 %, +2,71 %] |
| **0,050–0,100** | 997 | **+1,67 %** | **[+0,14 %, +3,09 %]** |
| **0,055–0,105** | 1 070 | **+1,82 %** | **[+0,35 %, +3,19 %]** |
| 0,060–0,110 | 1 117 | +1,43 % | [−0,04 %, +2,83 %] |
| **0,040–0,120** | 1 678 | **+1,70 %** | **[+0,22 %, +3,05 %]** |
| 0,030–0,150 | 3 028 | +0,71 % | [−0,38 %, +1,80 %] |

Trois fenêtres chevauchantes excluent zéro. **Par saison** : 2024 +1,55 %, 2025 +1,59 %.
**Par ligue** : 7 positives sur 9.

**Multiplicité** : t observé +2,03 ; sous H0 le meilleur des 18 cellules atteint t=1,11 en
médiane et 2,20 au 95ᵉ centile. **p family-wise = 9,9 %.** Ça ne passe pas.

## Ce que les données ne peuvent pas trancher

J'ai utilisé **le même prix pour parier et pour layer**. Un exchange a un écart entre les
deux, et ces données n'en contiennent qu'un seul. Combien la cellule en absorbe :

| Écart back/lay | Rendement (comm. 5 %) |
|---|---|
| 0 % | +1,67 % |
| 5 % | +1,27 % |
| 10 % | +0,91 % |
| 20 % | +0,28 % |
| **25 %** | **0,00 %** |
| 30 % | −0,25 % |

L'edge meurt à **25 % d'écart**. Sur une ligue liquide l'écart est de quelques pour cent ;
à cote 12 sur une petite ligue il peut dépasser 25 %. **Indéterminable ici.**

Et le profil d'exposition est brutal : **liabilité de 11 pour 1 unité gagnée ; une seule
perte efface 12 gains.**

Enfin, la fenêtre 2024-2026 est celle que l'étude exchange du 6 septembre a déjà lue.
Ce n'est pas une preuve vierge.

## Le line shopping est un mirage

| Source | Overround médian | Lignes impliquant un arbitrage |
|---|---|---|
| Moyenne marché | +6,24 % | 0,0 % |
| Pinnacle | +3,25 % | 0,0 % |
| **Maximum tous books** | **+0,32 %** | **41,3 %** |

Le prix maximum ramène la marge à 0,32 %, mais **41,3 % de ces lignes impliquent un
arbitrage** : ces prix n'existent pas simultanément. Le gain médian du maximum sur la
moyenne est de 5,88 % de cote — c'est un composite, pas une offre.

Même à ce plafond irréalisable, les favoris p > 0,70 ne rendent que +2,55 % (se 0,96).

## Verdict

Le lay des outsiders à l'exchange est **la chose la plus robuste trouvée en trois jours** :
plateau et non pic, stable par saison et par ligue, et son mécanisme est cohérent — un
marché pair-à-pair n'a aucune raison de corriger l'appétit du public pour les gros prix.

Il ne passe pas la barre : p family-wise 9,9 %, fenêtre déjà lue, et un écart back/lay que
ces données ne mesurent pas et qui peut le consommer entièrement.

C'est le seul résultat de ce projet dont l'échec ne vienne **pas** de la taille du biais.
