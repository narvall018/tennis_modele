#!/usr/bin/env python3
"""Ce qui marche quand les opérateurs internationaux sont accessibles.

Les rapports précédents prenaient les cinq opérateurs agréés comme ensemble
atteignable, ce qui n'est une contrainte que si l'on s'y tient. Pinnacle est à
3,02% de surmarge quand le seuil de rentabilité d'un gros favori est à 4,62%:
l'analyse change donc matériellement.

Deux stratégies, à ne pas confondre. En (B) Pinnacle est la référence de vérité,
il ne peut donc pas être simultanément l'endroit où l'on parie.

  A. parier les gros favoris au meilleur de deux books — exploite le biais
     favori-outsider, c'est-à-dire la façon dont un book range sa marge;
  B. parier là où un book mou dépasse le prix Pinnacle dévigué — exploite un
     désaccord entre deux books sur le même événement.

Ce script applique à chacune les deux contrôles qui les départagent: l'érosion
dans le temps, et la sensibilité aux années aberrantes.
"""
import numpy as np, pandas as pd
f = pd.read_csv("data/football/football_matches.csv.gz", low_memory=False)
f["match_date"]=pd.to_datetime(f["match_date"],errors="coerce")
f=f.dropna(subset=["match_date","result"])
f["month"],f["year"]=f["match_date"].dt.to_period("M"),f["match_date"].dt.year

def boot(v,m,draws=3000):
    m=np.asarray(m); g=[v[m==x] for x in np.unique(m)]
    rng=np.random.default_rng(0)
    s=[np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))]).mean() for _ in range(draws)]
    return float(np.percentile(s,2.5)),float(np.percentile(s,97.5))

rows=[]
for side in "HDA":
    b=pd.to_numeric(f[f"B365{side}"],errors="coerce"); p=pd.to_numeric(f[f"PS{side}"],errors="coerce")
    k=b.notna()&p.notna()&b.gt(1)&p.gt(1)
    rows.append(pd.DataFrame({"B":b[k],"S":p[k],"best2":np.maximum(b[k],p[k]),
        "won":(f["result"]==side).astype(float)[k],"month":f.loc[k,"month"],
        "year":f.loc[k,"year"],"idx":f.index[k]}))
L=pd.concat(rows,ignore_index=True)

print("=== A. favoris 1,20-1,35 (meilleur de 2), par époque ===")
c=L[L["best2"].between(1.20,1.35,"left")].copy(); c["gain"]=c["won"]*c["best2"]
for lo,hi,label in ((2005,2012,"2005-2012"),(2013,2018,"2013-2018"),(2019,2025,"2019-2025")):
    e=c[c["year"].between(lo,hi)]
    if len(e)<300: continue
    g=e["gain"].to_numpy(); a,b=boot(g,e["month"].to_numpy())
    print(f"  {label}: n={len(e):>5,}  r={g.mean():.4f}  [{a:.4f}, {b:.4f}]  {g.mean()-1:+.2%}")

print("\n=== B. value bet 2,0-4,0, sensibilité à 2012 ===")
tot=(1/L["S"]).groupby(L["idx"]).transform("sum")
V=L[(tot>1.0)&(tot<1.35)].copy()
V["p"]=(1/V["S"])/tot[V.index]; V["value"]=V["p"]*V["B"]-1.0
V["profit"]=np.where(V["won"]==1,V["B"]-1.0,-1.0)
band=V[(V["B"].between(2.0,4.0,"left"))&(V["value"]>0)]
for label, sub in (("toutes années", band),
                   ("sans 2012", band[band["year"]!=2012]),
                   ("2015 et après", band[band["year"]>=2015]),
                   ("2019 et après", band[band["year"]>=2019])):
    if len(sub)<300: continue
    p=sub["profit"].to_numpy(); a,b=boot(p,sub["month"].to_numpy())
    star=" *" if a>0 else ""
    print(f"  {label:>16s}: n={len(sub):>6,}  ROI {p.mean():>+7.2%}  [{a:+.2%}, {b:+.2%}]{star}")

print("\n=== volume et gain réaliste (stratégie B, 2019+) ===")
sub=band[band["year"]>=2019]
n_year=len(sub)/sub["year"].nunique()
roi=sub["profit"].mean(); cote=sub["B"].mean()
kelly=roi/(cote-1.0)
print(f"  {n_year:.0f} paris/an à cote moyenne {cote:.2f}, ROI {roi:+.2%}")
print(f"  Kelly complet {kelly:.1%}, quart de Kelly {kelly/4:.2%} de bankroll")
print(f"  croissance attendue: {n_year*roi*kelly/4:.1%} de bankroll par an")
print("\n* = borne basse au-dessus de zéro")
