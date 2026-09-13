#!/usr/bin/env python3
"""Les marchés annexes valent-ils mieux que le 1X2 ? Totaux, handicap, et le reste.

Question naturelle après avoir fermé le résultat sec: un marché moins couru —
totaux de buts, handicap asiatique — serait-il moins bien coté, donc battable ?

Deux mesures par marché: la surmarge, qui fixe la barre, et le value bet
sharp-contre-mou, qui dit si un book mou s'écarte assez du prix juste.

Ce que ce script ne teste pas, faute de cotes dans les données: les corners
(football-data ne fournit que les corners joués, jamais leur prix), les totaux
de jeux au tennis, et les props UFC — ce dernier marché est de toute façon clos,
un vrai biais de décision y étant annulé par 22% d'overround.

Le handicap asiatique se règle avec remboursements: une ligne entière peut
annuler le pari, une ligne en quart le coupe en deux. Ignorer ça fausse tout.
"""
import numpy as np, pandas as pd

f = pd.read_csv("data/football/football_matches.csv.gz", low_memory=False)
f["match_date"] = pd.to_datetime(f["match_date"], errors="coerce")
f = f.dropna(subset=["match_date","result"])
f["month"] = f["match_date"].dt.to_period("M")
f["gd"] = pd.to_numeric(f["home_goals"], errors="coerce") - pd.to_numeric(f["away_goals"], errors="coerce")

def boot(v, m, draws=2500):
    m=np.asarray(m); g=[v[m==x] for x in np.unique(m)]
    rng=np.random.default_rng(0)
    s=[np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))]).mean() for _ in range(draws)]
    return float(np.percentile(s,2.5)), float(np.percentile(s,97.5))

print("=== surmarge médiane par marché et par source ===")
print(f"{'marché':>22s} {'Bet365':>9s} {'Pinnacle':>9s} {'Avg':>9s} {'Betfair':>9s}")
MARKETS = {
 "1X2 (3 issues)":      {"Bet365":("B365H","B365D","B365A"),"Pinnacle":("PSH","PSD","PSA"),
                         "Avg":("AvgH","AvgD","AvgA"),"Betfair":("BFEH","BFED","BFEA")},
 "totaux 2,5 (2 issues)":{"Bet365":("B365>2.5","B365<2.5"),"Pinnacle":("P>2.5","P<2.5"),
                         "Avg":("Avg>2.5","Avg<2.5"),"Betfair":("BFE>2.5","BFE<2.5")},
 "handicap (2 issues)": {"Bet365":("B365AHH","B365AHA"),"Pinnacle":("PAHH","PAHA"),
                         "Avg":("AvgAHH","AvgAHA"),"Betfair":("BFEAHH","BFEAHA")},
}
for name, srcs in MARKETS.items():
    line = f"{name:>22s}"
    for src in ("Bet365","Pinnacle","Avg","Betfair"):
        cols = srcs.get(src)
        if not cols or cols[0] not in f: line += f"{'—':>9s}"; continue
        o = [pd.to_numeric(f[c], errors="coerce") for c in cols]
        k = np.logical_and.reduce([x.notna() & x.gt(1) for x in o])
        tot = sum(1/x[k] for x in o); tot = tot[(tot>0.85)&(tot<1.4)]
        line += f"{tot.median()-1:>9.2%}" if len(tot)>500 else f"{'—':>9s}"
    print(line)

def ah_profit(price, line, gd, on_home):
    """Règlement du handicap asiatique, remboursements et quarts compris."""
    edge = (gd + line) if on_home else -(gd + line)
    out = np.where(edge > 0.25, price - 1.0,
          np.where(edge == 0.25, (price-1.0)/2,
          np.where(edge == 0, 0.0,
          np.where(edge == -0.25, -0.5, -1.0))))
    return out

print("\n=== value bet Pinnacle -> Bet365, par marché (mêmes instants) ===")
print(f"{'marché':>22s} {'seuil':>6s} {'paris':>8s} {'ROI':>8s} {'IC 95%':>20s}")

def run(name, s_cols, b_cols, won_or_profit, is_ah=False, line=None):
    s = [pd.to_numeric(f[c], errors="coerce") for c in s_cols]
    b = [pd.to_numeric(f[c], errors="coerce") for c in b_cols]
    k = np.logical_and.reduce([x.notna() & x.gt(1) for x in s+b])
    tot = sum(1/x for x in s)
    k &= (tot>1.0)&(tot<1.4)
    if is_ah: k &= line.notna()
    idx = f.index[k]
    rows=[]
    for i,(sc,bc) in enumerate(zip(s,b)):
        p = (1/sc[idx])/tot[idx]
        if is_ah:
            profit = ah_profit(bc[idx].to_numpy(), line[idx].to_numpy(),
                               f.loc[idx,"gd"].to_numpy(), on_home=(i==0))
        else:
            profit = np.where(won_or_profit[idx].to_numpy()==i, bc[idx]-1.0, -1.0)
        rows.append(pd.DataFrame({"value":p*bc[idx]-1.0,"profit":profit,
                                  "month":f.loc[idx,"month"]}))
    legs = pd.concat(rows, ignore_index=True).dropna()
    for th in (0.0, 0.02):
        take = legs[legs["value"]>th]
        if len(take)<150: continue
        p=take["profit"].to_numpy(); a,bb=boot(p, take["month"].to_numpy())
        star=" *" if a>0 else ""
        print(f"{name:>22s} {th:>6.0%} {len(take):>8,} {p.mean():>+8.2%} [{a:+.2%}, {bb:+.2%}]{star}")

side = f["result"].map({"H":0,"D":1,"A":2})
run("1X2", ("PSH","PSD","PSA"), ("B365H","B365D","B365A"), side)
run("totaux 2,5", ("P>2.5","P<2.5"), ("B365>2.5","B365<2.5"),
    (pd.to_numeric(f["total_goals"],errors="coerce")<3).astype(float))
run("handicap", ("PAHH","PAHA"), ("B365AHH","B365AHA"), None,
    is_ah=True, line=pd.to_numeric(f["AHh"], errors="coerce"))
print("\n* = borne basse au-dessus de zéro")
