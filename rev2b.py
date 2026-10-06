src=open("revision.py").read(); exec(src.split("res={}")[0].replace("B=1000","B=500"))
tr=base[base.cycle.isin(OLD)]; mods=fit_tiers(tr); cd=cdc(base)
gb=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=50,random_state=0).fit(tr[T3],tr.y)
rows=[]
for c in NEW:
    m=(base.cycle==c).values; y=base.y.values[m]; w=base.WT_MEC.values[m]; n=m.sum(); idx=[rng0.integers(0,n,n) for _ in range(1000)]
    p0=mods["T0"].predict_proba(base.loc[m,T0])[:,1]
    q=qm(base,base.cycle.isin(OLD),base.cycle==c); ls=qm(base,base.cycle.isin(OLD),base.cycle==c,locscale=True)
    for lab,dd in [("No adjustment",base),("CDC bridging",cd),("Quantile mapping",q),("Location-scale",ls)]:
        p=mods["T3"].predict_proba(dd.loc[m,T3])[:,1]; pg=gb.predict_proba(dd.loc[m,T3])[:,1]
        r=(y.mean()/p.mean())/(y.mean()/p0.mean()); bs=[(p0[i].mean()/p[i].mean()) for i in idx]
        rw=(np.sum(w*p0)/np.sum(w*p))
        rows.append(dict(cycle=c,method=lab,ratio_T3_T0=f"{r:.3f} ({np.percentile(bs,2.5):.3f}–{np.percentile(bs,97.5):.3f})",
          GBM_OE=round(y.mean()/pg.mean(),3),weighted_T0_OE=round(np.sum(w*y)/np.sum(w*p0),3),weighted_T3_OE=round(np.sum(w*y)/np.sum(w*p),3),weighted_ratio=round(rw,3)))
# negative control
trn=base[base.cycle=="2011-2012"]; mn=fit_tiers(trn); m=(base.cycle=="2013-2014").values; y=base.y.values[m]; n=m.sum(); idx=[rng0.integers(0,n,n) for _ in range(1000)]
p0=mn["T0"].predict_proba(base.loc[m,T0])[:,1]; qn=qm(base,base.cycle=="2011-2012",base.cycle=="2013-2014")
for lab,dd in [("No adjustment",base),("Quantile mapping",qn)]:
    p=mn["T3"].predict_proba(dd.loc[m,T3])[:,1]; bs=[(p0[i].mean()/p[i].mean()) for i in idx]
    rows.append(dict(cycle="NC 2013-2014",method=lab,ratio_T3_T0=f"{p0.mean()/p.mean():.3f} ({np.percentile(bs,2.5):.3f}–{np.percentile(bs,97.5):.3f})"))
print("NC train n",len(trn),"events",int(trn.y.sum()),"test n",n,"events",int(y.sum()))
r=pd.DataFrame(rows); r.to_csv("res/rev2_ratio_methods.csv",index=False); print(r.to_string())
