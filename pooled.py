src=open("revision.py").read(); head=src.split("res={}")[0]
exec(head)
T3s=[c for c in T3 if c not in ("phosphorus_si","uric_acid_si","bicarbonate_si")]; T4=T3+["bun_si"]
ALL={"T0":T0,"T1":T1,"TR":TR,"T3":T3,"T3s":T3s,"T4":T4}
tr=base[base.cycle.isin(OLD)]; te=base[base.cycle.isin(NEW)]; y=te.y.values; n=len(y)
idx=[rng0.integers(0,n,n) for _ in range(1000)]
P={k:LR().fit(tr[f],tr.y).predict_proba(te[f])[:,1] for k,f in ALL.items()}
gb=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=50,random_state=0)
PG={k:gb.fit(tr[f],tr.y).predict_proba(te[f])[:,1] for k,f in ALL.items()}
rows=[]
for k in ALL:
    p=P[k]; a=[roc_auc_score(y[i],p[i]) for i in idx]; o=[y[i].mean()/p[i].mean() for i in idx]
    d=[roc_auc_score(y[i],p[i])-roc_auc_score(y[i],P["T0"][i]) for i in idx[:500]]
    rows.append(dict(tier=k,n_pred=len(ALL[k]),AUC=f"{roc_auc_score(y,p):.3f} ({np.percentile(a,2.5):.3f}–{np.percentile(a,97.5):.3f})",
      dAUC=f"{roc_auc_score(y,p)-roc_auc_score(y,P['T0']):.3f} ({np.percentile(d,2.5):.3f}–{np.percentile(d,97.5):.3f})",
      OE=f"{y.mean()/p.mean():.3f} ({np.percentile(o,2.5):.3f}–{np.percentile(o,97.5):.3f})",slope=round(cal_slope(y,p),3),
      GBM_AUC=round(roc_auc_score(y,PG[k]),3),GBM_OE=round(y.mean()/PG[k].mean(),3)))
pd.DataFrame(rows).to_csv("res/rev_table2_pooled.csv",index=False); print(pd.DataFrame(rows).to_string())
print("prev dev",round(tr.y.mean()*100,2),tr.y.sum(),"val",round(y.mean()*100,2),y.sum())
