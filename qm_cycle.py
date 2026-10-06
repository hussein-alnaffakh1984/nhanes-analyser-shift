import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
exec(open("chain.py").read().split("# Investigate")[0])
raw["stratum"]=raw.sex_female.astype(str)+"_"+pd.cut(raw.age,[17,40,60,120]).astype(str)
def qmap_cycle(dd,col,target_cycle):
    out=dd[col].copy()
    for s,g in dd.groupby("stratum"):
        ref=g.loc[g.cycle.isin(OLD),col].dropna().values; m=(g.cycle==target_cycle)&g[col].notna()
        if len(ref)<50 or m.sum()<20: continue
        r=g.loc[m,col].rank(pct=True).values; out.loc[g.index[m]]=np.quantile(ref,np.clip(r,0.001,0.999))
    return out
QM=["albumin_si","alt_si","ast_si","alp_si","uric_acid_si","cholesterol_si","triglycerides_si","potassium_si","total_protein_si","calcium_si","phosphorus_si","chloride_si","bicarbonate_si","sodium_si","glucose_si"]
rows=[]
for cyc in ["2017-2020","2021-2023"]:
    dq=raw.copy()
    for c in QM: dq[c]=qmap_cycle(raw,c,cyc)
    te=dq[dq.cycle==cyc]; p=mdl.predict_proba(te[T3])[:,1]; ci,sl=cal(te.y60.values,p)
    rows.append(dict(test_cycle=cyc,scenario="Label-free stratified quantile mapping",AUC=round(roc_auc_score(te.y60,p),3),O_E=round(te.y60.mean()/p.mean(),3),intercept=ci,slope=sl))
q=pd.concat([t,pd.DataFrame(rows)]).sort_values(["test_cycle","scenario"]); print(q.to_string())
q.to_csv("res/table12_chained_harmonisation_by_cycle.csv",index=False)
