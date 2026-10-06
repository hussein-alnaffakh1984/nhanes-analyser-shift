import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
exec(open("harmonize.py").read().split("feat_cols=")[0])   # reuse SI, EQ, back(), egfr(), labels()
d=labels(df.copy())
CHEM=["albumin_si","alt_si","ast_si","alp_si","uric_acid_si","cholesterol_si","triglycerides_si","potassium_si","calcium_si","total_protein_si","phosphorus_si","bicarbonate_si"]
CBC=["hemoglobin_si","hematocrit_si","mcv_si","rdw_si","wbc_si","platelets_si"]
# ---------- 1) Unsupervised drift detection: domain-classifier AUC between adjacent cycles ----------
cyc=["2011-2012","2013-2014","2015-2016","2017-2020","2021-2023"]; rows=[]
for a,b in zip(cyc[:-1],cyc[1:]):
    s=d[d.cycle.isin([a,b])]; y=(s.cycle==b).astype(int)
    for nm,f in [("Chemistry",CHEM),("CBC (control)",CBC),("Age+Sex (population)",["age","sex_female"])]:
        X=s[f]; p=cross_val_predict(HistGradientBoostingClassifier(max_iter=150,random_state=0),X,y,cv=3,method="predict_proba")[:,1]
        rows.append(dict(transition=f"{a} -> {b}",panel=nm,domain_AUC=round(roc_auc_score(y,p),3)))
dr=pd.DataFrame(rows).pivot(index="transition",columns="panel",values="domain_AUC"); print(dr); dr.to_csv("res/probe1_drift_detection.csv")
# ---------- 2) Label-free harmonisation: age/sex-stratified quantile mapping vs CDC paired equations ----------
OLD=["2011-2012","2013-2014","2015-2016"]; NEW17=["2017-2020"]
d["stratum"]=d.sex_female.astype(str)+"_"+pd.cut(d.age,[17,40,60,120]).astype(str)
def qmap(dd,col):
    out=dd[col].copy()
    for s,g in dd.groupby("stratum"):
        ref=g.loc[g.cycle.isin(OLD),col].dropna().values; m=g.cycle.isin(["2017-2020","2021-2023"])&g[col].notna()
        if len(ref)<50: continue
        src=g.loc[m,col].values; ranks=pd.Series(src).rank(pct=True).values
        out.loc[g.index[m]]=np.quantile(ref,np.clip(ranks,0.001,0.999))
    return out
cmp=[]
for c in ["albumin_si","alt_si","ast_si","uric_acid_si","cholesterol_si","alp_si"]:
    qm=qmap(d,c); m=d.cycle.isin(NEW17)&d[c].notna()
    cdc=EQ[c](d.loc[m,c]/SI[c])*SI[c]
    med_raw=d.loc[m,c].median(); med_q=qm[m].median(); med_cdc=cdc.median()
    cmp.append(dict(analyte=c,median_new_raw=round(med_raw,2),median_CDC_paired=round(med_cdc,2),
                    median_labelfree_QM=round(med_q,2),
                    corr_QM_vs_CDC=round(np.corrcoef(qm[m],cdc)[0,1],3),
                    agreement_pct=round(100*(1-abs(med_q-med_cdc)/abs(med_cdc-med_raw+1e-9)),1) if abs(med_cdc-med_raw)>1e-6 else None))
cm=pd.DataFrame(cmp); print(cm.to_string()); cm.to_csv("res/probe2_labelfree_vs_cdc.csv",index=False)
# NOTE: downstream calibration comparisons are in qm_cycle.py (per-cycle, chained CDC); EFLM augmentation in eflm_aug.py.
