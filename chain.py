import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar
exec(open("harmonize.py").read().split("feat_cols=")[0])   # SI, EQ (Cobas6000->DxC), egfr, labels
# BIOPRO_L backward: Cobas6000 = f(Cobas8000), conventional units; only where CDC recommends adjustment
EQL={"alp_si":lambda x:0.6098+0.9041*x,"alt_si":lambda x:-1.529+1.035*x,"bun_si":lambda x:0.09386+1.021*x,
     "bicarbonate_si":lambda x:1.37+1.022*x,"chloride_si":lambda x:9.137+0.919*x,"glucose_si":lambda x:-0.7821+0.9669*x,
     "ggt_si":lambda x:0.9735+0.9444*x,"iron_si":lambda x:-1.679+0.9918*x,"potassium_si":lambda x:0.2108+0.923*x,
     "ldh_si":lambda x:2.46+0.979*x,"sodium_si":lambda x:41.95+0.6983*x,"triglycerides_si":lambda x:-0.8463+0.9879*x}
SI2=dict(SI); SI2.update({"bicarbonate_si":1,"chloride_si":1,"glucose_si":0.0555,"potassium_si":1,"sodium_si":1})
def harm(d,feats,step2=True):
    d=d.copy()
    m21=d.cycle=="2021-2023"
    for c in feats:                      # step 1: Cobas8000 -> Cobas6000 (2021-23 only)
        if c in EQL: d.loc[m21,c]=EQL[c](d.loc[m21,c]/SI2[c])*SI2[c]
    if step2:
        m=d.cycle.isin(["2017-2020","2021-2023"])
        for c in feats:                  # step 2: Cobas6000 -> DxC660i
            if c in EQ: d.loc[m,c]=EQ[c](d.loc[m,c]/SI[c])*SI[c]
    return d
FEATS=[c for c in set(EQ)|set(EQL) if c not in ("creatinine_si","bun_si")]
OLD=["2011-2012","2013-2014","2015-2016"]
CBC=["hemoglobin_si","hematocrit_si","mcv_si","rdw_si","wbc_si","platelets_si"]
T3=["age","sex_female"]+CBC+["glucose_si","sodium_si","potassium_si","albumin_si","total_protein_si","calcium_si","phosphorus_si","uric_acid_si","chloride_si","bicarbonate_si","cholesterol_si","triglycerides_si","alt_si","ast_si","alp_si","hba1c_si"]
def LR(): return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=3000))
def cal(y,p):
    lp=np.log(p/(1-p)); sl=LogisticRegression(C=1e6).fit(lp.reshape(-1,1),y).coef_[0][0]
    ci=minimize_scalar(lambda a:-np.sum(y*(a+lp)-np.log1p(np.exp(a+lp))),bounds=(-5,5),method="bounded").x
    return round(ci,3),round(sl,3)
raw=labels(df.copy()); hj=labels(harm(df,FEATS))
mdl=LR().fit(raw[raw.cycle.isin(OLD)][T3],raw[raw.cycle.isin(OLD)].y60)
rows=[]
for cyc in ["2017-2020","2021-2023"]:
    for nm,d in [("Raw",raw),("CDC chained harmonisation",hj)]:
        te=d[d.cycle==cyc]; p=mdl.predict_proba(te[T3])[:,1]; ci,sl=cal(te.y60.values,p)
        rows.append(dict(test_cycle=cyc,scenario=nm,AUC=round(roc_auc_score(te.y60,p),3),O_E=round(te.y60.mean()/p.mean(),3),intercept=ci,slope=sl))
t=pd.DataFrame(rows); print(t.to_string()); t.to_csv("res/table12_chained_harmonisation_by_cycle.csv",index=False)
# Investigate undocumented 2013->2015 shift: per-analyte domain AUC and median change
CHEM=["albumin_si","alt_si","ast_si","alp_si","uric_acid_si","cholesterol_si","triglycerides_si","potassium_si","sodium_si","chloride_si","calcium_si","total_protein_si","phosphorus_si","bicarbonate_si","glucose_si","bun_si","creatinine_si"]
out=[]
for a,b in [("2013-2014","2015-2016"),("2015-2016","2017-2020"),("2017-2020","2021-2023")]:
    s=raw[raw.cycle.isin([a,b])]; y=(s.cycle==b).astype(int).values
    for c in CHEM:
        x=s[c].values; ok=~np.isnan(x)
        auc=roc_auc_score(y[ok],x[ok]); auc=max(auc,1-auc)
        out.append(dict(transition=f"{a}->{b}",analyte=c.replace("_si",""),univariate_domain_AUC=round(auc,3),
                        median_before=round(np.nanmedian(s.loc[s.cycle==a,c]),3),median_after=round(np.nanmedian(s.loc[s.cycle==b,c]),3)))
o=pd.DataFrame(out); o.to_csv("res/table13_per_analyte_shift.csv",index=False)
print(o.sort_values(["transition","univariate_domain_AUC"],ascending=[True,False]).groupby("transition").head(5).to_string())

# Sensitivity: drop CDC equations with weak bridging correlation (r<0.90: sodium r=0.645, chloride r=0.891)
for drop in [["sodium_si"],["sodium_si","chloride_si"],["sodium_si","chloride_si","potassium_si"]]:
    EQL2={k:v for k,v in EQL.items() if k not in drop}
    EQL_bak=EQL.copy(); EQL.clear(); EQL.update(EQL2)
    hj2=labels(harm(df,FEATS)); EQL.clear(); EQL.update(EQL_bak)
    te=hj2[hj2.cycle=="2021-2023"]; p=mdl.predict_proba(te[T3])[:,1]
    print("drop",drop,"2021-23 O/E=",round(te.y60.mean()/p.mean(),3),"AUC=",round(roc_auc_score(te.y60,p),3))
