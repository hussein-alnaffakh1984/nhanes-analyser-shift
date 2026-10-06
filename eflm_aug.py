import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
exec(open("harmonize.py").read().split("feat_cols=")[0])
# EFLM BV estimates (CVI, CVG, %), source noted
BV={ "albumin_si":(2.5,4.3,"EFLM BVD meta-analysis"),"calcium_si":(1.8,2.2,"EFLM BVD"),"glucose_si":(4.7,8.0,"EFLM BVD"),
 "potassium_si":(3.9,5.5,"EFLM BVD"),"sodium_si":(0.5,0.8,"EFLM BVD"),"total_protein_si":(2.6,3.7,"EFLM BVD"),
 "chloride_si":(1.0,1.3,"EFLM 2025 update (Bio-Rad)"),"cholesterol_si":(5.4,15.6,"EFLM 2025 update (Bio-Rad)"),
 "triglycerides_si":(19.8,34.3,"EFLM 2025 update (Bio-Rad)"),"alt_si":(12.6,30.0,"EFLM 2025 update (Bio-Rad)"),
 "ast_si":(8.4,19.4,"EFLM 2025 update (Bio-Rad)"),"uric_acid_si":(8.3,23.6,"EFLM BVD (Diaz-Garzon 2021 Table)"),
 "phosphorus_si":(7.8,10.7,"EFLM BVD (Diaz-Garzon 2021 Table)"),"hba1c_si":(1.2,5.4,"EFLM meta-analysis 2020"),
 "alp_si":(6.0,21.0,"PROXY: ALP liver-type, EFLM BVD"),"bicarbonate_si":(4.0,4.8,"PROXY: actual bicarbonate (whole blood), EFLM BVD")}
CDC_DIFF={"albumin_si":-4.42,"alp_si":9.74,"ast_si":-18.03,"alt_si":-15.46,"bicarbonate_si":1.74,"calcium_si":-2.12,
 "cholesterol_si":-3.21,"chloride_si":-1.44,"glucose_si":2.41,"potassium_si":-1.398,"sodium_si":0.80,"phosphorus_si":-1.80,
 "total_protein_si":0.36,"triglycerides_si":13.44,"uric_acid_si":2.37}
rows=[]
for k,(i,g,src) in BV.items():
    b=0.25*np.sqrt(i**2+g**2)
    rows.append(dict(analyte=k.replace("_si",""),CVI=i,CVG=g,bias_optimal=round(b/2,2),bias_desirable=round(b,2),bias_minimum=round(1.5*b,2),
                     CDC_observed_diff_pct=CDC_DIFF.get(k),exceeds_minimum=(abs(CDC_DIFF[k])>1.5*b) if k in CDC_DIFF else None,source=src))
aps=pd.DataFrame(rows); aps.to_csv("res/table10_EFLM_APS_vs_observed_shift.csv",index=False); print(aps.to_string())
d=labels(df.copy()); OLD=["2011-2012","2013-2014","2015-2016"]; NEW=["2017-2020","2021-2023"]
CBC=["hemoglobin_si","hematocrit_si","mcv_si","rdw_si","wbc_si","platelets_si"]
T3=["age","sex_female"]+CBC+["glucose_si","sodium_si","potassium_si","albumin_si","total_protein_si","calcium_si","phosphorus_si","uric_acid_si","chloride_si","bicarbonate_si","cholesterol_si","triglycerides_si","alt_si","ast_si","alp_si","hba1c_si"]
def LR(): return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=3000))
tr=d[d.cycle.isin(OLD)]; te=d[d.cycle.isin(NEW)]
res=[]
p=LR().fit(tr[T3],tr.y60).predict_proba(te[T3])[:,1]
res.append(dict(training="Standard",level="-",AUC=round(roc_auc_score(te.y60,p),3),OE_mean=round(te.y60.mean()/p.mean(),3),OE_sd=0))
for lvl,mult in [("optimal",0.5),("desirable",1.0),("minimum",1.5),("2x minimum",3.0)]:
    oes=[];aucs=[]
    for seed in range(5):
        rng=np.random.default_rng(seed); aug=[tr]
        for k in range(10):
            t=tr.copy()
            for c,(i,g,_) in BV.items(): t[c]=t[c]*(1+rng.normal(0,mult*0.25*np.sqrt(i**2+g**2)/100))
            aug.append(t)
        tra=pd.concat(aug); p=LR().fit(tra[T3],tra.y60).predict_proba(te[T3])[:,1]
        oes.append(te.y60.mean()/p.mean()); aucs.append(roc_auc_score(te.y60,p))
    res.append(dict(training="EFLM-APS augmentation",level=lvl,AUC=round(np.mean(aucs),3),OE_mean=round(np.mean(oes),3),OE_sd=round(np.std(oes),3)))
r=pd.DataFrame(res); r.to_csv("res/table11_EFLM_augmentation.csv",index=False); print(r.to_string())
