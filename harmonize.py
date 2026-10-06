import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.optimize import minimize_scalar
R="res/"; df=pd.read_csv("nhanes_ckd_dataset.csv")
NEW=["2017-2020","2021-2023"]; OLD=["2011-2012","2013-2014","2015-2016"]
# --- CDC BIOPRO_J backward equations: DxC660i = f(Cobas6000), conventional units ---
SI={"albumin_si":10,"alt_si":1,"ast_si":1,"alp_si":1,"ggt_si":1,"ldh_si":1,"cholesterol_si":0.02586,
    "triglycerides_si":0.01129,"iron_si":0.1791,"uric_acid_si":59.48,"bun_si":0.357,"creatinine_si":88.42}
EQ={"albumin_si":lambda x:1.044*x+0.01128,"ast_si":lambda x:1.018*x+3.762,"alt_si":lambda x:1.013*x+2.688,
    "bun_si":lambda x:1.001*x+0.4488,"cholesterol_si":lambda x:1.046*x-2.203,"creatinine_si":lambda x:1.051*x-0.06945,
    "ggt_si":lambda x:0.8042*x+2.363,"iron_si":lambda x:0.9776*x-4.494,"ldh_si":lambda x:0.8568*x+2.062,
    "triglycerides_si":lambda x:0.9655*x-7.020,"uric_acid_si":lambda x:0.9323*x+0.2326,
    "alp_si":lambda x:10**(1.001*np.log10(x)-0.04294)}
EQL={"alp_si":lambda x:0.6098+0.9041*x,"alt_si":lambda x:-1.529+1.035*x,"bun_si":lambda x:0.09386+1.021*x,
     "bicarbonate_si":lambda x:1.37+1.022*x,"chloride_si":lambda x:9.137+0.919*x,"glucose_si":lambda x:-0.7821+0.9669*x,
     "ggt_si":lambda x:0.9735+0.9444*x,"iron_si":lambda x:-1.679+0.9918*x,"potassium_si":lambda x:0.2108+0.923*x,
     "ldh_si":lambda x:2.46+0.979*x,"sodium_si":lambda x:41.95+0.6983*x,"triglycerides_si":lambda x:-0.8463+0.9879*x}
SI2=dict(SI); SI2.update({"bicarbonate_si":1,"chloride_si":1,"glucose_si":0.0555,"potassium_si":1,"sodium_si":1})
def back(d,cols):
    """Chained CDC harmonisation to DxC660i scale: 2021-23 Cobas8000->Cobas6000 (BIOPRO_L), then Cobas6000->DxC (BIOPRO_J)."""
    d=d.copy(); m21=d.cycle=="2021-2023"; m=d.cycle.isin(NEW)
    for c in cols:
        if c in EQL: d.loc[m21,c]=EQL[c](d.loc[m21,c]/SI2[c])*SI2[c]
    for c in cols:
        if c in EQ: d.loc[m,c]=EQ[c](d.loc[m,c]/SI[c])*SI[c]
    return d
def egfr(scr_umol,age,fem):
    s=scr_umol/88.42; k=np.where(fem,0.7,0.9); a=np.where(fem,-0.241,-0.302)
    return 142*np.minimum(s/k,1)**a*np.maximum(s/k,1)**-1.2*0.9938**age*np.where(fem,1.012,1)
def labels(d):
    e=egfr(d.creatinine_si,d.age,d.sex_female==1)
    d["egfr_h"]=e; d["y60"]=(e<60).astype(int)
    thr=np.where(d.age<40,75,np.where(d.age<=65,60,45)); d["yAge"]=(e<thr).astype(int); return d
feat_cols=[c for c in sorted(set(EQ)|set(EQL)) if c not in ("creatinine_si","bun_si")]
raw=labels(df.copy())
lab_h=labels(back(df,["creatinine_si"]))              # outcome harmonised only
feat_h=back(df,feat_cols); feat_h=labels(feat_h)      # features harmonised, raw outcome
both=labels(back(df,feat_cols+["creatinine_si"]))
# label shift
m=raw.cycle.isin(NEW)
ls=pd.DataFrame({"cycle":raw.cycle[m],"raw":raw.y60[m],"harm":lab_h.y60[m]})
shift=ls.groupby("cycle").agg(prev_raw=("raw","mean"),prev_harmonised=("harm","mean"),
      reclassified=("raw",lambda s: (s!=ls.loc[s.index,"harm"]).mean()))
shift=(shift*100).round(2); shift.to_csv(R+"table7_label_shift_creatinine_bridging.csv"); print(shift)
T0=["age","sex_female"]; T1=T0+["hemoglobin_si","hematocrit_si","mcv_si","rdw_si","wbc_si","platelets_si"]
T2=T1+["glucose_si","sodium_si","potassium_si"]
T3=T2+["albumin_si","total_protein_si","calcium_si","phosphorus_si","uric_acid_si","chloride_si","bicarbonate_si",
       "cholesterol_si","triglycerides_si","alt_si","ast_si","alp_si","hba1c_si"]
def LR(): return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=3000))
def cal(y,p):
    lp=np.log(p/(1-p)); sl=LogisticRegression(C=1e6).fit(lp.reshape(-1,1),y).coef_[0][0]
    cil=minimize_scalar(lambda a:-np.sum(y*(a+lp)-np.log1p(np.exp(a+lp))),bounds=(-5,5),method="bounded").x
    return round(cil,3),round(sl,3),round(y.mean()/p.mean(),3)
rows=[]
scen={"A Raw (no harmonisation)":raw,"B Features harmonised (CDC eq.)":feat_h,
      "C Outcome harmonised (creatinine eq.)":lab_h,"D Features + outcome harmonised":both}
for tn,f in [("T3 Extended",T3),("T1 CBC",T1)]:
    for sn,d in scen.items():
        tr=d[d.cycle.isin(OLD)]; te=d[d.cycle.isin(NEW)]
        m_=LR().fit(tr[f],tr.y60); p=m_.predict_proba(te[f])[:,1]; y=te.y60.values
        ci,sl,oe=cal(y,p)
        rows.append(dict(Tier=tn,Scenario=sn,AUC=round(roc_auc_score(y,p),3),Brier=round(brier_score_loss(y,p),4),
                         Calib_intercept=ci,Calib_slope=sl,O_E=oe))
    # E: model updating (logistic recalibration) on 20% of new data, eval on rest — raw data
    tr=raw[raw.cycle.isin(OLD)]; te=raw[raw.cycle.isin(NEW)].sample(frac=1,random_state=1)
    upd,ev=te.iloc[:int(.2*len(te))],te.iloc[int(.2*len(te)):]
    m_=LR().fit(tr[f],tr.y60); lpu=np.log(m_.predict_proba(upd[f])[:,1]/(1-m_.predict_proba(upd[f])[:,1]))
    rc=LogisticRegression(C=1e6).fit(lpu.reshape(-1,1),upd.y60)
    pe=m_.predict_proba(ev[f])[:,1]; lpe=np.log(pe/(1-pe)); pr=rc.predict_proba(lpe.reshape(-1,1))[:,1]
    for nm,pp in [("E0 Raw (same 80% eval set)",pe),("E Logistic recalibration (20% update set)",pr)]:
        ci,sl,oe=cal(ev.y60.values,pp)
        rows.append(dict(Tier=tn,Scenario=nm,AUC=round(roc_auc_score(ev.y60,pp),3),Brier=round(brier_score_loss(ev.y60,pp),4),
                         Calib_intercept=ci,Calib_slope=sl,O_E=oe))
h=pd.DataFrame(rows); h.to_csv(R+"table8_harmonisation_recalibration.csv",index=False); print(h.to_string())
# --- Age-adapted outcome (Delanaye) on fully harmonised data ---
T3s=[c for c in T3 if c not in ("phosphorus_si","uric_acid_si","bicarbonate_si")]
out=[]; d=both; tr=d[d.cycle.isin(OLD)]; te=d[d.cycle.isin(NEW)]
for yname in ["y60","yAge"]:
    for tn,f in [("T0 Age+Sex",T0),("T1 +CBC",T1),("T2 +Glu/Na/K",T2),("T3 Extended",T3),("T3s minus Phos/UA/HCO3",T3s),("T4 T3+BUN",T3+["bun_si"])]:
        p=LR().fit(tr[f],tr[yname]).predict_proba(te[f])[:,1]
        ci,sl,oe=cal(te[yname].values,p)
        out.append(dict(Outcome="eGFR<60" if yname=="y60" else "Age-adapted (75/60/45)",Tier=tn,
                        prevalence_test=round(te[yname].mean()*100,1),AUC=round(roc_auc_score(te[yname],p),3),O_E=oe))
a=pd.DataFrame(out); a.to_csv(R+"table9_age_adapted_outcome.csv",index=False); print(a.to_string())
print("age-adapted: % aged>65 among cases:",
      round((te.loc[te.yAge==1,"age"]>65).mean()*100,1)," vs eGFR<60:",round((te.loc[te.y60==1,"age"]>65).mean()*100,1))
