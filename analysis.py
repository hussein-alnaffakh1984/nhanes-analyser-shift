import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
import shap
rng=np.random.default_rng(42); R="res/"
df=pd.read_csv("nhanes_ckd_dataset.csv"); y="ckd_egfr_lt60"
s=df.creatinine_si/88.42; f=df.sex_female==1; k=np.where(f,.7,.9); al=np.where(f,-.241,-.302)
df[y]=((142*np.minimum(s/k,1)**al*np.maximum(s/k,1)**-1.2*.9938**df.age*np.where(f,1.012,1))<60).astype(int)
T0=["age","sex_female"]
T1=T0+["hemoglobin_si","hematocrit_si","mcv_si","rdw_si","wbc_si","platelets_si"]
T2=T1+["glucose_si","sodium_si","potassium_si"]
EXT=["albumin_si","total_protein_si","calcium_si","phosphorus_si","uric_acid_si","chloride_si","bicarbonate_si",
     "cholesterol_si","triglycerides_si","alt_si","ast_si","alp_si","hba1c_si"]
T3=T2+EXT
T3s=[f for f in T3 if f not in ("phosphorus_si","uric_acid_si","bicarbonate_si")]
T4=T3+["bun_si"]
sets={"T0 Age+Sex":T0,"T1 +CBC":T1,"T2 +Glu/Na/K":T2,"T3 Extended (no Cr/BUN)":T3,
      "T3s Extended minus Phos/UA/HCO3":T3s,"T4 Extended + BUN":T4}
tr=df[df.cycle.isin(["2011-2012","2013-2014","2015-2016"])]; te=df[df.cycle.isin(["2017-2020","2021-2023"])]
ytr,yte=tr[y].values,te[y].values
def LR(): return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=3000))
def GB(): return HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=50,random_state=0)
def calib(yv,p):
    lp=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    s=LogisticRegression(C=1e6,max_iter=1000).fit(lp.reshape(-1,1),yv); slope=s.coef_[0][0]
    from scipy.optimize import minimize_scalar
    f=lambda a: -np.sum(yv*(a+lp)-np.log1p(np.exp(a+lp)))
    cil=minimize_scalar(f,bounds=(-5,5),method="bounded").x
    return cil,slope,p.mean() and yv.mean()/p.mean()
B=300; idx=[rng.integers(0,len(yte),len(yte)) for _ in range(B)]
preds={}; rows=[]
for sn,f in sets.items():
    for mn,mk in [("LR",LR),("GBM",GB)]:
        m=mk().fit(tr[f],ytr); p=m.predict_proba(te[f])[:,1]; preds[(sn,mn)]=(m,p)
        auc=roc_auc_score(yte,p); br=brier_score_loss(yte,p); cil,sl,oe=calib(yte,p)
        bs=[roc_auc_score(yte[i],p[i]) for i in idx]
        rows.append(dict(Model=sn,Algorithm=mn,n_features=len(f),AUC=round(auc,3),
            AUC_95CI=f"{np.percentile(bs,2.5):.3f}-{np.percentile(bs,97.5):.3f}",
            Brier=round(br,4),Calib_intercept=round(cil,3),Calib_slope=round(sl,3),O_E=round(oe,3)))
perf=pd.DataFrame(rows); perf.to_csv(R+"table2_performance_temporal.csv",index=False); print(perf.to_string())
# Delta AUC vs T0 (LR)
base=preds[("T0 Age+Sex","LR")][1]; d=[]
for sn in sets:
    if sn.startswith("T0"): continue
    for mn in ["LR","GBM"]:
        p=preds[(sn,mn)][1]
        ds=[roc_auc_score(yte[i],p[i])-roc_auc_score(yte[i],base[i]) for i in idx]
        d.append(dict(Model=sn,Algorithm=mn,Delta_AUC_vs_AgeSex=round(roc_auc_score(yte,p)-roc_auc_score(yte,base),3),
                      CI95=f"{np.percentile(ds,2.5):.3f} to {np.percentile(ds,97.5):.3f}"))
dd=pd.DataFrame(d); dd.to_csv(R+"table3_incremental_value.csv",index=False); print(dd.to_string())
# Subgroups (LR T3 vs T0)
sub=[]; tt=te.copy()
groups={"All":tt.index==tt.index,"Male":tt.sex_female==0,"Female":tt.sex_female==1,"Age<60":tt.age<60,"Age>=60":tt.age>=60,
        "Age>=70":tt.age>=70,"Diabetes":tt.diabetes_dx==1,"No diabetes":tt.diabetes_dx==0}
for g,mask in groups.items():
    mask=np.asarray(mask)
    if yte[mask].sum()<20: continue
    r=dict(Subgroup=g,n=int(mask.sum()),events=int(yte[mask].sum()))
    for sn in ["T0 Age+Sex","T1 +CBC","T3 Extended (no Cr/BUN)"]:
        r[sn.split()[0]+"_AUC"]=round(roc_auc_score(yte[mask],preds[(sn,"LR")][1][mask]),3)
    sub.append(r)
sb=pd.DataFrame(sub); sb.to_csv(R+"table4_subgroups.csv",index=False); print(sb.to_string())
# Calibration plots
fig,ax=plt.subplots(1,2,figsize=(11,5))
for k,mn in enumerate(["LR","GBM"]):
    a=ax[k]; a.plot([0,0.6],[0,0.6],"k--",lw=1)
    for sn in ["T0 Age+Sex","T1 +CBC","T3 Extended (no Cr/BUN)"]:
        p=preds[(sn,mn)][1]; q=pd.qcut(p,10,duplicates="drop")
        g=pd.DataFrame({"p":p,"y":yte}).groupby(q,observed=True).mean()
        a.plot(g.p,g.y,"o-",label=sn)
    a.set_title(f"Calibration ({mn}) - temporal validation"); a.set_xlabel("Predicted risk"); a.set_ylabel("Observed proportion"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(R+"fig1_calibration.png",dpi=200); plt.close()
# DCA
th=np.linspace(0.01,0.40,80); N=len(yte); prev=yte.mean()
plt.figure(figsize=(7,5))
plt.plot(th,prev-(1-prev)*th/(1-th),"k:",label="Treat all"); plt.axhline(0,color="grey",lw=1,label="Treat none")
for sn in ["T0 Age+Sex","T1 +CBC","T3 Extended (no Cr/BUN)","T4 Extended + BUN"]:
    p=preds[(sn,"LR")][1]
    nb=[((p>=t)&(yte==1)).sum()/N-((p>=t)&(yte==0)).sum()/N*t/(1-t) for t in th]
    plt.plot(th,nb,label=sn)
plt.ylim(-0.02,prev+0.01); plt.xlabel("Threshold probability"); plt.ylabel("Net benefit"); plt.title("Decision curve analysis (LR, temporal validation)")
plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(R+"fig2_decision_curve.png",dpi=200); plt.close()
# SHAP (GBM T3)
m=preds[("T3 Extended (no Cr/BUN)","GBM")][0]
Xs=te[T3].sample(2000,random_state=0)
ex=shap.TreeExplainer(m); sv=ex.shap_values(Xs)
plt.figure(); shap.summary_plot(sv,Xs,show=False,max_display=15); plt.tight_layout(); plt.savefig(R+"fig3_shap_T3_gbm.png",dpi=200); plt.close()
imp=pd.DataFrame({"feature":T3,"mean_abs_SHAP":np.abs(sv).mean(0)}).sort_values("mean_abs_SHAP",ascending=False)
imp.to_csv(R+"table5_shap_importance.csv",index=False); print(imp.head(12).to_string())
# LR coefficients T3
lr=preds[("T3 Extended (no Cr/BUN)","LR")][0]
co=pd.DataFrame({"feature":T3,"std_coef":lr[-1].coef_[0],"OR_per_SD":np.exp(lr[-1].coef_[0])}).sort_values("std_coef",key=abs,ascending=False)
co.round(3).to_csv(R+"table6_lr_coefficients.csv",index=False)
# Table 1 descriptive
desc=[]
for c in ["age","sex_female","diabetes_dx","hypertension_dx","egfr"]+T3[2:]+["bun_si"]:
    for lab,d_ in [("Train",tr),("Test",te)]:
        pass
t1=df.assign(Set=np.where(df.cycle.isin(["2017-2020","2021-2023"]),"Temporal validation","Development"))
tab=t1.groupby(["Set",y])[["age","egfr","hemoglobin_si","albumin_si","glucose_si","potassium_si","uric_acid_si","phosphorus_si","bicarbonate_si"]].median().round(2)
tab["n"]=t1.groupby(["Set",y]).size(); tab["female_%"]=(t1.groupby(["Set",y]).sex_female.mean()*100).round(1)
tab.to_csv(R+"table1_characteristics_median.csv"); print(tab.to_string())
