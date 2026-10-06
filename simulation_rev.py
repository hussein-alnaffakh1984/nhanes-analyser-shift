import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
exec(open("harmonize.py").read().split("feat_cols=")[0])
d0=labels(df.copy()); d0=d0[d0.cycle.isin(["2011-2012","2013-2014","2015-2016"])].reset_index(drop=True)
CBC=["hemoglobin_si","hematocrit_si","mcv_si","rdw_si","wbc_si","platelets_si"]
T3=["age","sex_female"]+CBC+["glucose_si","sodium_si","potassium_si","albumin_si","total_protein_si","calcium_si","phosphorus_si","uric_acid_si","chloride_si","bicarbonate_si","cholesterol_si","triglycerides_si","alt_si","ast_si","alp_si","hba1c_si"]
BIASED=["albumin_si","ast_si","alt_si","alp_si","uric_acid_si","cholesterol_si","triglycerides_si"]
# forward (DxC -> Cobas6000) = inverse of backward linear eq; ALP log
FWD={"albumin_si":(1.044,0.01128),"ast_si":(1.018,3.762),"alt_si":(1.013,2.688),"cholesterol_si":(1.046,-2.203),
     "triglycerides_si":(0.9655,-7.020),"uric_acid_si":(0.9323,0.2326)}
def add_bias(d):
    d=d.copy()
    for c,(a,b) in FWD.items(): x=d[c]/SI[c]; d[c]=((x-b)/a)*SI[c]
    d["alp_si"]=10**((np.log10(d.alp_si)+0.04294)/1.001)
    return d
def remove_bias_oracle(d):
    d=d.copy()
    for c,(a,b) in FWD.items(): x=d[c]/SI[c]; d[c]=(a*x+b)*SI[c]
    d["alp_si"]=10**(1.001*np.log10(d.alp_si)-0.04294); return d
def qmap(ref,tgt,cols,mask_ref=None,mask_tgt=None,by=("sex_female",),age_strata=True):
    out=tgt.copy()
    r=ref if mask_ref is None else ref[mask_ref(ref)]
    t_fit=tgt if mask_tgt is None else tgt[mask_tgt(tgt)]
    keys=list(by)
    def strat(x):
        s=x[keys].astype(str).agg("_".join,axis=1)
        if age_strata: s=s+"_"+pd.cut(x.age,[17,40,60,120]).astype(str)
        return s
    rs,ts,tall=strat(r),strat(t_fit),strat(tgt)
    qs=np.linspace(0.01,0.99,99)
    for c in cols:
        for k in tall.unique():
            rv=r.loc[rs==k,c].dropna(); tv=t_fit.loc[ts==k,c].dropna()
            if len(rv)<30 or len(tv)<30:      # fall back to sex-only stratum
                kk=k.split("_")[0]; rv=r.loc[r.sex_female.astype(str)==kk,c].dropna(); tv=t_fit.loc[t_fit.sex_female.astype(str)==kk,c].dropna()
            qr,qt=np.quantile(rv,qs),np.quantile(tv,qs)
            m=(tall==k)&tgt[c].notna()
            out.loc[m,c]=np.interp(tgt.loc[m,c],qt,qr)   # monotone transfer function, linear extrapolation clipped
    return out
def LR(): return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=3000))
rows=[]
for seed in range(5):
    rng=np.random.default_rng(seed); idx=rng.permutation(len(d0)); h=len(d0)//2
    ref,tgt0=d0.iloc[idx[:h]].copy(),d0.iloc[idx[h:]].copy()
    mdl=LR().fit(ref[T3],ref.y60)
    for scen in ["Bias only","True case-mix shift only","Bias + case-mix shift","Moderate case-mix shift only","Bias + moderate case-mix shift"]:
        tgt=tgt0.copy()
        if "case-mix" in scen:   # oversample cases: +100% (extreme) or +25% (moderate, ~ageing/diabetes trend)
            cases=tgt[tgt.y60==1]; k=len(cases) if "Moderate" not in scen and "moderate" not in scen else int(0.25*len(cases))
            tgt=pd.concat([tgt,cases.sample(k,replace=True,random_state=seed)]).reset_index(drop=True)
        truth=tgt.copy()
        if "Bias" in scen: tgt=add_bias(tgt)
        meths={"No correction":tgt,
               "Oracle (true equations)":remove_bias_oracle(tgt) if "Bias" in scen else tgt,
               "Quantile mapping (all subjects)":qmap(ref,tgt,BIASED),
               "Anchored QM (low-risk: age<40, no diabetes)":qmap(ref,tgt,BIASED,
                    mask_ref=lambda x:(x.age<40)&(x.diabetes_dx!=1),mask_tgt=lambda x:(x.age<40)&(x.diabetes_dx!=1),age_strata=False)}
        for mn,dd in meths.items():
            p=mdl.predict_proba(dd[T3])[:,1]
            err=np.nanmean([np.nanmedian(np.abs(dd[c]-truth[c])/truth[c])*100 for c in BIASED])
            rows.append(dict(seed=seed,scenario=scen,method=mn,AUC=roc_auc_score(dd.y60,p),O_E=dd.y60.mean()/p.mean(),feature_error_pct=err))
r=pd.DataFrame(rows); s=r.groupby(["scenario","method"]).agg(AUC=("AUC","mean"),O_E=("O_E","mean"),O_E_sd=("O_E","std"),feature_error_pct=("feature_error_pct","mean")).round(3)
print(s.to_string()); s.to_csv("res/rev_simulation.csv")
