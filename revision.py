import pandas as pd, numpy as np, warnings, json; warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from scipy.optimize import minimize_scalar
from statsmodels.nonparametric.smoothers_lowess import lowess
exec(open("chain.py").read().split("FEATS=")[0])   # SI, EQ (C6000->DxC), EQL (C8000->C6000), SI2, egfr, labels
R="res/rev_"; rng0=np.random.default_rng(2026)
OLD=["2011-2012","2013-2014","2015-2016"]; NEW=["2017-2020","2021-2023"]
CBC=["hemoglobin_si","hematocrit_si","mcv_si","rdw_si","wbc_si","platelets_si"]
T0=["age","sex_female"]; T1=T0+CBC
TR=T1+["cholesterol_si","triglycerides_si","alt_si","ast_si","alp_si","hba1c_si"]          # realistic creatinine-free panel
T3=T1+["glucose_si","sodium_si","potassium_si","albumin_si","total_protein_si","calcium_si","phosphorus_si","uric_acid_si","chloride_si","bicarbonate_si","cholesterol_si","triglycerides_si","alt_si","ast_si","alp_si","hba1c_si"]
TIERS={"T0":T0,"T1":T1,"TR":TR,"T3":T3}
FEAT_ADJ=[c for c in sorted(set(EQ)|set(EQL)) if c not in ("creatinine_si","bun_si")]
# ---------- outcome: creatinine harmonised to DxC scale (primary) ----------
def harm_creat(d):
    d=d.copy(); m=d.cycle.isin(NEW); d.loc[m,"creatinine_si"]=EQ["creatinine_si"](d.loc[m,"creatinine_si"]/SI["creatinine_si"])*SI["creatinine_si"]; return d
raw0=df.copy(); e_raw=egfr(raw0.creatinine_si,raw0.age,raw0.sex_female==1)
hc=harm_creat(df); e_h=egfr(hc.creatinine_si,hc.age,hc.sex_female==1)
base=df.copy(); base["y_raw"]=(e_raw<60).astype(int); base["y"]=(e_h<60).astype(int)
thr=np.where(base.age<40,75,np.where(base.age<=65,60,45)); base["y_age"]=(e_h<thr).astype(int)
recl=[]
for c in NEW:
    m=base.cycle==c; a,b=base.y_raw[m],base.y[m]
    recl.append(dict(cycle=c,prev_unharmonised=round(a.mean()*100,2),prev_harmonised=round(b.mean()*100,2),
        down_pct=round(((a==1)&(b==0)).mean()*100,2),up_pct=round(((a==0)&(b==1)).mean()*100,2),gross_pct=round((a!=b).mean()*100,2),net_pp=round((a.mean()-b.mean())*100,2)))
pd.DataFrame(recl).to_csv(R+"reclassification.csv",index=False)
# ---------- helpers ----------
def LR(): return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=3000))
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def cal_int(y,p):
    lp=logit(p); return minimize_scalar(lambda a:-np.sum(y*(a+lp)-np.log1p(np.exp(a+lp))),bounds=(-5,5),method="bounded").x
def cal_slope(y,p): return LogisticRegression(C=1e6,max_iter=2000).fit(logit(p).reshape(-1,1),y).coef_[0][0]
def ici(y,p):
    sm=lowess(y,p,frac=0.3,it=0,return_sorted=False); e=np.abs(sm-p); return e.mean(),np.median(e),np.quantile(e,0.9)
B=1000
def boot_metrics(y,p,w=None,idx=None,extra=None):
    n=len(y); idx=idx if idx is not None else [rng0.integers(0,n,n) for _ in range(B)]
    oe=lambda yy,pp,ww=None: (yy.mean()/pp.mean()) if ww is None else (np.sum(ww*yy)/np.sum(ww*pp))
    est=dict(AUC=roc_auc_score(y,p),OE=oe(y,p,w),intercept=cal_int(y,p),slope=cal_slope(y,p))
    I=ici(y,p); est.update(ICI=I[0],E50=I[1],E90=I[2])
    bs={k:[] for k in ["AUC","OE","intercept","slope"]}
    for i in idx[:B]:
        yy,pp=y[i],p[i]; ww=None if w is None else w[i]
        bs["AUC"].append(roc_auc_score(yy,pp)); bs["OE"].append(oe(yy,pp,ww))
        if len(bs["intercept"])<300: bs["intercept"].append(cal_int(yy,pp)); bs["slope"].append(cal_slope(yy,pp))
    out={}
    for k,v in est.items():
        if k in bs: lo,hi=np.percentile(bs[k],[2.5,97.5]); out[k]=f"{v:.3f} ({lo:.3f}–{hi:.3f})"
        else: out[k]=round(v,3)
    return out,bs
def fit_tiers(train,ycol="y"): return {k:LR().fit(train[f],train[ycol]) for k,f in TIERS.items()}
# ---------- harmonisation functions ----------
def cdc(d,cols=FEAT_ADJ,chain=True,skip=()):
    d=d.copy(); m21=d.cycle=="2021-2023"; m=d.cycle.isin(NEW)
    if chain:
        for c in cols:
            if c in EQL and c not in skip: d.loc[m21,c]=EQL[c](d.loc[m21,c]/SI2[c])*SI2[c]
    for c in cols:
        if c in EQ and c not in skip: d.loc[m,c]=EQ[c](d.loc[m,c]/SI[c])*SI[c]
    return d
QMCOLS=["albumin_si","alt_si","ast_si","alp_si","uric_acid_si","cholesterol_si","triglycerides_si","potassium_si","total_protein_si","calcium_si","phosphorus_si","chloride_si","bicarbonate_si","sodium_si","glucose_si"]
def strat(x): return x.sex_female.astype(str)+"_"+pd.cut(x.age,[17,40,60,120]).astype(str)
def wquant(v,w,q):
    o=np.argsort(v); v,w=v[o],w[o]; cw=(np.cumsum(w)-0.5*w)/w.sum(); return np.interp(q,cw,v)
QS=np.linspace(0.005,0.995,199)
def qm(d,ref_mask,tgt_mask,fit_mask=None,cols=QMCOLS,weighted=False,locscale=False):
    """map values of rows in tgt_mask to reference distribution (rows ref_mask); mapping fitted on fit_mask (default tgt_mask)"""
    d=d.copy(); s=strat(d); fit_mask=tgt_mask if fit_mask is None else fit_mask
    for c in cols:
        for k in s.unique():
            r=d.loc[ref_mask&(s==k),c].dropna(); f=d.loc[fit_mask&(s==k),c].dropna(); mt=tgt_mask&(s==k)&d[c].notna()
            if len(r)<50 or len(f)<20 or mt.sum()==0: continue
            if locscale:
                d.loc[mt,c]=(d.loc[mt,c]-f.mean())/f.std()*r.std()+r.mean(); continue
            if weighted:
                qr=wquant(r.values,d.loc[r.index,"WT_MEC"].values,QS); qf=wquant(f.values,d.loc[f.index,"WT_MEC"].values,QS)
            else: qr,qf=np.quantile(r,QS),np.quantile(f,QS)
            d.loc[mt,c]=np.interp(d.loc[mt,c],qf,qr)
    return d
res={}
# ---------- 1. main temporal validation: per cycle, tiers x methods, harmonised outcome ----------
tr=base[base.cycle.isin(OLD)]; models=fit_tiers(tr)
methods={"Unadjusted":base,"CDC chained":cdc(base)}
for c in NEW:
    methods[f"QM_{c}"]=qm(base,base.cycle.isin(OLD),base.cycle==c)
    methods[f"LS_{c}"]=qm(base,base.cycle.isin(OLD),base.cycle==c,locscale=True)
rows=[]; bsstore={}
for c in NEW:
    m=(base.cycle==c).values; y=base.y.values[m]; idx=[rng0.integers(0,m.sum(),m.sum()) for _ in range(B)]
    for meth,key in [("Unadjusted","Unadjusted"),("CDC chained","CDC chained"),("Quantile mapping",f"QM_{c}"),("Location-scale",f"LS_{c}")]:
        dd=methods[key]
        for t,f in TIERS.items():
            if meth!="Unadjusted" and t in ("T0","T1"): continue
            p=models[t].predict_proba(dd[f].values[m] if False else dd.loc[m,f])[:,1]
            out,bs=boot_metrics(y,p,idx=idx); bsstore[(c,meth,t)]=np.array(bs["OE"])
            rows.append(dict(cycle=c,method=meth,tier=t,**out))
main=pd.DataFrame(rows); main.to_csv(R+"main_by_cycle.csv",index=False)
# analyser-attributable component: OE(T3)/OE(T0), OE(TR)/OE(T0), unadjusted
dec=[]
for c in NEW:
    for t in ["T1","TR","T3"]:
        r_=bsstore[(c,"Unadjusted",t)]/bsstore[(c,"Unadjusted","T0")]
        est=float(main[(main.cycle==c)&(main.method=="Unadjusted")&(main.tier==t)].OE.str.split().str[0].iloc[0])/float(main[(main.cycle==c)&(main.method=="Unadjusted")&(main.tier=="T0")].OE.str.split().str[0].iloc[0])
        dec.append(dict(cycle=c,contrast=f"{t}/T0",ratio=f"{est:.3f} ({np.percentile(r_,2.5):.3f}–{np.percentile(r_,97.5):.3f})"))
    for meth in ["CDC chained","Quantile mapping","Location-scale"]:
        dlt=bsstore[(c,meth,"T3")]-bsstore[(c,"Unadjusted","T3")]
        dec.append(dict(cycle=c,contrast=f"OE change T3: {meth} minus unadjusted",ratio=f"{dlt.mean():.3f} ({np.percentile(dlt,2.5):.3f}–{np.percentile(dlt,97.5):.3f})"))
    d1=bsstore[(c,"Quantile mapping","T3")]-1; d2=bsstore[(c,"CDC chained","T3")]-1
    dd_=np.abs(d1)-np.abs(d2); dec.append(dict(cycle=c,contrast="|OE-1| QM minus |OE-1| CDC (T3)",ratio=f"{dd_.mean():.3f} ({np.percentile(dd_,2.5):.3f}–{np.percentile(dd_,97.5):.3f})"))
pd.DataFrame(dec).to_csv(R+"decomposition.csv",index=False)
# ---------- 2. internal cross-validated performance 2011-2016 ----------
cvr=[]; yd=tr.y.values; skf=StratifiedKFold(5,shuffle=True,random_state=1)
for t,f in TIERS.items():
    p=cross_val_predict(LR(),tr[f],yd,cv=skf,method="predict_proba")[:,1]
    out,_=boot_metrics(yd,p); cvr.append(dict(tier=t,**out))
pd.DataFrame(cvr).to_csv(R+"internal_cv.csv",index=False)
# ---------- 3. negative control: train 2011-2012 -> test 2013-2014 ----------
nc=[]; trn=base[base.cycle=="2011-2012"]; mods=fit_tiers(trn); m=(base.cycle=="2013-2014").values; y=base.y.values[m]
qmn=qm(base,base.cycle=="2011-2012",base.cycle=="2013-2014")
for t,f in TIERS.items():
    for lab,dd in [("Unadjusted",base),("Quantile mapping",qmn)]:
        if lab!="Unadjusted" and t in ("T0","T1"): continue
        out,_=boot_metrics(y,mods[t].predict_proba(dd.loc[m,f])[:,1]); nc.append(dict(design="train 2011-2012, test 2013-2014",method=lab,tier=t,**out))
pd.DataFrame(nc).to_csv(R+"negative_control.csv",index=False)
# ---------- 4. CDC bridging diagnostics ----------
ref=base[base.cycle=="2015-2016"]; cd=cdc(base); cd1=cdc(base,chain=False); q21=methods["QM_2021-2023"]; q17=methods["QM_2017-2020"]
diag=[]
for a in FEAT_ADJ+["potassium_si","sodium_si","chloride_si"]:
    if a in [x["analyte"] for x in diag]: continue
    if a not in base: continue
    diag.append(dict(analyte=a.replace("_si",""),ref_2015_16=round(ref[a].median(),3),
      raw_2017_20=round(base[base.cycle=="2017-2020"][a].median(),3),cdc_2017_20=round(cd[cd.cycle=="2017-2020"][a].median(),3),qm_2017_20=round(q17[q17.cycle=="2017-2020"][a].median(),3),
      raw_2021_23=round(base[base.cycle=="2021-2023"][a].median(),3),cdc_chained_2021_23=round(cd[cd.cycle=="2021-2023"][a].median(),3),
      cdc_single_2021_23=round(cd1[cd1.cycle=="2021-2023"][a].median(),3),qm_2021_23=round(q21[q21.cycle=="2021-2023"][a].median(),3)))
pd.DataFrame(diag).to_csv(R+"bridging_medians.csv",index=False)
m=(base.cycle=="2021-2023").values; y=base.y.values[m]; loo=[]
p=models["T3"].predict_proba(cd.loc[m,T3])[:,1]; loo.append(dict(variant="Chained, all analytes",OE=round(y.mean()/p.mean(),3)))
p=models["T3"].predict_proba(cd1.loc[m,T3])[:,1]; loo.append(dict(variant="Single step (Cobas 6000->DxC only)",OE=round(y.mean()/p.mean(),3)))
for a in [x for x in FEAT_ADJ if x in T3]:
    dd=cdc(base,skip=(a,)); p=models["T3"].predict_proba(dd.loc[m,T3])[:,1]; loo.append(dict(variant=f"Chained, leave out {a.replace('_si','')}",OE=round(y.mean()/p.mean(),3)))
pd.DataFrame(loo).to_csv(R+"bridging_leave_one_out.csv",index=False)
# QM vs CDC transfer function comparison at reference quantiles (2017-2020)
cmp=[]
for a in ["albumin_si","alt_si","ast_si","alp_si","uric_acid_si","cholesterol_si","triglycerides_si","glucose_si"]:
    x=base.loc[base.cycle=="2017-2020",a].dropna().values; qx=np.quantile(x,[0.1,0.5,0.9])
    cdcv=EQ[a](qx/SI[a])*SI[a] if a in EQ else qx
    qref=np.quantile(base.loc[base.cycle.isin(OLD),a].dropna(),[0.1,0.5,0.9])
    cmp.append(dict(analyte=a.replace("_si",""),**{f"P{int(q*100)}_roche":round(v,2) for q,v in zip([.1,.5,.9],qx)},
      **{f"P{int(q*100)}_CDC":round(v,2) for q,v in zip([.1,.5,.9],cdcv)},**{f"P{int(q*100)}_QM":round(v,2) for q,v in zip([.1,.5,.9],qref)}))
pd.DataFrame(cmp).to_csv(R+"qm_vs_cdc_transfer.csv",index=False)
# prospective QM: mapping fitted on 2017-2020, applied to 2021-2023
qp=qm(base,base.cycle.isin(OLD),base.cycle=="2021-2023",fit_mask=base.cycle=="2017-2020")
p=models["T3"].predict_proba(qp.loc[m,T3])[:,1]; out,_=boot_metrics(y,p); pd.DataFrame([dict(method="QM fitted on 2017-2020, applied to 2021-2023",**out)]).to_csv(R+"qm_prospective.csv",index=False)
# ---------- 5. survey-weighted sensitivity (T3) ----------
wt=[]
qw={c:qm(base,base.cycle.isin(OLD),base.cycle==c,weighted=True) for c in NEW}
for c in NEW:
    m=(base.cycle==c).values; y=base.y.values[m]; w=base.WT_MEC.values[m]
    for lab,dd in [("Unadjusted",base),("CDC chained",methods["CDC chained"]),("Quantile mapping (weighted)",qw[c])]:
        p=models["T3"].predict_proba(dd.loc[m,T3])[:,1]; out,_=boot_metrics(y,p,w=w); wt.append(dict(cycle=c,method=lab,weighted_OE=out["OE"]))
pd.DataFrame(wt).to_csv(R+"weighted.csv",index=False)
# ---------- 6. domain classifier null ----------
CHEM=["albumin_si","alt_si","ast_si","alp_si","uric_acid_si","cholesterol_si","triglycerides_si","potassium_si","calcium_si","total_protein_si","phosphorus_si","bicarbonate_si"]
nul=[]
for c in base.cycle.unique():
    s=base[base.cycle==c]
    for rep in range(5):
        y=np.random.default_rng(rep).integers(0,2,len(s)); p=cross_val_predict(HistGradientBoostingClassifier(max_iter=150,random_state=0),s[CHEM],y,cv=3,method="predict_proba")[:,1]
        nul.append(dict(cycle=c,rep=rep,AUC=roc_auc_score(y,p)))
nl=pd.DataFrame(nul); nl.to_csv(R+"domain_null.csv",index=False)
# ---------- 7. recalibration: temporal + learning curve ----------
rc=[]; m17=(base.cycle=="2017-2020").values; m21=(base.cycle=="2021-2023").values
p17=models["T3"].predict_proba(base.loc[m17,T3])[:,1]; p21=models["T3"].predict_proba(base.loc[m21,T3])[:,1]; y17=base.y.values[m17]; y21=base.y.values[m21]
def recal(yu,pu,pt,kind):
    if kind=="intercept": a=cal_int(yu,pu); return 1/(1+np.exp(-(logit(pt)+a)))
    lr=LogisticRegression(C=1e6).fit(logit(pu).reshape(-1,1),yu); return lr.predict_proba(logit(pt).reshape(-1,1))[:,1]
idx21=[rng0.integers(0,m21.sum(),m21.sum()) for _ in range(B)]
out,_=boot_metrics(y21,p21,idx=idx21); rc.append(dict(update_data="none",kind="-",events="-",**out))
for kind in ["intercept","logistic"]:
    out,_=boot_metrics(y21,recal(y17,p17,p21,kind),idx=idx21); rc.append(dict(update_data="all 2017-2020",kind=kind,events=int(y17.sum()),**out))
prev=y17.mean()
for ev in [50,100,250,500]:
    for kind in ["intercept","logistic"]:
        oes=[];sls=[]
        for rep in range(50):
            g=np.random.default_rng(rep); n=int(ev/prev); i=g.choice(m17.sum(),n,replace=False)
            if y17[i].sum()<5: continue
            pr=recal(y17[i],p17[i],p21,kind); oes.append(y21.mean()/pr.mean()); sls.append(cal_slope(y21,pr))
        rc.append(dict(update_data=f"random 2017-2020 sample (~{ev} events)",kind=kind,events=ev,AUC="-",OE=f"{np.median(oes):.3f} ({np.percentile(oes,2.5):.3f}–{np.percentile(oes,97.5):.3f})",slope=f"{np.median(sls):.3f}"))
pd.DataFrame(rc).to_csv(R+"recalibration.csv",index=False)
# DCA on 2021-2023 after recalibration
pR=recal(y17,p17,p21,"logistic"); p0=models["T0"].predict_proba(base.loc[m21,T0])[:,1]; N=len(y21); dca=[]
for t in [0.05,0.10,0.20,0.30]:
    nb=lambda p: ((p>=t)&(y21==1)).sum()/N-((p>=t)&(y21==0)).sum()/N*t/(1-t)
    dca.append(dict(threshold=t,test_all=round(y21.mean()-(1-y21.mean())*t/(1-t),4),T0=round(nb(p0),4),T3_unadjusted=round(nb(p21),4),T3_recalibrated=round(nb(pR),4)))
pd.DataFrame(dca).to_csv(R+"dca_2021_recal.csv",index=False)
# ---------- 8. GBM calibration, missingness ----------
gb=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=50,random_state=0).fit(tr[T3],tr.y)
g=[]
for c in NEW:
    m=(base.cycle==c).values; out,_=boot_metrics(base.y.values[m],gb.predict_proba(base.loc[m,T3])[:,1]); g.append(dict(cycle=c,model="GBM T3 unadjusted",**out))
pd.DataFrame(g).to_csv(R+"gbm_calibration.csv",index=False)
base["period"]=np.where(base.cycle.isin(OLD),"Development","Validation")
miss=(base.groupby(["period","y"])[T3[2:]].apply(lambda x:x.isna().mean()*100).round(2).T); miss.to_csv(R+"missingness.csv")
# ---------- 9. age-adapted outcome on primary (harmonised outcome, unadjusted features) ----------
aa=[]; mta=fit_tiers(tr,"y_age"); mv=base.cycle.isin(NEW).values
for t,f in TIERS.items():
    out,_=boot_metrics(base.y_age.values[mv],mta[t].predict_proba(base.loc[mv,f])[:,1]); aa.append(dict(outcome="age-adapted",tier=t,prevalence=round(base.y_age[mv].mean()*100,1),**out))
pd.DataFrame(aa).to_csv(R+"age_adapted.csv",index=False)
# ---------- 10. subgroups (pooled validation, harmonised outcome) ----------
sg=[]; te=base[mv]
for g_,mask in [("All",np.ones(len(te),bool)),("Age <60",te.age.values<60),("Age ≥60",te.age.values>=60),("Age ≥70",te.age.values>=70),("Diabetes",te.diabetes_dx.values==1)]:
    for t,f in TIERS.items():
        y=te.y.values[mask]; p=models[t].predict_proba(te.loc[mask,f])[:,1]
        bs=[roc_auc_score(y[i],p[i]) for i in [rng0.integers(0,len(y),len(y)) for _ in range(B)]]
        sg.append(dict(group=g_,tier=t,n=int(mask.sum()),events=int(y.sum()),AUC=roc_auc_score(y,p),lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5)))
pd.DataFrame(sg).round(3).to_csv(R+"subgroups.csv",index=False)
print("done")
