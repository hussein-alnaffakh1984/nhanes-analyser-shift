src=open("revision.py").read(); exec(src.split("res={}")[0].replace("B=1000","B=1000"))
e=pd.read_csv("extra_vars.csv")[["SEQN","cycle","RIDEXMON","PHAFSTHR","PHDSESN"]]
base=base.merge(e,on=["SEQN","cycle"],how="left"); base["fast8"]=(base.PHAFSTHR>=8).astype(int); base.loc[base.PHAFSTHR.isna(),"fast8"]=-1
tr=base[base.cycle.isin(OLD)]; mods=fit_tiers(tr)
def ratio(m,dd,B=1000):
    y=base.y.values[m]; p=mods["T3"].predict_proba(dd.loc[m,T3])[:,1]; p0=mods["T0"].predict_proba(base.loc[m,T0])[:,1]; n=m.sum()
    bs=[p0[i].mean()/p[i].mean() for i in [rng0.integers(0,n,n) for _ in range(B)]]
    return round(y.mean()/p.mean(),3), f"{p0.mean()/p.mean():.3f} ({np.percentile(bs,2.5):.3f}–{np.percentile(bs,97.5):.3f})", int(n), int(y.sum())
rows=[]
# fasting-stratified quantile mapping (strata = sex x age x fasting>=8h vs <8h vs missing)
def strat_f(x): return x.sex_female.astype(str)+"_"+pd.cut(x.age,[17,40,60,120]).astype(str)+"_"+x.fast8.astype(str)
import types
for c in NEW:
    m=(base.cycle==c).values
    rows.append(dict(cycle=c,analysis="All, no adjustment",**dict(zip(["OE","T3/T0","n","events"],ratio(m,base)))))
    q=qm(base,base.cycle.isin(OLD),base.cycle==c); rows.append(dict(cycle=c,analysis="All, QM (sex-age strata)",**dict(zip(["OE","T3/T0","n","events"],ratio(m,q)))))
    globals()["strat"]=strat_f; qf=qm(base,base.cycle.isin(OLD),base.cycle==c); globals()["strat"]=lambda x: x.sex_female.astype(str)+"_"+pd.cut(x.age,[17,40,60,120]).astype(str)
    rows.append(dict(cycle=c,analysis="All, QM (sex-age-fasting strata)",**dict(zip(["OE","T3/T0","n","events"],ratio(m,qf)))))
    mf=m&(base.fast8.values==1)
    rows.append(dict(cycle=c,analysis="Fasting >=8 h, no adjustment",**dict(zip(["OE","T3/T0","n","events"],ratio(mf,base)))))
    ref8=base.cycle.isin(OLD)&(base.fast8==1); q8=qm(base,ref8,(base.cycle==c)&(base.fast8==1))
    rows.append(dict(cycle=c,analysis="Fasting >=8 h, QM (fasting reference)",**dict(zip(["OE","T3/T0","n","events"],ratio(mf,q8)))))
r=pd.DataFrame(rows); r.to_csv("res/rev3_fasting.csv",index=False); print(r.to_string())
# domain classifier 2015-2016 vs 2017-2020: covariates alone, chemistry, chemistry + covariates, chemistry within morning-fasting subgroup
CHEM=["albumin_si","alt_si","ast_si","alp_si","uric_acid_si","cholesterol_si","triglycerides_si","potassium_si","calcium_si","total_protein_si","phosphorus_si","bicarbonate_si"]
COV=["PHAFSTHR","PHDSESN","RIDEXMON"]
s=base[base.cycle.isin(["2015-2016","2017-2020"])].copy(); y=(s.cycle=="2017-2020").astype(int).values
def dauc(X,yy): p=cross_val_predict(HistGradientBoostingClassifier(max_iter=150,random_state=0),X,yy,cv=3,method="predict_proba")[:,1]; return round(roc_auc_score(yy,p),3)
dc=dict(covariates_only=dauc(s[COV],y),chemistry=dauc(s[CHEM],y),chemistry_plus_covariates=dauc(s[CHEM+COV],y))
mm=((s.PHDSESN<0.5)&(s.PHAFSTHR>=8)).values; dc["chemistry_morning_fasting8"]=dauc(s.loc[mm,CHEM],y[mm]); dc["n_morning_fasting8"]=int(mm.sum())
for k in ["glucose_si","triglycerides_si","potassium_si"]:
    dc[f"median_{k}_2015_16_fast8"]=round(s[(s.cycle=="2015-2016")&(s.PHAFSTHR>=8)][k].median(),2); dc[f"median_{k}_2017_20_fast8"]=round(s[(s.cycle=="2017-2020")&(s.PHAFSTHR>=8)][k].median(),2)
print(dc); pd.Series(dc).to_csv("res/rev3_domain_adjusted.csv")
print(base.groupby("cycle").apply(lambda x: pd.Series({"fast_median":x.PHAFSTHR.median(),"fast8_pct":round((x.PHAFSTHR>=8).mean()*100,1),"morning_pct":round((x.PHDSESN<0.5).mean()*100,1)})))
