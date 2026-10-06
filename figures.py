import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                     "savefig.dpi":600,"figure.dpi":150,"axes.titleweight":"bold","axes.titlesize":10})
C={"raw":"#7f7f7f","cdc":"#d95f02","qm":"#1b9e77","t0":"#999999","t1":"#e6ab02","t3":"#1f78b4","chem":"#d95f02","cbc":"#1b9e77","pop":"#7570b3"}
R="res/"; OUT="figs/"; import os; os.makedirs(OUT,exist_ok=True)
def save(fig,name): fig.savefig(OUT+name+".png",bbox_inches="tight"); fig.savefig(OUT+name+".pdf",bbox_inches="tight"); plt.close(fig)
# ---------- Figure 1: participant flow ----------
flow={}
for line in open("prep_report.txt"):
    if ":" in line and line.startswith("  "):
        k,v=line.strip().split(":"); flow[k]=int(v)
df=pd.read_csv("nhanes_ckd_dataset.csv")
dev=df.cycle.isin(["2011-2012","2013-2014","2015-2016"]).sum(); val=len(df)-dev
boxes=[(f"NHANES participants in merged files\n(2011–2023, five cycles)\nn = {flow['all_examined_in_files']:,}",0.5,0.92),
       (f"Adults aged ≥18 years\nn = {flow['adults_18plus']:,}",0.5,0.74),
       (f"With serum creatinine\nn = {flow['with_serum_creatinine']:,}",0.5,0.56),
       (f"Analytic sample\nn = {flow['analytic_sample']:,}",0.5,0.38),
       (f"Development\nBeckman DxC, 2011–2016\nn = {dev:,}",0.22,0.14),
       (f"Temporal validation\nRoche Cobas, 2017–2023\nn = {val:,}",0.78,0.14)]
fig,ax=plt.subplots(figsize=(6.2,5.6)); ax.axis("off")
for t,x,y in boxes:
    ax.text(x,y,t,ha="center",va="center",fontsize=8.5,bbox=dict(boxstyle="round,pad=0.5",fc="white",ec="black",lw=0.8))
for y0,y1 in [(0.87,0.79),(0.69,0.61),(0.51,0.43)]: ax.annotate("",xy=(0.5,y1),xytext=(0.5,y0),arrowprops=dict(arrowstyle="->",lw=0.8))
ax.annotate("",xy=(0.22,0.20),xytext=(0.5,0.33),arrowprops=dict(arrowstyle="->",lw=0.8))
ax.annotate("",xy=(0.78,0.20),xytext=(0.5,0.33),arrowprops=dict(arrowstyle="->",lw=0.8))
ax.text(0.80,0.47,f"Excluded:\npregnant at examination, n = {flow['excluded_pregnant']:,}\ndialysis in past 12 months, n = {flow['excluded_dialysis_12m']:,}",
        ha="left",va="center",fontsize=8,bbox=dict(boxstyle="round,pad=0.4",fc="#f2f2f2",ec="black",lw=0.6))
ax.annotate("",xy=(0.79,0.47),xytext=(0.5,0.47),arrowprops=dict(arrowstyle="->",lw=0.8))
ax.set_xlim(0,1.25); ax.set_ylim(0.03,1); save(fig,"Fig1_flow")
# ---------- Figure 2: discrimination by tier and age group (bootstrap CI) ----------
exec(open("harmonize.py").read().split("feat_cols=")[0])
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
d=labels(df.copy()); OLD=["2011-2012","2013-2014","2015-2016"]; tr=d[d.cycle.isin(OLD)]; te=d[~d.cycle.isin(OLD)]
CBC=["hemoglobin_si","hematocrit_si","mcv_si","rdw_si","wbc_si","platelets_si"]
T0=["age","sex_female"]; T1=T0+CBC
T3=T1+["glucose_si","sodium_si","potassium_si","albumin_si","total_protein_si","calcium_si","phosphorus_si","uric_acid_si","chloride_si","bicarbonate_si","cholesterol_si","triglycerides_si","alt_si","ast_si","alp_si","hba1c_si"]
LR=lambda: make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(max_iter=3000))
P={k:LR().fit(tr[f],tr.y60).predict_proba(te[f])[:,1] for k,f in [("T0",T0),("T1",T1),("T3",T3)]}
groups=[("All",np.ones(len(te),bool)),("Age <60",te.age.values<60),("Age ≥60",te.age.values>=60),("Age ≥70",te.age.values>=70),("Diabetes",te.diabetes_dx.values==1)]
rng=np.random.default_rng(42); rows=[]
for g,m in groups:
    y=te.y60.values[m]; idx=np.where(m)[0]
    for k in P:
        p=P[k][m]; bs=[]
        for _ in range(500):
            b=rng.integers(0,len(y),len(y))
            if y[b].min()!=y[b].max(): bs.append(roc_auc_score(y[b],p[b]))
        rows.append(dict(group=g,tier=k,n=len(y),events=int(y.sum()),AUC=roc_auc_score(y,p),lo=np.percentile(bs,2.5),hi=np.percentile(bs,97.5)))
fa=pd.DataFrame(rows); fa.round(3).to_csv(R+"tableS_subgroup_auc_ci.csv",index=False)
fig,ax=plt.subplots(figsize=(7.2,3.9)); lab={"T0":"Age + sex","T1":"+ CBC","T3":"+ extended chemistry"}; col={"T0":C["t0"],"T1":C["t1"],"T3":C["t3"]}
xs=np.arange(len(groups))
for i,k in enumerate(["T0","T1","T3"]):
    s=fa[fa.tier==k]; x=xs+(i-1)*0.22
    ax.errorbar(x,s.AUC,yerr=[s.AUC-s.lo,s.hi-s.AUC],fmt="o",ms=5,capsize=2.5,lw=1,color=col[k],label=lab[k])
ax.set_xticks(xs); ax.set_xticklabels([f"{g}\nn = {int(fa[fa.group==g].n.iloc[0]):,}\nevents = {int(fa[fa.group==g].events.iloc[0]):,}" for g,_ in groups],fontsize=7.5)
ax.set_ylabel("AUC (95% CI), 2017–2023"); ax.set_ylim(0.55,0.97); ax.axhline(0.5,lw=0)
ax.legend(frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(0.5,1.0)); ax.grid(axis="y",lw=0.3,alpha=0.6); save(fig,"Fig2_discrimination_by_group")
# ---------- Figure 3: drift detection (grouped bars) ----------
pr=pd.read_csv(R+"probe1_drift_detection.csv")
inst=["(platform not verified)","DxC 800 → DxC 800/660i","DxC 660i → Cobas 6000","Cobas 6000 → Cobas 8000"]
fig,ax=plt.subplots(figsize=(7.0,3.6)); x=np.arange(len(pr)); w=0.26
for i,(colname,cc,lb) in enumerate([("Chemistry",C["chem"],"Chemistry panel"),("CBC (control)",C["cbc"],"Complete blood count"),("Age+Sex (population)",C["pop"],"Age and sex")]):
    b=ax.bar(x+(i-1)*w,pr[colname]-0.5,w,bottom=0.5,color=cc,label=lb)
    for xi,v in zip(x+(i-1)*w,pr[colname]): ax.text(xi,v+0.008,f"{v:.2f}",ha="center",va="bottom",fontsize=6.5)
ax.axhline(0.5,color="black",lw=0.6)
ax.set_xticks(x); ax.set_xticklabels([f"{t.replace(' -> ',' → ')}\n{ins}" for t,ins in zip(pr.transition,inst)],fontsize=7.2)
ax.set_ylabel("Domain-classifier AUC\n(0.5 = no distinguishable shift)"); ax.set_ylim(0.45,1.05)
ax.legend(frameon=False,fontsize=7.5,ncol=3,loc="lower center",bbox_to_anchor=(0.5,1.0)); save(fig,"Fig3_drift_detection")
# ---------- Figure 4: calibration per cycle (raw / CDC chained / label-free) ----------
exec(open("chain.py").read().split("# Investigate")[0])
raw["stratum"]=raw.sex_female.astype(str)+"_"+pd.cut(raw.age,[17,40,60,120]).astype(str)
QM=["albumin_si","alt_si","ast_si","alp_si","uric_acid_si","cholesterol_si","triglycerides_si","potassium_si","total_protein_si","calcium_si","phosphorus_si","chloride_si","bicarbonate_si","sodium_si","glucose_si"]
def qmc(dd,col,cyc):
    out=dd[col].copy()
    for s,g in dd.groupby("stratum"):
        ref=g.loc[g.cycle.isin(OLD),col].dropna().values; m=(g.cycle==cyc)&g[col].notna()
        if len(ref)<50 or m.sum()<20: continue
        out.loc[g.index[m]]=np.quantile(ref,np.clip(g.loc[m,col].rank(pct=True).values,0.001,0.999))
    return out
fig,axs=plt.subplots(1,2,figsize=(7,3.4),sharey=True)
for ax,cyc in zip(axs,["2017-2020","2021-2023"]):
    dq=raw.copy()
    for c in QM: dq[c]=qmc(raw,c,cyc)
    for nm,dd,cc,mk in [("Raw","raw",C["raw"],"o"),("CDC chained harmonisation","cdc",C["cdc"],"s"),("Label-free stratified quantile mapping","qm",C["qm"],"^")]:
        src={"raw":raw,"cdc":hj,"qm":dq}[dd]; t=src[src.cycle==cyc]; p=mdl.predict_proba(t[T3])[:,1]
        q=pd.qcut(p,10,duplicates="drop"); g=pd.DataFrame({"p":p,"y":t.y60.values}).groupby(q,observed=True).mean()
        oe=t.y60.mean()/p.mean()
        ax.plot(g.p,g.y,marker=mk,ms=4,lw=1,color=cc,label=f"{ {'raw':'Unadjusted','cdc':'CDC bridging (chained)','qm':'Label-free mapping'}[dd]} (O/E {oe:.2f})")
    ax.plot([0,0.6],[0,0.6],"k--",lw=0.6); ax.set_xlim(0,0.6); ax.set_ylim(0,0.6); ax.set_title(cyc.replace("-","–")); ax.set_xlabel("Predicted risk (decile mean)")
    ax.legend(frameon=False,fontsize=7,loc="upper left")
axs[0].set_ylabel("Observed proportion"); save(fig,"Fig4_calibration_by_cycle")
# ---------- Figure 5: observed analyser bias vs EFLM APS ----------
a=pd.read_csv(R+"table10_EFLM_APS_vs_observed_shift.csv").dropna(subset=["CDC_observed_diff_pct"])
a["absdiff"]=a.CDC_observed_diff_pct.abs(); a=a.sort_values("absdiff")
names={"alt":"ALT","ast":"AST","alp":"ALP†","uric_acid":"Uric acid","total_protein":"Total protein","bicarbonate":"Bicarbonate†"}
fig,ax=plt.subplots(figsize=(6.2,4.4)); y=np.arange(len(a))
ax.barh(y,a.bias_minimum,color="#e8e8e8",label="EFLM minimum bias specification")
ax.barh(y,a.bias_desirable,color="#bdbdbd",label="EFLM desirable bias specification")
ex=a.absdiff>a.bias_minimum
ax.scatter(a.absdiff[~ex],y[~ex.values],color="#2c7fb8",zorder=3,s=24,label="Observed bias within minimum specification")
ax.scatter(a.absdiff[ex],y[ex.values],color=C["chem"],zorder=3,s=24,label="Observed bias exceeding minimum specification")
ax.set_yticks(y); ax.set_yticklabels([names.get(n,n.replace("_"," ").capitalize()) for n in a.analyte],fontsize=8)
ax.set_xlabel("Absolute mean difference, Roche Cobas 6000 vs Beckman DxC 660i (%)")
ax.legend(frameon=False,fontsize=7,loc="lower right",bbox_to_anchor=(1.0,0.06))
fig.text(0.99,-0.01,"Observed bias: CDC NHANES bridging study (n = 248 serum samples). † EFLM proxy values (see Methods).",ha="right",va="top",fontsize=6.5)
save(fig,"Fig5_bias_vs_APS")
# ---------- Figure 6: simulation ----------
s=pd.read_csv(R+"table14_simulation_bias_vs_true_shift.csv")
meth=["No correction","Oracle (true equations)","Quantile mapping (all subjects)"]; ml={"No correction":"Unadjusted","Oracle (true equations)":"Oracle (true equations)","Quantile mapping (all subjects)":"Label-free mapping"}
sc=["Bias only","True case-mix shift only","Bias + case-mix shift"]
fig,ax=plt.subplots(figsize=(6.4,3.3)); w=0.25; cols=[C["raw"],"#4d4d4d",C["qm"]]
for i,mn in enumerate(meth):
    v=[s[(s.scenario==c)&(s.method==mn)] for c in sc]
    ax.bar(np.arange(3)+(i-1)*w,[x.O_E.iloc[0] for x in v],w,yerr=[x.O_E_sd.iloc[0] for x in v],capsize=2,color=cols[i],label=ml[mn],error_kw=dict(lw=0.8))
ax.axhline(1,color="black",lw=0.6,ls="--"); ax.set_xticks(range(3)); ax.set_xticklabels(["Analyser bias only","True case-mix shift only","Both"],fontsize=8)
ax.set_ylabel("Observed/expected ratio (mean ± SD, 5 runs)"); ax.legend(frameon=False,fontsize=7.5,ncol=3,loc="lower center",bbox_to_anchor=(0.5,1.0)); save(fig,"Fig6_simulation")
print("figures done"); print(fa.round(3).to_string())
