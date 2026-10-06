import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":600})
R="res/"; O="figs/"
def save(f,n): f.savefig(O+n+".png",bbox_inches="tight"); f.savefig(O+n+".pdf",bbox_inches="tight"); plt.close(f)
# Fig2 subgroups with realistic tier
s=pd.read_csv(R+"rev_subgroups.csv"); groups=list(dict.fromkeys(s.group)); tiers=[("T0","Age + sex","#999999"),("T1","+ CBC","#e6ab02"),("TR","+ CBC, lipids, liver enzymes, HbA1c","#66a61e"),("T3","+ extended chemistry","#1f78b4")]
fig,ax=plt.subplots(figsize=(7.4,4.0)); xs=np.arange(len(groups))
for i,(t,l,c) in enumerate(tiers):
    q=s[s.tier==t].set_index("group").loc[groups]; x=xs+(i-1.5)*0.18
    ax.errorbar(x,q.AUC,yerr=[q.AUC-q.lo,q.hi-q.AUC],fmt="o",ms=4.5,capsize=2,lw=1,color=c,label=l)
ax.set_xticks(xs); ax.set_xticklabels([f"{g}\nn = {int(s[(s.group==g)].n.iloc[0]):,}\nevents = {int(s[(s.group==g)].events.iloc[0]):,}" for g in groups],fontsize=7.5)
ax.set_ylabel("AUC (95% CI), 2017–2023"); ax.set_ylim(0.55,0.97); ax.grid(axis="y",lw=0.3,alpha=0.6)
ax.legend(frameon=False,ncol=2,fontsize=7.5,loc="lower center",bbox_to_anchor=(0.5,1.0)); save(fig,"Fig2_discrimination_by_group")
# Fig3 with null band
pr=pd.read_csv(R+"probe1_drift_detection.csv"); nl=pd.read_csv(R+"rev_domain_null.csv")
inst=["DxC 800 → DxC 800","DxC 800 → DxC 800/660i","DxC 660i → Cobas 6000","Cobas 6000 → Cobas 8000"]
fig,ax=plt.subplots(figsize=(7.0,3.6)); x=np.arange(len(pr)); w=0.26
ax.axhspan(nl.AUC.min(),nl.AUC.max(),color="#dddddd",zorder=0,label=f"Within-cycle null ({nl.AUC.min():.2f}–{nl.AUC.max():.2f})")
for i,(cn,cc,lb) in enumerate([("Chemistry","#d95f02","Chemistry panel"),("CBC (control)","#1b9e77","Complete blood count"),("Age+Sex (population)","#7570b3","Age and sex")]):
    ax.bar(x+(i-1)*w,pr[cn]-0.45,w,bottom=0.45,color=cc,label=lb)
    for xi,v in zip(x+(i-1)*w,pr[cn]): ax.text(xi,v+0.008,f"{v:.2f}",ha="center",va="bottom",fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels([f"{t.replace(' -> ',' → ')}\n{ins}" for t,ins in zip(pr.transition,inst)],fontsize=7.2)
ax.set_ylabel("Domain-classifier AUC"); ax.set_ylim(0.45,1.05); ax.legend(frameon=False,fontsize=7,ncol=2,loc="lower center",bbox_to_anchor=(0.5,1.0)); save(fig,"Fig3_drift_detection")
# Fig5 signed
a=pd.read_csv(R+"table10_EFLM_APS_vs_observed_shift.csv").dropna(subset=["CDC_observed_diff_pct"]); a=a.reindex(a.CDC_observed_diff_pct.abs().sort_values().index)
nm={"alt":"ALT","ast":"AST","alp":"ALP†","uric_acid":"Uric acid","total_protein":"Total protein","bicarbonate":"Bicarbonate†"}
fig,ax=plt.subplots(figsize=(6.4,4.6)); y=np.arange(len(a))
ax.barh(y,2*a.bias_minimum,left=-a.bias_minimum,color="#e8e8e8",label="EFLM minimum specification (±)")
ax.barh(y,2*a.bias_desirable,left=-a.bias_desirable,color="#bdbdbd",label="EFLM desirable specification (±)")
ex=a.CDC_observed_diff_pct.abs()>a.bias_minimum
ax.scatter(a.CDC_observed_diff_pct[~ex],y[~ex.values],color="#2c7fb8",s=24,zorder=3,label="Within minimum specification")
ax.scatter(a.CDC_observed_diff_pct[ex],y[ex.values],color="#d95f02",s=24,zorder=3,label="Exceeds minimum specification")
ax.axvline(0,color="black",lw=0.6); ax.set_yticks(y); ax.set_yticklabels([nm.get(n,n.replace("_"," ").capitalize()) for n in a.analyte],fontsize=8)
ax.set_xlabel("Mean difference, Roche Cobas 6000 vs Beckman DxC 660i (%)"); ax.legend(frameon=False,fontsize=7,loc="lower left"); save(fig,"Fig5_bias_vs_APS")
# Fig6 simulation
s=pd.read_csv(R+"rev_simulation.csv"); sc=[("Bias only","Analyser bias only"),("Moderate case-mix shift only","Case-mix +25%"),("Bias + moderate case-mix shift","Bias + case-mix +25%"),("True case-mix shift only","Case-mix +100%")]
me=[("No correction","Unadjusted","#9e9e9e"),("Oracle (true equations)","Oracle","#404040"),("Quantile mapping (all subjects)","Quantile mapping","#1b9e77"),("Anchored QM (low-risk: age<40, no diabetes)","QM, low-risk subgroup","#a6dba0")]
fig,ax=plt.subplots(figsize=(7.0,3.5)); w=0.2
for i,(m,l,c) in enumerate(me):
    v=[s[(s.scenario==k)&(s.method==m)].iloc[0] for k,_ in sc]
    ax.bar(np.arange(4)+(i-1.5)*w,[q.O_E for q in v],w,yerr=[q.O_E_sd for q in v],capsize=2,color=c,label=l,error_kw=dict(lw=0.7))
ax.axhline(1,color="black",lw=0.6,ls="--"); ax.set_xticks(range(4)); ax.set_xticklabels([l for _,l in sc],fontsize=8)
ax.set_ylabel("Observed/expected ratio (mean ± SD)"); ax.legend(frameon=False,fontsize=7.5,ncol=4,loc="lower center",bbox_to_anchor=(0.5,1.0)); save(fig,"Fig6_simulation")
print("ok")
