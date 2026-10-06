src=open("revision.py").read(); exec(src.split("res={}")[0])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
b=base.copy(); b["Set"]=np.where(b.cycle.isin(OLD),"Dev","Val")
v=[("age","Age",0),("hemoglobin_si","Hb",0),("albumin_si","Alb",0),("uric_acid_si","UA",0),("potassium_si","K",1),("glucose_si","Glu",2)]
b["egfr_h"]=e_h
for (s,y),g in b.groupby(["Set","y"]):
    r=[s,y,len(g),round(g.sex_female.mean()*100,1),round((g.diabetes_dx==1).mean()*100,1),round((g.hypertension_dx==1).mean()*100,1)]
    for c,l,dp in v+[("egfr_h","eGFR",1)]:
        q=g[c].quantile([.25,.5,.75]).values; r.append(f"{q[1]:.{dp}f} ({q[0]:.{dp}f}–{q[2]:.{dp}f})")
    print(r)
miss=b.groupby(["Set","y"])[T3[2:]].apply(lambda x:x.isna().mean()*100).round(1).T; print(miss.max())
tr=b[b.cycle.isin(OLD)]; te=b[b.cycle.isin(NEW)]; y=te.y.values; N=len(y); th=np.linspace(0.02,0.30,57)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":600})
fig,ax=plt.subplots(figsize=(6,4)); ax.plot(th,y.mean()-(1-y.mean())*th/(1-th),"k:",lw=1,label="Test all"); ax.axhline(0,color="grey",lw=0.8,label="Test none")
for k,f,c in [("Age + sex",T0,"#999999"),("+ CBC, lipids, liver enzymes, HbA1c",TR,"#66a61e"),("+ extended chemistry",T3,"#1f78b4")]:
    p=LR().fit(tr[f],tr.y).predict_proba(te[f])[:,1]; nb=[((p>=t)&(y==1)).sum()/N-((p>=t)&(y==0)).sum()/N*t/(1-t) for t in th]; ax.plot(th,nb,color=c,lw=1.4,label=k)
    print(k,[round(nb[i],4) for i in [6,16,36]], "thr",[round(th[i],2) for i in [6,16,36]])
ax.set_ylim(-0.01,0.08); ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit"); ax.legend(frameon=False,fontsize=8)
fig.savefig("figs/FigS1_decision_curve.png",bbox_inches="tight"); fig.savefig("figs/FigS1_decision_curve.pdf",bbox_inches="tight")
