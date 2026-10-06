src=open("revision.py").read(); head=src.split("res={}")[0]; exec(head)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":600})
tr=base[base.cycle.isin(OLD)]; mdl=LR().fit(tr[T3],tr.y); cd=cdc(base)
fig=plt.figure(figsize=(7.4,4.4)); gs=fig.add_gridspec(2,2,height_ratios=[4,1],hspace=0.08)
grid=np.linspace(0,0.5,101)
for j,c in enumerate(NEW):
    ax=fig.add_subplot(gs[0,j]); axh=fig.add_subplot(gs[1,j],sharex=ax); m=(base.cycle==c).values; y=base.y.values[m]
    q=qm(base,base.cycle.isin(OLD),base.cycle==c)
    for lab,dd,col in [("Unadjusted",base,"#7f7f7f"),("CDC bridging (chained)",cd,"#d95f02"),("Quantile mapping",q,"#1b9e77")]:
        p=mdl.predict_proba(dd.loc[m,T3])[:,1]
        f=lambda yy,pp: np.interp(grid,*lowess(yy,pp,frac=0.3,it=0).T)
        est=f(y,p); g=np.random.default_rng(0); bs=np.array([f(y[i],p[i]) for i in [g.integers(0,len(y),len(y)) for _ in range(150)]])
        lo,hi=np.percentile(bs,[2.5,97.5],axis=0); mx=np.quantile(p,0.99); k=grid<=mx
        ax.plot(grid[k],est[k],color=col,lw=1.4,label=f"{lab} (O/E {y.mean()/p.mean():.2f})"); ax.fill_between(grid[k],lo[k],hi[k],color=col,alpha=0.18,lw=0)
        if lab=="Unadjusted": axh.hist(p,bins=60,range=(0,0.5),color="#7f7f7f")
    ax.plot([0,0.5],[0,0.5],"k--",lw=0.6); ax.set_xlim(0,0.5); ax.set_ylim(0,0.5); ax.set_title(c.replace("-","–")); ax.legend(frameon=False,fontsize=7,loc="upper left")
    plt.setp(ax.get_xticklabels(),visible=False); axh.set_xlabel("Predicted risk"); axh.set_ylabel("Count", fontsize=8)
    if j==0: ax.set_ylabel("Observed proportion (loess)")
fig.savefig("figs/Fig4_calibration_by_cycle.png",bbox_inches="tight"); fig.savefig("figs/Fig4_calibration_by_cycle.pdf",bbox_inches="tight")
