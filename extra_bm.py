"""Additional analyses requested in the final verification:
(1) fasting time by period and T3/T0 ratio in participants fasting >= 8 h (design-based CI, 1000 replicates);
(2) domain classifier with all 15 chemistry analytes (sodium, chloride and glucose added to the 12 of the main analysis).
Outputs: res_bm/fasting.csv, res_bm/domain_classifier_15.csv"""
import os
src = open("bm_analysis.py").read().split("def cycle_analysis")[0]
exec(src)
e = pd.read_csv("extra_vars.csv")[["SEQN", "cycle", "PHAFSTHR"]]
fx = base[["SEQN", "cycle"]].merge(e, on=["SEQN", "cycle"], how="left").PHAFSTHR.values
rows = []
for lab, mk in [("Development 2011-2016", DEV), ("Validation 2017-2023", base.cycle.isin(NEW).values)]:
    v = fx[mk]; rows.append(dict(group=lab, n=int(mk.sum()), fasting_h_median=np.nanmedian(v),
                                  fasting_h_IQR=f"{np.nanpercentile(v,25):.0f}–{np.nanpercentile(v,75):.0f}",
                                  pct_fasting_ge8=round(np.nanmean(v >= 8) * 100, 1), missing=int(np.isnan(v).sum()), ratio_T3_T0_fasting_ge8=""))
for i, cyc in enumerate(NEW):
    m = (base.cycle == cyc).values; mf = m & (fx >= 8)
    sub = base[mf]; y = y_all[mf]; p0 = P["T0"][mf]; p3 = P["T3"][mf]
    rw = RaoWu(sub, SEED + 3000 + i); br, b3 = [], []
    for _ in range(R):
        w = rw.draw(); br.append(np.sum(w * p0) / np.sum(w * p3)); b3.append(oe(y, p3, w))
    rows.append(dict(group=f"{cyc}, fasting >= 8 h", n=int(mf.sum()), fasting_h_median=np.nanmedian(fx[m]),
                     fasting_h_IQR="", pct_fasting_ge8=round(np.nanmean(fx[m] >= 8) * 100, 1), missing=int(np.isnan(fx[m]).sum()),
                     ratio_T3_T0_fasting_ge8=fmt(p0.sum() / p3.sum(), br), OE_T3=fmt(oe(y, p3, np.ones(mf.sum())), b3), events=int(y.sum())))
pd.DataFrame(rows).to_csv(OUT + "fasting.csv", index=False); print(pd.DataFrame(rows).to_string())
from sklearn.model_selection import cross_val_predict as cvp
CH15 = list(QMCOLS)
def dauc(X, yy):
    p = cvp(HistGradientBoostingClassifier(max_iter=150, random_state=0), X, yy, cv=3, method="predict_proba")[:, 1]; return roc_auc_score(yy, p)
bk = base.copy(); bk["potassium_r1"] = bk.potassium_si.round(1); CH15r = [c if c != "potassium_si" else "potassium_r1" for c in CH15]
cyc = ["2011-2012", "2013-2014", "2015-2016", "2017-2020", "2021-2023"]; out = []
for a, b in zip(cyc[:-1], cyc[1:]):
    s = bk[bk.cycle.isin([a, b])]; yy = (s.cycle == b).astype(int).values
    nul = [dauc(s[CH15], np.random.default_rng(r).permutation(yy)) for r in range(20)]
    out.append(dict(transition=f"{a} -> {b}", chemistry15=round(dauc(s[CH15], yy), 3), chemistry15_K_rounded=round(dauc(s[CH15r], yy), 3),
                    perm_null_max=round(max(nul), 3)))
pd.DataFrame(out).to_csv(OUT + "domain_classifier_15.csv", index=False); print(pd.DataFrame(out).to_string()); print(CH15)
