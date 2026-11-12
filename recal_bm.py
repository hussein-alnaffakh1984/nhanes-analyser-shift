"""Recalibration on 2017-2020 (Cobas 6000) evaluated on 2021-2023 (Cobas 8000), design-based CIs."""
import numpy as np, pandas as pd, os, json
src = open("bm_analysis.py").read().split("# ------------------------------------------------------------------ 1. main analysis")[0]
os.environ.setdefault("BM_OUT", "res_bm/"); exec(src)
m17 = (base.cycle == "2017-2020").values; m21 = (base.cycle == "2021-2023").values
y17, y21 = y_all[m17], y_all[m21]; p17, p21 = P["T3"][m17], P["T3"][m21]; p0_21 = P["T0"][m21]
def recal(yu, pu, pt, kind):
    if kind == "intercept":
        a = citl(yu, pu, np.ones(len(yu))); return 1 / (1 + np.exp(-(logit(pt) + a)))
    lr = LogisticRegression(C=1e8, max_iter=2000).fit(logit(pu).reshape(-1, 1), yu); return lr.predict_proba(logit(pt).reshape(-1, 1))[:, 1]
rw = RaoWu(base[m21], SEED + 2000); mults = [rw.draw() for _ in range(R)]
rows = []
def summarise(lab, kind, ev, p):
    o = [oe(y21, p, mu) for mu in mults]; s = [slope(y21, p, mu) for mu in mults[:300]]
    rows.append(dict(update=lab, kind=kind, update_events=ev, OE_2021_23=fmt(oe(y21, p, np.ones(len(y21))), o),
                     slope_2021_23=fmt(slope(y21, p, np.ones(len(y21))), s), ratio_T3_T0=fmt(p0_21.sum() / p.sum(), [np.sum(mu * p0_21) / np.sum(mu * p) for mu in mults])))
summarise("none", "-", "-", p21)
for kind in ["intercept", "logistic"]:
    summarise("all 2017-2020", kind, int(y17.sum()), recal(y17, p17, p21, kind))
prev = y17.mean()
for ev in [50, 100, 250, 500]:
    for kind in ["intercept", "logistic"]:
        oes = []; sls = []
        for rep in range(200):
            g = np.random.default_rng(rep); n = int(ev / prev); i = g.choice(m17.sum(), n, replace=False)
            if y17[i].sum() < 5: continue
            pr = recal(y17[i], p17[i], p21, kind); oes.append(y21.mean() / pr.mean()); sls.append(slope(y21, pr, np.ones(len(y21))))
        rows.append(dict(update=f"random 2017-2020 subsample (~{ev} events), 200 repetitions", kind=kind, update_events=ev,
                         OE_2021_23=f"{np.median(oes):.3f} ({np.percentile(oes, 2.5):.3f}–{np.percentile(oes, 97.5):.3f})",
                         slope_2021_23=f"{np.median(sls):.3f} ({np.percentile(sls, 2.5):.3f}–{np.percentile(sls, 97.5):.3f})", ratio_T3_T0="-"))
pd.DataFrame(rows).to_csv(OUT + "recalibration.csv", index=False)
# net benefit 2021-2023 before/after logistic recalibration
pR = recal(y17, p17, p21, "logistic"); N = len(y21); d = []
for t in [0.05, 0.10, 0.15]:
    nb = lambda p: ((p >= t) & (y21 == 1)).sum() / N - ((p >= t) & (y21 == 0)).sum() / N * t / (1 - t)
    d.append(dict(threshold=t, treat_all=round(y21.mean() - (1 - y21.mean()) * t / (1 - t), 4), T0=round(nb(p0_21), 4),
                  T3_unadjusted=round(nb(p21), 4), T3_recalibrated=round(nb(pR), 4)))
pd.DataFrame(d).to_csv(OUT + "dca_recalibration_2021.csv", index=False)
print(pd.DataFrame(rows).to_string()); print(pd.DataFrame(d).to_string())
