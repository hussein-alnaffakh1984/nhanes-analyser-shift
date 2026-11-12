"""Revised analysis (Biochemia Medica submission).

One seeded script for every number in the revised manuscript:
  * design-based 95% CIs (Rao-Wu rescaling bootstrap of PSUs within strata, 1,000 replicates),
    unweighted (primary) and survey-weighted estimates;
  * statistical alignment (quantile mapping, location-scale) evaluated out of sample by
    5-fold cross-fitting within each validation cycle and prospectively (2017-2020 -> 2021-2023),
    with the alignment re-estimated in every bootstrap replicate;
  * paired bootstrap CIs for differences between alignment methods;
  * negative controls (2011-12 -> 2013-14; 2011-14 -> 2015-16) used as empirical nulls;
  * sensitivity analyses: extended reference model, age splines, gradient boosting,
    creatinine conversion direction, unconverted creatinine, training on 2011-2014 only;
  * Shapley attribution of the laboratory-input component to analyte groups;
  * race/ethnicity subgroups, decision curves (2-15%), sample-size check.
Outputs: res_bm/*.csv and res_bm/key_numbers.json
"""
import json, time, warnings, os
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from itertools import combinations
from math import factorial
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, SplineTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss
from statsmodels.nonparametric.smoothers_lowess import lowess

exec(open("revision.py").read().split("res={}")[0])   # df, base (y on harmonised creatinine scale), tiers, cdc(), EQ, EQL, SI, ...
OUT = os.environ.get("BM_OUT", "res_bm/")
import os; os.makedirs(OUT, exist_ok=True)
R = int(os.environ.get("BM_R", 1000))  # bootstrap replicates
SEED = 20261112
KEY = {}                     # key numbers for the manuscript
t_start = time.time()

# ------------------------------------------------------------------ helpers
def fmt(est, arr, d=3):
    lo, hi = np.percentile(arr, [2.5, 97.5]); return f"{est:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"

def oe(y, p, w): return np.sum(w * y) / np.sum(w * p)

def citl(y, p, w):
    lp = logit(p)
    f = lambda a: -np.sum(w * (y * (a + lp) - np.log1p(np.exp(a + lp))))
    return minimize_scalar(f, bounds=(-5, 5), method="bounded").x

def slope(y, p, w):
    return LogisticRegression(C=1e8, max_iter=2000).fit(logit(p).reshape(-1, 1), y, sample_weight=w).coef_[0][0]

def ici_e90(y, p):
    sm = lowess(y, p, frac=0.3, it=0, delta=0.002, return_sorted=False); e = np.abs(sm - p)
    return e.mean(), np.quantile(e, 0.9)

class RaoWu:
    """Rao-Wu rescaling bootstrap: in each stratum draw n_h-1 PSUs with replacement,
    multiplier = n_h/(n_h-1) x times drawn.  Returns replicate multipliers (R x n)."""
    def __init__(self, frame, seed):
        self.rng = np.random.default_rng(seed)
        key = frame.strata.astype(int).astype(str) + "_" + frame.psu.astype(int).astype(str)
        self.cl = pd.factorize(key)[0]
        st = frame.strata.astype(int).values
        self.groups = []
        for h in np.unique(st):
            psus = np.unique(self.cl[st == h]); self.groups.append(psus)
        self.ncl = self.cl.max() + 1
    def draw(self):
        mult = np.zeros(self.ncl)
        for psus in self.groups:
            n = len(psus)
            pick = self.rng.choice(psus, n - 1, replace=True)
            np.add.at(mult, pick, n / (n - 1))
        return mult[self.cl]

# ------------------------------------------------------------------ fast alignment (label-free)
STRAT = pd.factorize(strat(base))[0]
QS = np.linspace(0.005, 0.995, 199)
DEV = base.cycle.isin(OLD).values
GROUPS6 = {
    "Electrolytes (Na, K, Cl, total CO2)": ["sodium_si", "potassium_si", "chloride_si", "bicarbonate_si"],
    "Albumin, total protein, calcium": ["albumin_si", "total_protein_si", "calcium_si"],
    "Urate, phosphate": ["uric_acid_si", "phosphorus_si"],
    "Glucose": ["glucose_si"],
    "Lipids": ["cholesterol_si", "triglycerides_si"],
    "Liver enzymes (ALT, AST, ALP)": ["alt_si", "ast_si", "alp_si"],
}
ALIGN_COLS = QMCOLS  # 15 chemistry analytes

def wq(v, w, q):
    o = np.argsort(v); v, w = v[o], w[o]; cw = (np.cumsum(w) - 0.5 * w) / w.sum(); return np.interp(q, cw, v)

def ref_stats(ref_mask, weights=None):
    """reference distribution (fixed): quantiles, mean, sd per (analyte, stratum)."""
    st = {}
    for c in ALIGN_COLS:
        x = base[c].values
        for k in np.unique(STRAT):
            m = ref_mask & (STRAT == k) & ~np.isnan(x)
            if m.sum() < 50: continue
            if weights is None:
                q = np.quantile(x[m], QS); mu, sd = x[m].mean(), x[m].std(ddof=1)
            else:
                ww = weights[m]; q = wq(x[m], ww, QS); mu = np.average(x[m], weights=ww)
                sd = np.sqrt(np.average((x[m] - mu) ** 2, weights=ww))
            st[(c, k)] = (q, mu, sd)
    return st

def align(X, rows, fit_w, ref, cols, method, folds=None, apply_rows=None):
    """X: DataFrame copy to modify (all rows). rows: bool mask of rows forming the fit population.
    fit_w: weights (len n) used when fitting the target distribution (replicate x survey weights).
    folds: None -> fit on all `rows`, apply to `apply_rows` (default rows); else int array (fold id, -1 outside)
    cross-fitting: rows in fold f are mapped with a transfer function fitted on rows not in f."""
    apply_rows = rows if apply_rows is None else apply_rows
    out = {}
    for c in cols:
        x = base[c].values; newv = X[c].values.copy()
        for k in np.unique(STRAT):
            if (c, k) not in ref: continue
            qr, mur, sdr = ref[(c, k)]
            base_m = rows & (STRAT == k) & ~np.isnan(x) & (fit_w > 0)
            if folds is None:
                parts = [(base_m, apply_rows & (STRAT == k) & ~np.isnan(x))]
            else:
                parts = [(base_m & (folds != f), rows & (STRAT == k) & ~np.isnan(x) & (folds == f)) for f in range(folds.max() + 1)]
            for fm, am in parts:
                if fm.sum() < 20 or am.sum() == 0: continue
                if method == "qm":
                    qf = wq(x[fm], fit_w[fm], QS); newv[am] = np.interp(x[am], qf, qr)
                else:
                    mu = np.average(x[fm], weights=fit_w[fm]); sd = np.sqrt(np.average((x[fm] - mu) ** 2, weights=fit_w[fm]))
                    newv[am] = (x[am] - mu) / sd * sdr + mur
        out[c] = newv
    for c, v in out.items(): X[c] = v
    return X

# ------------------------------------------------------------------ models
tr = base[DEV]
models = {t: LR().fit(tr[f], tr.y) for t, f in TIERS.items()}
def GB(): return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, random_state=0)
gbm = {t: GB().fit(tr[TIERS[t]], tr.y) for t in ["T0", "T3"]}
# extended reference: + race/ethnicity, self-reported diabetes and hypertension
XCOV = ["race_eth", "diabetes_dx", "hypertension_dx"]
def LRx(feats):
    num = [f for f in feats if f != "race_eth"]
    ct = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), ["race_eth"])])
    return make_pipeline(ct, LogisticRegression(max_iter=3000))
modx = {"T0x": LRx(T0 + XCOV).fit(tr[T0 + XCOV], tr.y), "T3x": LRx(T3 + XCOV).fit(tr[T3 + XCOV], tr.y)}
# age as restricted spline
def LRs(feats):
    rest = [f for f in feats if f != "age"]
    ct = ColumnTransformer([("age", SplineTransformer(n_knots=4, degree=3), ["age"])] +
                           ([("rest", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), rest)] if rest else []))
    return make_pipeline(ct, LogisticRegression(max_iter=3000))
mods = {"T0s": LRs(T0).fit(tr[T0], tr.y), "T3s": LRs(T3).fit(tr[T3], tr.y)}
# training on 2011-2014 only (excludes 2015-2016 when the DxC 660i was added)
tr14 = base[base.cycle.isin(["2011-2012", "2013-2014"])]
mod14 = {t: LR().fit(tr14[TIERS[t]], tr14.y) for t in ["T0", "T3"]}

# model specification table (for Methods / Supplement)
coef = models["T3"][-1].coef_[0]
pd.DataFrame({"predictor": T3, "standardised_coef": coef.round(4), "OR_per_SD": np.exp(coef).round(3)}).to_csv(OUT + "lr_T3_coefficients.csv", index=False)
pd.DataFrame({"predictor": T0, "standardised_coef": models["T0"][-1].coef_[0].round(4)}).to_csv(OUT + "lr_T0_coefficients.csv", index=False)
KEY["lr_T3_intercept"] = float(models["T3"][-1].intercept_[0])

REF_U = ref_stats(DEV)
REF_W = ref_stats(DEV, weights=base.WT_MEC.values)
REF_NC1 = ref_stats(base.cycle.eq("2011-2012").values)
REF_NC2 = ref_stats(base.cycle.isin(["2011-2012", "2013-2014"]).values)

y_all = base.y.values
P = {t: models[t].predict_proba(base[TIERS[t]])[:, 1] for t in TIERS}
PG = {t: gbm[t].predict_proba(base[TIERS[t]])[:, 1] for t in gbm}
PX = {"T0x": modx["T0x"].predict_proba(base[T0 + XCOV])[:, 1], "T3x": modx["T3x"].predict_proba(base[T3 + XCOV])[:, 1]}
PS = {"T0s": mods["T0s"].predict_proba(base[T0])[:, 1], "T3s": mods["T3s"].predict_proba(base[T3])[:, 1]}
P14 = {t: mod14[t].predict_proba(base[TIERS[t]])[:, 1] for t in mod14}
CDCD = cdc(base)
P_cdc = {t: models[t].predict_proba(CDCD[TIERS[t]])[:, 1] for t in ["TR", "T3"]}
# paired-sample mean % differences (new minus old), CDC bridging tables BIOPRO_J (Cobas 6000 vs DxC 660i, n=248)
# and BIOPRO_L (Cobas 8000 vs Cobas 6000, n=152), applied to all 15 analytes regardless of the adjustment recommendation
D_J = {"albumin_si": -4.42, "alp_si": 9.74, "ast_si": -18.03, "alt_si": -15.46, "bicarbonate_si": 1.74, "calcium_si": -2.12,
       "cholesterol_si": -3.21, "chloride_si": -1.44, "glucose_si": 2.41, "potassium_si": -1.398, "sodium_si": 0.80,
       "phosphorus_si": -1.80, "total_protein_si": 0.36, "triglycerides_si": 13.44, "uric_acid_si": 2.37}
D_L = {"albumin_si": 1.48, "alp_si": 9.68, "ast_si": 0.59, "alt_si": 6.96, "bicarbonate_si": -7.65, "calcium_si": 0.84,
       "cholesterol_si": 0.61, "chloride_si": -0.80, "glucose_si": 4.31, "potassium_si": 2.69, "sodium_si": 0.60,
       "phosphorus_si": -2.23, "total_protein_si": 3.13, "triglycerides_si": 2.22, "uric_acid_si": -0.04}
def paired_diff(d, skip_L=()):
    d = d.copy(); m17 = d.cycle.isin(NEW); m21 = d.cycle == "2021-2023"
    for c, v in D_J.items(): d.loc[m17, c] = d.loc[m17, c] / (1 + v / 100)
    for c, v in D_L.items():
        if c not in skip_L: d.loc[m21, c] = d.loc[m21, c] / (1 + v / 100)
    return d
PDD = paired_diff(base); PDX = paired_diff(base, skip_L=("bicarbonate_si",))
P_pd = {t: models[t].predict_proba(PDD[TIERS[t]])[:, 1] for t in ["TR", "T3"]}
P_pdx = {t: models[t].predict_proba(PDX[TIERS[t]])[:, 1] for t in ["TR", "T3"]}
# single-step CDC (Cobas 6000 -> DxC only) and chained CDC without the Cobas 8000 total CO2 equation
CDC1 = cdc(base, chain=False); P_cdc1 = {t: models[t].predict_proba(CDC1[TIERS[t]])[:, 1] for t in ["TR", "T3"]}
CDCX = cdc(base, skip=("bicarbonate_si",))   # total CO2 has a CDC equation only for Cobas 8000 -> Cobas 6000
P_cdcx = {t: models[t].predict_proba(CDCX[TIERS[t]])[:, 1] for t in ["TR", "T3"]}
FOLDS_SEED = np.random.default_rng(SEED + 7)

def pred_aligned(method, rows, fit_w, ref, cols=ALIGN_COLS, folds=None, apply_rows=None, tiers=("TR", "T3")):
    X = align(base[TIERS["T3"]].copy(), rows, fit_w, ref, cols, method, folds, apply_rows)
    return {t: models[t].predict_proba(X[TIERS[t]])[:, 1] for t in tiers}

def make_folds(rows, rng, k=5):
    f = np.full(len(base), -1); idx = np.where(rows)[0]; f[idx] = rng.permutation(np.arange(len(idx)) % k); return f

# ------------------------------------------------------------------ 1. main analysis by validation cycle
def cycle_analysis(cyc, weighted, seed):
    m = (base.cycle == cyc).values; y = y_all[m]
    sw = base.WT_MEC.values if weighted else np.ones(len(base))
    rw = RaoWu(base[m], seed)
    ref = REF_W if weighted else REF_U
    # point estimates ---------------------------------------------------------
    w = sw[m]
    pt = {}
    for t in TIERS: pt[("none", t)] = P[t][m]
    for t in ["TR", "T3"]: pt[("cdc", t)] = P_cdc[t][m]
    for t in ["TR", "T3"]:
        pt[("pd", t)] = P_pd[t][m]; pt[("pdx", t)] = P_pdx[t][m]; pt[("cdc1", t)] = P_cdc1[t][m]; pt[("cdcx", t)] = P_cdcx[t][m]
    full_w = sw.copy()
    for meth in ["qm", "ls"]:
        insample = pred_aligned(meth, m, full_w, ref)                       # fitted and applied to the same rows
        for t in ["TR", "T3"]: pt[(meth + "_in", t)] = insample[t][m]
        cf = {t: np.zeros(m.sum()) for t in ["TR", "T3"]}; nrep = 20        # cross-fitted: average of 20 fold assignments
        for r_ in range(nrep):
            pr = pred_aligned(meth, m, full_w, ref, folds=make_folds(m, np.random.default_rng(SEED + 100 + r_)))
            for t in cf: cf[t] += pr[t][m] / nrep
        for t in cf: pt[(meth + "_cf", t)] = cf[t]
    for t in ["T0", "T3"]: pt[("gbm", t)] = PG[t][m]
    for t in PX: pt[("x", t)] = PX[t][m]
    for t in PS: pt[("s", t)] = PS[t][m]
    for t in P14: pt[("dev1114", t)] = P14[t][m]
    est = {k: oe(y, p, w) for k, p in pt.items()}
    # bootstrap ----------------------------------------------------------------
    bs = {k: [] for k in pt}; bs_auc = {k: [] for k in [("none", "T0"), ("none", "T3")]}
    bs_cal = {k: [] for k in [("none", "T3"), ("cdc", "T3"), ("qm_cf", "T3"), ("ls_cf", "T3")]}
    rng = np.random.default_rng(seed + 1)
    for r_ in range(R):
        mult = rw.draw(); ww = w * mult
        fw = np.zeros(len(base)); fw[m] = ww
        for k, p in pt.items():
            if k[0] in ("qm_cf", "ls_cf", "qm_in", "ls_in"): continue
            bs[k].append(oe(y, p, ww))
        for meth in ["qm", "ls"]:
            pr = pred_aligned(meth, m, fw, ref, folds=make_folds(m, rng))
            pi = pred_aligned(meth, m, fw, ref)
            for t in ["TR", "T3"]:
                bs[(meth + "_cf", t)].append(oe(y, pr[t][m], ww)); bs[(meth + "_in", t)].append(oe(y, pi[t][m], ww))
            if meth == "qm": p_qm = pr["T3"][m]
            else: p_ls = pr["T3"][m]
        keep = ww > 0
        for k in bs_auc: bs_auc[k].append(roc_auc_score(y[keep], pt[k][keep], sample_weight=ww[keep]))
        if r_ < R:
            for k, p in [(("none", "T3"), pt[("none", "T3")]), (("cdc", "T3"), pt[("cdc", "T3")]), (("qm_cf", "T3"), p_qm), (("ls_cf", "T3"), p_ls)]:
                bs_cal[k].append((citl(y[keep], p[keep], ww[keep]), slope(y[keep], p[keep], ww[keep])))
    bs = {k: np.array(v) for k, v in bs.items()}
    return dict(m=m, y=y, w=w, pt=pt, est=est, bs=bs, bs_auc={k: np.array(v) for k, v in bs_auc.items()},
                bs_cal={k: np.array(v) for k, v in bs_cal.items()})

results = {}
for weighted in [False, True]:
    for i, cyc in enumerate(NEW):
        results[(cyc, weighted)] = cycle_analysis(cyc, weighted, SEED + 10 * i + (5 if weighted else 0))
        print(f"cycle {cyc} weighted={weighted} done {time.time()-t_start:.0f}s", flush=True)

LAB = {"none": "No adjustment", "cdc": "CDC bridging (chained in 2021–2023)", "cdc1": "CDC bridging, single step (Cobas 6000 equations only)",
       "cdcx": "CDC bridging, chained, without the total CO2 equation", "pd": "Paired mean-difference correction (all 15 analytes)",
       "pdx": "Paired mean-difference correction, without Cobas 8000 total CO2", "qm_cf": "Quantile mapping, cross-fitted",
       "ls_cf": "Location–scale, cross-fitted", "qm_in": "Quantile mapping, in-sample", "ls_in": "Location–scale, in-sample"}
rows = []
for (cyc, weighted), r in results.items():
    y, w = r["y"], r["w"]
    r0 = r["bs"][("none", "T0")]
    for meth in ["none", "cdc", "cdc1", "cdcx", "pd", "pdx", "qm_cf", "ls_cf", "qm_in", "ls_in"]:
        k3 = (meth, "T3"); kR = (meth, "TR")
        ratio_est = r["est"][k3] / r["est"][("none", "T0")]; ratio_bs = r["bs"][k3] / r0
        ratioR_est = r["est"][kR] / r["est"][("none", "T0")]; ratioR_bs = r["bs"][kR] / r0
        row = dict(cycle=cyc, weighted=weighted, method=LAB[meth],
                   OE_T3=fmt(r["est"][k3], r["bs"][k3]), ratio_T3_T0=fmt(ratio_est, ratio_bs),
                   OE_TR=fmt(r["est"][kR], r["bs"][kR]), ratio_TR_T0=fmt(ratioR_est, ratioR_bs))
        if k3 in r["bs_cal"]:
            p = r["pt"][k3]; cal = r["bs_cal"][k3]
            row["CITL_T3"] = fmt(citl(y, p, w), cal[:, 0]); row["slope_T3"] = fmt(slope(y, p, w), cal[:, 1])
            if not weighted:
                i_, e90 = ici_e90(y, p); row["ICI_T3"] = round(i_, 3); row["E90_T3"] = round(e90, 3)
                row["Brier_T3"] = round(brier_score_loss(y, p), 4)
        rows.append(row)
        KEY[f"{cyc}|{'w' if weighted else 'u'}|{meth}|ratio"] = [round(ratio_est, 3)] + list(np.round(np.percentile(ratio_bs, [2.5, 97.5]), 3))
        KEY[f"{cyc}|{'w' if weighted else 'u'}|{meth}|OE_T3"] = [round(r["est"][k3], 3)] + list(np.round(np.percentile(r["bs"][k3], [2.5, 97.5]), 3))
    for t in ["T0", "T1", "TR"]:
        KEY[f"{cyc}|{'w' if weighted else 'u'}|none|OE_{t}"] = [round(r["est"][("none", t)], 3)] + list(np.round(np.percentile(r["bs"][("none", t)], [2.5, 97.5]), 3))
    for k in r["bs_auc"]:
        a = roc_auc_score(y, r["pt"][k], sample_weight=w)
        KEY[f"{cyc}|{'w' if weighted else 'u'}|AUC_{k[1]}"] = [round(a, 3)] + list(np.round(np.percentile(r["bs_auc"][k], [2.5, 97.5]), 3))
    # paired differences between methods (|ln ratio|)
    lr_ = {meth: np.abs(np.log(r["bs"][(meth, "T3")] / r0)) for meth in ["cdc", "pd", "qm_cf", "ls_cf", "qm_in", "ls_in"]}
    lr_e = {meth: abs(np.log(r["est"][(meth, "T3")] / r["est"][("none", "T0")])) for meth in lr_}
    for a_, b_ in [("ls_cf", "qm_cf"), ("ls_cf", "cdc"), ("qm_cf", "cdc"), ("qm_cf", "qm_in"), ("ls_cf", "ls_in"), ("pd", "cdc"), ("ls_cf", "pd")]:
        d = lr_[a_] - lr_[b_]; p2 = 2 * min((d > 0).mean(), (d < 0).mean())
        rows.append(dict(cycle=cyc, weighted=weighted, method=f"DIFF |ln ratio|: {LAB[a_]} minus {LAB[b_]}",
                         ratio_T3_T0=fmt(lr_e[a_] - lr_e[b_], d), OE_T3=f"bootstrap P={max(p2, 1/R):.3f}"))
        KEY[f"{cyc}|{'w' if weighted else 'u'}|diff|{a_}-{b_}"] = [round(lr_e[a_] - lr_e[b_], 3)] + list(np.round(np.percentile(d, [2.5, 97.5]), 3)) + [round(max(p2, 1/R), 3)]
    # sensitivity models (ratio T3/T0 within the same family)
    for fam, a_, b_, lab in [("gbm", "T3", "T0", "Gradient boosting T3/T0"), ("x", "T3x", "T0x", "Extended reference (T3x/T0x: + race/ethnicity, diabetes, hypertension)"),
                             ("s", "T3s", "T0s", "Age as restricted cubic spline (T3/T0)"), ("dev1114", "T3", "T0", "Development 2011–2014 only (T3/T0)")]:
        e_ = r["est"][(fam, a_)] / r["est"][(fam, b_)]; b = r["bs"][(fam, a_)] / r["bs"][(fam, b_)]
        rows.append(dict(cycle=cyc, weighted=weighted, method="SENS " + lab, ratio_T3_T0=fmt(e_, b),
                         OE_T3=fmt(r["est"][(fam, a_)], r["bs"][(fam, a_)])))
        KEY[f"{cyc}|{'w' if weighted else 'u'}|sens|{fam}"] = [round(e_, 3)] + list(np.round(np.percentile(b, [2.5, 97.5]), 3)) + [round(r["est"][(fam, a_)], 3)]
main = pd.DataFrame(rows); main.to_csv(OUT + "main_by_cycle.csv", index=False)
print(f"main done {time.time()-t_start:.0f}s", flush=True)

# ------------------------------------------------------------------ 2. prospective alignment: fitted on 2017-2020, applied to 2021-2023
def prospective(weighted, seed):
    m17 = (base.cycle == "2017-2020").values; m21 = (base.cycle == "2021-2023").values; y = y_all[m21]
    sw = base.WT_MEC.values if weighted else np.ones(len(base)); ref = REF_W if weighted else REF_U
    out = {}
    rw17, rw21 = RaoWu(base[m17], seed), RaoWu(base[m21], seed + 1)
    for meth in ["qm", "ls"]:
        p = pred_aligned(meth, m17, sw, ref, apply_rows=m21)["T3"][m21]
        est = oe(y, p, sw[m21]) / oe(y, P["T0"][m21], sw[m21]); bs = []; bso = []
        for _ in range(R // 2):
            fw = np.zeros(len(base)); fw[m17] = sw[m17] * rw17.draw(); w21 = sw[m21] * rw21.draw()
            pp = pred_aligned(meth, m17, fw, ref, apply_rows=m21)["T3"][m21]
            bs.append(oe(y, pp, w21) / oe(y, P["T0"][m21], w21)); bso.append(oe(y, pp, w21))
        out[meth] = (est, np.array(bs), oe(y, p, sw[m21]), np.array(bso))
    return out
pros = []
for weighted in [False, True]:
    o = prospective(weighted, SEED + 300 + weighted)
    for meth, (e_, b, eo, bo) in o.items():
        pros.append(dict(weighted=weighted, method=meth, ratio_T3_T0=fmt(e_, b), OE_T3=fmt(eo, bo)))
        KEY[f"prospective|{'w' if weighted else 'u'}|{meth}"] = [round(e_, 3)] + list(np.round(np.percentile(b, [2.5, 97.5]), 3)) + [round(eo, 3)]
pd.DataFrame(pros).to_csv(OUT + "prospective_alignment.csv", index=False)
print(f"prospective done {time.time()-t_start:.0f}s", flush=True)

# ------------------------------------------------------------------ 3. negative controls (empirical null)
def neg_control(train_cycles, test_cycle, ref, seed, weighted=False):
    trn = base[base.cycle.isin(train_cycles)]; mm = {t: LR().fit(trn[TIERS[t]], trn.y) for t in ["T0", "TR", "T3"]}
    m = (base.cycle == test_cycle).values; y = y_all[m]; sw = base.WT_MEC.values if weighted else np.ones(len(base)); w = sw[m]
    p0 = mm["T0"].predict_proba(base.loc[m, T0])[:, 1]; p3 = mm["T3"].predict_proba(base.loc[m, T3])[:, 1]
    def al(meth, fw, folds):
        X = align(base[T3].copy(), m, fw, ref, ALIGN_COLS, meth, folds); return mm["T3"].predict_proba(X.loc[m, T3])[:, 1]
    pqm = np.mean([al("qm", sw, make_folds(m, np.random.default_rng(SEED + 500 + i))) for i in range(10)], axis=0)
    pls = np.mean([al("ls", sw, make_folds(m, np.random.default_rng(SEED + 600 + i))) for i in range(10)], axis=0)
    rw = RaoWu(base[m], seed); rng = np.random.default_rng(seed + 9); b = {k: [] for k in ["none", "qm", "ls", "oe0", "oe3"]}
    for _ in range(R):
        mult = rw.draw(); ww = w * mult; fw = np.zeros(len(base)); fw[m] = ww
        b["none"].append(np.sum(ww * p0) / np.sum(ww * p3)); b["oe0"].append(oe(y, p0, ww)); b["oe3"].append(oe(y, p3, ww))
        b["qm"].append(np.sum(ww * p0) / np.sum(ww * al("qm", fw, make_folds(m, rng))))
        b["ls"].append(np.sum(ww * p0) / np.sum(ww * al("ls", fw, make_folds(m, rng))))
    res = dict(design=f"train {'+'.join(train_cycles)} -> test {test_cycle}", weighted=weighted, n_train=len(trn), events_train=int(trn.y.sum()),
               n_test=int(m.sum()), events_test=int(y.sum()),
               OE_T0=fmt(oe(y, p0, w), b["oe0"]), OE_T3=fmt(oe(y, p3, w), b["oe3"]),
               ratio_none=fmt(np.sum(w * p0) / np.sum(w * p3), b["none"]),
               ratio_qm_cf=fmt(np.sum(w * p0) / np.sum(w * pqm), b["qm"]), ratio_ls_cf=fmt(np.sum(w * p0) / np.sum(w * pls), b["ls"]))
    return res
ncs = [neg_control(["2011-2012"], "2013-2014", REF_NC1, SEED + 700),
       neg_control(["2011-2012", "2013-2014"], "2015-2016", REF_NC2, SEED + 710),
       neg_control(["2011-2012"], "2013-2014", ref_stats(base.cycle.eq("2011-2012").values, weights=base.WT_MEC.values), SEED + 720, weighted=True)]
pd.DataFrame(ncs).to_csv(OUT + "negative_controls.csv", index=False)
for i, r in enumerate(ncs): KEY[f"NC{i+1}"] = r
print(f"negative controls done {time.time()-t_start:.0f}s", flush=True)

# ------------------------------------------------------------------ 4. Shapley attribution of the laboratory-input component (cross-fitted QM, unweighted)
shap_rows = []
gnames = list(GROUPS6)
for cyc in NEW:
    m = (base.cycle == cyc).values; y = y_all[m]; one = np.ones(len(base)); p0 = P["T0"][m]
    folds = [make_folds(m, np.random.default_rng(SEED + 800 + i)) for i in range(5)]
    cache = {}
    def lnratio(S):
        if S in cache: return cache[S]
        if not S: v = np.log(np.sum(p0) / np.sum(P["T3"][m]))
        else:
            cols = sum([GROUPS6[g] for g in S], [])
            pp = np.mean([models["T3"].predict_proba(align(base[T3].copy(), m, one, REF_U, cols, "qm", f).loc[m, T3])[:, 1] for f in folds], axis=0)
            v = np.log(np.sum(p0) / np.sum(pp))
        cache[S] = v; return v
    n = len(gnames); full = lnratio(tuple(gnames)); empty = lnratio(())
    for g in gnames:
        phi = 0.0; others = [x for x in gnames if x != g]
        for k in range(n):
            for S in combinations(others, k):
                wgt = factorial(k) * factorial(n - k - 1) / factorial(n)
                S1 = tuple(sorted(S + (g,), key=gnames.index)); S0 = tuple(sorted(S, key=gnames.index))
                phi += wgt * (lnratio(S1) - lnratio(S0))
        shap_rows.append(dict(cycle=cyc, group=g, shapley_ln_ratio=round(phi, 4),
                              share_of_total_pct=round(100 * phi / (full - empty), 1),
                              one_at_a_time_ln_ratio=round(lnratio((g,)) - empty, 4)))
    shap_rows.append(dict(cycle=cyc, group="TOTAL (all groups aligned)", shapley_ln_ratio=round(full - empty, 4),
                          share_of_total_pct=100.0, one_at_a_time_ln_ratio=np.nan))
    KEY[f"{cyc}|lnratio_unadjusted"] = round(empty, 4); KEY[f"{cyc}|lnratio_allaligned"] = round(full, 4)
pd.DataFrame(shap_rows).to_csv(OUT + "shapley_attribution.csv", index=False)
print(f"shapley done {time.time()-t_start:.0f}s", flush=True)

# ------------------------------------------------------------------ 5. creatinine scale sensitivity
# (a) unconverted Roche creatinine for the outcome; (b) development creatinine converted to the Roche (enzymatic) scale
def egfr_lt60(cr, age, fem): return (egfr(cr, age, fem) < 60).astype(int)
fem = (base.sex_female == 1).values
y_raw = egfr_lt60(df.creatinine_si.values, base.age.values, fem)
cr_dev_to_roche = np.where(DEV, ((df.creatinine_si.values / SI["creatinine_si"] + 0.06945) / 1.051) * SI["creatinine_si"], df.creatinine_si.values)
y_roche = egfr_lt60(cr_dev_to_roche, base.age.values, fem)
cs = []
for lab, yy in [("Outcome from unconverted creatinine (validation on Roche scale, development on Beckman scale)", y_raw),
                ("Development creatinine converted to the Roche scale; validation unconverted", y_roche)]:
    mm = {t: LR().fit(base.loc[DEV, TIERS[t]], yy[DEV]) for t in ["T0", "T3"]}
    for i, cyc in enumerate(NEW):
        m = (base.cycle == cyc).values; y = yy[m]
        p0 = mm["T0"].predict_proba(base.loc[m, T0])[:, 1]; p3 = mm["T3"].predict_proba(base.loc[m, T3])[:, 1]
        rw = RaoWu(base[m], SEED + 900 + i); b0, b3, br = [], [], []
        for _ in range(R):
            ww = rw.draw(); b0.append(oe(y, p0, ww)); b3.append(oe(y, p3, ww)); br.append(np.sum(ww * p0) / np.sum(ww * p3))
        one = np.ones(m.sum())
        cs.append(dict(analysis=lab, cycle=cyc, prevalence_dev=round(yy[DEV].mean() * 100, 2), prevalence_val=round(y.mean() * 100, 2),
                       OE_T0=fmt(oe(y, p0, one), b0), OE_T3=fmt(oe(y, p3, one), b3), ratio=fmt(np.sum(p0) / np.sum(p3), br)))
pd.DataFrame(cs).to_csv(OUT + "creatinine_scale_sensitivity.csv", index=False)
print(f"creatinine sensitivity done {time.time()-t_start:.0f}s", flush=True)

# ------------------------------------------------------------------ 6. pooled validation: discrimination, race/ethnicity, DCA, internal CV
mv = base.cycle.isin(NEW).values; yv = y_all[mv]
rwv = RaoWu(base[mv], SEED + 1000)
mults = [rwv.draw() for _ in range(R)]
pool = []
for t in ["T0", "T1", "TR", "T3"]:
    p = P[t][mv]; pg = PG.get(t)
    a = [roc_auc_score(yv[mu > 0], p[mu > 0], sample_weight=mu[mu > 0]) for mu in mults]
    o = [oe(yv, p, mu) for mu in mults]
    d = [roc_auc_score(yv[mu > 0], p[mu > 0], sample_weight=mu[mu > 0]) - roc_auc_score(yv[mu > 0], P["T0"][mv][mu > 0], sample_weight=mu[mu > 0]) for mu in mults[:R]] if t != "T0" else None
    row = dict(tier=t, n_predictors=len(TIERS[t]), AUC=fmt(roc_auc_score(yv, p), a), OE=fmt(yv.mean() / p.mean(), o),
               dAUC_vs_T0=fmt(roc_auc_score(yv, p) - roc_auc_score(yv, P["T0"][mv]), d) if d else "reference",
               Brier=round(brier_score_loss(yv, p), 4))
    if pg is not None: row.update(GBM_AUC=round(roc_auc_score(yv, pg[mv]), 3), GBM_OE=round(yv.mean() / pg[mv].mean(), 3))
    pool.append(row)
pd.DataFrame(pool).to_csv(OUT + "pooled_validation.csv", index=False)
# internal cross-validation
from sklearn.model_selection import StratifiedKFold, cross_val_predict
icv = []
for t in TIERS:
    p = cross_val_predict(LR(), tr[TIERS[t]], tr.y, cv=StratifiedKFold(5, shuffle=True, random_state=1), method="predict_proba")[:, 1]
    icv.append(dict(tier=t, AUC=round(roc_auc_score(tr.y, p), 3), OE=round(tr.y.mean() / p.mean(), 3), Brier=round(brier_score_loss(tr.y, p), 4)))
pd.DataFrame(icv).to_csv(OUT + "internal_cv.csv", index=False)
# race/ethnicity (NHANES RIDRETH3: 1 Mexican American, 2 Other Hispanic, 3 Non-Hispanic White, 4 Non-Hispanic Black, 6 Non-Hispanic Asian, 7 Other/multiracial)
RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other or multiracial"}
rr = []
race_v = base.race_eth.values[mv]
for code, lab in RACE.items():
    g = race_v == code; yg = yv[g]
    row = dict(group=lab, n=int(g.sum()), events=int(yg.sum()))
    for t in ["T0", "T3"]:
        p = P[t][mv][g]
        a = [roc_auc_score(yg[mu[g] > 0], p[mu[g] > 0], sample_weight=mu[g][mu[g] > 0]) for mu in mults[:500]]
        o = [oe(yg, p, mu[g]) for mu in mults[:500]]
        row[f"AUC_{t}"] = fmt(roc_auc_score(yg, p), a); row[f"OE_{t}"] = fmt(yg.mean() / p.mean(), o)
    rr.append(row)
pd.DataFrame(rr).to_csv(OUT + "race_ethnicity.csv", index=False)
# decision curves, thresholds 2-15%
dca = []
for th in [0.02, 0.05, 0.075, 0.10, 0.125, 0.15]:
    N = len(yv); row = dict(threshold=th, treat_all=round(yv.mean() - (1 - yv.mean()) * th / (1 - th), 4))
    for t in ["T0", "TR", "T3"]:
        p = P[t][mv]; row[t] = round(((p >= th) & (yv == 1)).sum() / N - ((p >= th) & (yv == 0)).sum() / N * th / (1 - th), 4)
    dca.append(row)
pd.DataFrame(dca).to_csv(OUT + "decision_curve.csv", index=False)
# sample size (Riley et al. 2021, external validation, O/E precision)
ss = []
for cyc in NEW:
    m = (base.cycle == cyc).values; n = m.sum(); phi = y_all[m].mean()
    se = np.sqrt((1 - phi) / (n * phi)); n_req = (1 - phi) / (phi * 0.051 ** 2)
    ss.append(dict(cycle=cyc, n=int(n), events=int(y_all[m].sum()), prevalence=round(phi, 4), SE_lnOE=round(se, 4),
                   expected_95CI_width_OE_at_1=round(np.exp(1.96 * se) - np.exp(-1.96 * se), 3), n_required_for_width_0_2=int(np.ceil(n_req))))
dev_events = int(tr.y.sum()); ss.append(dict(cycle="development 2011-2016", n=int(len(tr)), events=dev_events,
                                             prevalence=round(tr.y.mean(), 4), SE_lnOE=None, expected_95CI_width_OE_at_1=None,
                                             n_required_for_width_0_2=f"events per predictor parameter T3 = {dev_events/len(T3):.1f}"))
pd.DataFrame(ss).to_csv(OUT + "sample_size.csv", index=False)
print(f"pooled done {time.time()-t_start:.0f}s", flush=True)

# ------------------------------------------------------------------ 7. domain classifier with permutation null and the potassium-precision check
from sklearn.model_selection import cross_val_predict as cvp
CHEM = ["albumin_si", "alt_si", "ast_si", "alp_si", "uric_acid_si", "cholesterol_si", "triglycerides_si", "potassium_si", "calcium_si", "total_protein_si", "phosphorus_si", "bicarbonate_si"]
CBCc = CBC; dom = []
cycles = ["2011-2012", "2013-2014", "2015-2016", "2017-2020", "2021-2023"]
def dauc(X, yy, seed=0):
    p = cvp(HistGradientBoostingClassifier(max_iter=150, random_state=0), X, yy, cv=3, method="predict_proba")[:, 1]; return roc_auc_score(yy, p)
bk = base.copy(); bk["potassium_r1"] = bk.potassium_si.round(1)
CHEMr = [c if c != "potassium_si" else "potassium_r1" for c in CHEM]
for a, b in zip(cycles[:-1], cycles[1:]):
    s = bk[bk.cycle.isin([a, b])]; yy = (s.cycle == b).astype(int).values
    row = dict(transition=f"{a} -> {b}", chemistry=round(dauc(s[CHEM], yy), 3), chemistry_K_rounded=round(dauc(s[CHEMr], yy), 3),
               chemistry_without_K=round(dauc(s[[c for c in CHEM if c != 'potassium_si']], yy), 3),
               CBC=round(dauc(s[CBCc], yy), 3), age_sex=round(dauc(s[T0], yy), 3))
    nul = []
    for rep in range(20):
        yp = np.random.default_rng(rep).permutation(yy); nul.append(dauc(s[CHEM], yp))
    row["perm_null_chem_max"] = round(max(nul), 3); row["perm_null_chem_mean"] = round(np.mean(nul), 3)
    nul2 = [dauc(s[T0], np.random.default_rng(100 + rep).permutation(yy)) for rep in range(20)]
    row["perm_null_agesex_max"] = round(max(nul2), 3)
    dom.append(row)
pd.DataFrame(dom).to_csv(OUT + "domain_classifier.csv", index=False)
prec = {c: base.groupby("cycle")[c].nunique().to_dict() for c in ["potassium_si", "sodium_si", "calcium_si"]}
pd.DataFrame(prec).to_csv(OUT + "distinct_values_by_cycle.csv")
print(f"domain done {time.time()-t_start:.0f}s", flush=True)

# ------------------------------------------------------------------ 8. Table 1 (median (IQR); age median (min-max))
b2 = base.copy(); b2["egfr_h"] = e_h; b2["Set"] = np.where(DEV, "Development", "Validation")
t1 = []
for (s_, yv_), g in b2.groupby(["Set", "y"]):
    row = dict(set=s_, outcome=int(yv_), n=len(g), female_pct=round(g.sex_female.mean() * 100, 1),
               diabetes_pct=round((g.diabetes_dx == 1).mean() * 100, 1), hypertension_pct=round((g.hypertension_dx == 1).mean() * 100, 1),
               age=f"{g.age.median():.0f} ({g.age.min():.0f}–{g.age.max():.0f})")
    for c, dp in [("egfr_h", 1), ("hemoglobin_si", 0), ("albumin_si", 0), ("uric_acid_si", 0), ("potassium_si", 1), ("sodium_si", 0), ("chloride_si", 0), ("bicarbonate_si", 0), ("glucose_si", 2)]:
        q = g[c].quantile([.25, .5, .75]).values; row[c] = f"{q[1]:.{dp}f} ({q[0]:.{dp}f}–{q[2]:.{dp}f})"
    t1.append(row)
pd.DataFrame(t1).to_csv(OUT + "table1.csv", index=False)
KEY["age_top80_pct"] = round((base.age >= 80).mean() * 100, 1)

with open(OUT + "key_numbers.json", "w") as f: json.dump(KEY, f, indent=1, default=str)
print(f"ALL DONE {time.time()-t_start:.0f}s")
