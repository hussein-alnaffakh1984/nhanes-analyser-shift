"""Model-based allowable bias: the relative bias of each chemistry analyte that changes the O/E ratio of the
extended model (T3) by 5% or 10% in the development data, compared with EFLM biological-variation limits
and with the inter-platform differences of the CDC bridging studies. Output: res_bm/model_based_allowable_bias.csv"""
src = open("bm_analysis.py").read().split("def cycle_analysis")[0]; exec(src)
from scipy.optimize import brentq
aps = pd.read_csv(OUT + "aps_vs_bridging.csv")
NAME = {"sodium_si": "Sodium", "potassium_si": "Potassium", "chloride_si": "Chloride", "bicarbonate_si": "Total CO2", "albumin_si": "Albumin",
        "total_protein_si": "Total protein", "calcium_si": "Calcium", "uric_acid_si": "Urate", "phosphorus_si": "Phosphate", "glucose_si": "Glucose",
        "alt_si": "ALT", "ast_si": "AST", "alp_si": "ALP", "cholesterol_si": "Cholesterol", "triglycerides_si": "Triglycerides"}
X = tr[T3].copy(); y = tr.y.values; m3 = models["T3"]; base_oe = y.mean() / m3.predict_proba(X)[:, 1].mean()
def oe_bias(col, b):
    Z = X.copy(); Z[col] = Z[col] * (1 + b / 100); return y.mean() / m3.predict_proba(Z)[:, 1].mean() / base_oe
rows = []
for c, nm in NAME.items():
    s1 = (oe_bias(c, 1) - 1) * 100                       # % change in O/E per +1% bias
    def lim(t):
        f = lambda b: abs(oe_bias(c, b) - 1) * 100 - t
        try: return brentq(f, 0, 60) if s1 != 0 else np.nan
        except ValueError: return np.nan
    def lim_neg(t):
        f = lambda b: abs(oe_bias(c, -b) - 1) * 100 - t
        try: return brentq(f, 0, 60)
        except ValueError: return np.nan
    a = aps[aps.analyte == nm].iloc[0]
    b5 = min(lim(5), lim_neg(5)); b10 = min(lim(10), lim_neg(10))
    rows.append(dict(analyte=nm, OE_change_pct_per_1pct_bias=round(s1, 2), bias_for_5pct_OE=round(b5, 2), bias_for_10pct_OE=round(b10, 2),
                     EFLM_desirable=a.APS_desirable, EFLM_minimum=a.APS_minimum,
                     OE_change_at_desirable=round(max(abs(oe_bias(c, a.APS_desirable) - 1), abs(oe_bias(c, -a.APS_desirable) - 1)) * 100, 1) if a.APS_desirable == a.APS_desirable else np.nan,
                     OE_change_at_minimum=round(max(abs(oe_bias(c, a.APS_minimum) - 1), abs(oe_bias(c, -a.APS_minimum) - 1)) * 100, 1) if a.APS_minimum == a.APS_minimum else np.nan,
                     bridging_2017=a.J_mean_diff, OE_change_at_bridging_2017=round((oe_bias(c, a.J_mean_diff) - 1) * 100, 1),
                     bridging_2021=a.L_mean_diff, OE_change_at_bridging_2021=round((oe_bias(c, a.L_mean_diff) - 1) * 100, 1)))
r = pd.DataFrame(rows).sort_values("bias_for_5pct_OE"); r.to_csv(OUT + "model_based_allowable_bias.csv", index=False)
pd.set_option("display.width", 250); print(r.to_string(index=False))
# joint effects (development data, model fixed), with design-based CI for the mean predicted risk
rw = RaoWu(tr, SEED + 4000); mults = [rw.draw() for _ in range(R)]
p_base = m3.predict_proba(X)[:, 1]
def joint(D):
    p = m3.predict_proba(D)[:, 1]; est = p_base.mean() / p.mean()
    return fmt(est, [np.sum(w * p_base) / np.sum(w * p) for w in mults])
Zd, Zm, Zj, Zc, Zn = X.copy(), X.copy(), X.copy(), X.copy(), X.copy()
for c, nm in NAME.items():
    a = aps[aps.analyte == nm].iloc[0]
    Zj[c] *= 1 + a.J_mean_diff / 100; Zc[c] *= (1 + a.J_mean_diff / 100) * (1 + a.L_mean_diff / 100)
    if not a.J_adjustment_recommended: Zn[c] *= 1 + a.J_mean_diff / 100
    if a.APS_desirable == a.APS_desirable:
        sgn = -np.sign(oe_bias(c, 1) - 1); Zd[c] *= 1 + sgn * a.APS_desirable / 100; Zm[c] *= 1 + sgn * a.APS_minimum / 100
jr = [("11 analytes biased at the desirable limit, direction lowering O/E", joint(Zd)),
      ("11 analytes biased at the minimum limit, direction lowering O/E", joint(Zm)),
      ("All 15 analytes at the 2017 bridging mean differences", joint(Zj)),
      ("Only the 8 analytes without a 2017 adjustment equation, at their 2017 mean differences", joint(Zn)),
      ("All 15 analytes at the chained 2017 and 2021 bridging mean differences", joint(Zc))]
pd.DataFrame(jr, columns=["scenario", "expected_OE_relative_to_unbiased"]).to_csv(OUT + "model_based_joint_bias.csv", index=False)
for a_, b_ in jr: print(a_, b_)
print("n without J equation:", int((~aps.J_adjustment_recommended.astype(bool)).sum()))

# ---- uncertainty and model dependence of the per-analyte sensitivity
# (a) refit T3 in 200 bootstrap samples of PSUs within strata; (b) gradient boosting T3; (c) sodium and chloride biased together
rng = np.random.default_rng(SEED + 4100)
key = (tr.strata.astype(int).astype(str) + "_" + tr.psu.astype(int).astype(str)).values; st = tr.strata.astype(int).values
idx_by = {k: np.where(key == k)[0] for k in np.unique(key)}; psu_by = {h: np.unique(key[st == h]) for h in np.unique(st)}
def per1(model, D, yv, c):
    Z = D.copy(); Z[c] = Z[c] * 1.01; return (model.predict_proba(D)[:, 1].mean() / model.predict_proba(Z)[:, 1].mean() - 1) * 100
boot = {c: [] for c in NAME}
for b in range(200):
    ii = np.concatenate([idx_by[k] for h in psu_by for k in rng.choice(psu_by[h], len(psu_by[h]), replace=True)])
    Db = tr.iloc[ii]; mb = LR().fit(Db[T3], Db.y)
    for c in NAME: boot[c].append(per1(mb, Db[T3], Db.y.values, c))
g3 = gbm["T3"]
unc = pd.DataFrame([dict(analyte=NAME[c], per1_logistic=round(per1(m3, X, y, c), 2), boot_lo=round(np.percentile(boot[c], 2.5), 2),
                         boot_hi=round(np.percentile(boot[c], 97.5), 2), per1_gradient_boosting=round(per1(g3, X, y, c), 2)) for c in NAME])
unc.to_csv(OUT + "model_based_bias_uncertainty.csv", index=False); print(unc.to_string(index=False))
Z = X.copy(); Z["sodium_si"] *= 1.008; Z["chloride_si"] *= 1.008
common = p_base.mean() / m3.predict_proba(Z)[:, 1].mean()
pd.DataFrame([dict(scenario="Sodium and chloride both biased by +0.8%", expected_OE_relative_to_unbiased=round(common, 3)),
              dict(scenario="Sodium alone biased by +0.8%", expected_OE_relative_to_unbiased=round(oe_bias("sodium_si", 0.8), 3))]).to_csv(OUT + "model_based_common_mode.csv", index=False)
print("common mode Na+Cl +0.8%:", round(common, 3), "Na alone:", round(oe_bias("sodium_si", 0.8), 3))
