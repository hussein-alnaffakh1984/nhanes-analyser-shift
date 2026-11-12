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
# joint: all 11 analytes with APS biased at the desirable limit in the direction that lowers O/E, and all bridging 2017 differences together
Z = X.copy(); Zm = X.copy(); Zj = X.copy()
for c, nm in NAME.items():
    a = aps[aps.analyte == nm].iloc[0]
    Zj[c] = Zj[c] * (1 + a.J_mean_diff / 100)
    if a.APS_desirable == a.APS_desirable:
        sgn = -np.sign(oe_bias(c, 1) - 1); Z[c] = Z[c] * (1 + sgn * a.APS_desirable / 100); Zm[c] = Zm[c] * (1 + sgn * a.APS_minimum / 100)
for lab, D in [("11 analytes at desirable limit, worst direction", Z), ("11 analytes at minimum limit, worst direction", Zm), ("all 15 at 2017 bridging differences", Zj)]:
    print(lab, round(y.mean() / m3.predict_proba(D)[:, 1].mean() / base_oe, 3))
