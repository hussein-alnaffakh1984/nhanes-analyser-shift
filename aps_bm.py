"""Inter-platform bias from the CDC bridging studies vs EFLM biological-variation APS.

Bridging data: NHANES BIOPRO_J (Roche Cobas 6000 vs Beckman DxC 660i, n=248 samples from 2016) and
BIOPRO_L (Cobas 8000 vs Cobas 6000, n=152).  % DIFF = new minus old (sign verified from the equations).
Forward equations (conventional units):  J: Cobas6000 = slope*DxC + intercept ;  L: Cobas8000 = intercept + slope*Cobas6000.
"Adjustment not recommended" = no equation published (L criterion: 95% CI of slope includes 1 and of intercept includes 0).
Biological variation: EFLM Biological Variation Database estimates (as compiled, August 2025); only analytes with a
verified estimate are compared with APS.  APS: desirable bias = 0.25*sqrt(CVI^2+CVG^2); optimum 0.5x; minimum 1.5x.
"""
import numpy as np, pandas as pd, os
os.makedirs("res_bm", exist_ok=True)
BV = {  # CVI, CVG (%) - EFLM BVD, compiled August 2025
    "Albumin": (2.5, 4.3), "Calcium": (1.8, 2.2), "Glucose": (4.7, 8.0), "Potassium": (3.9, 5.3), "Sodium": (0.6, 0.7),
    "Total protein": (2.6, 4.4), "Chloride": (1.0, 1.3), "Cholesterol": (5.4, 15.6), "Triglycerides": (19.8, 34.3),
    "ALT": (12.6, 30.0), "AST": (8.4, 19.4)}
# name: (J %diff, J forward (slope, intercept) or None, L %diff, L forward (slope, intercept) or None, conventional unit,
#        decision levels (conventional), SI factor, SI unit)
BR = {
    "Albumin":       (-4.42, (0.9581, -0.0108), 1.48, None, "g/dL", [3.5], 10, "g/L"),
    "ALP":           (9.74, "log", 9.68, (1.106, -0.6745), "U/L", [130], 1, "U/L"),
    "AST":           (-18.03, (0.9822, -3.695), 0.59, None, "U/L", [40], 1, "U/L"),
    "ALT":           (-15.46, (0.9869, -2.653), 6.96, (0.9663, 1.477), "U/L", [40], 1, "U/L"),
    "Total CO2":     (1.74, None, -7.65, (0.9788, -1.341), "mmol/L", [22], 1, "mmol/L"),
    "Calcium":       (-2.12, None, 0.84, None, "mg/dL", [8.5, 10.5], 0.2495, "mmol/L"),
    "Cholesterol":   (-3.21, (0.9556, 2.105), 0.61, None, "mg/dL", [200], 0.02586, "mmol/L"),
    "Chloride":      (-1.44, None, -0.80, (1.088, -9.942), "mmol/L", [98, 107], 1, "mmol/L"),
    "Glucose":       (2.41, None, 4.31, (1.034, 0.8089), "mg/dL", [126], 0.0555, "mmol/L"),
    "Potassium":     (-1.398, None, 2.69, (1.083, -0.2283), "mmol/L", [3.5, 5.0], 1, "mmol/L"),
    "Sodium":        (0.80, None, 0.60, (1.432, -60.07), "mmol/L", [135, 145], 1, "mmol/L"),
    "Phosphate":     (-1.80, None, -2.23, None, "mg/dL", [2.5, 4.5], 0.3229, "mmol/L"),
    "Total protein": (0.36, None, 3.13, None, "g/dL", [6.0, 8.0], 10, "g/L"),
    "Triglycerides": (13.44, (1.036, 7.271), 2.22, (1.012, 0.8567), "mg/dL", [150], 0.01129, "mmol/L"),
    "Urate":         (2.37, (1.073, -0.2495), -0.04, None, "mg/dL", [6.0], 59.48, "µmol/L"),
}
def fwd(eq, x):
    if eq is None: return None
    if eq == "log": return 10 ** (0.9986 * np.log10(x) + 0.04288)        # J ALP, log10 Deming
    s, i = eq; return s * x + i
rows = []
for a, (dj, ej, dl, el, u, levels, f, usi) in BR.items():
    cvi, cvg = BV.get(a, (np.nan, np.nan))
    des = 0.25 * np.sqrt(cvi ** 2 + cvg ** 2)
    r = dict(analyte=a, CVI=cvi, CVG=cvg, APS_optimum=round(des / 2, 2), APS_desirable=round(des, 2), APS_minimum=round(1.5 * des, 2),
             J_mean_diff=dj, J_adjustment_recommended=ej is not None, L_mean_diff=dl, L_adjustment_recommended=el is not None,
             chained_mean_diff=round(((1 + dj / 100) * (1 + dl / 100) - 1) * 100, 2))
    for lab, d in [("J", dj), ("L", dl), ("chained", r["chained_mean_diff"])]:
        r[f"{lab}_exceeds_minimum"] = (abs(d) > 1.5 * des) if not np.isnan(des) else None
        r[f"{lab}_exceeds_desirable"] = (abs(d) > des) if not np.isnan(des) else None
    dl_txt = []
    for x in levels:
        bj = (fwd(ej, x) - x) / x * 100 if ej is not None else np.nan
        bl = (fwd(el, x) - x) / x * 100 if el is not None else np.nan
        dl_txt.append(f"{x * f:.3g} {usi}: J {bj:+.1f}% / L {bl:+.1f}%")
    r["bias_at_decision_levels_from_equations"] = "; ".join(dl_txt)
    rows.append(r)
t = pd.DataFrame(rows); t.to_csv("res_bm/aps_vs_bridging.csv", index=False)
pd.set_option("display.width", 250); print(t.drop(columns=["bias_at_decision_levels_from_equations"]).to_string())
print(t[["analyte", "bias_at_decision_levels_from_equations"]].to_string())
v = t.dropna(subset=["CVI"])
for lab in ["J", "L", "chained"]:
    ex = v[v[f"{lab}_exceeds_minimum"] == True].analyte.tolist()
    print(lab, "exceeds minimum APS:", len(ex), "of", len(v), ex)
exJ = v[(v.J_exceeds_minimum == True) & (v.J_adjustment_recommended)].analyte.tolist()
exL = v[(v.L_exceeds_minimum == True) & (v.L_adjustment_recommended)].analyte.tolist()
print("J exceeds minimum AND equation published:", exJ); print("L exceeds minimum AND equation published:", exL)
