"""Population mean shift (validation cycle vs 2011-2016) compared with the paired-sample mean difference (CDC bridging)."""
import pandas as pd, numpy as np
d = pd.read_csv("nhanes_ckd_dataset.csv"); dev = d[d.cycle.isin(["2011-2012", "2013-2014", "2015-2016"])]
J = {"sodium_si": 0.80, "potassium_si": -1.398, "chloride_si": -1.44, "bicarbonate_si": 1.74, "albumin_si": -4.42, "total_protein_si": 0.36,
     "calcium_si": -2.12, "uric_acid_si": 2.37, "phosphorus_si": -1.80, "glucose_si": 2.41, "alt_si": -15.46, "ast_si": -18.03, "alp_si": 9.74,
     "cholesterol_si": -3.21, "triglycerides_si": 13.44}
L = {"sodium_si": 0.60, "potassium_si": 2.69, "chloride_si": -0.80, "bicarbonate_si": -7.65, "albumin_si": 1.48, "total_protein_si": 3.13,
     "calcium_si": 0.84, "uric_acid_si": -0.04, "phosphorus_si": -2.23, "glucose_si": 4.31, "alt_si": 6.96, "ast_si": 0.59, "alp_si": 9.68,
     "cholesterol_si": 0.61, "triglycerides_si": 2.22}
NAME = {"sodium_si": "Sodium", "potassium_si": "Potassium", "chloride_si": "Chloride", "bicarbonate_si": "Total CO2", "albumin_si": "Albumin",
        "total_protein_si": "Total protein", "calcium_si": "Calcium", "uric_acid_si": "Urate", "phosphorus_si": "Phosphate", "glucose_si": "Glucose",
        "alt_si": "ALT", "ast_si": "AST", "alp_si": "ALP", "cholesterol_si": "Cholesterol", "triglycerides_si": "Triglycerides"}
rows = []
for c in J:
    m0 = dev[c].mean()
    p17 = (d[d.cycle == "2017-2020"][c].mean() / m0 - 1) * 100; p21 = (d[d.cycle == "2021-2023"][c].mean() / m0 - 1) * 100
    ch = ((1 + J[c] / 100) * (1 + L[c] / 100) - 1) * 100
    rows.append(dict(analyte=NAME[c], paired_2017=J[c], population_2017_20=round(p17, 2), paired_chained_2021=round(ch, 2), population_2021_23=round(p21, 2)))
t = pd.DataFrame(rows); t.to_csv("res_bm/population_vs_paired.csv", index=False); print(t.to_string())
