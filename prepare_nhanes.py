# ===== NHANES data preparation v2 (adds pregnancy & dialysis exclusions + participant flow) =====
# Copy this whole cell into ONE notebook cell and run it.
import subprocess, sys
subprocess.run([sys.executable,"-m","pip","install","-q","pandas","requests"])

import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------
# Cycle definitions: begin year -> (folder label, file suffix/prefix, MEC weight)
# --------------------------------------------------------------------------
CYCLES = {
    2011: {"label": "2011-2012", "name": lambda f: f"{f}_G", "wt": "WTMEC2YR"},
    2013: {"label": "2013-2014", "name": lambda f: f"{f}_H", "wt": "WTMEC2YR"},
    2015: {"label": "2015-2016", "name": lambda f: f"{f}_I", "wt": "WTMEC2YR"},
    # 2017-March 2020 pre-pandemic combined file (replaces 2017-2018 _J)
    2017: {"label": "2017-2020", "name": lambda f: f"P_{f}", "wt": "WTMECPRP"},
    2021: {"label": "2021-2023", "name": lambda f: f"{f}_L", "wt": "WTMEC2YR"},
}

COMPONENTS = ["DEMO", "BIOPRO", "CBC", "ALB_CR", "GHB", "DIQ", "BPQ", "KIQ_U"]

# Two URL layouts (CDC moved files in 2024); both are tried.
URL_PATTERNS = [
    "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt",
    "https://wwwn.cdc.gov/Nchs/Nhanes/{label}/{name}.XPT",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (research data download)"}

# --------------------------------------------------------------------------
# Variables: NHANES name -> (clean name, conventional unit, SI factor, SI unit, role)
# SI value = conventional value * factor   (factor None = already SI / no conversion)
# --------------------------------------------------------------------------
LAB_VARS = {
    # ---- BIOPRO (serum chemistry) ----
    "LBXSCR":   ("creatinine",      "mg/dL",  88.42,  "umol/L", "leakage"),
    "LBXSBU":   ("bun",             "mg/dL",  0.357,  "mmol/L", "leakage_review"),
    "LBXSAL":   ("albumin",         "g/dL",   10.0,   "g/L",    "feature"),
    "LBXSTP":   ("total_protein",   "g/dL",   10.0,   "g/L",    "feature"),
    "LBXSGB":   ("globulin",        "g/dL",   10.0,   "g/L",    "feature"),
    "LBXSUA":   ("uric_acid",       "mg/dL",  59.48,  "umol/L", "feature"),
    "LBXSPH":   ("phosphorus",      "mg/dL",  0.3229, "mmol/L", "feature"),
    "LBXSCA":   ("calcium",         "mg/dL",  0.2495, "mmol/L", "feature"),
    "LBXSGL":   ("glucose",         "mg/dL",  0.0555, "mmol/L", "feature"),
    "LBXSNASI": ("sodium",          "mmol/L", None,   "mmol/L", "feature"),
    "LBXSKSI":  ("potassium",       "mmol/L", None,   "mmol/L", "feature"),
    "LBXSCLSI": ("chloride",        "mmol/L", None,   "mmol/L", "feature"),
    "LBXSC3SI": ("bicarbonate",     "mmol/L", None,   "mmol/L", "feature"),
    "LBXSOSSI": ("osmolality",      "mmol/kg", None,  "mmol/kg", "feature"),
    "LBXSATSI": ("alt",             "U/L",    None,   "U/L",    "feature"),
    "LBXSASSI": ("ast",             "U/L",    None,   "U/L",    "feature"),
    "LBXSAPSI": ("alp",             "U/L",    None,   "U/L",    "feature"),
    "LBXSGTSI": ("ggt",             "U/L",    None,   "U/L",    "feature"),
    "LBXSLDSI": ("ldh",             "U/L",    None,   "U/L",    "feature"),
    "LBXSTB":   ("total_bilirubin", "mg/dL",  17.1,   "umol/L", "feature"),
    "LBXSCH":   ("cholesterol",     "mg/dL",  0.02586, "mmol/L", "feature"),
    "LBXSTR":   ("triglycerides",   "mg/dL",  0.01129, "mmol/L", "feature"),
    "LBXSIR":   ("iron",            "ug/dL",  0.1791, "umol/L", "feature"),
    # ---- CBC ----
    "LBXHGB":   ("hemoglobin",      "g/dL",   10.0,   "g/L",    "feature"),
    "LBXHCT":   ("hematocrit",      "%",      None,   "%",      "feature"),
    "LBXMCVSI": ("mcv",             "fL",     None,   "fL",     "feature"),
    "LBXRDW":   ("rdw",             "%",      None,   "%",      "feature"),
    "LBXWBCSI": ("wbc",             "10^9/L", None,   "10^9/L", "feature"),
    "LBXPLTSI": ("platelets",       "10^9/L", None,   "10^9/L", "feature"),
    # ---- GHB ----
    "LBXGH":    ("hba1c",           "%",      None,   "%",      "feature"),
    # ---- ALB_CR (urine) : used for the outcome only ----
    "URDACT":   ("uacr",            "mg/g",   None,   "mg/g",   "outcome_input"),
}

DEMO_VARS = {
    "RIDAGEYR": "age",
    "RIAGENDR": "sex",          # 1 = male, 2 = female
    "RIDRETH3": "race_eth",     # descriptive only, NOT used in eGFR (2021 race-free)
    "SDMVPSU":  "psu",
    "SDMVSTRA": "strata",
    "RIDEXPRG": "pregnant_exam",  # 1 = pregnant at exam (lab test/self-report), 2 = not, 3 = cannot ascertain
}

# Questionnaire comorbidities (1 = yes, 2 = no; 3/7/9 -> borderline/refused/unknown)
Q_VARS = {
    "DIQ010": "diabetes_dx",
    "BPQ020": "hypertension_dx",
    "KIQ025": "dialysis_12m",     # received dialysis in past 12 months (1 = yes)
}


# --------------------------------------------------------------------------
def download_xpt(year: int, comp: str, cache: Path) -> pd.DataFrame | None:
    cyc = CYCLES[year]
    name = cyc["name"](comp)
    local = cache / f"{name}.xpt"

    if not local.exists():
        for pat in URL_PATTERNS:
            url = pat.format(year=year, label=cyc["label"], name=name)
            try:
                r = requests.get(url, headers=HEADERS, timeout=120)
                # CDC returns an HTML error page with status 200 for missing files
                if r.status_code == 200 and not r.content[:200].lstrip().startswith(b"<"):
                    local.write_bytes(r.content)
                    print(f"  downloaded {name}  ({len(r.content)/1e6:.1f} MB)")
                    break
            except requests.RequestException as e:
                print(f"  ! {url}: {e}")
            time.sleep(0.5)
        else:
            print(f"  ! {name} not available for {cyc['label']} - skipped")
            return None
    else:
        print(f"  cached     {name}")

    try:
        df = pd.read_sas(local, format="xport", encoding="latin-1")
    except Exception as e:
        print(f"  ! could not read {local.name}: {e}")
        return None
    df["SEQN"] = df["SEQN"].astype("int64")
    return df


def egfr_ckd_epi_2021(scr_mgdl: pd.Series, age: pd.Series, female: pd.Series) -> pd.Series:
    """Race-free CKD-EPI 2021 creatinine equation (Inker et al., NEJM 2021)."""
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr_mgdl / kappa
    egfr = (142.0
            * np.minimum(ratio, 1.0) ** alpha
            * np.maximum(ratio, 1.0) ** -1.200
            * 0.9938 ** age
            * np.where(female, 1.012, 1.0))
    return pd.Series(egfr, index=scr_mgdl.index)


def build_cycle(year: int, cache: Path) -> pd.DataFrame | None:
    print(f"\n[{CYCLES[year]['label']}]")
    tables = {c: download_xpt(year, c, cache) for c in COMPONENTS}
    if tables["DEMO"] is None or tables["BIOPRO"] is None:
        print("  ! DEMO or BIOPRO missing - cycle skipped")
        return None

    wt = CYCLES[year]["wt"]
    keep_demo = ["SEQN"] + [v for v in DEMO_VARS if v in tables["DEMO"].columns]
    if wt in tables["DEMO"].columns:
        keep_demo.append(wt)
    df = tables["DEMO"][keep_demo].copy()

    for comp in ["BIOPRO", "CBC", "GHB", "ALB_CR", "DIQ", "BPQ", "KIQ_U"]:
        t = tables[comp]
        if t is None:
            continue
        wanted = [v for v in list(LAB_VARS) + list(Q_VARS) if v in t.columns]
        if wanted:
            df = df.merge(t[["SEQN"] + wanted], on="SEQN", how="left")

    df = df.rename(columns={wt: "WT_MEC"})
    df.insert(1, "cycle", CYCLES[year]["label"])
    return df


def finalize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.rename(columns=DEMO_VARS).rename(columns=Q_VARS)

    # participant flow (for the TRIPOD/STROBE flow diagram)
    flow = {"all_examined_in_files": len(df)}
    df = df[df["age"] >= 18].copy(); flow["adults_18plus"] = len(df)
    df = df[df["LBXSCR"].notna()].copy(); flow["with_serum_creatinine"] = len(df)
    if "pregnant_exam" in df.columns:
        n0 = len(df); df = df[df["pregnant_exam"] != 1].copy(); flow["excluded_pregnant"] = n0 - len(df)
    else:
        flow["excluded_pregnant"] = "variable not available"
    if "dialysis_12m" in df.columns:
        n0 = len(df); df = df[df["dialysis_12m"] != 1].copy(); flow["excluded_dialysis_12m"] = n0 - len(df)
    else:
        flow["excluded_dialysis_12m"] = "variable not available"
    flow["analytic_sample"] = len(df)
    df.attrs["flow"] = flow
    female = df["sex"] == 2

    # ---- outcome ----
    df["egfr"] = egfr_ckd_epi_2021(df["LBXSCR"], df["age"], female).round(1)
    df["ckd_egfr_lt60"] = (df["egfr"] < 60).astype(int)
    uacr = df["URDACT"] if "URDACT" in df.columns else pd.Series(np.nan, index=df.index)
    df["albuminuria_ge30"] = np.where(uacr.isna(), np.nan, (uacr >= 30).astype(float))
    # primary KDIGO label: eGFR < 60 OR UACR >= 30 (single-visit approximation)
    df["ckd_kdigo"] = ((df["egfr"] < 60) | (uacr >= 30)).astype(int)
    df.loc[uacr.isna() & (df["egfr"] >= 60), "ckd_kdigo"] = np.nan  # undeterminable

    # comorbidities -> 1/0/NaN
    for c in Q_VARS.values():
        if c in df.columns:
            df[c] = df[c].map({1: 1, 2: 0})
    df["sex_female"] = female.astype(int)

    # ---- unit conversion to SI + data dictionary ----
    dict_rows = []
    for nh, (clean, unit, factor, si_unit, role) in LAB_VARS.items():
        if nh not in df.columns:
            continue
        col = f"{clean}_si"
        df[col] = df[nh] * factor if factor else df[nh]
        dict_rows.append({"variable": col, "nhanes_var": nh, "unit": si_unit,
                          "conventional_unit": unit, "si_factor": factor or 1.0,
                          "role": role})
    df = df.drop(columns=[c for c in LAB_VARS if c in df.columns])

    for v, role, unit in [("age", "feature", "years"),
                          ("sex_female", "feature", "1=female"),
                          ("diabetes_dx", "covariate", "1=yes"),
                          ("hypertension_dx", "covariate", "1=yes"),
                          ("egfr", "outcome_input", "mL/min/1.73m2"),
                          ("ckd_egfr_lt60", "outcome_alt", "binary"),
                          ("albuminuria_ge30", "outcome_input", "binary"),
                          ("ckd_kdigo", "outcome_primary", "binary"),
                          ("race_eth", "descriptive_only", "NHANES code"),
                          ("WT_MEC", "survey_weight", ""),
                          ("psu", "survey_design", ""), ("strata", "survey_design", "")]:
        if v in df.columns:
            dict_rows.append({"variable": v, "nhanes_var": "", "unit": unit,
                              "conventional_unit": "", "si_factor": "", "role": role})
    df.attrs["flow"] = flow
    return df, pd.DataFrame(dict_rows)


def write_report(df: pd.DataFrame, dd: pd.DataFrame, path: Path) -> None:
    buf = io.StringIO()
    buf.write("NHANES CKD development cohort - preparation report\n")
    buf.write("=" * 60 + "\n\n")
    buf.write("Participant flow:\n")
    for k, v in df.attrs.get("flow", {}).items():
        buf.write(f"  {k}: {v}\n")
    buf.write(f"\nAnalytic sample: {len(df):,}\n\n")

    g = df.groupby("cycle").agg(
        n=("SEQN", "size"),
        ckd_kdigo_pct=("ckd_kdigo", lambda s: 100 * s.mean()),
        egfr_lt60_pct=("ckd_egfr_lt60", lambda s: 100 * s.mean()),
        uacr_missing_pct=("albuminuria_ge30", lambda s: 100 * s.isna().mean()),
    ).round(1)
    buf.write("Per cycle (unweighted):\n" + g.to_string() + "\n\n")

    labelled = df["ckd_kdigo"].notna()
    buf.write(f"Primary outcome (ckd_kdigo) determinable: {labelled.sum():,}  "
              f"| events: {int(df.loc[labelled, 'ckd_kdigo'].sum()):,}\n\n")

    feats = dd.loc[dd["role"].isin(["feature", "leakage_review"]), "variable"]
    miss = (df[feats].isna().mean() * 100).round(1).sort_values(ascending=False)
    buf.write("Missingness of candidate features (%):\n" + miss.to_string() + "\n\n")
    buf.write("REMINDER: 'creatinine_si' defines the outcome -> exclude from features.\n"
              "           'bun_si' -> evaluate models with and without it.\n")
    path.write_text(buf.getvalue(), encoding="utf-8")
    print("\n" + buf.getvalue())



cache, out = Path("nhanes_xpt"), Path("output")
cache.mkdir(exist_ok=True); out.mkdir(exist_ok=True)
parts = [p for y in CYCLES for p in [build_cycle(y, cache)] if p is not None]
df, dd = finalize(pd.concat(parts, ignore_index=True))
df.to_csv(out/"nhanes_ckd_dataset.csv", index=False)
dd.to_csv(out/"data_dictionary.csv", index=False)
write_report(df, dd, out/"prep_report.txt")
import shutil
shutil.make_archive("nhanes_output_v2", "zip", out)
try:
    from google.colab import files
    files.download("nhanes_output_v2.zip")
except Exception:
    print("Done. File ready: nhanes_output_v2.zip")
