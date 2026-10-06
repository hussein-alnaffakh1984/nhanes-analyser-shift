# ===== NHANES extra variables for round-2 revision (run as ONE Kaggle cell; Internet must be ON) =====
# Downloads: DEMO (RIDEXMON exam period), FASTQX (fasting hours, session), BIOPRO_J IDs (2017-2018),
# and the BIOPRO_G / BIOPRO_H documentation pages (to verify the 2011-2012 analyser).
# Output: /kaggle/working/nhanes_extra.zip
import io, time, zipfile, requests, pandas as pd
from pathlib import Path

H = {"User-Agent": "Mozilla/5.0 (research data download)"}
CYC = {2011: ("2011-2012", "_G", ""), 2013: ("2013-2014", "_H", ""), 2015: ("2015-2016", "_I", ""),
       2017: ("2017-2020", "", "P_"), 2021: ("2021-2023", "_L", "")}
PAT = ["https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{y}/DataFiles/{n}.xpt",
       "https://wwwn.cdc.gov/Nchs/Nhanes/{lab}/{n}.XPT"]
out = Path("/kaggle/working"); cache = out / "xpt_extra"; cache.mkdir(parents=True, exist_ok=True)

def get(y, name, lab):
    f = cache / f"{name}.xpt"
    if not f.exists():
        for p in PAT:
            try:
                r = requests.get(p.format(y=y, n=name, lab=lab), headers=H, timeout=120)
                if r.status_code == 200 and not r.content[:200].lstrip().startswith(b"<"):
                    f.write_bytes(r.content); print("downloaded", name); break
            except requests.RequestException as e:
                print("!", name, e)
            time.sleep(0.5)
        else:
            print("! not available:", name); return None
    return pd.read_sas(f, format="xport", encoding="latin-1")

rows = []
for y, (lab, suf, pre) in CYC.items():
    demo = get(y, f"{pre}DEMO{suf}", lab)
    fast = get(y, f"{pre}FASTQX{suf}", lab)
    d = demo[["SEQN"] + [c for c in ["RIDEXMON"] if c in demo]].copy()
    if fast is not None:
        keep = ["SEQN"] + [c for c in ["PHAFSTHR", "PHAFSTMN", "PHDSESN"] if c in fast]
        d = d.merge(fast[keep], on="SEQN", how="left")
    d["cycle"] = lab
    rows.append(d)
extra = pd.concat(rows, ignore_index=True)

# 2017-2018 participants (Cobas 6000 throughout 2017-March 2020): flag SEQNs present in BIOPRO_J
bj = get(2017, "BIOPRO_J", "2017-2018")
extra["in_2017_2018"] = extra.SEQN.isin(set(bj.SEQN)) if bj is not None else pd.NA
extra.to_csv(out / "extra_vars.csv", index=False)
for c in ["RIDEXMON","PHAFSTHR","PHDSESN"]:
    if c in extra: print(c, extra.groupby("cycle")[c].apply(lambda s: round(s.notna().mean(),3)).to_dict())
print("2017-2018 flagged:", int(extra["in_2017_2018"].fillna(False).sum()))

# Documentation pages for the 2011-2012 and 2013-2014 biochemistry profile
docs = {}
for y, n in [(2011, "BIOPRO_G"), (2013, "BIOPRO_H")]:
    for u in [f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{y}/DataFiles/{n}.htm",
              f"https://wwwn.cdc.gov/Nchs/Nhanes/{y}-{y+1}/{n}.htm"]:
        try:
            r = requests.get(u, headers=H, timeout=60)
            if r.status_code == 200 and len(r.text) > 2000:
                docs[n] = r.text; (out / f"{n}.htm").write_text(r.text, encoding="utf-8"); print("saved", n); break
        except requests.RequestException as e:
            print("!", u, e)

with zipfile.ZipFile(out / "nhanes_extra.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out / "extra_vars.csv", "extra_vars.csv")
    for n in docs: z.write(out / f"{n}.htm", f"{n}.htm")
print("\nDONE -> download /kaggle/working/nhanes_extra.zip from the Output panel")
