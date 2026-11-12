"""Figures for the Biochemia Medica submission (greyscale, 600 dpi TIFF + PNG + PDF)."""
import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 600, "axes.linewidth": 0.7})
O = "figs_bm/"; os.makedirs(O, exist_ok=True); R = "res_bm/"
def save(f, n):
    f.savefig(O + n + ".png", bbox_inches="tight"); f.savefig(O + n + ".pdf", bbox_inches="tight")
    f.savefig(O + n + ".tif", bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"}); plt.close(f)
WHICH = os.environ.get("FIGS", "1234S")

# ---------------------------------------------------------------- Figure 1: inter-platform bias vs EFLM APS
if "1" in WHICH:
    a = pd.read_csv(R + "aps_vs_bridging.csv")
    order = ["Sodium", "Chloride", "Calcium", "Albumin", "Total protein", "Potassium", "Glucose", "Cholesterol", "AST", "ALT",
             "Triglycerides", "Urate", "Phosphate", "ALP", "Total CO2"]
    a = a.set_index("analyte").loc[order[::-1]].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
    for ax, lab, title in [(axes[0], "J", "A  Roche Cobas 6000 vs Beckman DxC 660i\n(n = 248 paired samples)"),
                           (axes[1], "L", "B  Roche Cobas 8000 vs Roche Cobas 6000\n(n = 152 paired samples)")]:
        y = np.arange(len(a))
        has = a.CVI.notna().values
        ax.barh(y[has], 2 * a.APS_minimum[has], left=-a.APS_minimum[has], color="#e3e3e3", height=0.7, label="Minimum APS (±)")
        ax.barh(y[has], 2 * a.APS_desirable[has], left=-a.APS_desirable[has], color="#b5b5b5", height=0.7, label="Desirable APS (±)")
        d = a[f"{lab}_mean_diff"].values; rec = a[f"{lab}_adjustment_recommended"].values.astype(bool)
        ax.scatter(d[rec], y[rec], marker="o", s=26, color="black", zorder=3, label="CDC equation published")
        ax.scatter(d[~rec], y[~rec], marker="o", s=26, facecolor="white", edgecolor="black", lw=0.9, zorder=3, label="Adjustment not recommended")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(title, fontsize=8.5, loc="left"); ax.set_xlabel("Mean difference, new vs previous analyser (%)")
        ax.set_xlim(-20, 16); ax.grid(axis="x", lw=0.3, color="#cccccc")
    lbl = [n + ("*" if not h else "") for n, h in zip(a.analyte, a.CVI.notna())]
    axes[0].set_yticks(np.arange(len(a))); axes[0].set_yticklabels(lbl)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, "Figure1")

# ---------------------------------------------------------------- Figure 2: T3/T0 ratio forest plot
if "2" in WHICH:
    m = pd.read_csv(R + "main_by_cycle.csv"); nc = pd.read_csv(R + "negative_controls.csv"); pr = pd.read_csv(R + "prospective_alignment.csv")
    def parse(s):
        e, rest = s.split(" ("); lo, hi = rest.rstrip(")").split("–"); return float(e), float(lo), float(hi)
    rows = []  # (label, unweighted triple, weighted triple or None, group)
    rows.append(("2011–2012 → 2013–2014 (DxC 800, no change)", parse(nc.ratio_none[0]), parse(nc.ratio_none[2]), "Negative controls"))
    rows.append(("2011–2014 → 2015–2016 (DxC 660i added)", parse(nc.ratio_none[1]), None, "Negative controls"))
    meths = [("No adjustment", "No adjustment"), ("CDC bridging equations", "CDC bridging (chained in 2021–2023)"),
             ("Paired mean-difference correction", "Paired mean-difference correction (all 15 analytes)"),
             ("Quantile mapping (cross-fitted)", "Quantile mapping, cross-fitted"), ("Location–scale (cross-fitted)", "Location–scale, cross-fitted")]
    for cyc, grp in [("2017-2020", "2017–2020, Cobas 6000"), ("2021-2023", "2021–2023, Cobas 8000")]:
        for lab, key in meths:
            u = m[(m.cycle == cyc) & (m.weighted == False) & (m.method == key)].ratio_T3_T0.iloc[0]
            w = m[(m.cycle == cyc) & (m.weighted == True) & (m.method == key)].ratio_T3_T0.iloc[0]
            rows.append((lab, parse(u), parse(w), grp))
    for meth, lab in [("qm", "Quantile mapping fitted on 2017–2020"), ("ls", "Location–scale fitted on 2017–2020")]:
        u = pr[(pr.weighted == False) & (pr.method == meth)].ratio_T3_T0.iloc[0]; w = pr[(pr.weighted == True) & (pr.method == meth)].ratio_T3_T0.iloc[0]
        rows.append((lab, parse(u), parse(w), "2021–2023, Cobas 8000"))
    fig, ax = plt.subplots(figsize=(7.0, 6.2)); y = 0; yt = []; yl = []; groups_seen = []
    for lab, u, w, g in rows:
        if g not in groups_seen:
            y -= 0.6; ax.text(0.505, y, g, fontsize=8.5, fontweight="bold", va="center"); groups_seen.append(g); y -= 0.8
        ax.errorbar(u[0], y + 0.15, xerr=[[u[0] - u[1]], [u[2] - u[0]]], fmt="o", color="black", ms=4.5, capsize=2, lw=0.9)
        if w is not None:
            ax.errorbar(w[0], y - 0.15, xerr=[[w[0] - w[1]], [w[2] - w[0]]], fmt="s", color="#7a7a7a", mfc="white", ms=4.2, capsize=2, lw=0.9)
        yt.append(y); yl.append(lab); y -= 1
    ax.axvline(1, color="black", lw=0.7, ls="--")
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8); ax.set_xlim(0.5, 1.75)
    ax.set_xlabel("Ratio of O/E, extended model (T3) / age–sex model (T0)  (95% CI)")
    ax.errorbar([], [], fmt="o", color="black", label="Unweighted (primary)"); ax.errorbar([], [], fmt="s", color="#7a7a7a", mfc="white", label="Survey-weighted")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    save(fig, "Figure2")

# ---------------------------------------------------------------- Figure 3: calibration curves (needs model predictions)
if "3" in WHICH:
    src = open("bm_analysis.py").read().split("# ------------------------------------------------------------------ 1. main analysis")[0]
    exec(src); R = "res_bm/"
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharey=True)
    styles = [("No adjustment", "-", "black"), ("CDC bridging", "--", "black"), ("Location–scale (cross-fitted)", ":", "#555555")]
    for ax, cyc, n_lab in [(axes[0], "2017-2020", "A  2017–2020, Roche Cobas 6000"), (axes[1], "2021-2023", "B  2021–2023, Roche Cobas 8000")]:
        mm = (base.cycle == cyc).values; y = y_all[mm]; one = np.ones(len(base))
        p_ls = np.mean([pred_aligned("ls", mm, one, REF_U, folds=make_folds(mm, np.random.default_rng(SEED + 100 + i)))["T3"][mm] for i in range(5)], axis=0)
        preds = {"No adjustment": P["T3"][mm], "CDC bridging": P_cdc["T3"][mm], "Location–scale (cross-fitted)": p_ls}
        grid = np.linspace(0.005, 0.6, 120)
        for lab, ls, col in styles:
            p = preds[lab]; sm = lowess(y, p, frac=0.3, it=0, return_sorted=True)
            g = np.interp(grid, sm[:, 0], sm[:, 1]); keep = grid <= np.quantile(p, 0.995)
            # bootstrap band (200 participant-level resamples) for the unadjusted curve only
            if lab == "No adjustment":
                rng = np.random.default_rng(1); bands = []
                for _ in range(200):
                    i = rng.integers(0, len(y), len(y)); s2 = lowess(y[i], p[i], frac=0.3, it=0, delta=0.002, return_sorted=True)
                    bands.append(np.interp(grid, s2[:, 0], s2[:, 1]))
                lo, hi = np.percentile(bands, [2.5, 97.5], axis=0); ax.fill_between(grid[keep], lo[keep], hi[keep], color="#dddddd", lw=0)
            ax.plot(grid[keep], g[keep], ls=ls, color=col, lw=1.4, label=lab)
        ax.plot([0, 0.6], [0, 0.6], color="#999999", lw=0.7)
        ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6); ax.set_title(n_lab, fontsize=8.5, loc="left")
        ax.set_xlabel("Predicted risk, extended model (T3)")
        ins = ax.inset_axes([0.0, 0.0, 1.0, 0.12]); ins.hist(P["T3"][mm], bins=60, range=(0, 0.6), color="#bbbbbb"); ins.axis("off")
    axes[0].set_ylabel("Observed proportion (loess)"); axes[1].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(); save(fig, "Figure3")

# ---------------------------------------------------------------- Figure 4: Shapley attribution
if "4" in WHICH:
    s = pd.read_csv(R + "shapley_attribution.csv"); s = s[~s.group.str.startswith("TOTAL")]
    groups = list(dict.fromkeys(s.group)); y = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (cyc, col, hatch) in enumerate([("2017-2020", "black", None), ("2021-2023", "#a0a0a0", None)]):
        v = s[s.cycle == cyc].set_index("group").loc[groups].shapley_ln_ratio
        ax.barh(y + (0.2 if i == 0 else -0.2), v, height=0.38, color=col, label=cyc.replace("-", "–"))
    ax.axvline(0, color="black", lw=0.6); ax.set_yticks(y); ax.set_yticklabels(groups); ax.invert_yaxis()
    ax.set_xlabel("Shapley contribution to the change in ln(T3/T0 ratio) after alignment")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right"); fig.tight_layout(); save(fig, "Figure4")

# ---------------------------------------------------------------- Supplementary figures
if "S" in WHICH:
    d = pd.read_csv(R + "domain_classifier.csv"); d15 = pd.read_csv(R + "domain_classifier_15.csv")
    d["chemistry"] = d15.chemistry15.values; d["chemistry_K_rounded"] = d15.chemistry15_K_rounded.values
    d["perm_null_chem_max"] = np.maximum(d.perm_null_chem_max.values, d15.perm_null_max.values)
    fig, ax = plt.subplots(figsize=(7.0, 3.4)); x = np.arange(len(d)); w = 0.2
    cols = [("chemistry", "15 chemistry analytes (as released)", "#303030"), ("chemistry_K_rounded", "Chemistry, potassium rounded to 0.1", "#7a7a7a"),
            ("CBC", "Complete blood count", "#b0b0b0"), ("age_sex", "Age and sex", "#e0e0e0")]
    for i, (c, l, col) in enumerate(cols):
        ax.bar(x + (i - 1.5) * w, d[c] - 0.45, w, bottom=0.45, color=col, edgecolor="black", lw=0.4, label=l)
    ax.axhline(d.perm_null_chem_max.max(), color="black", ls=":", lw=0.8, label="Maximum of permutation null")
    ax.set_xticks(x); ax.set_xticklabels([t.replace(" -> ", " →\n") for t in d.transition], fontsize=7.5)
    ax.set_ylim(0.45, 1.0); ax.set_ylabel("Domain-classifier AUC"); ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    save(fig, "FigureS2_domain_classifier")
    dc = pd.read_csv(R + "decision_curve.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for c, l, ls in [("treat_all", "Test all", ":"), ("T0", "Age and sex (T0)", "-."), ("TR", "Realistic panel (TR)", "--"), ("T3", "Extended panel (T3)", "-")]:
        ax.plot(dc.threshold * 100, dc[c], ls=ls, color="black", lw=1.2, label=l)
    ax.axhline(0, color="#999999", lw=0.6); ax.set_xlabel("Threshold probability (%)"); ax.set_ylabel("Net benefit")
    ax.legend(frameon=False, fontsize=7.5); save(fig, "FigureS3_decision_curve")
    sm = pd.read_csv(R + "simulation.csv")
    sc = [("Bias only", "Analyser bias only"), ("Moderate case-mix shift only", "Case mix +25%"), ("Bias + moderate case-mix shift", "Bias + case mix +25%"),
          ("True case-mix shift only", "Case mix +100%"), ("Bias + case-mix shift", "Bias + case mix +100%")]
    me = [("No correction", "No correction", "#e0e0e0"), ("Oracle (true equations)", "Oracle correction", "#303030"),
          ("Quantile mapping (all subjects)", "Quantile mapping", "#8a8a8a"), ("Anchored QM (low-risk: age<40, no diabetes)", "QM, low-risk subgroup", "#bdbdbd")]
    fig, ax = plt.subplots(figsize=(7.2, 3.4)); w = 0.2
    for i, (k, l, col) in enumerate(me):
        v = [sm[(sm.scenario == s_) & (sm.method == k)].iloc[0] for s_, _ in sc]
        ax.bar(np.arange(len(sc)) + (i - 1.5) * w, [q.O_E for q in v], w, yerr=[q.O_E_sd for q in v], capsize=2, color=col, edgecolor="black", lw=0.4,
               label=l, error_kw=dict(lw=0.7))
    ax.axhline(1, color="black", lw=0.6, ls="--"); ax.set_xticks(range(len(sc))); ax.set_xticklabels([l for _, l in sc], fontsize=7.5)
    ax.set_ylabel("O/E ratio (mean ± SD, 200 repetitions)"); ax.legend(frameon=False, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, "FigureS4_simulation")
print("figures done:", WHICH)
