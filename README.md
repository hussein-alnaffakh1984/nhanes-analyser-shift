# Analyser changes and the calibration of a routine-chemistry prediction model (NHANES)

Code and derived data for the manuscript *"Calibration of a routine-chemistry prediction model after laboratory analyser changes: paired-sample bridging equations versus statistical alignment"* (Al-Naffakh HAH, Reishaan MA, Radhi AD, Khattak UF; Al-Kafeel University, Najaf, Iraq). The work is dedicated to the memory of the late Dr. Abdulsahib Saad Jubran.

## Submitted version (Biochemia Medica)
The analyses reported in the submitted manuscript are produced by the scripts below; outputs are in `res_bm/` (tables, CSV/JSON) and `figs_bm/` (figures, 600 dpi).

| Script | Content |
| --- | --- |
| `bm_analysis.py` | Main analysis: model tiers, T3/T0 ratio, CDC bridging equations, paired mean-difference correction, cross-fitted statistical alignment, prospective transfer, negative controls, Shapley attribution, sensitivity analyses, pooled validation, race/ethnicity, decision curves, domain classifier; design-based (Rao–Wu) bootstrap with 1000 replicates |
| `aps_bm.py` | Inter-platform bias of the CDC bridging studies vs EFLM biological-variation specifications |
| `aps_model_bm.py` | Model-based allowable bias per analyte, joint and common-mode bias, bootstrap refitting and gradient boosting |
| `recal_bm.py` | Recalibration on 2017–2020 evaluated on 2021–2023 |
| `pop_vs_paired.py` | Population changes vs paired-sample differences |
| `extra_bm.py` | Fasting sensitivity analysis and 15-analyte domain classifier |
| `sim_bm.py` | Simulation of analyser bias and case-mix change (200 repetitions) |
| `figs_bm.py` | Figures and supplementary figures |

Run order: `python bm_analysis.py`, then `aps_bm.py`, `aps_model_bm.py`, `recal_bm.py`, `pop_vs_paired.py`, `extra_bm.py`, `sim_bm.py`, `figs_bm.py` (`BM_R` sets the number of bootstrap replicates). The full main analysis takes about 45 minutes. The scripts described in the sections below belong to earlier versions of the analysis and are kept for transparency.

## What the code does
- Builds an adult analytic sample (n = 29,706) from five NHANES cycles (2011–2023), excluding pregnancy and dialysis.
- Trains creatinine-free models (age, sex, CBC, routine chemistry) on Beckman DxC data (2011–2016) and validates them on Roche Cobas data (2017–2023).
- Compares CDC bridging equations, label-free stratified quantile mapping and location–scale harmonisation, using the O/E ratio and the T3/T0 ratio.
- Runs drift detection, EFLM performance-specification comparison, per-analyte attribution, fasting sensitivity analyses and a simulation.

## Data
All inputs are public NHANES data (CDC/NCHS, public domain).
- `nhanes_ckd_dataset.csv` – analytic dataset (SHA-256 begins `63c9548468be63fc`).
- `extra_vars.csv` – fasting time, examination session and examination period.
- To rebuild both from the CDC website: `python prepare_nhanes.py` then `python prepare_extra_vars.py` (internet required).

## How to reproduce
```bash
pip install -r requirements.txt
bash run_all.sh
```
Outputs are written to `res/` (tables) and `figs/` (figures at 600 dpi, PNG and PDF). All random processes are seeded. A full run takes roughly 30–60 minutes on a laptop.
Pre-computed outputs are included in `results/` and `figures/`.

## Main scripts
| Script | Content |
| --- | --- |
| `analysis.py`, `harmonize.py`, `chain.py`, `qm_cycle.py` | Model tiers, CDC bridging equations, quantile mapping |
| `novelty_probe.py` | Domain-classifier drift detection |
| `eflm_aug.py` | EFLM bias specifications vs observed inter-platform bias |
| `revision.py`, `pooled.py`, `t1.py` | Harmonised outcome, Tables 1–3, internal validation, negative control, recalibration, weighted analysis |
| `rev2.py`, `rev2b.py` | Per-analyte attribution (Table S2), T3/T0 ratios by method |
| `rev3.py` | Fasting and session sensitivity analyses (Table S3) |
| `simulation.py`, `simulation_rev.py` | Simulation of analyser bias and case-mix shift |
| `figures.py`, `figures_rev.py`, `fig4.py` | Figures 1–6 and S1 |

## Software
Python 3.13 (submitted version; earlier versions 3.12); package versions in `requirements.txt`.

## Licence
MIT (code). NHANES data are in the public domain.

## Contact
Hussein Ali Hussein Al-Naffakh – hussein.alnaffakh@alkafeel.edu.iq
