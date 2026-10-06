# Laboratory analyser changes and loss of calibration in creatinine-free machine learning models (NHANES)

Code and derived data for the manuscript *"Laboratory analyser changes and loss of calibration in creatinine-free machine learning models for reduced kidney function in NHANES"* (Al-Naffakh HAH, Jubran AS; Al-Kafeel University, Najaf, Iraq).

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
Python 3.12; package versions in `requirements.txt`.

## Licence
MIT (code). NHANES data are in the public domain.

## Contact
Hussein Ali Hussein Al-Naffakh – hussein.alnaffakh@alkafeel.edu.iq
