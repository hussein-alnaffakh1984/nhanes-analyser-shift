#!/usr/bin/env bash
# Reproduces every number, table and figure in the manuscript.
# Inputs (included): nhanes_ckd_dataset.csv, extra_vars.csv
# To rebuild the inputs from CDC: python prepare_nhanes.py ; python prepare_extra_vars.py
set -e
mkdir -p res figs
python -c "import hashlib;print('dataset sha256:',hashlib.sha256(open('nhanes_ckd_dataset.csv','rb').read()).hexdigest())"
for s in analysis.py harmonize.py novelty_probe.py eflm_aug.py chain.py qm_cycle.py simulation.py \
         simulation_rev.py revision.py pooled.py t1.py rev2.py rev2b.py rev3.py figures.py qm_cycle.py figures_rev.py fig4.py; do
  echo "== $s"; python "$s" > "res/log_${s%.py}.txt" 2>&1
done
echo "All outputs written to res/ and figs/"
