#!/usr/bin/env bash
# Reproduce every table and figure.  Usage:
#   ./run_all.sh            Stage 2 onward from the committed processed data + frozen footprint (~5 min)
#   ./run_all.sh --phase1   also rebuild data/processed from the raw LaDe file (see data/README.md)
#   ./run_all.sh --stage1   also re-solve Stage 1 (needs a Gurobi licence; ~6 h)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-python}"
for arg in "$@"; do
  case "$arg" in
    --phase1) "$PY" scripts/01_prep_phase1.py ;;
    --stage1) "$PY" scripts/02_stage1_siting.py ;;
    *) echo "unknown option $arg"; exit 1 ;;
  esac
done
"$PY" scripts/03_build_scenarios.py
"$PY" scripts/04_stage2_dro.py
"$PY" scripts/05_evaluate.py            # needs Δc for the upgrade step ...
"$PY" scripts/06_upgrade_outage.py
"$PY" scripts/05_evaluate.py            # ... then re-evaluates including the upgraded designs
"$PY" scripts/07_scaling.py
"$PY" scripts/08_figures_tables.py
echo "Done: results/tables and results/figures"
