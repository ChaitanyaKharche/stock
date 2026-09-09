#!/bin/bash
# The bar, and the lookahead audit. CPU only -- NO GPU IS REQUESTED ON PURPOSE.
#
#   sbatch trade_analysis/hpc/submit_baseline.sbatch
#
# RUN THIS BEFORE THE SWEEP. Section 6 of research/vrp_preregistration.md requires beating
# the best HAR variant on QLIKE before a single GPU hour is justified, and HAR is OLS on
# eight features -- it fits in seconds on a core. Requesting a GPU to discover that a
# linear model already wins is the expensive way to learn it.
#
# It also runs the audit, which deliberately injects a one-minute lookahead. Read the
# result the opposite way to the obvious guess:
#     audit BETTER than honest -> PASS. The honest features did not contain that minute.
#     audit SAME  as honest    -> the minute was already inside the honest run. Leak.

#SBATCH --job-name=vrp-baseline
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

REPO="${REPO:-$HOME/vrp/rep}"
VRP_ROOT="${VRP_ROOT:-$HOME/vrp}"
cd "$REPO"
mkdir -p "$VRP_ROOT/results"

source "$HOME/.bashrc" 2>/dev/null || true
conda activate vrp

if [ ! -d "$VRP_ROOT/data/vrp" ] || [ -z "$(ls -A "$VRP_ROOT/data/vrp" 2>/dev/null)" ]; then
  echo "NO DATA at $VRP_ROOT/data/vrp"
  echo "The parquet frame is deliberately NOT in git -- it is derived and would sit in"
  echo "every clone forever. Push it up from the laptop first:"
  echo "  rsync -avP data/vrp/ kharche.c@login.explorer.northeastern.edu:~/vrp/data/vrp/"
  exit 1
fi
echo "sessions available: $(ls "$VRP_ROOT/data/vrp"/*.parquet 2>/dev/null | wc -l)"

echo
echo "================ HONEST ================"
python -m trade_analysis.hpc.har_baseline --data "$VRP_ROOT/data/vrp" \
  | tee "$VRP_ROOT/results/baseline_honest.txt"

echo
echo "================ AUDIT (lookahead deliberately injected) ================"
python -m trade_analysis.hpc.har_baseline --data "$VRP_ROOT/data/vrp" --audit \
  | tee "$VRP_ROOT/results/baseline_audit.txt"

echo
echo "Compare the two QLIKE tables. Audit must be BETTER. If they are the same, the"
echo "honest pipeline already saw that minute and every downstream result is void."
