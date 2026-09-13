#!/bin/bash
#SBATCH --job-name=momo_sweep
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-499
# ABSOLUTE --output, and it is not cosmetic. A relative path resolves against the SUBMISSION
# directory -- the repo -- and that is exactly how 24 stray .out/.err files once landed in
# the working tree and tripped a provenance gate. The directory must exist BEFORE sbatch:
# Slurm opens the file before the script runs, so a mkdir inside is too late.
#   once, by hand:   mkdir -p ~/momo/logs
#SBATCH --output=/home/%u/momo/logs/momo_%A_%a.out
#
# research/momo_sweep_preregistration.md -- exhaustive standalone backtest of MOMO_CHASE.
#
#   python -m trade_analysis.momo_sweep.pack --out $HOME/momo/data     # once
#   python -m trade_analysis.momo_sweep.sweep --validate               # MUST PASS FIRST
#   sbatch trade_analysis/momo_sweep/submit_sweep.sbatch
#   python -m trade_analysis.momo_sweep.collect --results $HOME/momo/results
#
# PARTITION: `short` is CPU and allows 50 concurrent jobs at up to 48h; `gpu` caps a user at
# 4 running jobs and 8h. There is no GPU work here at all -- it is a Python loop over bars.
# 500 tasks therefore run in 10 waves of 50.
#
# SIZING, from a measured benchmark rather than a guess. 0.253 ms per cell per session.
# DISCOVERY is 1,499 QQQ sessions, not the full 3,815 -- so a cell costs 0.38 s, not 0.96 s,
# and 2,822,400 cells is ~300 CPU-hours rather than the ~756 the full sample would imply.
# At 500 shards that is ~5,645 cells and ~36 min of compute per task, plus ~45 s for the
# one-time prepare() pass that every task pays in full.
# --time is 6h -- a 10x margin -- because a task killed at the wall clock leaves a TRUNCATED
# shard that still looks like a perfectly good file to the collector.
#
# GETTING THE DATA UP. The bar cache is ~18 MB packed, so unlike the VRP panel it is small.
# Run this ON THE LAPTOP (PowerShell), cd first and pass BARE FILENAMES -- `scp C:\path\f`
# makes scp parse `C:` as a hostname, and a Git-Bash `/c/Users/...` path is not something
# PowerShell's scp.exe can stat. Both of those mistakes have been made here before.
#
#   cd C:\Users\chaitanyakharche\Documents\stock\data\momo
#   scp QQQ.npz SPY.npz kharche.c@login.explorer.northeastern.edu:/home/kharche.c/momo/data/
#
# NOT to /scratch -- it is wiped in full on the first Tuesday of every month with no
# access-time condition, and that purge already destroyed this project's TFT source once.

set -euo pipefail

REPO="${REPO:-$HOME/momo/rep}"
ROOT="${ROOT:-$HOME/momo}"
DATA="${DATA:-$ROOT/data}"
OUT="${OUT:-$ROOT/results}"

cd "$REPO"
mkdir -p "$OUT" "$ROOT/logs"

# ---- fail loudly on missing input, BEFORE Slurm can record COMPLETED --------------------
for f in QQQ.npz SPY.npz; do
  if [[ ! -f "$DATA/$f" ]]; then
    echo "FATAL: $DATA/$f not found. scp it up -- see the header of this file." >&2
    exit 2
  fi
done

# ---- the grid must verify its own size --------------------------------------------------
# --array is hand-written above, which is exactly how a sweep silently runs 500 of 800
# shards and reports a p-value over the part that ran. Ask the code, and refuse on a
# disagreement rather than trusting the header.
N_CELLS=$(python - <<'PY'
from trade_analysis.momo_sweep.grid import build_grid
print(len(build_grid()))
PY
)
N_TASKS="${SLURM_ARRAY_TASK_COUNT:-500}"
echo "grid        $N_CELLS cells over $N_TASKS shards"
if [[ "$N_CELLS" -lt "$N_TASKS" ]]; then
  echo "FATAL: more shards ($N_TASKS) than cells ($N_CELLS)." >&2
  exit 3
fi

# ---- environment ------------------------------------------------------------------------
if [[ -f "$REPO/uv.lock" ]] && command -v uv >/dev/null 2>&1; then
  export UV_PROJECT_ENVIRONMENT="$ROOT/.venv"
  uv sync --locked --offline
  PY_BIN="$ROOT/.venv/bin/python"
else
  echo "NOTE: no uv.lock -- falling back to conda, which is NOT a pinned environment."
  module load anaconda3/2024.06 2>/dev/null || true
  source activate vrp 2>/dev/null || conda activate vrp
  PY_BIN="$(command -v python)"
fi

# This job is a Python bar loop, not BLAS. Extra threads buy nothing and oversubscribe the
# 2 allocated CPUs when 50 tasks share a node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "host        $(hostname)"
echo "interpreter $PY_BIN"
echo "numpy       $("$PY_BIN" -c 'import numpy;print(numpy.__version__)')"
echo "git         $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (DIRTY)')"
echo "task        ${SLURM_ARRAY_TASK_ID:-0} of $N_TASKS"

# ---- srun the payload, with NOTHING after it --------------------------------------------
# Slurm records the exit status of the BATCH SCRIPT. A trailing `echo` turns a Python
# traceback into COMPLETED, which is the single most common way a failed sweep cell reads as
# a finished one.
# --start/--end are the pre-registration's DISCOVERY window and are passed EXPLICITLY rather
# than left to the defaults, so that reading this file tells you what the run covered.
# QQQ only: SPY's history begins 2022-01-03 and is entirely inside the held-out set.
srun "$PY_BIN" -m trade_analysis.momo_sweep.sweep \
  --data "$DATA" \
  --symbols QQQ \
  --start 2016-01-01 \
  --end 2021-12-31 \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" \
  --nshards "$N_TASKS" \
  --out "$OUT"
