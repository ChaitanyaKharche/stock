#!/bin/bash
#SBATCH --job-name=arm_a_ssd
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --array=0-11
#SBATCH --output=arm_a_%A_%a.out
#
# Arm A of research/conditional_0dte_preregistration.md: one task per entry time.
#
#   sbatch trade_analysis/hpc/submit_arm_a.sbatch
#   python -m trade_analysis.hpc.collect_arm_a --results $HOME/vrp/results/arm_a
#
# PARTITION: `short`, CPU, deliberately. `gpu` caps a user at 4 RUNNING JOBS and 8h,
# `short` allows 50 and 48h (rc.northeastern.edu/partitions, checked 2026-09-11), so a
# 12-task array runs in ONE wave here and three waves there. There is no GPU work in this
# job at all -- it is pandas and a histogram.
#
# THE DATA IS NOT IN GIT AND MUST NOT BE. The Vilkov panel is licensed
# "academic replication only; redistribution of raw exchange data is not permitted", so it
# lives outside the repo and is rsynced by hand:
#
# Run this ON THE LAPTOP, in Git Bash. Not on the cluster, and not in PowerShell.
#
#   scp /c/Users/chaitanyakharche/Documents/research_data/0dte-strategies/data/data_opt.parquet \
#       kharche.c@login.explorer.northeastern.edu:/home/kharche.c/vrp/data/
#
# The first version of this comment said `rsync -av ~/Documents/...` and was wrong twice
# over: Git for Windows ships no rsync, and that source path exists only on the laptop, so
# pasted into a cluster shell it gives
#   change_dir "/home/kharche.c/Documents/..." failed: No such file or directory
# which is exactly what happened the first time someone followed it.
#
# 252 MB, one file. NOT to /scratch -- that is wiped in full on the first Tuesday of every
# month with no access-time condition.

set -euo pipefail

REPO="${REPO:-$HOME/vrp/rep}"
VRP_ROOT="${VRP_ROOT:-$HOME/vrp}"
PANEL="${PANEL:-$VRP_ROOT/data/data_opt.parquet}"
OUT="${OUT:-$VRP_ROOT/results/arm_a}"

cd "$REPO"
mkdir -p "$OUT"

# ---- fail loudly on a missing input, before Slurm records COMPLETED --------------------
if [[ ! -f "$PANEL" ]]; then
  echo "FATAL: panel not found at $PANEL" >&2
  echo "rsync it up first -- see the header of this file." >&2
  exit 2
fi

# ---- the grid must verify its own size --------------------------------------------------
# --array is hand-written in the header, which is exactly how a sweep silently runs 12 of
# 24 cells. Ask the code how many slots there are and refuse if they disagree.
N_SLOTS=$(python - <<'PY'
from trade_analysis.hpc.arm_a_ssd import SLOTS
print(len(SLOTS))
PY
)
if [[ "${SLURM_ARRAY_TASK_COUNT:-$N_SLOTS}" != "$N_SLOTS" ]]; then
  echo "FATAL: --array gives ${SLURM_ARRAY_TASK_COUNT} tasks but the code defines $N_SLOTS slots." >&2
  echo "Fix the --array range; do NOT edit it to match a guess." >&2
  exit 3
fi

# ---- environment, pinned; a resolver disagreement is a hard failure ---------------------
if [[ -f "$REPO/uv.lock" ]] && command -v uv >/dev/null 2>&1; then
  export UV_PROJECT_ENVIRONMENT="$VRP_ROOT/.venv"
  uv sync --locked --offline
  PY_BIN="$VRP_ROOT/.venv/bin/python"
else
  # conda is the fallback ONLY. It is how this project silently resolved torch 2.6.0+cu124
  # when 2.9 was required, so it announces itself rather than passing for a pinned env.
  echo "NOTE: no uv.lock -- falling back to conda, which is NOT a pinned environment."
  module load anaconda3/2024.06 2>/dev/null || true
  source activate vrp 2>/dev/null || conda activate vrp
  PY_BIN="$(command -v python)"
fi

# tiny models scale NEGATIVELY with many threads; pin to what was actually allocated
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

echo "host        $(hostname)"
echo "interpreter $PY_BIN"
echo "git         $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (DIRTY)')"
echo "task        ${SLURM_ARRAY_TASK_ID:-none} of $N_SLOTS"
echo "threads     $OMP_NUM_THREADS"

# ---- srun the payload, with NOTHING after it -------------------------------------------
# Slurm records the exit status of the BATCH SCRIPT. A trailing `echo` here turns a Python
# traceback into COMPLETED, which is the single most common way a failed sweep cell reads
# as a finished one. srun also gets a per-step exit code into sacct.
srun "$PY_BIN" -m trade_analysis.hpc.arm_a_ssd \
  --panel "$PANEL" \
  --slot-index "${SLURM_ARRAY_TASK_ID:-0}" \
  --out "$OUT"
