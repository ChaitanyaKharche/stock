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
# ALL FOUR §6 PRE-CONDITIONS MUST PASS BEFORE THIS IS SUBMITTED. In order:
#   python -m trade_analysis.momo_sweep.pack --out $HOME/momo/data     # once
#   python -m trade_analysis.momo_sweep.sweep --validate               # cond. 1: V0 n=8,598
#   python -m trade_analysis.momo_sweep.stats_test                     # cond. 2: statistics
#   python -m trade_analysis.momo_sweep.audit                          # cond. 3 + 4
#   sbatch trade_analysis/momo_sweep/submit_sweep.sbatch
#   python -m trade_analysis.momo_sweep.collect --results $HOME/momo/results --expect 500
#
# `audit` is the lookahead test (peeking must help MATERIALLY, or the honest path is already
# peeking) and the direction placebo (a coin flip must make nothing, or the fill/exit
# convention is manufacturing P&L). Both had no implementation until 2026-09-15, which is
# the real reason this job sat built-but-unsubmitted rather than any cluster problem.
#
# PARTITION: `short` is CPU and allows 50 concurrent jobs at up to 48h; `gpu` caps a user at
# 4 running jobs and 8h. There is no GPU work here at all -- it is a Python loop over bars.
# 500 tasks therefore run in 10 waves of 50.
#
# SIZING, and the first two numbers here were both measured on the WRONG MACHINE.
# The traded window is 1,497 QQQ sessions -- the loader reports 1,499 in-window and
# iter_prepared consumes the first two as indicator seed (range(PREFIX, ...) with PREFIX=2),
# which audit.py and run_shard agree on exactly.
# A laptop benchmark gave 0.151 ms per cell-run (revising an earlier 0.253 ms guess) and an
# implied ~177 CPU-hours. Measured ON A COMPUTE NODE 2026-09-15, the same audit workload
# took 808 s where the laptop took 205 s -- explorer cores are ~3.5x slower than a modern
# laptop's boost clock, which is the normal direction for cluster hardware and is exactly
# why a sizing benchmark has to be run where the job runs.
# Cluster-measured: ~0.51 ms per cell-run, so 2,822,400 cells x 1,497 sessions is
# ~700 CPU-hours. At 500 shards that is ~5,645 cells and ~80 min per task, plus a few
# minutes for the bootstrap. 10 waves of 50 on `short` is ~13 h of wall clock.
# --time is 6h -- a 4.5x margin over the cluster-measured 80 min, which is the number that
# matters -- because a task killed at the wall clock leaves a TRUNCATED shard that still
# looks like a perfectly good file to the collector. Had --time been set from the laptop
# benchmark with a tight margin, this is where that error would have cost the whole run.
# Peak RSS is ~1.2 GB against the 8G request, dominated by the (reps x cells) bootstrap
# array in studentize(): 10,000 x 5,645 x 8 B = 451 MB, plus a transient of the same size.
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

# ---- environment ------------------------------------------------------------------------
# THIS BLOCK USED TO ACTIVATE A CONDA ENV NAMED `vrp` THAT DOES NOT EXIST ON DISCOVERY.
# There is no uv.lock or pyproject.toml in this repo, so the uv branch was never taken and
# every one of the 500 tasks would have reached `conda activate vrp` and died. Recorded
# rather than quietly fixed because "the fallback path was never exercised" is the general
# shape of the bug, not a typo in an env name.
#
# The requirement is genuinely tiny and that is what makes this robust: the sweep imports
# NUMPY and nothing else. `trade_analysis/__init__.py` declares `__all__` and a version with
# no imports, so the `config.py`-raises-on-import chain is never touched, and Hansen's SPA
# is implemented in `stats.py` rather than taken from `arch`. Any Python 3.10+ with numpy
# runs this. So: take an explicit override, else try the envs that ACTUALLY exist on the
# box, else whatever python is on PATH -- then PROVE it by importing.
set +u
if [[ -n "${MOMO_PY:-}" ]]; then
  PY_BIN="$MOMO_PY"
else
  module load anaconda3/2024.06 2>/dev/null || module load miniconda3 2>/dev/null || true
  for env in trade-venv trade-env; do
    if conda activate "$env" 2>/dev/null || source activate "$env" 2>/dev/null; then
      echo "conda env  $env"
      break
    fi
  done
  PY_BIN="$(command -v python3 || command -v python)"
fi
set -u

# Prove the interpreter can actually run the payload, BEFORE srun. A task that fails inside
# srun on an ImportError is indistinguishable in `sacct` from one that failed on the data,
# and at 500 tasks that ambiguity costs a whole debugging cycle.
if ! "$PY_BIN" -c "import numpy, trade_analysis.momo_sweep.engine" 2>/dev/null; then
  echo "FATAL: $PY_BIN cannot import numpy + trade_analysis.momo_sweep.engine" >&2
  echo "  The sweep needs ONLY numpy -- so try the cheapest option FIRST:" >&2
  echo "    module load anaconda3/2024.06   # its base env already ships numpy" >&2
  echo "    MOMO_PY=\$(command -v python) sbatch ..." >&2
  echo "  If base numpy is somehow missing, a venv is far lighter than a conda env:" >&2
  echo "    python -m venv ~/momo/.venv && ~/momo/.venv/bin/pip install numpy" >&2
  echo "  DO NOT \`conda create ... numpy\` ON A LOGIN NODE. The defaults channel pulls the" >&2
  echo "  whole MKL stack (intel-openmp, mkl, mkl_fft, mkl_random, tbb, tbb-devel) and the" >&2
  echo "  login node SIGKILLs it during \"Verifying transaction\" -- observed 2026-09-15." >&2
  echo "  MKL is useless here anyway: this is a Python bar loop and the job pins" >&2
  echo "  OMP_NUM_THREADS=1. If you must use conda, add \`nomkl\` and run it under srun." >&2
  echo "  Also confirm \$REPO ($REPO) is the repo root -- the import is package-relative." >&2
  exit 4
fi

# ---- the grid must verify its own size --------------------------------------------------
# --array is hand-written above, which is exactly how a sweep silently runs 500 of 800
# shards and reports a p-value over the part that ran. Ask the code, and refuse on a
# disagreement rather than trusting the header.
N_CELLS=$("$PY_BIN" - <<'PY'
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

# Belt and braces against submission from inside another job: even with `srun` gone, a
# leaked CPU-bind request should not be able to reach anything that launches a process.
unset SLURM_CPU_BIND SLURM_CPU_BIND_LIST SLURM_CPU_BIND_TYPE SLURM_CPU_BIND_VERBOSE 2>/dev/null || true

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

# ---- exec the payload, with NOTHING after it --------------------------------------------
# NO `srun`, AND THAT IS A FIX, NOT A SIMPLIFICATION. Array job 10373735 failed 15 tasks in
# a row on:
#     srun: error: CPU binding outside of job step allocation,
#           allocated CPUs are: 0x2000400
#     srun: error: Unable to satisfy cpu bind request
# The job had been submitted from inside an `srun --pty bash` session, and that session's
# SLURM_CPU_BIND / SLURM_CPU_BIND_LIST / SLURM_CPU_BIND_TYPE were inherited by the batch
# script. `srun` read them and tried to bind each task to a CPU mask belonging to the PARENT
# job on a DIFFERENT node. Resubmitting from a login node avoids it, but relying on that is
# relying on nobody ever submitting from a compute node again.
#
# `srun` buys nothing here. This is `--ntasks=1`: one Python process, no MPI, no multi-node
# step. Dropping it removes the entire step-launch failure class, and `sacct` still records
# the batch step's State, Elapsed, MaxRSS and ExitCode.
#
# `exec` replaces the shell, so the interpreter's exit status IS the batch script's and
# nothing can run after it by construction. Slurm records the script's status, and a
# trailing command is the single most common way a Python traceback reads as COMPLETED --
# with `exec` that is not a discipline to remember, it is impossible.
#
# --start/--end are the pre-registration's DISCOVERY window and are passed EXPLICITLY rather
# than left to the defaults, so that reading this file tells you what the run covered.
# QQQ only: SPY's history begins 2022-01-03 and is entirely inside the held-out set.
exec "$PY_BIN" -m trade_analysis.momo_sweep.sweep \
  --data "$DATA" \
  --symbols QQQ \
  --start 2016-01-01 \
  --end 2021-12-31 \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" \
  --nshards "$N_TASKS" \
  --out "$OUT"
