#!/bin/bash
# Zero-shot foundation models vs HAR, on the frozen validation block.
#
#   bash  trade_analysis/hpc/fetch_zeroshot.sh        <- login node, once, needs internet
#   sbatch trade_analysis/hpc/submit_zeroshot.sbatch  <- this
#
# A GPU IS REQUESTED HERE, unlike the HAR baseline, because this is ~15,000 forward passes
# through a 200M-parameter encoder rather than OLS on eight features. It is still a small
# job: one GPU, well under an hour.
#
# Runs honest and audit in the same job. Read the audit backwards from the obvious guess:
#     audit BETTER than honest -> PASS. The honest context did not contain that origin.
#     audit SAME  as honest    -> the origin was already inside the honest run. Void.

#SBATCH --job-name=vrp-zeroshot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# ABSOLUTE log path. A relative --output resolves against the SUBMISSION directory,
# which is the git clone, so Slurm wrote its .out/.err straight into the working tree.
# That is where the 24 stray vrp-*.out/.err files in ~/vrp/rep came from, and it is
# what tripped submit_arm_a's own provenance gate on its first cluster run. %u is the
# username. The directory must exist BEFORE sbatch -- Slurm opens the file before the
# script body runs, so an in-script mkdir is too late:   mkdir -p ~/vrp/logs
#SBATCH --output=/home/%u/vrp/logs/%x_%j.out
#SBATCH --error=/home/%u/vrp/logs/%x_%j.err

set -euo pipefail

REPO="${REPO:-$HOME/vrp/rep}"
VRP_ROOT="${VRP_ROOT:-$HOME/vrp}"
cd "$REPO"
mkdir -p "$VRP_ROOT/results"

ENV_NAME="${ENV_NAME:-vrp}"
PY="$HOME/.conda/envs/$ENV_NAME/bin/python"
if [ ! -x "$PY" ]; then
  echo "no interpreter at $PY -- see trade_analysis/hpc/README.md"
  exit 1
fi
echo "interpreter: $PY"

MODEL="${MODEL:-amazon/chronos-bolt-base}"

# FAIL NOW, LEGIBLY, rather than after the GPU is allocated and the data is loaded.
# On 2026-09-09 the fetch step was OOM-killed on the login node and the job was submitted
# anyway; without this check that surfaces as an ImportError forty seconds in, buried
# under a stack trace, having spent a GPU allocation to discover a missing package.
if ! "$PY" -c "import chronos" 2>/dev/null; then
  echo "chronos is not installed in env '$ENV_NAME'."
  echo "Install it ON A COMPUTE NODE -- the login node will OOM-kill pip:"
  echo "  srun --partition=short --time=00:30:00 --mem=16G --pty bash"
  echo "  cd ~/vrp/rep && bash trade_analysis/hpc/fetch_zeroshot.sh"
  exit 1
fi

# Prefer the cache, but do not FORCE offline. Discovery's compute nodes do have outbound
# network -- setup_env.sh pip-installs torch on one -- so pinning HF_HUB_OFFLINE=1 with an
# empty cache would turn a slow download into a hard failure for no reason. Offline is set
# only when the weights are already present, where it is a genuine safety net: it makes a
# cache miss an instant error instead of a silent re-download mid-job.
if "$PY" - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(sys.argv[1], local_files_only=True)
except Exception:
    sys.exit(1)
PY
then
  echo "weights cached; running offline"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  echo "weights NOT cached; allowing the job to download them"
fi

if [ ! -d "$VRP_ROOT/data/vrp" ] || [ -z "$(ls -A "$VRP_ROOT/data/vrp" 2>/dev/null)" ]; then
  echo "NO DATA at $VRP_ROOT/data/vrp -- push the frame up from the laptop first"
  exit 1
fi
echo "sessions available: $(ls "$VRP_ROOT/data/vrp"/*.parquet 2>/dev/null | wc -l)"
"$PY" -c "import torch; print('cuda:', torch.cuda.is_available(), '|',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

SAFE=$(echo "$MODEL" | tr '/' '_')

# Two context lengths, because context is a FAIRNESS parameter here, not a tuning knob.
# HAR is handed rv_prev_day / rv_prev_5 / rv_prev_22 -- explicit daily, weekly and MONTHLY
# aggregates of past variance. At 60 origins per session, 512 origins is about 8.5
# sessions and contains nothing like a 22-session lag. Reporting that chronos lost at 512
# alone would be beating a model we under-informed. 1320 origins = 22 sessions exactly,
# and chronos-bolt was trained at context 2048, so it is well inside range.
for CTX in ${CONTEXTS:-512 1320}; do
  echo
  echo "================ HONEST : $MODEL  context=$CTX ================"
  "$PY" -m trade_analysis.hpc.zeroshot_baseline \
    --data "$VRP_ROOT/data/vrp" --model "$MODEL" --context "$CTX" \
    | tee "$VRP_ROOT/results/zeroshot_${SAFE}_ctx${CTX}_honest.txt"
done
# Both controls. `saturate` is the one that decides whether this arm is believable:
# the whole context becomes the target, so any model that reads its input must return
# roughly that value and QLIKE must collapse toward zero. `recency` replaces only the
# last value and is weak by construction -- one point in 1320, against a model that
# instance-normalises over all of it -- but it is the closest analogue to the HAR audit,
# so it is kept for comparability.
for MODE in saturate recency; do
  echo
  echo "================ AUDIT [$MODE], context=1320 ================"
  "$PY" -m trade_analysis.hpc.zeroshot_baseline \
    --data "$VRP_ROOT/data/vrp" --model "$MODEL" --context 1320 --audit "$MODE" \
    | tee "$VRP_ROOT/results/zeroshot_${SAFE}_audit_${MODE}.txt"
done

echo
echo "Compare the two QLIKE tables. Audit must be BETTER."
echo "Then read the DM p-value, not the QLIKE level: at ~250 sessions a few thousandths"
echo "of QLIKE is noise, and section 6 is decided on the test, not on the ordering."
