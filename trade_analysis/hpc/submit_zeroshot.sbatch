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
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

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

# No network on this node. Weights must already be cached by fetch_zeroshot.sh; this makes
# a missing one an instant error rather than a wall-clock-eating connect timeout.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [ ! -d "$VRP_ROOT/data/vrp" ] || [ -z "$(ls -A "$VRP_ROOT/data/vrp" 2>/dev/null)" ]; then
  echo "NO DATA at $VRP_ROOT/data/vrp -- push the frame up from the laptop first"
  exit 1
fi
echo "sessions available: $(ls "$VRP_ROOT/data/vrp"/*.parquet 2>/dev/null | wc -l)"
"$PY" -c "import torch; print('cuda:', torch.cuda.is_available(), '|',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

MODEL="${MODEL:-amazon/chronos-bolt-base}"
SAFE=$(echo "$MODEL" | tr '/' '_')

echo
echo "================ HONEST : $MODEL ================"
"$PY" -m trade_analysis.hpc.zeroshot_baseline \
  --data "$VRP_ROOT/data/vrp" --model "$MODEL" \
  | tee "$VRP_ROOT/results/zeroshot_${SAFE}_honest.txt"

echo
echo "================ AUDIT (one origin of future injected) ================"
"$PY" -m trade_analysis.hpc.zeroshot_baseline \
  --data "$VRP_ROOT/data/vrp" --model "$MODEL" --audit \
  | tee "$VRP_ROOT/results/zeroshot_${SAFE}_audit.txt"

echo
echo "Compare the two QLIKE tables. Audit must be BETTER."
echo "Then read the DM p-value, not the QLIKE level: at ~250 sessions a few thousandths"
echo "of QLIKE is noise, and section 6 is decided on the test, not on the ordering."
