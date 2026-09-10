#!/bin/bash
# Install chronos and cache its weights, WITHOUT disturbing the env that produced the
# verified baseline.
#
# RUN IT ON A COMPUTE NODE, NOT THE LOGIN NODE:
#
#   srun --partition=short --time=00:30:00 --mem=16G --pty bash
#   cd ~/vrp/rep && bash trade_analysis/hpc/fetch_zeroshot.sh
#
# The login node killed this on 2026-09-09 -- "Killed", SIGKILL, the cgroup OOM reaper.
# pip's resolver is the memory hog: given `--upgrade` and a package that depends on
# torch, it downloads and unpacks candidate torch wheels to compare them, which is
# gigabytes on a node capped in the hundreds of megabytes. Discovery's compute nodes have
# both the memory and outbound network (setup_env.sh already pip-installs torch on one),
# so a compute node is the right place for all of it.
#
# No HF token is needed or wanted. Every model here is a public repo. Do NOT run
# `export HF_TOKEN=...` to make this work -- it lands in ~/.bash_history in plaintext on
# a shared filesystem, which has already happened once on this account.
set -euo pipefail

ENV_NAME="${ENV_NAME:-vrp}"
PY="$HOME/.conda/envs/$ENV_NAME/bin/python"
if [ ! -x "$PY" ]; then
  echo "no interpreter at $PY -- build the env first:"
  echo "  srun --partition=short --time=00:40:00 --mem=16G --pty bash"
  echo "  cd ~/vrp/rep && bash trade_analysis/hpc/setup_env.sh"
  exit 1
fi
echo "interpreter: $PY"

if [[ "$(hostname)" == login* || "$(hostname)" == explorer-* ]]; then
  echo
  echo "This looks like a LOGIN node ($(hostname)). pip will be OOM-killed here."
  echo "Get a compute node first:"
  echo "  srun --partition=short --time=00:30:00 --mem=16G --pty bash"
  echo "  cd ~/vrp/rep && bash trade_analysis/hpc/fetch_zeroshot.sh"
  exit 1
fi

# PIN THE ENV BEFORE TOUCHING IT. The baseline number -- HAR-RV 0.430642, reproduced on
# this env and on a laptop -- is only meaningful while the env stays put. If pip swaps
# torch or numpy underneath it, that reproduction silently stops being evidence.
BEFORE=$("$PY" -c "import torch,numpy;print(torch.__version__,numpy.__version__)")
echo "before: torch/numpy = $BEFORE"

echo
echo "-- package --"
# --no-deps, then the deps by hand. The resolver is what got killed, and it has no
# business reconsidering a torch that is already installed and already validated.
"$PY" -m pip install --no-cache-dir --quiet "transformers>=4.44" accelerate
"$PY" -m pip install --no-cache-dir --quiet --no-deps chronos-forecasting
"$PY" -c "import chronos; print('chronos', getattr(chronos, '__version__', 'installed'))"

AFTER=$("$PY" -c "import torch,numpy;print(torch.__version__,numpy.__version__)")
echo "after:  torch/numpy = $AFTER"
if [ "$BEFORE" != "$AFTER" ]; then
  echo
  echo "*** THE ENV MOVED: $BEFORE -> $AFTER ***"
  echo "Re-run the HAR baseline before believing any zero-shot number against it:"
  echo "  sbatch trade_analysis/hpc/submit_baseline.sbatch"
  echo "HAR-RV must still read 0.430642. If it does not, the comparison is void."
fi

# chronos-bolt-base is the default: ~200 MB, public, fast enough that 15k origins fit in
# a short-partition job. Add repo ids here to compare more -- each is scored by the same
# code on the same rows, so they are directly comparable.
MODELS="${MODELS:-amazon/chronos-bolt-base amazon/chronos-bolt-small}"

echo
echo "-- weights --"
for m in $MODELS; do
  echo "fetching $m"
  "$PY" - "$m" <<'PY'
import sys
from huggingface_hub import snapshot_download
print("  cached at", snapshot_download(sys.argv[1]))
PY
done

echo
echo "-- cache size (the README measured this at 44 GB once; keep an eye on it) --"
du -sh "$HOME/.cache/huggingface" 2>/dev/null || true

echo
echo "Done. Leave the compute node (exit), then submit:"
echo "  sbatch trade_analysis/hpc/submit_zeroshot.sbatch"
