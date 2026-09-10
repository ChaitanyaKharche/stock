#!/bin/bash
# RUN THIS ON THE LOGIN NODE. It is the only step here that needs the internet.
#
#   bash trade_analysis/hpc/fetch_zeroshot.sh
#
# Discovery's compute nodes have no outbound network. A job that tries to pull weights
# from huggingface.co does not fail fast -- it hangs on a connect timeout, burns its wall
# clock and dies with a stack trace that reads like a library bug. So the package and the
# weights are fetched here, up front, and the batch job runs with HF_HUB_OFFLINE=1 so a
# missing file is an immediate, legible error instead of a hang.
#
# No HF token is needed or wanted. Every model below is a public repo. Do NOT run
# `export HF_TOKEN=...` to make this work -- it lands in ~/.bash_history in plaintext on a
# shared filesystem, which has already happened once on this account.
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

echo
echo "-- package --"
"$PY" -m pip install --quiet --upgrade chronos-forecasting
"$PY" -c "import chronos; print('chronos', getattr(chronos, '__version__', 'installed'))"

# chronos-bolt-base is the default: ~200 MB, public, and fast enough that 15k origins fit
# in a short-partition job. Add repo ids here as you want to compare them -- each is
# scored by the same code on the same rows, so they are directly comparable.
MODELS="${MODELS:-amazon/chronos-bolt-base amazon/chronos-bolt-small}"

echo
echo "-- weights --"
for m in $MODELS; do
  echo "fetching $m"
  "$PY" - "$m" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1])
print("  cached at", p)
PY
done

echo
echo "-- cache size (the README measured this at 44 GB once; keep an eye on it) --"
du -sh "$HOME/.cache/huggingface" 2>/dev/null || true

echo
echo "Done. Now submit the offline job:"
echo "  sbatch trade_analysis/hpc/submit_zeroshot.sbatch"
