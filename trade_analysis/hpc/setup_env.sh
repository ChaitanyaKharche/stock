#!/bin/bash
# One-time environment build on Discovery.
#
#   bash trade_analysis/hpc/setup_env.sh
#
# Conda rather than venv because this account already runs conda (`(base)` in the prompt,
# and ~/.conda/envs holds trade-venv / legal-env / 8674-env from earlier work).
#
# RUN IT ON A COMPUTE NODE, not the login node. Installing torch pulls ~2.5 GB and pegs a
# core for several minutes; login nodes are shared and this is exactly the sort of job that
# gets a polite email from RC:
#
#   srun --partition=short --time=00:40:00 --mem=16G --pty bash
#
# The env lives in $HOME (~/.conda/envs/vrp), which persists. Nothing here touches
# /scratch, which is wiped IN FULL on the first Tuesday of every month -- no access-time
# condition, so using a file does not protect it. See the header of submit_sweep.sbatch.
set -euo pipefail

ENV_NAME="${ENV_NAME:-vrp}"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env '$ENV_NAME' already exists -- activating and checking it instead"
else
  echo "creating conda env '$ENV_NAME' (python 3.11)"
  conda create -y -n "$ENV_NAME" python=3.11
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# CPU-side first: har_baseline.py needs ONLY these, and it is the step that runs before
# any GPU is justified. If the torch install below fails, the baseline still works.
pip install --quiet --upgrade pip
pip install --quiet numpy pandas pyarrow scipy

# torch >= 2.9 ships torch.optim.Muon natively; train_vrp falls back to AdamW and says so
# if it is absent, so a mismatch here degrades rather than breaks.
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124

echo
echo "--- verification ---"
python - <<'PY'
import importlib
for m in ("numpy", "pandas", "pyarrow", "scipy"):
    try:
        importlib.import_module(m)
        print(f"  {m:<10} ok")
    except ImportError as e:
        print(f"  {m:<10} MISSING ({e})")
try:
    import torch
    print(f"  torch      {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    print(f"  Muon       {'native' if hasattr(torch.optim, 'Muon') else 'absent -> AdamW'}")
except ImportError:
    print("  torch      MISSING -- the CPU baseline still runs; the GPU sweep will not")
PY

echo
echo "cuda_available=False here is EXPECTED on a login or CPU node."
echo "It only matters inside the GPU job, where submit_sweep.sbatch prints nvidia-smi."
echo
echo "next:  sbatch trade_analysis/hpc/submit_baseline.sbatch"
