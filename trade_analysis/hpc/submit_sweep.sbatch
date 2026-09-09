#!/bin/bash
# Seed + hyperparameter sweep for the VRP variance forecaster.
#
#   sbatch --array=0-23 trade_analysis/hpc/submit_sweep.sbatch
#
# A SWEEP, NOT ONE RUN, and the reason matters. A single seed that beats HAR is not a
# result: with ~250 validation sessions the seed-to-seed spread in QLIKE is comparable to
# the effect being looked for. Running 4 seeds per configuration and reporting the
# DISTRIBUTION is what separates "this model works" from "this seed worked".
#
# One GPU, small model, short job. This problem is ~368 training sessions and a 20k
# parameter network; requesting more is queue time spent for nothing, and a reviewer who
# sees an H100-week on a 3 MB dataset will read it as inexperience rather than rigour.

#SBATCH --job-name=vrp-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1            # ANY gpu. Pinning h100/h200/a100 only buys queue
#                                 time, and a 20k-parameter MLP does not care.
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%A_%a.out    # relative to sbatch's cwd; keep it off scratch
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

# EVERYTHING PERSISTENT LIVES IN $HOME. /scratch on Discovery is purged after 45 days of
# no access, and it has already eaten this project once: the whole trading directory
# under /scratch/kharche.c/, including the ORIGINAL TFT model-definition source, is gone.
# The checkpoints survived only because they had been copied into the git repo.
#
# So: code from git, data rsynced up, results written to $HOME. Scratch is scratch --
# fine as fast working space DURING a job, never the place a result waits for you.
VRP_ROOT="${VRP_ROOT:-$HOME/vrp}"
cd "$VRP_ROOT"
mkdir -p "$VRP_ROOT/logs" "$VRP_ROOT/results"

module purge
module load cuda/12.4 || true
source "$HOME/venvs/vrp/bin/activate"

# 24 cells: 2 optimisers x 3 widths x 4 seeds.
OPTS=(adamw muon)
WIDTHS=(32 64 128)
SEEDS=(0 1 2 3)

i=${SLURM_ARRAY_TASK_ID:-0}
seed=${SEEDS[$(( i % 4 ))]}
width=${WIDTHS[$(( (i / 4) % 3 ))]}
opt=${OPTS[$(( (i / 12) % 2 ))]}

echo "cell $i: optimizer=$opt width=$width seed=$seed"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m trade_analysis.hpc.train_vrp \
  --data "$VRP_ROOT/data/vrp" \
  --optimizer "$opt" \
  --width "$width" \
  --seed "$seed" \
  --epochs 120 \
  --out "$VRP_ROOT/results/vrp_${opt}_w${width}_s${seed}.json"
