#!/usr/bin/env bash
#SBATCH --job-name=fejer-cheb
#SBATCH --partition=gpu               # <-- set to your cluster's A100 partition
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00               # Phase 1 ~= 4h on A100; leave headroom
#SBATCH --output=slurm-%x-%j.out
#
# Submit one phase per job, e.g.:
#     sbatch experiments/slurm_run.sbatch experiments/run_phase1.sh
#     FEJER_GAMMA=3.0 sbatch experiments/slurm_run.sbatch experiments/run_phase2.sh
#
set -euo pipefail

module load cuda 2>/dev/null || true         # adjust to your cluster's module system
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-fejer}"

export WANDB_MODE="${WANDB_MODE:-offline}"    # 'online' if you ran `wandb login`

SCRIPT="${1:?usage: sbatch slurm_run.sbatch <experiments/run_phaseN.sh>}"
echo "host=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
bash "$SCRIPT"
