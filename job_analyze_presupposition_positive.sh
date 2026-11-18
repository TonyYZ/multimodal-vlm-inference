#!/bin/bash
#SBATCH --job-name=ppos-analysis
#SBATCH --partition=gpu-p1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

WORK="${WORK:-/store/work/users/$USER}"
REPO_ROOT="$WORK/multimodal-vlm-inference"

cd "$REPO_ROOT"
source /home/yzhou/venv/multimodal-vlm/bin/activate

python code/compare_model_performance.py \
    --human-effects materials/stats/TSC_EXP_2_human_effects.csv
