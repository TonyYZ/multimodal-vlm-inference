#!/bin/bash
#SBATCH --job-name=feed_experiment
#SBATCH --partition=gpu-p1,gpu-p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.log

set -euo pipefail

: "${WORK:?WORK is not set; load the Oberon environment before submitting}"
REPO_ROOT="$WORK/multimodal-vlm-inference"

cd "$REPO_ROOT"
source ~/venv/multimodal-vlm/bin/activate

export HF_HOME="$WORK/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$WORK/torch"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME"

echo "Running on $(hostname)"
echo "Python: $(which python)"
echo "Python version: $(python -V)"
echo "Start time: $(date)"

python code/feed_experiment.py

echo "End time: $(date)"
