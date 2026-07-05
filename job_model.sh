#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=%x-%j.log

set -e

cd /store/scratch/yzhou/multimodal-vlm-inference
source /home/yzhou/venv/multimodal-vlm/bin/activate

# store huggingface cache in scratch
export HF_HOME=/store/scratch/yzhou/huggingface
export HF_HUB_CACHE=/store/scratch/yzhou/huggingface/hub
export TORCH_HOME=/store/scratch/yzhou/torch

MODEL_ID="$1"
RUNNER="${2:-}"

echo "Running on: $(hostname)"
echo "Model: $MODEL_ID"
echo "Runner: ${RUNNER:-default}"
echo "Python: $(which python)"
echo "Start: $(date)"

if [ -n "$RUNNER" ]; then
    python -u ./code/feed_experiment.py \
        --model-id "$MODEL_ID" \
        --runner "$RUNNER"
else
    python -u ./code/feed_experiment.py \
        --model-id "$MODEL_ID"
fi

echo "End: $(date)"
