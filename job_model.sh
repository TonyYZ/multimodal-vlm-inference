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

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch job_model.sh MODEL_ID [feed_experiment.py options...]"
    exit 2
fi

MODEL_ID="$1"
shift

EXTRA_ARGS=("$@")
if [ "${#EXTRA_ARGS[@]}" -gt 0 ] && [[ "${EXTRA_ARGS[0]}" != --* ]]; then
    EXTRA_ARGS=(--runner "${EXTRA_ARGS[0]}" "${EXTRA_ARGS[@]:1}")
fi

echo "Running on: $(hostname)"
echo "Model: $MODEL_ID"
echo "Extra args: ${EXTRA_ARGS[*]:-(none)}"
echo "Python: $(which python)"
echo "Start: $(date)"

python -u ./code/feed_experiment.py \
    --model-id "$MODEL_ID" \
    "${EXTRA_ARGS[@]}"

echo "End: $(date)"
