#!/bin/bash
#SBATCH --partition=gpu-p1,gpu-p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

: "${WORK:?WORK is not set; load the Oberon environment before submitting}"
REPO_ROOT="$WORK/multimodal-vlm-inference"

if [ "$#" -lt 1 ]; then
    echo "Usage: job_describe.sh MODEL_ID [describe_materials.py options...]"
    exit 2
fi

MODEL_ID="$1"
shift

cd "$REPO_ROOT"
source /home/yzhou/venv/multimodal-vlm/bin/activate

export HF_HOME="$WORK/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$WORK/torch"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" results/descriptions

echo "Running on: $(hostname)"
echo "Model: $MODEL_ID"
echo "Arguments: $*"
echo "Python: $(which python)"
echo "Start: $(date)"

python -u ./code/describe_materials.py --model-id "$MODEL_ID" "$@"

echo "End: $(date)"
