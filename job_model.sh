#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=%x-%j.log

set -e

: "${WORK:?WORK is not set; load the Oberon environment before submitting}"
REPO_ROOT="$WORK/multimodal-vlm-inference"

cd "$REPO_ROOT"
source /home/yzhou/venv/multimodal-vlm/bin/activate

# Keep model caches in the user work area.
export HF_HOME="$WORK/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$WORK/torch"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME"

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
