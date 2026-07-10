#!/bin/bash
#SBATCH --job-name=gesture-omni
#SBATCH --partition=gpu-p1,gpu-p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail

cd /store/scratch/yzhou/multimodal-vlm-inference
module load ffmpeg/6.1.1-gyns
source /home/yzhou/venv/multimodal-vlm/bin/activate

export HF_HOME=/store/scratch/yzhou/huggingface
export HF_HUB_CACHE=/store/scratch/yzhou/huggingface/hub
export TORCH_HOME=/store/scratch/yzhou/torch

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TORCH_HOME" results

echo "Running on: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $(which python)"
echo "ffmpeg: $(which ffmpeg)"
ffmpeg -version | head -n 1
python - <<'PY'
import torch
print("Torch CUDA available before model load:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
echo "Start: $(date)"

python -u ./code/test_gesture_omni.py "$@"

echo "End: $(date)"
