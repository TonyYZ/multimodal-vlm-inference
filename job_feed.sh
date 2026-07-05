#!/bin/bash
#SBATCH --job-name=feed_experiment
#SBATCH --partition=gpu-p1,gpu-p2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.log

cd /store/scratch/yzhou/multimodal-vlm-inference
source ~/venv/multimodal-vlm/bin/activate

echo "Running on $(hostname)"
echo "Python: $(which python)"
echo "Python version: $(python -V)"
echo "Start time: $(date)"

python code/feed_experiment.py

echo "End time: $(date)"
