#!/bin/bash

set -euo pipefail

: "${WORK:?WORK is not set; load the Oberon environment before submitting}"
REPO_ROOT="$WORK/multimodal-vlm-inference"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash submit_descriptions.sh MODEL_ID [describe_materials.py options...]"
    exit 2
fi

MODEL_ID="$1"
shift

RESOURCES=(--partition=gpu-p1,gpu-p2 --gres=gpu:1 --mem=64G --time=02:00:00)

case "$MODEL_ID" in
    *Qwen2.5-VL-32B*|*Qwen3.8-27B*)
        MODEL_TAG="large"
        RESOURCES=(--partition=gpu-p1 --gres=gpu:2 --mem=128G --time=04:00:00)
        ;;
    *Qwen2.5-VL-3B*) MODEL_TAG="qwen25-3b" ;;
    *Qwen2.5-VL-7B*) MODEL_TAG="qwen25-7b" ;;
    *gemma-4-12B*) MODEL_TAG="gemma12b" ;;
    *Qwen3-VL-2B*) MODEL_TAG="qwen3-2b" ;;
    *Qwen3-VL-8B*) MODEL_TAG="qwen3-8b" ;;
    *) MODEL_TAG="vlm" ;;
esac

INPUT_MODE="split"
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    if [ "${ARGS[$i]}" = "--input-mode" ] && ((i + 1 < ${#ARGS[@]})); then
        INPUT_MODE="${ARGS[$((i + 1))]}"
        break
    fi
done

if [ "$MODEL_TAG" = "large" ]; then
    case "$MODEL_ID" in
        *Qwen2.5-VL-32B*) MODEL_TAG="qwen25-32b" ;;
        *Qwen3.8-27B*) MODEL_TAG="qwen38-27b" ;;
    esac
fi

mkdir -p "$REPO_ROOT/logs" "$REPO_ROOT/results/animation/perception/descriptions"

sbatch \
    --job-name="desc-${MODEL_TAG}-${INPUT_MODE}" \
    --chdir="$REPO_ROOT" \
    --output="$REPO_ROOT/logs/%x-%j.log" \
    "${RESOURCES[@]}" \
    "$REPO_ROOT/job_describe.sh" \
    "$MODEL_ID" \
    "$@"
