#!/bin/bash

set -euo pipefail

WORK="${WORK:-/store/work/users/$USER}"
export WORK
REPO_ROOT="$WORK/multimodal-vlm-inference"
INDEX="processed/vlm_sequences/presupposition_positive_index.jsonl"

cd "$REPO_ROOT"
mkdir -p logs results

job_ids=()

submit_model() {
    local model_id="$1"
    local runner="$2"
    local slug="$3"
    local short_name="$4"
    local partition="$5"
    local gpus="$6"
    local memory="$7"
    local time_limit="$8"

    for mode in split pure-video pure-text; do
        local mode_slug="${mode//-/_}"
        local mode_short
        case "$mode" in
            split) mode_short="s" ;;
            pure-video) mode_short="v" ;;
            pure-text) mode_short="t" ;;
        esac

        local output="results/animation/inference/ratings/${slug}_${mode_slug}_presupposition_positive_sequence_ratings.jsonl"
        local args=(
            "$model_id"
            --input-mode "$mode"
            --sequence-index "$INDEX"
            --output "$output"
        )
        if [[ "$runner" != "auto" ]]; then
            args+=(--runner "$runner")
        fi

        local job_id
        job_id=$(sbatch --parsable \
            --job-name="ppos-${short_name}-${mode_short}" \
            --partition="$partition" \
            --gres="gpu:${gpus}" \
            --mem="$memory" \
            --time="$time_limit" \
            --output="logs/%x-%j.log" \
            job_model.sh "${args[@]}")
        job_ids+=("${job_id%%;*}")
        printf '%s\t%s\t%s\n' "$job_id" "$model_id" "$mode"
    done
}

submit_model "Qwen/Qwen2.5-VL-3B-Instruct"  auto   "qwen__qwen2.5-vl-3b-instruct"  "q25-3b"  "gpu-p1,gpu-p2" 1 64G  00:20:00
submit_model "Qwen/Qwen2.5-VL-7B-Instruct"  auto   "qwen__qwen2.5-vl-7b-instruct"  "q25-7b"  "gpu-p1,gpu-p2" 1 64G  00:20:00
submit_model "Qwen/Qwen2.5-VL-32B-Instruct" auto   "qwen__qwen2.5-vl-32b-instruct" "q25-32b" "gpu-p1"        2 128G 01:00:00
submit_model "Qwen/Qwen3-VL-2B-Instruct"    auto   "qwen__qwen3-vl-2b-instruct"    "q3-2b"   "gpu-p1,gpu-p2" 1 64G  00:20:00
submit_model "Qwen/Qwen3-VL-8B-Instruct"    auto   "qwen__qwen3-vl-8b-instruct"    "q3-8b"   "gpu-p1,gpu-p2" 1 64G  00:30:00
submit_model "Qwen/Qwen3.8-27B"             auto   "qwen__qwen3.8-27b"             "q38-27b" "gpu-p1"        2 128G 01:30:00
submit_model "google/gemma-4-12B-it"         gemma4 "google__gemma-4-12b-it"         "gemma"   "gpu-p1,gpu-p2" 1 64G  00:30:00

dependency=$(IFS=:; echo "${job_ids[*]}")
analysis_job=$(sbatch --parsable \
    --job-name=ppos-analysis \
    --partition=gpu-p1 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:10:00 \
    --dependency="afterok:${dependency}" \
    --output="logs/%x-%j.log" \
    --wrap="cd '$REPO_ROOT' && source /home/yzhou/venv/multimodal-vlm/bin/activate && python code/compare_model_performance.py --human-effects materials/stats/TSC_EXP_2_human_effects.csv")

echo "Analysis job: $analysis_job"
