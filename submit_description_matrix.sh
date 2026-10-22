#!/bin/bash

set -euo pipefail

: "${WORK:?WORK is not set; load the Oberon environment before submitting}"
REPO_ROOT="$WORK/multimodal-vlm-inference"
EXPECTED_ROWS=24

MODELS=(
    "Qwen/Qwen2.5-VL-3B-Instruct"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "Qwen/Qwen2.5-VL-32B-Instruct"
    "Qwen/Qwen3-VL-2B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3.8-27B"
    "google/gemma-4-12B-it"
)

if [ "${1:-split}" = "all" ]; then
    MODES=(split pure-video pure-text)
else
    MODES=("${1:-split}")
fi

slug() {
    printf '%s' "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's|/|__|g; s|[^a-z0-9_.-]+|_|g; s|^_+||; s|_+$||'
}

cd "$REPO_ROOT"
for model in "${MODELS[@]}"; do
    for mode in "${MODES[@]}"; do
        output="results/descriptions/$(slug "$model")_${mode//-/_}_descriptions.jsonl"
        valid_rows=0
        if [ -f "$output" ]; then
            valid_rows=$(python -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    rows = [json.loads(line) for line in source if line.strip()]
print(sum(row.get("prompt_version") == "3" and bool(row.get("response")) for row in rows))
' "$output")
        fi
        if [ "$valid_rows" -eq "$EXPECTED_ROWS" ]; then
            echo "Skipping complete output: $output"
            continue
        fi

        bash submit_descriptions.sh \
            "$model" \
            --input-mode "$mode" \
            --max-new-tokens 1024 \
            --output "$output"
    done
done
