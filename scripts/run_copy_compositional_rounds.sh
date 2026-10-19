#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

DTK_ENV=${DTK_ENV:-/opt/dtk-26.04/env.sh}
ARTIFACT_DIR=${ARTIFACT_DIR:-$HOME/tujie/apps/tinydl-artifacts}
DATASET_DIR=${DATASET_DIR:-$HOME/tujie/apps/datasets}
DEVICE=${DEVICE:-4}
ROUNDS=${ROUNDS:-3}
STEPS_PER_ROUND=${STEPS_PER_ROUND:-5000}
START_STEP=${START_STEP:-75000}
START_MODEL=${START_MODEL:-$ARTIFACT_DIR/model-zh-v1-format-copy-open-75k.tdl}

# shellcheck disable=SC1090
source "$DTK_ENV"
export HIP_VISIBLE_DEVICES=$DEVICE

TRAIN_BINARY=${TRAIN_BINARY:-./build/tinydl_train}
GENERATE_BINARY=${GENERATE_BINARY:-./build/tinydl_generate}
TOKENIZER=${TOKENIZER:-$ARTIFACT_DIR/tokenizer_zh_8k_eos.model}
TRAIN_TEXT=${TRAIN_TEXT:-$DATASET_DIR/corpus_zh_format_copy_compositional_train.txt}
VALIDATION_TEXT=${VALIDATION_TEXT:-$DATASET_DIR/corpus_zh_format_copy_compositional_validation.txt}
TRAIN_CACHE=${TRAIN_CACHE:-$DATASET_DIR/corpus_zh_format_copy_compositional_train.tdlcache}
VALIDATION_CACHE=${VALIDATION_CACHE:-$DATASET_DIR/corpus_zh_format_copy_compositional_validation.tdlcache}

resume_model=$START_MODEL
current_step=$START_STEP

run_eval() {
    local model=$1
    local label=$2
    local evaluation=$3
    python3 scripts/run_instruction_eval.py \
        --binary "$GENERATE_BINARY" \
        --model "$model" \
        --tokenizer "$TOKENIZER" \
        --evaluation-file "$evaluation" \
        --output "$ARTIFACT_DIR/$label.eval.jsonl" \
        --max-new-tokens 64 \
        --temperature 0.2 \
        --top-k 1 \
        --seed 42 \
        --device "$DEVICE" \
        --dcu
}

for ((round = 1; round <= ROUNDS; ++round)); do
    next_step=$((current_step + STEPS_PER_ROUND))
    label="model-zh-v1-format-copy-compositional-${next_step}"
    output_model="$ARTIFACT_DIR/$label.tdl"

    echo "=== round $round/$ROUNDS: $current_step -> $next_step ==="
    "$TRAIN_BINARY" \
        "$TRAIN_TEXT" \
        "$TOKENIZER" \
        "$output_model" \
        "$STEPS_PER_ROUND" \
        --validation-text "$VALIDATION_TEXT" \
        --resume "$resume_model" \
        --learning-rate 3e-5 \
        --min-learning-rate 1e-5 \
        --warmup-steps 0 \
        --gradient-accumulation-steps 4 \
        --answer-only-loss \
        --training-token-cache "$TRAIN_CACHE" \
        --validation-token-cache "$VALIDATION_CACHE" \
        --validation-interval 500 \
        --checkpoint-interval 1000 \
        --dcu

    run_eval "$output_model" "$label.copy2" \
        data/instruction_eval_format_copy2_v1.jsonl
    run_eval "$output_model" "$label.copy-count" \
        data/instruction_eval_format_copy_count_v1.jsonl
    run_eval "$output_model" "$label.compositional" \
        data/instruction_eval_format_copy_compositional_v1.jsonl
    run_eval "$output_model" "$label.lexical-ood" \
        data/instruction_eval_format_copy_count_ood_v1.jsonl

    resume_model=$output_model
    current_step=$next_step
done

echo "completed $ROUNDS rounds; final model: $resume_model"
