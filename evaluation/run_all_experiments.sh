#!/bin/bash
# Copyright 2026 Miguel Ángel González Santamarta
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==========================================================================
# run_all_experiments.sh
#
# Runs the NL2KG evaluation for every model, with and without GBNF grammar.
# Each experiment:
#   1. Generates the YAML config for the model
#   2. Updates nl2kg.yaml with use_gbnf=true/false
#   3. Launches the NL2KG system
#   4. Waits for the action server
#   5. Runs the evaluation
#   6. Kills the system
#
# Usage:
#   cd ~/ros2_ws/src/nl2kg/evaluation
#   bash run_all_experiments.sh [--dataset PATH] [--no-embeddings] [--sleep SECS]
#
# Results are saved under results/<model>-{grammar,no_grammar}.json
# ==========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${HOME}/ros2_ws"
BRINGUP_PARAMS="${WS_DIR}/src/nl2kg/nl2kg_bringup/params"
EVAL_DIR="${SCRIPT_DIR}"
RESULTS_DIR="${EVAL_DIR}/results"
DATASET="${EVAL_DIR}/dataset/kg_dialogue_500.json"
EXTRA_ARGS=""
SLEEP_AFTER_LAUNCH=30  # seconds to wait after launching before evaluation

# Model definitions: (short_name  repo  filename)
# Add or remove models here to change the experiment matrix.
MODELS=(
    # "gemma-3-4b-it            bartowski/google_gemma-3-4b-it-GGUF           google_gemma-3-4b-it-Q4_K_M.gguf"
    # "Nemotron3-Nano-4B        bartowski/Nemotron-Mini-4B-Instruct-GGUF      Nemotron-Mini-4B-Instruct-Q4_K_M.gguf"
    # "Phi-4-mini-instruct      bartowski/microsoft_Phi-4-mini-instruct-GGUF  microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
    # "Qwen-3-4B                bartowski/Qwen_Qwen3-4B-GGUF                  Qwen_Qwen3-4B-Q4_K_M.gguf"
    "Qwen-3.5-4B              bartowski/Qwen_Qwen3.5-4B-GGUF                Qwen_Qwen3.5-4B-Q4_K_M.gguf"
    # "Qwen-3.5-9B              bartowski/Qwen_Qwen3.5-9B-GGUF                Qwen_Qwen3.5-9B-Q4_K_M.gguf"
    # "Llama-3.1-8B             bartowski/Meta-Llama-3.1-8B-Instruct-GGUF     Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)

# Grammar modes: (suffix  use_gbnf_value)
GRAMMAR_MODES=(
    "grammar      true"
    "no_grammar   false"
)

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --no-embeddings)
            EXTRA_ARGS="--no-embeddings"
            shift
            ;;
        --sleep)
            SLEEP_AFTER_LAUNCH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--dataset PATH] [--no-embeddings] [--sleep SECS]"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    echo ""
}

generate_model_yaml() {
    local repo="$1"
    local filename="$2"
    local output_path="$3"

    cat > "${output_path}" <<EOF
/**:
  ros__parameters:
    model:
      repo: ${repo}
      filename: ${filename}

    context:
      n_ctx: 8192
      n_batch: 256
      n_predict: -1

    gpu:
      n_gpu_layers: -1

    cpu:
      n_threads: -1
EOF
}

update_nl2kg_yaml() {
    local use_gbnf="$1"

    cat > "${BRINGUP_PARAMS}/nl2kg.yaml" <<EOF
nl2kg_node:
  ros__parameters:
    temperature: 0.0
    use_gbnf: ${use_gbnf}
    use_structured_output: false
    enable_rag: false
EOF
}

wait_for_action_server() {
    local max_wait=120
    local waited=0
    echo "  Waiting for NL2KG action server (max ${max_wait}s)..."

    while ! ros2 action list 2>/dev/null | grep -q "/nl2kg_node/nl2kg"; do
        sleep 2
        waited=$((waited + 2))
        if [ "${waited}" -ge "${max_wait}" ]; then
            echo "  ERROR: Action server not available after ${max_wait}s"
            return 1
        fi
    done
    echo "  Action server is ready (waited ${waited}s)"

    # Additional sleep to ensure model is fully loaded
    echo "  Waiting ${SLEEP_AFTER_LAUNCH}s for model warm-up..."
    sleep "${SLEEP_AFTER_LAUNCH}"
}

kill_ros_system() {
    echo "  Shutting down ROS system..."

    # Kill the launch process if it exists
    if [[ -n "${LAUNCH_PID:-}" ]] && kill -0 "${LAUNCH_PID}" 2>/dev/null; then
        kill "${LAUNCH_PID}" 2>/dev/null || true
        wait "${LAUNCH_PID}" 2>/dev/null || true
    fi

    # Make sure all related processes are terminated
    pkill -f "ros2 launch nl2kg_bringup" 2>/dev/null || true
    pkill -f "llama_node" 2>/dev/null || true
    pkill -f "nl2kg_node" 2>/dev/null || true
    pkill -f "embedding_node" 2>/dev/null || true
    pkill -f "reranker_node" 2>/dev/null || true

    sleep 5
    echo "  System shut down."
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

log "NL2KG Evaluation — All Experiments"

echo "Workspace:     ${WS_DIR}"
echo "Dataset:       ${DATASET}"
echo "Results dir:   ${RESULTS_DIR}"
echo "Models:        ${#MODELS[@]}"
echo "Grammar modes: ${#GRAMMAR_MODES[@]}"
echo "Total runs:    $(( ${#MODELS[@]} * ${#GRAMMAR_MODES[@]} ))"

# Check dataset exists
if [[ ! -f "${DATASET}" ]]; then
    echo ""
    echo "Dataset not found. Generating..."
    cd "${EVAL_DIR}"
    python3 generate_dataset.py -n 500 -s 42 -o "${DATASET}"
fi

# Create results directory
mkdir -p "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

TOTAL_RUNS=$(( ${#MODELS[@]} * ${#GRAMMAR_MODES[@]} ))
RUN_NUM=0
FAILED_RUNS=()

for model_line in "${MODELS[@]}"; do
    # Parse model definition
    read -r model_name model_repo model_file <<< "${model_line}"

    for grammar_line in "${GRAMMAR_MODES[@]}"; do
        read -r grammar_suffix use_gbnf <<< "${grammar_line}"

        RUN_NUM=$((RUN_NUM + 1))
        EXPERIMENT_NAME="${model_name}-${grammar_suffix}"
        OUTPUT_FILE="${RESULTS_DIR}/results-${EXPERIMENT_NAME}.json"

        log "[${RUN_NUM}/${TOTAL_RUNS}] ${EXPERIMENT_NAME}"

        # Skip if results already exist
        if [[ -f "${OUTPUT_FILE}" ]]; then
            echo "  Results already exist: ${OUTPUT_FILE}"
            echo "  Skipping. Delete the file to re-run."
            continue
        fi

        # 1. Generate model YAML
        MODEL_YAML="/tmp/nl2kg_eval_model_${model_name}.yaml"
        generate_model_yaml "${model_repo}" "${model_file}" "${MODEL_YAML}"
        echo "  Model config: ${MODEL_YAML}"

        # 2. Update nl2kg.yaml
        update_nl2kg_yaml "${use_gbnf}"
        echo "  Grammar: use_gbnf=${use_gbnf}"

        # 3. Launch the system
        echo "  Launching NL2KG system..."
        ros2 launch nl2kg_bringup nl2kg.launch.py \
            model_params:="${MODEL_YAML}" enable_embedding:=true \
            > "/tmp/nl2kg_eval_launch_${EXPERIMENT_NAME}.log" 2>&1 &
        LAUNCH_PID=$!
        echo "  Launch PID: ${LAUNCH_PID}"

        # 4. Wait for action server
        if ! wait_for_action_server; then
            echo "  FAILED: Could not start system for ${EXPERIMENT_NAME}"
            FAILED_RUNS+=("${EXPERIMENT_NAME}")
            kill_ros_system
            continue
        fi

        # 5. Run evaluation
        echo "  Running evaluation..."
        cd "${EVAL_DIR}"

        if python3 evaluate.py \
            --dataset "${DATASET}" \
            --output "${OUTPUT_FILE}" \
            ${EXTRA_ARGS}; then
            echo "  SUCCESS: Results saved to ${OUTPUT_FILE}"
        else
            echo "  FAILED: Evaluation returned non-zero exit code"
            FAILED_RUNS+=("${EXPERIMENT_NAME}")
        fi

        # 6. Kill the system
        kill_ros_system

    done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

log "All Experiments Complete"

echo "Total runs:     ${TOTAL_RUNS}"
echo "Failed runs:    ${#FAILED_RUNS[@]}"

if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed experiments:"
    for name in "${FAILED_RUNS[@]}"; do
        echo "  - ${name}"
    done
fi

echo ""
echo "Results saved in: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"*.json 2>/dev/null || echo "  (no result files found)"

echo ""
echo "To analyze results, run:"
echo "  cd ${EVAL_DIR}"
echo "  python3 analyze_results.py --results-dir ${RESULTS_DIR}"
