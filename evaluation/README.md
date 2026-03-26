# NL2KG Evaluation

This directory contains the complete evaluation framework for the NL2KG system. It provides tools for dataset generation, model evaluation with comprehensive metrics, automated experimentation across multiple models and configurations, and results analysis.

## Directory Structure

```
evaluation/
├── generate_dataset.py       # Generate the KG-Dialogue-500 benchmark
├── evaluate.py               # Run evaluation on a single model configuration
├── metrics.py                # Comprehensive metrics (F1, VRAM, timing, embeddings, …)
├── analyze_results.py        # Compare results across all experiments
├── run_all_experiments.sh    # Bash script to run all models × grammar modes
├── dataset/                  # Generated benchmark datasets
│   └── kg_dialogue_500.json
├── results/                  # Evaluation results (one JSON per experiment)
│   ├── results-Qwen-3.5-4B-grammar.json
│   ├── results-Qwen-3.5-4B-no_grammar.json
│   └── …
└── README.md                 # This file
```

## Quick Start

### 1. Generate the Dataset

```bash
cd ~/ros2_ws/src/nl2kg/evaluation
python3 generate_dataset.py -n 500 -s 42 -o dataset/kg_dialogue_500.json
```

The dataset contains 500 samples across 8 categories. All utterances are written as natural human speech — no graph, node, or edge terminology:

| Category            | Weight | Description                                                          |
| ------------------- | ------ | -------------------------------------------------------------------- |
| Entity introduction | 75     | Announcing a new robot, object or person ("There's a robot named X") |
| Location update     | 62     | Stating where something is ("Send Alpha to the lab")                 |
| State/property      | 51     | Property updates in plain speech ("Alpha's battery is at 80%")       |
| Entity queries      | 62     | Natural questions about entities ("What is Alpha?")                  |
| Location queries    | 75     | Location and relation questions ("Where is Alpha?")                  |
| Entity forgetting   | 62     | Removing entities ("Forget about Alpha — it's gone")                 |
| Relation forgetting | 51     | Removing relations ("Alpha left the kitchen")                        |
| Compound statements | 62     | Rich sentences that imply multiple facts at once                     |

#### Setup operations

Several categories require the knowledge graph to already contain certain
entities before the utterance makes sense (e.g. you cannot set a property on
a node that does not exist, or remove an edge that was never created).
For those samples the dataset includes a `setup_operations` field — a list
of KG operations that are executed **before** the NL input is sent.

| Category            | Setup operations                            |
| ------------------- | ------------------------------------------- |
| Entity introduction | _(none — clean graph)_                      |
| Location update     | _(none — NL2KG auto-creates nodes)_         |
| State/property      | Creates the target robot node               |
| Entity queries      | Creates the entity being queried            |
| Location queries    | Creates robot node, location node, and edge |
| Entity forgetting   | Creates the node to be removed              |
| Relation forgetting | Creates robot node, location node, and edge |
| Compound statements | _(none — NL2KG auto-creates nodes)_         |

At evaluation time the evaluator **clears the entire graph before each
sample** and then applies the setup operations, ensuring every sample
starts from a deterministic, known graph state.

### 2. Run a Single Evaluation

First, launch the NL2KG system in one terminal:

```bash
ros2 launch nl2kg_bringup nl2kg.launch.py
```

Then run the evaluation in another terminal:

```bash
cd ~/ros2_ws/src/nl2kg/evaluation
python3 evaluate.py \
    --dataset dataset/kg_dialogue_500.json \
    --output results/results-MyModel-grammar.json
```

### 3. Run All Experiments (Automated)

The `run_all_experiments.sh` script automates evaluation across all models with and without GBNF grammar:

```bash
cd ~/ros2_ws/src/nl2kg/evaluation
bash run_all_experiments.sh
```

Options:

| Flag              | Default                        | Description                              |
| ----------------- | ------------------------------ | ---------------------------------------- |
| `--dataset PATH`  | `dataset/kg_dialogue_500.json` | Path to benchmark dataset                |
| `--no-embeddings` | _(not set)_                    | Disable embedding similarity             |
| `--sleep SECS`    | `30`                           | Wait time after launch for model warm-up |

The script will:

1. Generate the dataset if it doesn't exist
2. For each model × grammar mode combination:
   - Generate the model YAML config
   - Update `nl2kg.yaml` with `use_gbnf=true/false`
   - Launch the NL2KG system
   - Wait for the action server to be ready
   - Run the full evaluation
   - Shut down the system
3. Skip experiments whose result files already exist

#### Models

| Model               | Repo                                            |
| ------------------- | ----------------------------------------------- |
| gemma-3-4b-it       | bartowski/google_gemma-3-4b-it-GGUF             |
| Llama-3.1-8B        | bartowski/Meta-Llama-3.1-8B-Instruct-GGUF       |
| Nemotron3-Nano-4B   | bartowski/nvidia_Nemotron-Mini-4B-Instruct-GGUF |
| Phi-4-mini-instruct | bartowski/microsoft_Phi-4-mini-instruct-GGUF    |
| Qwen-3-4B           | bartowski/Qwen_Qwen3-4B-GGUF                    |
| Qwen-3.5-4B         | unsloth/Qwen3.5-4B-GGUF                         |
| Qwen-3.5-9B         | bartowski/Qwen_Qwen3.5-9B-GGUF                  |

Each model is tested with **grammar** (GBNF constrained decoding) and **no grammar** (unconstrained), yielding 14 total experiments.

### 4. Analyze Results

```bash
cd ~/ros2_ws/src/nl2kg/evaluation
python3 analyze_results.py --results-dir results/
```

Optionally export to CSV:

```bash
python3 analyze_results.py --results-dir results/ --csv results/summary.csv
```

The analysis script produces:

- **Comparative table** — All models × grammar modes side by side
- **Grammar ablation** — Delta between GBNF and unconstrained for each model
- **Per-category breakdown** — F1 per category for every experiment
- **Per-operation-type breakdown** — F1 per operation type (create_node, create_edge, etc.)

## Metrics

All metrics are computed by `metrics.py` and aggregated by `compute_metrics()`. The result JSON written by `evaluate.py` and read by `analyze_results.py` contains the following keys.

---

### Output validity

These metrics measure whether the LLM produced a well-formed response. They are computed over **all** samples (including those with errors), not just valid responses.

| Key                    | Type        | Description                                                                                          |
| ---------------------- | ----------- | ---------------------------------------------------------------------------------------------------- |
| `total_samples`        | int         | Total number of samples sent to the model                                                            |
| `valid_responses`      | int         | Samples that produced a parseable, schema-valid result                                               |
| `json_parse_errors`    | int         | Samples whose raw LLM output could not be parsed as JSON                                             |
| `other_errors`         | int         | Samples that failed for other reasons (e.g. action server error)                                     |
| `json_validity_rate`   | float [0–1] | `valid_json / total_samples` — includes all samples                                                  |
| `schema_validity_rate` | float [0–1] | `schema_valid / total_samples` — JSON valid **and** contains `intent`, `operations`, `response` keys |

---

### Intent accuracy

| Key               | Type        | Description                                                                                                                                                |
| ----------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `intent_accuracy` | float [0–1] | Fraction of valid responses where the predicted intent exactly matches the gold intent. Possible intents: `assert`, `query`, `remove`, `modify`, `unclear` |

---

### Operation matching

Operations are normalised before comparison: field values are lower-cased and trimmed; fields irrelevant to the operation type are zeroed out (e.g. `edge_type` is ignored for `create_node`). Matching uses **set-based** comparison (order-independent).

| Key                  | Type        | Description                                                                                                                     |
| -------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `mean_ops_precision` | float [0–1] | Mean precision across valid samples: `TP / (TP + FP)` where TP = correctly predicted operations, FP = predicted but not in gold |
| `mean_ops_recall`    | float [0–1] | Mean recall across valid samples: `TP / (TP + FN)` where FN = gold operations not predicted                                     |
| `mean_ops_f1`        | float [0–1] | Mean harmonic mean of precision and recall: `2·P·R / (P + R)`. Both empty sets → 1.0                                            |
| `exact_match_rate`   | float [0–1] | Fraction of valid samples where the intent is correct **and** the predicted operation set exactly equals the gold set           |

---

### Semantic similarity

| Key                             | Type                  | Description                                                                                                                                              |
| ------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mean_embedding_similarity`     | float [0–1] or `null` | Mean cosine similarity between the BGE embedding of the predicted operations (serialised) and the gold operations. `null` when `--no-embeddings` is used |
| `min_embedding_similarity`      | float or `null`       | Minimum cosine similarity across valid samples                                                                                                           |
| `max_embedding_similarity`      | float or `null`       | Maximum cosine similarity across valid samples                                                                                                           |
| `mean_response_edit_similarity` | float [0–1]           | Mean of `1 − (Levenshtein distance / max_len)` between predicted and gold `response` strings. 1.0 = identical, 0.0 = completely different                |

---

### Timing (`timing` sub-object)

All values are in **seconds** and are computed over valid responses only.

| Key                       | Type  | Description                                                                      |
| ------------------------- | ----- | -------------------------------------------------------------------------------- |
| `timing.mean_latency_s`   | float | Average wall-clock time per sample from sending the goal to receiving the result |
| `timing.median_latency_s` | float | Median latency — robust to outliers                                              |
| `timing.min_latency_s`    | float | Fastest sample                                                                   |
| `timing.max_latency_s`    | float | Slowest sample                                                                   |
| `timing.p95_latency_s`    | float | 95th-percentile latency (tail performance)                                       |
| `timing.p99_latency_s`    | float | 99th-percentile latency (worst-case tail)                                        |
| `timing.total_time_s`     | float | Sum of all per-sample latencies                                                  |

---

### VRAM (`vram` sub-object)

Measured via `nvidia-smi --query-gpu=memory.used` at the start and end of each sample. Values are in **MB** and aggregate across all GPUs. `null` when no GPU is available.

| Key                 | Type            | Description                                                |
| ------------------- | --------------- | ---------------------------------------------------------- |
| `vram.mean_used_mb` | float or `null` | Mean VRAM consumption recorded during valid samples        |
| `vram.min_used_mb`  | float or `null` | Minimum VRAM recorded across all valid samples             |
| `vram.max_used_mb`  | float or `null` | Maximum VRAM recorded across all valid samples (peak load) |
| `vram.total_mb`     | float or `null` | Total GPU VRAM capacity reported by `nvidia-smi`           |

---

### Per-category breakdown (`per_category` sub-object)

One entry per dataset category. Each entry contains:

| Key                         | Type            | Description                                        |
| --------------------------- | --------------- | -------------------------------------------------- |
| `count`                     | int             | Number of valid samples in this category           |
| `intent_accuracy`           | float [0–1]     | Intent accuracy for this category                  |
| `mean_ops_f1`               | float [0–1]     | Mean operation F1 for this category                |
| `exact_match_rate`          | float [0–1]     | Exact match rate for this category                 |
| `mean_latency_s`            | float           | Mean per-sample latency for this category          |
| `mean_embedding_similarity` | float or `null` | Mean cosine embedding similarity for this category |

---

### Per-operation-type breakdown (`per_op_type` sub-object)

One entry per operation type seen in the dataset (`create_node`, `create_edge`, `remove_node`, `remove_edge`, `set_property`, `query`). Metrics are computed by aggregating TP/FP/FN counts globally across all samples and then computing precision/recall/F1 once, making them **macro-averaged at type level** rather than sample level.

| Key         | Type        | Description                                                  |
| ----------- | ----------- | ------------------------------------------------------------ |
| `precision` | float [0–1] | `TP / (TP + FP)` for this operation type across all samples  |
| `recall`    | float [0–1] | `TP / (TP + FN)` for this operation type across all samples  |
| `f1`        | float [0–1] | Harmonic mean of precision and recall                        |
| `tp`        | int         | True positives — correctly predicted operations of this type |
| `fp`        | int         | False positives — predicted but not in gold                  |
| `fn`        | int         | False negatives — in gold but not predicted                  |
