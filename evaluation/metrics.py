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

"""Evaluation metrics for NL2KG benchmark.

Provides comprehensive metrics including:
  - Intent accuracy
  - Operation F1 (per-op exact match)
  - Exact match rate
  - JSON validity rate
  - Embedding similarity (cosine similarity between predicted and expected)
  - Timing statistics (per-sample latency)
  - VRAM usage tracking
  - Per-category breakdown
  - Operation-type accuracy
  - Levenshtein edit distance on responses
"""

import json
import subprocess
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Operation normalization & F1
# ---------------------------------------------------------------------------


def normalize_ops(ops: list[dict]) -> list[tuple]:
    """Convert operations to hashable tuples for set comparison.

    Empty strings and None are treated as equivalent (both become "").
    Only fields relevant to each operation type are included to avoid
    spurious mismatches from extra empty fields.

    Location prepositions "in" and "at" are treated as equivalent because
    they express the same spatial relationship and LLMs (especially under
    grammar constraints) may output one instead of the other.
    """

    # Synonymous location-edge types → canonical form
    _EDGE_SYNONYMS = {"in": "at"}

    result = []
    for op in ops:
        op_type = (op.get("op", "") or "").strip().lower()
        name = (op.get("name", "") or "").strip().lower()
        node_type = (op.get("node_type", "") or "").strip().lower()
        edge_type = (op.get("edge_type", "") or "").strip().lower()
        source = (op.get("source", "") or "").strip().lower()
        target = (op.get("target", "") or "").strip().lower()
        key_val = (op.get("key", "") or "").strip().lower()
        value = (op.get("value", "") or "").strip().lower()

        # Normalize synonymous edge types
        edge_type = _EDGE_SYNONYMS.get(edge_type, edge_type)

        if op_type == "create_node":
            key = (op_type, name, node_type, "", "", "", "", "")
        elif op_type in ("create_edge", "remove_edge"):
            key = (op_type, "", "", edge_type, source, target, "", "")
        elif op_type == "remove_node":
            key = (op_type, name, "", "", "", "", "", "")
        elif op_type == "set_property":
            key = (op_type, name, "", "", source, target, key_val, value)
        elif op_type == "query":
            key = (op_type, "", "", "", "", "", "", "")
        else:
            key = (
                op_type,
                name,
                node_type,
                edge_type,
                source,
                target,
                key_val,
                value,
            )
        result.append(key)
    return result


def compute_f1(pred: list[tuple], gold: list[tuple]) -> float:
    """Compute F1 score between predicted and gold operation sets."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    pred_set = set(pred)
    gold_set = set(gold)

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_precision_recall(pred: list[tuple], gold: list[tuple]) -> tuple[float, float]:
    """Compute precision and recall between predicted and gold operation sets."""
    if not pred and not gold:
        return 1.0, 1.0
    if not pred:
        return 0.0, 0.0
    if not gold:
        return 0.0, 0.0

    pred_set = set(pred)
    gold_set = set(gold)
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    return precision, recall


# ---------------------------------------------------------------------------
# JSON validity
# ---------------------------------------------------------------------------


def check_json_validity(raw_text: str) -> bool:
    """Check if a raw LLM output string is valid JSON."""
    if raw_text is None:
        return False
    try:
        json.loads(raw_text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def check_schema_validity(raw_text: str) -> bool:
    """Check if a raw LLM output is valid JSON AND conforms to expected schema."""
    if raw_text is None:
        return False
    try:
        data = json.loads(raw_text)
        if not isinstance(data, dict):
            return False
        if "intent" not in data:
            return False
        if "operations" not in data:
            return False
        if not isinstance(data["operations"], list):
            return False
        if "response" not in data:
            return False
        return True
    except (json.JSONDecodeError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Embedding similarity
# ---------------------------------------------------------------------------


def compute_embedding_similarity(
    pred_text: str,
    gold_text: str,
    embedding_model=None,
) -> Optional[float]:
    """Compute cosine similarity between predicted and expected operation sets.

    Uses embedding_model.embed_documents() if available. Falls back to None.
    """
    if embedding_model is None or pred_text is None or gold_text is None:
        return None

    try:
        embeddings = embedding_model.embed_documents([pred_text, gold_text])
        if len(embeddings) != 2:
            return None

        import numpy as np

        a = np.array(embeddings[0])
        b = np.array(embeddings[1])
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception:
        return None


def serialize_operations(operations: list[dict]) -> str:
    """Serialize a list of operation dicts into a canonical string for embedding."""
    parts = []
    for op in operations:
        op_type = op.get("op", "")
        fields = []
        for k in [
            "op",
            "name",
            "node_type",
            "edge_type",
            "source",
            "target",
            "key",
            "value",
        ]:
            v = op.get(k, "") or ""
            if v:
                fields.append(f"{k}={v}")
        parts.append(" ".join(fields))
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def normalized_edit_similarity(s1: str, s2: str) -> float:
    """Compute 1 - (edit_distance / max_len) — a similarity in [0, 1]."""
    if not s1 and not s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(s1, s2) / max_len


# ---------------------------------------------------------------------------
# VRAM measurement
# ---------------------------------------------------------------------------


def get_gpu_vram_usage_mb() -> Optional[float]:
    """Query current GPU VRAM usage in MB via nvidia-smi.

    Returns the total used VRAM across all GPUs, or None if unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().split("\n")
        total = sum(float(line.strip()) for line in lines if line.strip())
        return total
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def get_gpu_vram_total_mb() -> Optional[float]:
    """Query total GPU VRAM capacity in MB via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        lines = result.stdout.strip().split("\n")
        total = sum(float(line.strip()) for line in lines if line.strip())
        return total
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


# ---------------------------------------------------------------------------
# Operation-type accuracy
# ---------------------------------------------------------------------------


def compute_op_type_accuracy(results: list[dict]) -> dict[str, dict]:
    """Compute per-operation-type precision, recall, and F1.

    Groups operations by their 'op' field (create_node, create_edge, etc.)
    and computes metrics for each type separately.
    """
    # Collect per-type TP, FP, FN
    type_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )

    for r in results:
        if r.get("error") is not None:
            continue

        pred_ops = (
            r.get("predicted", {}).get("operations", []) if r.get("predicted") else []
        )
        gold_ops = r.get("expected", {}).get("operations", [])

        pred_by_type: dict[str, list] = defaultdict(list)
        gold_by_type: dict[str, list] = defaultdict(list)

        for op in pred_ops:
            pred_by_type[(op.get("op", "") or "").lower()].append(op)
        for op in gold_ops:
            gold_by_type[(op.get("op", "") or "").lower()].append(op)

        all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())
        for op_type in all_types:
            if not op_type:
                continue
            p_ops = normalize_ops(pred_by_type.get(op_type, []))
            g_ops = normalize_ops(gold_by_type.get(op_type, []))

            p_set = set(p_ops)
            g_set = set(g_ops)

            tp = len(p_set & g_set)
            type_stats[op_type]["tp"] += tp
            type_stats[op_type]["fp"] += len(p_set) - tp
            type_stats[op_type]["fn"] += len(g_set) - tp

    # Compute metrics
    result = {}
    for op_type, stats in sorted(type_stats.items()):
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        result[op_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return result


# ---------------------------------------------------------------------------
# Aggregated metrics
# ---------------------------------------------------------------------------


def compute_metrics(results: list[dict]) -> dict:
    """Aggregate per-sample results into comprehensive summary metrics."""
    n = len(results)
    if n == 0:
        return {}

    valid = [r for r in results if r.get("error") is None]
    errors = [r for r in results if r.get("error") is not None]

    # Basic metrics
    metrics: dict = {
        "total_samples": n,
        "valid_responses": len(valid),
        "timeout_errors": sum(1 for r in errors if r.get("error") == "timeout"),
        "json_parse_errors": sum(
            1 for r in errors if r.get("error") == "json_parse_error"
        ),
        "other_errors": sum(
            1 for r in errors if r.get("error") not in ("timeout", "json_parse_error")
        ),
    }

    if not valid:
        return metrics

    # Intent accuracy
    metrics["intent_accuracy"] = sum(r["intent_correct"] for r in valid) / len(valid)

    # Operation F1
    metrics["mean_ops_f1"] = sum(r["ops_f1"] for r in valid) / len(valid)

    # Operation precision & recall
    metrics["mean_ops_precision"] = sum(r.get("ops_precision", 0.0) for r in valid) / len(
        valid
    )
    metrics["mean_ops_recall"] = sum(r.get("ops_recall", 0.0) for r in valid) / len(valid)

    # Exact match rate
    metrics["exact_match_rate"] = sum(r["exact_match"] for r in valid) / len(valid)

    # JSON validity rate
    json_valid_count = sum(1 for r in valid if r.get("json_valid", False))
    metrics["json_validity_rate"] = json_valid_count / n  # over all samples

    # Schema validity rate
    schema_valid_count = sum(1 for r in valid if r.get("schema_valid", False))
    metrics["schema_validity_rate"] = schema_valid_count / n

    # Embedding similarity
    emb_sims = [
        r["embedding_similarity"]
        for r in valid
        if r.get("embedding_similarity") is not None
    ]
    if emb_sims:
        metrics["mean_embedding_similarity"] = sum(emb_sims) / len(emb_sims)
        metrics["min_embedding_similarity"] = min(emb_sims)
        metrics["max_embedding_similarity"] = max(emb_sims)
    else:
        metrics["mean_embedding_similarity"] = None

    # Response edit similarity
    edit_sims = [
        r["response_edit_similarity"]
        for r in valid
        if r.get("response_edit_similarity") is not None
    ]
    if edit_sims:
        metrics["mean_response_edit_similarity"] = sum(edit_sims) / len(edit_sims)

    # Timing statistics
    times = [r["latency_s"] for r in valid if r.get("latency_s") is not None]
    if times:
        metrics["timing"] = {
            "mean_latency_s": sum(times) / len(times),
            "min_latency_s": min(times),
            "max_latency_s": max(times),
            "median_latency_s": sorted(times)[len(times) // 2],
            "total_time_s": sum(times),
            "p95_latency_s": sorted(times)[int(len(times) * 0.95)],
            "p99_latency_s": sorted(times)[int(len(times) * 0.99)],
        }

    # VRAM
    vram_samples = [r["vram_used_mb"] for r in valid if r.get("vram_used_mb") is not None]
    if vram_samples:
        metrics["vram"] = {
            "mean_used_mb": sum(vram_samples) / len(vram_samples),
            "max_used_mb": max(vram_samples),
            "min_used_mb": min(vram_samples),
        }
    vram_total = next(
        (r.get("vram_total_mb") for r in results if r.get("vram_total_mb") is not None),
        None,
    )
    if vram_total is not None:
        metrics.setdefault("vram", {})["total_mb"] = vram_total

    # Per-category breakdown
    by_cat: dict[str, list] = defaultdict(list)
    for r in valid:
        by_cat[r["category"]].append(r)

    metrics["per_category"] = {}
    for cat, cat_results in sorted(by_cat.items()):
        cn = len(cat_results)
        cat_metrics: dict = {
            "count": cn,
            "intent_accuracy": sum(r["intent_correct"] for r in cat_results) / cn,
            "mean_ops_f1": sum(r["ops_f1"] for r in cat_results) / cn,
            "exact_match_rate": sum(r["exact_match"] for r in cat_results) / cn,
        }
        # Category timing
        cat_times = [
            r["latency_s"] for r in cat_results if r.get("latency_s") is not None
        ]
        if cat_times:
            cat_metrics["mean_latency_s"] = sum(cat_times) / len(cat_times)

        # Category embedding similarity
        cat_emb = [
            r["embedding_similarity"]
            for r in cat_results
            if r.get("embedding_similarity") is not None
        ]
        if cat_emb:
            cat_metrics["mean_embedding_similarity"] = sum(cat_emb) / len(cat_emb)

        metrics["per_category"][cat] = cat_metrics

    # Per-operation-type breakdown
    metrics["per_op_type"] = compute_op_type_accuracy(results)

    return metrics
