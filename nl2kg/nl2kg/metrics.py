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


from collections import defaultdict


def normalize_ops(ops: list[dict]) -> list[tuple]:
    """Convert operations to hashable tuples for set comparison.

    Empty strings and None are treated as equivalent (both become "").
    Only fields relevant to each operation type are included to avoid
    spurious mismatches from extra empty fields.
    """
    result = []
    for op in ops:
        op_type = op.get("op", "") or ""
        name = op.get("name", "") or ""
        node_type = op.get("node_type", "") or ""
        edge_type = op.get("edge_type", "") or ""
        source = op.get("source", "") or ""
        target = op.get("target", "") or ""
        key_val = op.get("key", "") or ""
        value = op.get("value", "") or ""

        # Normalize to only the relevant fields per operation type
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
            key = (op_type, name, node_type, edge_type, source, target, key_val, value)
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


def compute_metrics(results: list[dict]) -> dict:
    """Aggregate per-sample results into summary metrics."""
    n = len(results)
    if n == 0:
        return {}

    valid = [r for r in results if r.get("error") is None]

    metrics = {
        "total_samples": n,
        "valid_responses": len(valid),
        "timeout_errors": n - len(valid),
        "intent_accuracy": sum(r["intent_correct"] for r in valid) / max(len(valid), 1),
        "mean_ops_f1": sum(r["ops_f1"] for r in valid) / max(len(valid), 1),
        "exact_match_rate": sum(r["exact_match"] for r in valid) / max(len(valid), 1),
    }

    # Per-category breakdown
    by_cat: dict[str, list] = defaultdict(list)
    for r in valid:
        by_cat[r["category"]].append(r)

    metrics["per_category"] = {}
    for cat, cat_results in sorted(by_cat.items()):
        cn = len(cat_results)
        metrics["per_category"][cat] = {
            "count": cn,
            "intent_accuracy": sum(r["intent_correct"] for r in cat_results) / cn,
            "mean_ops_f1": sum(r["ops_f1"] for r in cat_results) / cn,
            "exact_match_rate": sum(r["exact_match"] for r in cat_results) / cn,
        }

    return metrics
