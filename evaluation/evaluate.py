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

"""Evaluate NL2KG pipeline on the KG-Dialogue benchmark.

Runs each sample through the NL2KG node (via ROS 2 action), compares
structural output against ground truth, and computes comprehensive metrics:
  - Intent accuracy
  - Operation F1 / precision / recall
  - Operation-type accuracy
  - Exact match rate
  - JSON validity & schema validity rate
  - Embedding similarity (cosine) between predicted and expected operations
  - Per-sample latency (wall-clock time)
  - GPU VRAM usage per sample
  - Response edit similarity (Levenshtein-based)
  - Per-category breakdown

Before each sample the evaluator **clears the knowledge graph** and
optionally applies ``setup_operations`` stored in the sample so that
the graph is in the exact state required by the utterance.

Each sample waits indefinitely for the action server to respond
(no per-sample timeout). The evaluation blocks until all samples complete.

Usage:
    python3 evaluate.py \\
        --dataset dataset/kg_dialogue_500.json \\
        --output results/results-MyModel.json
"""

import json
import time
import argparse
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from knowledge_graph import KnowledgeGraph

from nl2kg_msgs.action import NL2KG
from metrics import (
    normalize_ops,
    compute_f1,
    compute_precision_recall,
    compute_metrics,
    check_json_validity,
    check_schema_validity,
    compute_embedding_similarity,
    serialize_operations,
    normalized_edit_similarity,
    get_gpu_vram_usage_mb,
    get_gpu_vram_total_mb,
)


class Evaluator(Node):

    def __init__(
        self,
        samples: list[dict],
        use_embeddings: bool = True,
    ) -> None:
        super().__init__("nl2kg_evaluator")
        self.samples = samples

        self._action_client = ActionClient(self, NL2KG, "/nl2kg_node/nl2kg")
        self._results: list[dict] = []

        # Embedding model for similarity computation
        self._embedding_model = None
        if use_embeddings:
            try:
                from llama_ros.langchain import LlamaROSEmbeddings

                self._embedding_model = LlamaROSEmbeddings()
                self.get_logger().info("Embedding model loaded for similarity metrics")
            except Exception as e:
                self.get_logger().warn(
                    f"Could not load embedding model: {e}. "
                    "Embedding similarity will be skipped."
                )

        # VRAM baseline
        self._vram_total = get_gpu_vram_total_mb()
        if self._vram_total is not None:
            self.get_logger().info(f"GPU VRAM total: {self._vram_total:.0f} MB")

        # Knowledge Graph handle — used to clear/setup graph between samples
        self._graph = KnowledgeGraph.get_instance()
        self.get_logger().info("Connected to shared KnowledgeGraph instance")

    def run(self) -> list[dict]:
        self.get_logger().info("Waiting for NL2KG action server...")
        self._action_client.wait_for_server()
        self.get_logger().info("Connected to action server")

        for i, sample in enumerate(self.samples):
            self.get_logger().info(
                f"[{i + 1}/{len(self.samples)}] {sample['nl_input'][:60]}..."
            )
            result = self._evaluate_one(sample)
            self._results.append(result)

            # Progress log every 50 samples
            if (i + 1) % 50 == 0:
                valid = [r for r in self._results if r.get("error") is None]
                if valid:
                    avg_f1 = sum(r["ops_f1"] for r in valid) / len(valid)
                    avg_time = sum(
                        r["latency_s"] for r in valid if r.get("latency_s")
                    ) / max(len(valid), 1)
                    self.get_logger().info(
                        f"  Progress: {i + 1}/{len(self.samples)} | "
                        f"Mean F1={avg_f1:.3f} | Mean latency={avg_time:.2f}s"
                    )

        return self._results

    # ------------------------------------------------------------------
    # Graph management (clear + setup before each sample)
    # ------------------------------------------------------------------
    def _clear_graph(self) -> None:
        """Remove all edges and nodes from the knowledge graph."""
        g = self._graph
        # Remove edges first (they reference nodes)
        edges = g.get_edges()
        if edges:
            g.remove_edges(edges)
        # Then remove nodes
        nodes = g.get_nodes()
        if nodes:
            g.remove_nodes(nodes)

    def _apply_setup_operations(self, setup_ops: list[dict]) -> None:
        """Execute a list of KG operations to prepare graph state."""
        g = self._graph
        for op in setup_ops:
            op_type = op.get("op", "")
            try:
                if op_type == "create_node":
                    if not g.has_node(op["name"]):
                        g.create_node(op["name"], op.get("node_type", "unknown"))
                elif op_type == "create_edge":
                    src, tgt = op["source"], op["target"]
                    # Auto-create source/target nodes if missing
                    if not g.has_node(src):
                        g.create_node(src, "unknown")
                    if not g.has_node(tgt):
                        g.create_node(tgt, "unknown")
                    if not g.has_edge(op["edge_type"], src, tgt):
                        g.create_edge(op["edge_type"], src, tgt)
                elif op_type == "set_property":
                    name = op["name"]
                    if g.has_node(name):
                        from nl2kg.utils import cast_value

                        node = g.get_node(name)
                        node.set_property(op["key"], cast_value(op["value"]))
                        g.update_node(node)
            except Exception as e:
                self.get_logger().warn(f"Setup op {op_type} failed: {e}")

    def _evaluate_one(self, sample: dict) -> dict:
        # Clear graph and apply setup operations
        self._clear_graph()
        setup_ops = sample.get("setup_operations", [])
        if setup_ops:
            self._apply_setup_operations(setup_ops)
            # Small delay to let graph updates propagate via the topic
            time.sleep(0.5)

        # Measure VRAM before
        vram_before = get_gpu_vram_usage_mb()

        # Measure time
        t0 = time.perf_counter()

        goal = NL2KG.Goal()
        goal.input_text = sample["nl_input"]

        send_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        if send_future.result() is None:
            latency = time.perf_counter() - t0
            return self._error_result(sample, "goal_send_failed", latency)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            latency = time.perf_counter() - t0
            return self._error_result(sample, "goal_rejected", latency)

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        latency = time.perf_counter() - t0

        if result_future.result() is None:
            return self._error_result(sample, "result_failed", latency)

        action_result = result_future.result().result

        # Measure VRAM after
        vram_after = get_gpu_vram_usage_mb()

        return self._evaluate_result(
            sample, action_result, latency, vram_before, vram_after
        )

    def _error_result(self, sample: dict, error: str, latency: float) -> dict:
        return {
            "id": sample.get("id"),
            "category": sample["category"],
            "nl_input": sample["nl_input"],
            "intent_correct": False,
            "ops_f1": 0.0,
            "ops_precision": 0.0,
            "ops_recall": 0.0,
            "exact_match": False,
            "json_valid": False,
            "schema_valid": False,
            "embedding_similarity": None,
            "response_edit_similarity": None,
            "latency_s": latency,
            "vram_used_mb": None,
            "vram_total_mb": self._vram_total,
            "predicted": None,
            "expected": sample["expected"],
            "error": error,
        }

    def _evaluate_result(
        self,
        sample: dict,
        result: NL2KG.Result,
        latency: float,
        vram_before: float | None,
        vram_after: float | None,
    ) -> dict:
        expected = sample["expected"]

        # Intent accuracy
        intent_correct = result.intent == expected["intent"]

        # Convert action result operations to dicts
        pred_ops_dicts = [
            {
                "op": op.op,
                "name": op.name,
                "node_type": op.node_type,
                "edge_type": op.edge_type,
                "source": op.source,
                "target": op.target,
                "key": op.key,
                "value": op.value,
            }
            for op in result.operations
        ]
        gold_ops_dicts = expected.get("operations", [])

        # Normalize for comparison
        pred_ops = normalize_ops(pred_ops_dicts)
        gold_ops = normalize_ops(gold_ops_dicts)

        # F1, precision, recall
        ops_f1 = compute_f1(pred_ops, gold_ops)
        ops_precision, ops_recall = compute_precision_recall(pred_ops, gold_ops)

        # Exact match
        exact = intent_correct and ops_f1 == 1.0

        # JSON validity — reconstruct what the raw output would look like
        raw_json_str = json.dumps(
            {
                "intent": result.intent,
                "operations": pred_ops_dicts,
                "response": result.response,
            }
        )
        json_valid = check_json_validity(raw_json_str)
        schema_valid = check_schema_validity(raw_json_str)

        # Embedding similarity between predicted and expected operations
        embedding_sim = None
        if self._embedding_model is not None:
            pred_text = serialize_operations(pred_ops_dicts)
            gold_text = serialize_operations(gold_ops_dicts)
            if pred_text or gold_text:
                embedding_sim = compute_embedding_similarity(
                    pred_text, gold_text, self._embedding_model
                )

        # Response edit similarity
        gold_response = expected.get("response", "")
        response_edit_sim = None
        if gold_response and not gold_response.startswith("(answer depends"):
            response_edit_sim = normalized_edit_similarity(result.response, gold_response)

        # VRAM
        vram_used = None
        if vram_after is not None:
            vram_used = vram_after

        return {
            "id": sample.get("id"),
            "category": sample["category"],
            "nl_input": sample["nl_input"],
            "intent_correct": intent_correct,
            "ops_f1": ops_f1,
            "ops_precision": ops_precision,
            "ops_recall": ops_recall,
            "exact_match": exact,
            "json_valid": json_valid,
            "schema_valid": schema_valid,
            "embedding_similarity": embedding_sim,
            "response_edit_similarity": response_edit_sim,
            "latency_s": latency,
            "vram_used_mb": vram_used,
            "vram_total_mb": self._vram_total,
            "predicted": {
                "intent": result.intent,
                "operations": pred_ops_dicts,
                "response": result.response,
            },
            "expected": expected,
            "error": None,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NL2KG pipeline with comprehensive metrics"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to benchmark JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/results.json",
        help="Output results file",
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Disable embedding similarity computation",
    )
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text())

    rclpy.init()
    evaluator = Evaluator(
        dataset,
        use_embeddings=not args.no_embeddings,
    )

    try:
        results = evaluator.run()
    finally:
        evaluator.destroy_node()
        rclpy.shutdown()

    metrics = compute_metrics(results)

    output = {"metrics": metrics, "results": results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    # Print summary
    print("\n" + "=" * 60)
    print("  NL2KG Evaluation Results")
    print("=" * 60)
    print(f"  Samples:             {metrics['total_samples']}")
    print(f"  Valid responses:     {metrics['valid_responses']}")
    print(f"  JSON parse errors:   {metrics.get('json_parse_errors', 0)}")
    print()
    print(f"  JSON validity rate:  {metrics.get('json_validity_rate', 0):.3f}")
    print(f"  Schema validity:     {metrics.get('schema_validity_rate', 0):.3f}")
    print(f"  Intent accuracy:     {metrics.get('intent_accuracy', 0):.3f}")
    print(f"  Mean ops precision:  {metrics.get('mean_ops_precision', 0):.3f}")
    print(f"  Mean ops recall:     {metrics.get('mean_ops_recall', 0):.3f}")
    print(f"  Mean ops F1:         {metrics.get('mean_ops_f1', 0):.3f}")
    print(f"  Exact match rate:    {metrics.get('exact_match_rate', 0):.3f}")

    if metrics.get("mean_embedding_similarity") is not None:
        print(f"  Mean embed. sim.:    {metrics['mean_embedding_similarity']:.3f}")

    if metrics.get("mean_response_edit_similarity") is not None:
        print(f"  Mean resp. edit sim: {metrics['mean_response_edit_similarity']:.3f}")

    # Timing
    timing = metrics.get("timing", {})
    if timing:
        print()
        print("  Timing:")
        print(f"    Mean latency:      {timing['mean_latency_s']:.2f} s")
        print(f"    Median latency:    {timing['median_latency_s']:.2f} s")
        print(f"    P95 latency:       {timing['p95_latency_s']:.2f} s")
        print(f"    P99 latency:       {timing['p99_latency_s']:.2f} s")
        print(f"    Min latency:       {timing['min_latency_s']:.2f} s")
        print(f"    Max latency:       {timing['max_latency_s']:.2f} s")
        print(f"    Total time:        {timing['total_time_s']:.1f} s")

    # VRAM
    vram = metrics.get("vram", {})
    if vram:
        print()
        print("  VRAM:")
        if vram.get("total_mb") is not None:
            print(f"    Total capacity:    {vram['total_mb']:.0f} MB")
        print(f"    Mean used:         {vram.get('mean_used_mb', 0):.0f} MB")
        print(f"    Max used:          {vram.get('max_used_mb', 0):.0f} MB")

    # Per-category
    print()
    print("  Per-category breakdown:")
    print(
        f"  {'Category':<20s}  {'N':>4s}  {'Intent':>7s}  "
        f"{'F1':>6s}  {'EM':>6s}  {'Latency':>8s}"
    )
    print("  " + "-" * 58)
    for cat, cm in metrics.get("per_category", {}).items():
        lat_str = f"{cm['mean_latency_s']:.2f}s" if "mean_latency_s" in cm else "N/A"
        print(
            f"  {cat:<20s}  {cm['count']:>4d}  {cm['intent_accuracy']:>7.2f}  "
            f"{cm['mean_ops_f1']:>6.2f}  {cm['exact_match_rate']:>6.2f}  "
            f"{lat_str:>8s}"
        )

    # Per-op-type
    per_op = metrics.get("per_op_type", {})
    if per_op:
        print()
        print("  Per-operation-type breakdown:")
        print(
            f"  {'Op Type':<18s}  {'Prec':>6s}  {'Recall':>7s}  "
            f"{'F1':>6s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}"
        )
        print("  " + "-" * 56)
        for op_type, om in per_op.items():
            print(
                f"  {op_type:<18s}  {om['precision']:>6.2f}  "
                f"{om['recall']:>7.2f}  {om['f1']:>6.2f}  "
                f"{om['tp']:>5d}  {om['fp']:>5d}  {om['fn']:>5d}"
            )

    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
