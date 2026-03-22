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

Runs each sample through the NL2KG node (via ROS 2 action),
compares structural output against ground truth, and computes metrics:
  - Intent accuracy
  - Operation F1 (per-op exact match)
  - Operation-type accuracy
  - Overall exact match
  - Per-category breakdown

Usage:
    python3 scripts/evaluate.py --dataset kg_dialogue_200.json --output results.json
"""

import json
import argparse
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nl2kg_msgs.action import NL2KG
from nl2kg.metrics import normalize_ops, compute_f1, compute_metrics


class Evaluator(Node):

    def __init__(self, samples: list[dict], timeout: float = 30.0) -> None:
        super().__init__("nl2kg_evaluator")
        self.samples = samples
        self.timeout = timeout

        self._action_client = ActionClient(self, NL2KG, "/nl2kg_node/nl2kg")
        self._results: list[dict] = []

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
        return self._results

    def _evaluate_one(self, sample: dict) -> dict:
        goal = NL2KG.Goal()
        goal.input_text = sample["nl_input"]

        send_future = self._action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=self.timeout)

        if not send_future.done() or send_future.result() is None:
            return self._timeout_result(sample)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            return self._timeout_result(sample, error="goal_rejected")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=self.timeout)

        if not result_future.done() or result_future.result() is None:
            return self._timeout_result(sample)

        action_result = result_future.result().result
        return self._evaluate_result(sample, action_result)

    @staticmethod
    def _timeout_result(sample: dict, error: str = "timeout") -> dict:
        return {
            "id": sample.get("id"),
            "category": sample["category"],
            "intent_correct": False,
            "ops_f1": 0.0,
            "exact_match": False,
            "predicted": None,
            "error": error,
        }

    @staticmethod
    def _evaluate_result(sample: dict, result: NL2KG.Result) -> dict:
        expected = sample["expected"]

        # Intent accuracy
        intent_correct = result.intent == expected["intent"]

        # Convert action result operations to dicts for comparison
        pred_ops = normalize_ops(
            [
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
        )
        gold_ops = normalize_ops(expected.get("operations", []))
        ops_f1 = compute_f1(pred_ops, gold_ops)

        exact = intent_correct and ops_f1 == 1.0

        return {
            "id": sample.get("id"),
            "category": sample["category"],
            "nl_input": sample["nl_input"],
            "intent_correct": intent_correct,
            "ops_f1": ops_f1,
            "exact_match": exact,
            "predicted": {
                "intent": result.intent,
                "operations": [
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
                ],
                "response": result.response,
            },
            "expected": expected,
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NL2KG pipeline")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to benchmark JSON"
    )
    parser.add_argument(
        "--output", type=str, default="results.json", help="Output results file"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Per-sample timeout (s)"
    )
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text())

    rclpy.init()
    evaluator = Evaluator(dataset, timeout=args.timeout)

    try:
        results = evaluator.run()
    finally:
        evaluator.destroy_node()
        rclpy.shutdown()

    metrics = compute_metrics(results)

    output = {"metrics": metrics, "results": results}
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False))

    # Print summary
    print("\n=== NL2KG Evaluation Results ===")
    print(f"  Samples:          {metrics['total_samples']}")
    print(f"  Valid responses:  {metrics['valid_responses']}")
    print(f"  Intent accuracy:  {metrics['intent_accuracy']:.3f}")
    print(f"  Mean ops F1:      {metrics['mean_ops_f1']:.3f}")
    print(f"  Exact match rate: {metrics['exact_match_rate']:.3f}")
    print("\nPer-category:")
    for cat, cm in metrics.get("per_category", {}).items():
        print(
            f"  {cat:20s}  n={cm['count']:3d}  "
            f"intent={cm['intent_accuracy']:.2f}  "
            f"F1={cm['mean_ops_f1']:.2f}  "
            f"EM={cm['exact_match_rate']:.2f}"
        )
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
