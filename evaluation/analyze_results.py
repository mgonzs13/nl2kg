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

"""Analyze and compare results across multiple NL2KG evaluation experiments.

Reads all result JSON files from a directory and produces:
  - A comparative summary table (models × grammar modes)
  - Per-category heatmaps
  - Timing and VRAM comparisons
  - CSV export for further analysis

Usage:
    python3 analyze_results.py --results-dir results/
    python3 analyze_results.py --results-dir results/ --csv results/summary.csv
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict


def load_results(results_dir: str) -> dict[str, dict]:
    """Load all result JSON files from a directory.

    Returns a dict mapping experiment name → full result data.
    """
    results = {}
    results_path = Path(results_dir)

    for fpath in sorted(results_path.glob("results-*.json")):
        # Extract experiment name from filename
        # e.g., results-Qwen-3.5-4B-grammar.json → Qwen-3.5-4B-grammar
        name = fpath.stem.replace("results-", "")
        try:
            data = json.loads(fpath.read_text())
            results[name] = data
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: Could not load {fpath}: {e}")

    return results


def extract_summary_row(name: str, data: dict) -> dict:
    """Extract a summary row from a single experiment's results."""
    m = data.get("metrics", {})
    timing = m.get("timing", {})
    vram = m.get("vram", {})

    # Determine model and grammar mode from name
    if name.endswith("-grammar"):
        model = name[: -len("-grammar")]
        grammar = "yes"
    elif name.endswith("-no_grammar"):
        model = name[: -len("-no_grammar")]
        grammar = "no"
    else:
        model = name
        grammar = "unknown"

    return {
        "experiment": name,
        "model": model,
        "grammar": grammar,
        "total_samples": m.get("total_samples", 0),
        "valid_responses": m.get("valid_responses", 0),
        "timeout_errors": m.get("timeout_errors", 0),
        "json_parse_errors": m.get("json_parse_errors", 0),
        "json_validity_rate": m.get("json_validity_rate", 0),
        "schema_validity_rate": m.get("schema_validity_rate", 0),
        "intent_accuracy": m.get("intent_accuracy", 0),
        "mean_ops_precision": m.get("mean_ops_precision", 0),
        "mean_ops_recall": m.get("mean_ops_recall", 0),
        "mean_ops_f1": m.get("mean_ops_f1", 0),
        "exact_match_rate": m.get("exact_match_rate", 0),
        "mean_embedding_similarity": m.get("mean_embedding_similarity"),
        "mean_response_edit_similarity": m.get("mean_response_edit_similarity"),
        "mean_latency_s": timing.get("mean_latency_s"),
        "median_latency_s": timing.get("median_latency_s"),
        "p95_latency_s": timing.get("p95_latency_s"),
        "total_time_s": timing.get("total_time_s"),
        "vram_mean_mb": vram.get("mean_used_mb"),
        "vram_max_mb": vram.get("max_used_mb"),
        "vram_total_mb": vram.get("total_mb"),
    }


def print_comparison_table(rows: list[dict]) -> None:
    """Print a formatted comparison table of all experiments."""

    # Sort by grammar (yes first), then by exact_match_rate descending
    rows_sorted = sorted(
        rows, key=lambda r: (r["grammar"] != "yes", -r["exact_match_rate"])
    )

    print("\n" + "=" * 120)
    print("  COMPARATIVE RESULTS — ALL EXPERIMENTS")
    print("=" * 120)

    # Header
    header = (
        f"{'Model':<25s} {'GBNF':>4s} │ "
        f"{'JSON%':>5s} {'Schema%':>7s} {'Intent':>6s} {'Prec':>5s} "
        f"{'Rec':>5s} {'F1':>5s} {'EM':>5s} │ "
        f"{'EmbSim':>6s} {'Lat(s)':>6s} {'VRAM':>6s}"
    )
    print(header)
    print("─" * 120)

    for r in rows_sorted:
        emb = (
            f"{r['mean_embedding_similarity']:.3f}"
            if r["mean_embedding_similarity"] is not None
            else "N/A"
        )
        lat = f"{r['mean_latency_s']:.2f}" if r["mean_latency_s"] is not None else "N/A"
        vram = f"{r['vram_mean_mb']:.0f}" if r["vram_mean_mb"] is not None else "N/A"

        print(
            f"{r['model']:<25s} {r['grammar']:>4s} │ "
            f"{r['json_validity_rate']:>5.3f} {r['schema_validity_rate']:>7.3f} "
            f"{r['intent_accuracy']:>6.3f} {r['mean_ops_precision']:>5.3f} "
            f"{r['mean_ops_recall']:>5.3f} {r['mean_ops_f1']:>5.3f} "
            f"{r['exact_match_rate']:>5.3f} │ "
            f"{emb:>6s} {lat:>6s} {vram:>6s}"
        )

    print("─" * 120)


def print_grammar_ablation(rows: list[dict]) -> None:
    """Print a grammar vs. no-grammar comparison for each model."""

    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_model[r["model"]][r["grammar"]] = r

    print("\n" + "=" * 100)
    print("  GRAMMAR ABLATION — GBNF vs. Unconstrained")
    print("=" * 100)

    header = (
        f"{'Model':<25s} │ "
        f"{'Δ JSON%':>7s} {'Δ Schema':>8s} {'Δ Intent':>8s} "
        f"{'Δ F1':>6s} {'Δ EM':>6s} │ "
        f"{'Δ Lat(s)':>8s}"
    )
    print(header)
    print("─" * 100)

    for model, modes in sorted(by_model.items()):
        if "yes" not in modes or "no" not in modes:
            continue

        g = modes["yes"]
        n = modes["no"]

        def _delta(key):
            gv = g.get(key)
            nv = n.get(key)
            if gv is None or nv is None:
                return "N/A"
            diff = gv - nv
            sign = "+" if diff >= 0 else ""
            return f"{sign}{diff:.3f}"

        def _delta_inv(key):
            """For metrics where lower is better (latency)."""
            gv = g.get(key)
            nv = n.get(key)
            if gv is None or nv is None:
                return "N/A"
            diff = gv - nv
            sign = "+" if diff >= 0 else ""
            return f"{sign}{diff:.2f}"

        print(
            f"{model:<25s} │ "
            f"{_delta('json_validity_rate'):>7s} "
            f"{_delta('schema_validity_rate'):>8s} "
            f"{_delta('intent_accuracy'):>8s} "
            f"{_delta('mean_ops_f1'):>6s} "
            f"{_delta('exact_match_rate'):>6s} │ "
            f"{_delta_inv('mean_latency_s'):>8s}"
        )

    print("─" * 100)
    print("  Positive Δ = grammar is better; negative Δ = unconstrained is better")


def print_per_category(rows: list[dict], all_results: dict[str, dict]) -> None:
    """Print per-category breakdown for each experiment."""

    print("\n" + "=" * 100)
    print("  PER-CATEGORY F1 SCORES")
    print("=" * 100)

    # Collect all categories
    all_cats = set()
    for data in all_results.values():
        per_cat = data.get("metrics", {}).get("per_category", {})
        all_cats.update(per_cat.keys())
    cats = sorted(all_cats)

    # Header
    header = f"{'Experiment':<35s} │ " + " ".join(f"{c[:12]:>12s}" for c in cats)
    print(header)
    print("─" * (37 + 13 * len(cats)))

    for r in sorted(rows, key=lambda x: (x["grammar"] != "yes", x["model"])):
        name = r["experiment"]
        data = all_results.get(name, {})
        per_cat = data.get("metrics", {}).get("per_category", {})

        vals = []
        for c in cats:
            cm = per_cat.get(c, {})
            f1 = cm.get("mean_ops_f1")
            vals.append(f"{f1:.3f}" if f1 is not None else "N/A")

        print(f"{name:<35s} │ " + " ".join(f"{v:>12s}" for v in vals))

    print("─" * (37 + 13 * len(cats)))


def print_per_op_type(all_results: dict[str, dict]) -> None:
    """Print per-operation-type F1 for each experiment."""

    print("\n" + "=" * 100)
    print("  PER-OPERATION-TYPE F1 SCORES")
    print("=" * 100)

    # Collect all op types
    all_ops = set()
    for data in all_results.values():
        per_op = data.get("metrics", {}).get("per_op_type", {})
        all_ops.update(per_op.keys())
    ops = sorted(all_ops)

    if not ops:
        print("  No per-operation-type data available.")
        return

    header = f"{'Experiment':<35s} │ " + " ".join(f"{o[:14]:>14s}" for o in ops)
    print(header)
    print("─" * (37 + 15 * len(ops)))

    for name in sorted(all_results.keys()):
        data = all_results[name]
        per_op = data.get("metrics", {}).get("per_op_type", {})

        vals = []
        for o in ops:
            om = per_op.get(o, {})
            f1 = om.get("f1")
            vals.append(f"{f1:.3f}" if f1 is not None else "N/A")

        print(f"{name:<35s} │ " + " ".join(f"{v:>14s}" for v in vals))

    print("─" * (37 + 15 * len(ops)))


def export_csv(rows: list[dict], output_path: str) -> None:
    """Export summary rows to CSV."""
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and compare NL2KG evaluation results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV output path for summary table",
    )
    args = parser.parse_args()

    # Load all results
    all_results = load_results(args.results_dir)

    if not all_results:
        print(f"No result files found in: {args.results_dir}")
        print("Expected files like: results-ModelName-grammar.json")
        return

    print(f"Loaded {len(all_results)} experiment(s) from: {args.results_dir}")

    # Build summary rows
    rows = [extract_summary_row(name, data) for name, data in all_results.items()]

    # Print tables
    print_comparison_table(rows)
    print_grammar_ablation(rows)
    print_per_category(rows, all_results)
    print_per_op_type(all_results)

    # Export CSV
    if args.csv:
        export_csv(rows, args.csv)


if __name__ == "__main__":
    main()
