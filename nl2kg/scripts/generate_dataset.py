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

"""Generate the KG-Dialogue-500 benchmark dataset.

The dataset covers 8 categories of NL ↔ KG interactions:
  1. Node creation (assert)
  2. Edge creation (assert)
  3. Property setting (modify)
  4. Node queries (query)
  5. Edge/relationship queries (query)
  6. Node removal (remove)
  7. Edge removal (remove)
  8. Multi-operation commands (assert/modify)

Each sample contains:
  - nl_input:   natural language utterance
  - expected:   ground truth KGResponse (intent + operations + reference answer)
  - category:   one of the 8 categories above
"""

import json
import random
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Domain vocabulary
# ---------------------------------------------------------------------------
ROBOT_NAMES = ["robot1", "robot2", "tiago", "pepper", "spot", "ur5"]
LOCATIONS = [
    "kitchen",
    "bedroom",
    "living_room",
    "corridor",
    "office",
    "garage",
    "lab",
    "warehouse",
]
OBJECT_NAMES = ["cup", "book", "box", "bottle", "tool", "plate", "charger", "tray"]
OBJECT_TYPES = ["cup", "book", "box", "bottle", "tool", "plate", "charger", "tray"]
PERSON_NAMES = ["alice", "bob", "carlos", "diana"]
EDGE_TYPES_LOC = ["at", "in", "near"]
EDGE_TYPES_REL = ["holds", "carries", "sees", "faces"]
PROPERTIES = {
    "battery": ["100", "80", "50", "20", "10"],
    "speed": ["0.5", "1.0", "1.5", "2.0"],
    "status": ["idle", "busy", "charging", "moving"],
    "color": ["red", "blue", "green", "white"],
    "weight": ["0.5", "1.2", "3.0", "5.0"],
}

# ---------------------------------------------------------------------------
# Template-based generation helpers
# ---------------------------------------------------------------------------


def _pick(lst):
    return random.choice(lst)


def _create_node_sample():
    """Category 1: node creation."""
    templates = [
        ("Add a robot called {name}.", "robot", "assert"),
        ("There is a {type} named {name}.", None, "assert"),
        ("Register a new {type} called {name} in the system.", None, "assert"),
        ("Create a node for {name}, it is a {type}.", None, "assert"),
        ("I want to add {name} as a {type} to the graph.", None, "assert"),
    ]
    tmpl, override_type, intent = _pick(templates)
    if override_type == "robot":
        name = _pick(ROBOT_NAMES)
        ntype = "robot"
    else:
        cat = random.choice(["object", "person", "location"])
        if cat == "object":
            name = _pick(OBJECT_NAMES)
            ntype = _pick(OBJECT_TYPES)
        elif cat == "person":
            name = _pick(PERSON_NAMES)
            ntype = "person"
        else:
            name = _pick(LOCATIONS)
            ntype = "location"

    nl = tmpl.format(name=name, type=ntype)
    ops = [{"op": "create_node", "name": name, "node_type": ntype}]
    return {
        "nl_input": nl,
        "expected": {
            "intent": intent,
            "operations": ops,
            "response": f"Created node '{name}' of type '{ntype}'.",
        },
        "category": "node_creation",
    }


def _create_edge_sample():
    """Category 2: edge creation."""
    templates = [
        "{robot} is {rel} the {loc}.",
        "Move {robot} to {loc}.",
        "{robot} is now {rel} {loc}.",
        "Place {robot} {rel} {loc}.",
        "The robot {robot} went to {loc}.",
    ]
    robot = _pick(ROBOT_NAMES)
    loc = _pick(LOCATIONS)
    rel = _pick(EDGE_TYPES_LOC)
    nl = _pick(templates).format(robot=robot, loc=loc, rel=rel)
    ops = [{"op": "create_edge", "edge_type": rel, "source": robot, "target": loc}]
    return {
        "nl_input": nl,
        "expected": {
            "intent": "assert",
            "operations": ops,
            "response": f"{robot} is now {rel} {loc}.",
        },
        "category": "edge_creation",
    }


def _set_property_sample():
    """Category 3: property setting."""
    templates = [
        "Set {robot}'s {key} to {value}.",
        "The {key} of {robot} is {value}.",
        "Update {robot}: {key} = {value}.",
        "Change the {key} of {robot} to {value}.",
        "{robot} now has {key} {value}.",
    ]
    robot = _pick(ROBOT_NAMES)
    key = _pick(list(PROPERTIES.keys()))
    value = _pick(PROPERTIES[key])
    nl = _pick(templates).format(robot=robot, key=key, value=value)
    ops = [{"op": "set_property", "name": robot, "key": key, "value": value}]
    return {
        "nl_input": nl,
        "expected": {
            "intent": "modify",
            "operations": ops,
            "response": f"Set {key} of {robot} to {value}.",
        },
        "category": "property_setting",
    }


def _query_node_sample():
    """Category 4: node queries."""
    templates = [
        "What robots are in the system?",
        "List all nodes.",
        "Is there a robot called {name}?",
        "What type is {name}?",
        "How many robots are there?",
        "Show me all objects in the graph.",
        "Tell me about {name}.",
    ]
    name = _pick(ROBOT_NAMES + OBJECT_NAMES)
    nl = _pick(templates).format(name=name)
    return {
        "nl_input": nl,
        "expected": {
            "intent": "query",
            "operations": [],
            "response": "(answer depends on current graph state)",
        },
        "category": "node_query",
    }


def _query_edge_sample():
    """Category 5: relationship queries."""
    templates = [
        "Where is {robot}?",
        "What is {robot} doing?",
        "What does {robot} hold?",
        "Which robots are in the {loc}?",
        "Is {robot} in the {loc}?",
        "What is connected to {name}?",
        "Show me the relationships of {robot}.",
    ]
    robot = _pick(ROBOT_NAMES)
    loc = _pick(LOCATIONS)
    name = _pick(ROBOT_NAMES + OBJECT_NAMES)
    nl = _pick(templates).format(robot=robot, loc=loc, name=name)
    return {
        "nl_input": nl,
        "expected": {
            "intent": "query",
            "operations": [],
            "response": "(answer depends on current graph state)",
        },
        "category": "edge_query",
    }


def _remove_node_sample():
    """Category 6: node removal."""
    templates = [
        "Remove {name} from the graph.",
        "Delete the node {name}.",
        "{name} is no longer in the system.",
        "Take {name} out of the knowledge graph.",
        "Get rid of {name}.",
    ]
    name = _pick(ROBOT_NAMES + OBJECT_NAMES)
    nl = _pick(templates).format(name=name)
    ops = [{"op": "remove_node", "name": name}]
    return {
        "nl_input": nl,
        "expected": {
            "intent": "remove",
            "operations": ops,
            "response": f"Removed node '{name}'.",
        },
        "category": "node_removal",
    }


def _remove_edge_sample():
    """Category 7: edge removal."""
    templates = [
        "{robot} is no longer at {loc}.",
        "Remove the connection between {robot} and {loc}.",
        "{robot} left {loc}.",
        "Delete the '{rel}' edge from {robot} to {loc}.",
        "{robot} is not {rel} {loc} anymore.",
    ]
    robot = _pick(ROBOT_NAMES)
    loc = _pick(LOCATIONS)
    rel = _pick(EDGE_TYPES_LOC)
    nl = _pick(templates).format(robot=robot, loc=loc, rel=rel)
    ops = [{"op": "remove_edge", "edge_type": rel, "source": robot, "target": loc}]
    return {
        "nl_input": nl,
        "expected": {
            "intent": "remove",
            "operations": ops,
            "response": f"Removed edge '{rel}' from {robot} to {loc}.",
        },
        "category": "edge_removal",
    }


def _multi_op_sample():
    """Category 8: multi-operation commands."""
    templates = [
        "Add {robot} in the {loc} with battery {bat}.",
        "Create {robot}, place it at {loc}, and set its status to {status}.",
        "{robot} is a robot at {loc} carrying a {obj}.",
        "Register {robot} at {loc} with speed {speed}.",
    ]
    robot = _pick(ROBOT_NAMES)
    loc = _pick(LOCATIONS)
    obj = _pick(OBJECT_NAMES)
    bat = _pick(PROPERTIES["battery"])
    status = _pick(PROPERTIES["status"])
    speed = _pick(PROPERTIES["speed"])

    idx = random.randint(0, len(templates) - 1)
    nl = templates[idx].format(
        robot=robot, loc=loc, bat=bat, status=status, obj=obj, speed=speed
    )

    ops = [{"op": "create_node", "name": robot, "node_type": "robot"}]

    if idx == 0:
        ops += [
            {"op": "create_edge", "edge_type": "at", "source": robot, "target": loc},
            {"op": "set_property", "name": robot, "key": "battery", "value": bat},
        ]
    elif idx == 1:
        ops += [
            {"op": "create_edge", "edge_type": "at", "source": robot, "target": loc},
            {"op": "set_property", "name": robot, "key": "status", "value": status},
        ]
    elif idx == 2:
        ops += [
            {"op": "create_edge", "edge_type": "at", "source": robot, "target": loc},
            {"op": "create_node", "name": obj, "node_type": obj},
            {"op": "create_edge", "edge_type": "carries", "source": robot, "target": obj},
        ]
    else:
        ops += [
            {"op": "create_edge", "edge_type": "at", "source": robot, "target": loc},
            {"op": "set_property", "name": robot, "key": "speed", "value": speed},
        ]

    return {
        "nl_input": nl,
        "expected": {
            "intent": "assert",
            "operations": ops,
            "response": f"Done. {robot} registered with the given configuration.",
        },
        "category": "multi_operation",
    }


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

GENERATORS = [
    (_create_node_sample, 30),
    (_create_edge_sample, 30),
    (_set_property_sample, 25),
    (_query_node_sample, 25),
    (_query_edge_sample, 25),
    (_remove_node_sample, 20),
    (_remove_edge_sample, 20),
    (_multi_op_sample, 25),
]


def generate_dataset(n: int = 500, seed: int = 42) -> list[dict]:
    random.seed(seed)
    dataset: list[dict] = []

    # Proportional generation
    total_weight = sum(w for _, w in GENERATORS)
    for gen_fn, weight in GENERATORS:
        count = max(1, round(n * weight / total_weight))
        for _ in range(count):
            dataset.append(gen_fn())

    # Trim or pad to exactly n
    random.shuffle(dataset)
    dataset = dataset[:n]
    while len(dataset) < n:
        gen_fn, _ = random.choice(GENERATORS)
        dataset.append(gen_fn())

    # Add IDs
    for i, sample in enumerate(dataset, 1):
        sample["id"] = i

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate KG-Dialogue-500 dataset")
    parser.add_argument("-n", type=int, default=500, help="Number of samples")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="kg_dialogue_200.json",
        help="Output file path",
    )
    args = parser.parse_args()

    dataset = generate_dataset(n=args.n, seed=args.seed)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))

    # Print category distribution
    from collections import Counter

    cats = Counter(s["category"] for s in dataset)
    print(f"Generated {len(dataset)} samples → {out_path}")
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()
