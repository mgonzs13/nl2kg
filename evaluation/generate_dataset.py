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

This generator produces natural, human-sounding utterances that a person
would say to a robot assistant in everyday conversation. Sentences are
deliberately free of graph/node/edge terminology — they read like natural
speech and the robot must infer the underlying KG operations from context.

Eight categories are covered:
  1. Entity introduction  (assert)   — telling the robot about a new thing/person
  2. Location update      (assert)   — saying where something currently is
  3. State/property update(modify)   — informing the robot of a changed attribute
  4. Entity queries       (query)    — asking what exists or what something is
  5. Location/relation queries(query)— asking where things are or what they do
  6. Entity forgetting    (remove)   — telling the robot to forget something
  7. Relation forgetting  (remove)   — telling the robot a location link is gone
  8. Compound statements  (assert)   — rich sentences conveying multiple facts

Each sample contains:
  - nl_input:   natural language utterance (human-to-robot speech)
  - expected:   ground truth KGResponse (intent + operations + reference answer)
  - category:   one of the 8 categories above
  - setup_operations: (optional) list of KG operations to execute BEFORE
    the sample so that the graph is in the correct state.  For example a
    "set_property" sample first creates the target node, and a
    "remove_edge" sample first creates the source/target nodes and the
    edge.  Samples that operate on an empty graph omit this field.
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
# Helpers
# ---------------------------------------------------------------------------


def _pick(lst):
    return random.choice(lst)


def _article(word: str) -> str:
    """Return 'a' or 'an' depending on the first letter."""
    return "an" if word[0].lower() in "aeiou" else "a"


def _capitalize_name(name: str) -> str:
    """Capitalize a name appropriately for proper nouns and common nouns.

    - Person names (alice, bob, etc.) → Alice, Bob (capitalize first letter)
    - Robot names (robot1, tiago, etc.) → Robot1, Tiago (capitalize first letter)
    - Locations (kitchen, living_room, etc.) → Kitchen, Living_room (capitalize first letter)
    - Objects (cup, book, etc.) → Cup, Book (capitalize first letter)
    """
    if not name:
        return name
    return name[0].upper() + name[1:]


# ---------------------------------------------------------------------------
# Category 1: Entity introduction  (assert → create_node)
# People naturally announce or mention a new thing/person to the robot.
# No mention of "nodes", "graph", "register", "create", etc.
# ---------------------------------------------------------------------------


def _create_node_sample():
    templates_robot = [
        "Just so you know, {name} is one of our robots.",
        "We have a robot called {name} now.",
        "Meet {name} — it's the new robot on the team.",
        "There's a robot named {name} working with us.",
        "I wanted to let you know about {name}. It's a robot.",
        "{name} has just been deployed. It's a robot.",
        "Hey, {name} joined the fleet today. It's a robot.",
        "You should be aware that {name} is a robot operating here.",
        "We're bringing {name} online. It's a robot.",
        "I'd like you to remember {name} — it's a robot in our system.",
        "Please add {name} to the system. It's a robot.",
        "Please remember that {name} is one of our robots.",
        "A robot called {name} is now part of the team.",
        "We've got {name} on the team as a robot.",
        "I'm telling you about {name}. It operates as a robot.",
    ]

    templates_object = [
        "There's {art} {type} called {name} around.",
        "We have {art} {type} called {name}.",
        "{name} is {art} {type} we're tracking.",
        "I spotted {art} {type} called {name}.",
        "Heads up — there's {art} {type} called {name}.",
        "You should know about this {type}. It's called {name}.",
        "I wanted to mention the {type} called {name}.",
        "We have something called {name} — it's {art} {type}.",
        "{name} is {art} {type} we're keeping track of.",
        "I'd like you to remember {name}. It's {art} {type}.",
        "Please note that {name} is {art} {type}.",
        "{name} is {art} {type} that we need to track.",
        "Just so you know, {name} is {art} {type}.",
        "We've got {art} {type} named {name}.",
        "There's {art} {type} here. Its name is {name}.",
    ]

    templates_person = [
        "{name} is part of the team.",
        "I want you to know about {name}. They're a person on the team.",
        "{name} just joined us. They are a person.",
        "Meet {name} — they're a person on our team.",
        "You should know {name} — they're a member of the staff.",
        "Meet {name}. They're a person we work with.",
        "{name} has been assigned to the team.",
        "I'd like you to remember {name}. They're a person.",
        "{name} will be around. They're a person you should know.",
        "A person called {name} joined the team.",
    ]

    templates_location = [
        "The {name} is one of the places you need to know about.",
        "Just so you know, there's a {name} in this building.",
        "Please add the {name} to your map.",
        "We have a location called {name}.",
        "We have a {name} in this building.",
        "The {name} is a location you should remember.",
        "I'd like you to remember the {name}.",
        "There's a room called {name} in this facility.",
        "A place called {name} exists in this building.",
        "Note that there's a {name} you should know about.",
    ]

    cat = random.choice(["robot", "object", "person", "location"])

    if cat == "robot":
        name = _capitalize_name(_pick(ROBOT_NAMES))
        ntype = "robot"
        nl = _pick(templates_robot).format(name=name)
    elif cat == "object":
        ntype = _pick(OBJECT_TYPES)
        name = _capitalize_name(ntype)  # object name matches its type to avoid confusion
        art = _article(ntype)
        nl = _pick(templates_object).format(name=name, type=ntype, art=art)
    elif cat == "person":
        name = _capitalize_name(_pick(PERSON_NAMES))
        ntype = "person"
        nl = _pick(templates_person).format(name=name)
    else:
        name = _capitalize_name(_pick(LOCATIONS))
        ntype = "location"
        nl = _pick(templates_location).format(name=name)

    ops = [{"op": "create_node", "name": name, "node_type": ntype}]
    return {
        "nl_input": nl,
        "expected": {
            "intent": "assert",
            "operations": ops,
            "response": f"Got it, I'll remember that {name} is a {ntype}.",
        },
        "category": "node_creation",
    }


# ---------------------------------------------------------------------------
# Category 2: Location update  (assert → create_edge)
# People describe where something currently is, without mentioning edges.
# ---------------------------------------------------------------------------


def _create_edge_sample():
    templates_explicit_prep = [
        "{robot} is {prep} the {loc}.",
        "{robot} is currently {prep} the {loc}.",
        "I can see {robot} {prep} the {loc}.",
        "It looks like {robot} is {prep} the {loc} right now.",
        "{robot} ended up {prep} the {loc}.",
        "I just spotted {robot} {prep} the {loc}.",
        "{robot} has been stationed {prep} the {loc}.",
        "As far as I can tell, {robot} is {prep} the {loc}.",
        "{robot} seems to be {prep} the {loc} at the moment.",
        "From what I can see, {robot} is {prep} the {loc}.",
        "I noticed {robot} {prep} the {loc} just now.",
        "{robot} is {prep} the {loc}, just to let you know.",
    ]

    templates_movement = [
        "{robot} went to the {loc} a moment ago.",
        "{robot} just arrived at the {loc}.",
        "{robot} has reached the {loc}.",
        "{robot} went to the {loc} earlier.",
        "{robot} made its way to the {loc}.",
        "{robot} moved to the {loc} recently.",
        "{robot} headed over to the {loc}.",
        "{robot} is now over at the {loc}.",
        "I sent {robot} to the {loc}.",
        "We moved {robot} to the {loc}.",
        "I put {robot} at the {loc}.",
        "{robot} was deployed to the {loc}.",
        "{robot} ended up at the {loc}.",
    ]

    robot = _capitalize_name(_pick(ROBOT_NAMES))
    loc = _capitalize_name(_pick(LOCATIONS))

    if random.random() < 0.6:
        prep = _pick(EDGE_TYPES_LOC)
        nl = _pick(templates_explicit_prep).format(robot=robot, loc=loc, prep=prep)
        rel = prep
    else:
        rel = "at"
        nl = _pick(templates_movement).format(robot=robot, loc=loc)

    # Ground truth includes the create_node ops because the graph is always
    # cleared before each sample, so the model will (correctly) create the
    # robot and location nodes before linking them.
    ops = [
        {"op": "create_node", "name": robot, "node_type": "robot"},
        {"op": "create_node", "name": loc, "node_type": "location"},
        {"op": "create_edge", "edge_type": rel, "source": robot, "target": loc},
    ]
    return {
        "nl_input": nl,
        "expected": {
            "intent": "assert",
            "operations": ops,
            "response": f"Understood, {robot} is now {rel} the {loc}.",
        },
        "category": "edge_creation",
    }


# ---------------------------------------------------------------------------
# Category 3: State / property update  (modify → set_property)
# People report a changed attribute of a robot — no "property" mention.
# ---------------------------------------------------------------------------


def _set_property_sample():
    templates = {
        "battery": [
            "{robot}'s battery is at {value} percent.",
            "{robot} has {value} percent battery left.",
            "The battery on {robot} is now {value}.",
            "{robot} just finished charging — it's at {value} now.",
            "I checked {robot}'s battery. It's {value}.",
            "{robot} is showing {value} percent on the battery indicator.",
            "Battery level for {robot}: {value}.",
            "{robot} dropped to {value} percent battery.",
            "The battery life on {robot} is currently {value}.",
            "{robot}'s power is at {value} percent right now.",
        ],
        "speed": [
            "{robot} is moving at {value} metres per second.",
            "{robot}'s speed has changed to {value}.",
            "I've set {robot}'s speed to {value}.",
            "{robot} is now travelling at {value} m/s.",
            "The speed of {robot} is {value} at the moment.",
            "{robot} slowed down — it's going at {value} now.",
            "{robot} sped up to {value}.",
            "Can you make {robot} go at {value} metres per second?",
            "{robot} should be running at {value} m/s.",
            "Update {robot}'s speed to {value}.",
        ],
        "status": [
            "{robot} is {value} at the moment.",
            "{robot} went {value} just now.",
            "I think {robot} is {value}.",
            "{robot} is currently {value}.",
            "{robot} became {value} a moment ago.",
            "It looks like {robot} is {value}.",
            "{robot}'s current state is {value}.",
            "{robot} seems to be {value} right now.",
            "{robot} switched to {value}.",
            "Just so you know, {robot} is {value}.",
        ],
        "color": [
            "{robot}'s color is {value}.",
            "The color of {robot} is {value}.",
            "{robot} has been painted {value}.",
            "{robot} has a {value} color.",
            "I've set {robot}'s color to {value}.",
            "Update {robot}'s color to {value}.",
            "{robot}'s color has changed to {value}.",
            "They painted {robot} {value}.",
            "Please note that {robot}'s color is {value}.",
            "{robot} is now {value} in color.",
        ],
        "weight": [
            "{robot} weighs {value} kilograms.",
            "The weight of {robot} is {value} kg.",
            "{robot} is {value} kg.",
            "We measured {robot} — it comes in at {value} kg.",
            "{robot} has a mass of {value} kilograms.",
            "{robot}'s weight is about {value} kg.",
            "I'd say {robot} is roughly {value} kilos.",
            "{robot} tips the scales at {value} kg.",
            "They told me {robot} is {value} kilograms.",
            "{robot} is listed at {value} kg.",
        ],
    }

    robot = _capitalize_name(_pick(ROBOT_NAMES))
    key = _pick(list(PROPERTIES.keys()))
    value = _pick(PROPERTIES[key])
    nl = _pick(templates[key]).format(robot=robot, value=value)
    ops = [{"op": "set_property", "name": robot, "key": key, "value": value}]
    return {
        "nl_input": nl,
        "expected": {
            "intent": "modify",
            "operations": ops,
            "response": f"Got it, I've updated {robot}'s {key} to {value}.",
        },
        "category": "property_setting",
        "setup_operations": [
            {"op": "create_node", "name": robot, "node_type": "robot"},
        ],
    }


# ---------------------------------------------------------------------------
# Category 4: Entity queries  (query)
# People ask what the robot knows about the things in the environment.
# ---------------------------------------------------------------------------


def _query_node_sample():
    # --- Named-entity questions (the entity must exist) ---
    templates_named = [
        "Do you know about {name}?",
        "What is {name}?",
        "Can you tell me about {name}?",
        "What do you know about {name}?",
        "Have you seen {name} around?",
        "Is {name} one of the robots?",
        "Who or what is {name}?",
        "Do you recognise {name}?",
        "Is {name} something you know?",
        "Is {name} part of our team?",
        "Tell me what you know about {name}.",
        "What kind of thing is {name}?",
        "Do you have any information on {name}?",
        "Is there a robot called {name} around here?",
        "Any idea who {name} is?",
    ]

    # --- Open-ended questions (no specific entity required) ---
    templates_open = [
        "What robots are you aware of?",
        "Can you list the robots you know?",
        "Which robots are currently active?",
        "How many robots do we have?",
        "What objects do you know about?",
        "Who is on the team right now?",
        "Which people do you know about?",
        "Do you know about anything around here?",
    ]

    if random.random() < 0.65:
        # Named query — set up the entity so the graph is not empty
        cat = random.choice(["robot", "object", "person"])
        if cat == "robot":
            name = _capitalize_name(_pick(ROBOT_NAMES))
            ntype = "robot"
        elif cat == "object":
            name = _capitalize_name(_pick(OBJECT_NAMES))
            ntype = _pick(OBJECT_TYPES)
        else:
            name = _capitalize_name(_pick(PERSON_NAMES))
            ntype = "person"
        nl = _pick(templates_named).format(name=name)
        setup = [{"op": "create_node", "name": name, "node_type": ntype}]
    else:
        # Open-ended query — optionally populate a couple of entities
        nl = _pick(templates_open)
        setup = [
            {
                "op": "create_node",
                "name": _capitalize_name(_pick(ROBOT_NAMES)),
                "node_type": "robot",
            },
            {
                "op": "create_node",
                "name": _capitalize_name(_pick(OBJECT_NAMES)),
                "node_type": _pick(OBJECT_TYPES),
            },
        ]

    return {
        "nl_input": nl,
        "expected": {
            "intent": "query",
            "operations": [],
            "response": "(answer depends on current graph state)",
        },
        "category": "node_query",
        "setup_operations": setup,
    }


# ---------------------------------------------------------------------------
# Category 5: Location / relation queries  (query)
# People ask where things are or what a robot is doing.
# ---------------------------------------------------------------------------


def _query_edge_sample():
    templates = [
        "Where is {robot}?",
        "Where did {robot} end up?",
        "Do you know where {robot} is?",
        "Can you tell me {robot}'s location?",
        "Which room is {robot} in?",
        "Is {robot} in the {loc}?",
        "Has {robot} been to the {loc}?",
        "What is {robot} up to right now?",
        "What is {robot} carrying?",
        "Is {robot} holding anything?",
        "What's {robot} doing at the moment?",
        "Is anyone in the {loc}?",
        "Who's in the {loc} right now?",
        "Are there any robots in the {loc}?",
        "What can you tell me about {robot}'s whereabouts?",
        "I wonder where {robot} is.",
        "Has {robot} moved recently?",
        "Is {robot} still near the {loc}?",
        "What's going on with {robot}?",
        "Is {robot} near {name}?",
        "Do you know if {robot} is near the {loc}?",
        "Which direction is {robot} facing?",
        "Can you see {robot} from here?",
    ]
    robot = _capitalize_name(_pick(ROBOT_NAMES))
    loc = _capitalize_name(_pick(LOCATIONS))
    name = _capitalize_name(_pick(ROBOT_NAMES + OBJECT_NAMES))
    rel = _pick(EDGE_TYPES_LOC)
    nl = _pick(templates).format(robot=robot, loc=loc, name=name)
    return {
        "nl_input": nl,
        "expected": {
            "intent": "query",
            "operations": [],
            "response": "(answer depends on current graph state)",
        },
        "category": "edge_query",
        "setup_operations": [
            {"op": "create_node", "name": robot, "node_type": "robot"},
            {"op": "create_node", "name": loc, "node_type": "location"},
            {"op": "create_edge", "edge_type": rel, "source": robot, "target": loc},
        ],
    }


# ---------------------------------------------------------------------------
# Category 6: Entity forgetting  (remove → remove_node)
# People tell the robot to forget about something — no "node" mention.
# ---------------------------------------------------------------------------


def _remove_node_sample():
    templates = [
        "{name} is gone. You can forget about it.",
        "{name} isn't here anymore. Please forget it.",
        "We no longer have {name}. Stop tracking it.",
        "Forget about {name} — it's not around anymore.",
        "{name} left the team. You don't need to remember it.",
        "{name} is no longer relevant. Please drop it.",
        "{name} has been removed from the environment.",
        "We got rid of {name}. Please update your knowledge.",
        "{name} isn't part of this anymore. Forget it.",
        "I need you to forget everything about {name}.",
        "{name} doesn't exist here anymore.",
        "Please remove {name} from the knowledge graph.",
        "Please stop keeping track of {name}.",
        "{name} was taken away. Update accordingly.",
        "Drop {name} from your records.",
        "Erase {name} from the system, please.",
        "Please delete {name} from the graph.",
        "{name} should be removed. Forget about it.",
    ]
    name = _capitalize_name(_pick(ROBOT_NAMES + OBJECT_NAMES + PERSON_NAMES))
    nl = _pick(templates).format(name=name)
    ops = [{"op": "remove_node", "name": name}]

    # Determine the type for the setup node
    if name in ROBOT_NAMES:
        ntype = "robot"
    elif name in PERSON_NAMES:
        ntype = "person"
    else:
        ntype = name  # objects use their own name as type

    return {
        "nl_input": nl,
        "expected": {
            "intent": "remove",
            "operations": ops,
            "response": f"Understood, I'll forget about {name}.",
        },
        "category": "node_removal",
        "setup_operations": [
            {"op": "create_node", "name": name, "node_type": ntype},
        ],
    }


# ---------------------------------------------------------------------------
# Category 7: Relation forgetting  (remove → remove_edge)
# People say a location association no longer holds — no "edge" mention.
# ---------------------------------------------------------------------------


def _remove_edge_sample():
    templates_explicit_prep = [
        "{robot} is no longer {prep} the {loc}.",
        "{robot} isn't {prep} the {loc} anymore.",
        "{robot} moved away from the {loc}.",
        "{robot} is not {prep} the {loc} now.",
        "{robot} has left the area {prep} the {loc}.",
        "I don't think {robot} is {prep} the {loc} any more.",
        "{robot} was {prep} the {loc} but isn't now.",
    ]

    templates_generic = [
        "{robot} left the {loc}.",
        "{robot} is no longer at the {loc}.",
        "{robot} isn't in the {loc} anymore.",
        "{robot} has moved away from the {loc}.",
        "{robot} is somewhere else now — not the {loc}.",
        "{robot} vacated the {loc}.",
        "{robot} is not at the {loc} as far as I can tell.",
        "{robot} departed from the {loc}.",
        "{robot} is gone from the {loc}.",
        "I don't see {robot} at the {loc} anymore.",
        "{robot} and the {loc} — that's no longer the case.",
        "{robot} has been relocated away from the {loc}.",
    ]

    robot = _capitalize_name(_pick(ROBOT_NAMES))
    loc = _capitalize_name(_pick(LOCATIONS))

    if random.random() < 0.4:
        prep = _pick(EDGE_TYPES_LOC)
        nl = _pick(templates_explicit_prep).format(robot=robot, loc=loc, prep=prep)
        rel = prep
    else:
        rel = "at"
        nl = _pick(templates_generic).format(robot=robot, loc=loc)

    ops = [{"op": "remove_edge", "edge_type": rel, "source": robot, "target": loc}]
    return {
        "nl_input": nl,
        "expected": {
            "intent": "remove",
            "operations": ops,
            "response": f"Got it, I'll note that {robot} is no longer {rel} the {loc}.",
        },
        "category": "edge_removal",
        "setup_operations": [
            {"op": "create_node", "name": robot, "node_type": "robot"},
            {"op": "create_node", "name": loc, "node_type": "location"},
            {"op": "create_edge", "edge_type": rel, "source": robot, "target": loc},
        ],
    }


# ---------------------------------------------------------------------------
# Category 8: Compound statements  (assert → multiple ops)
# Rich, natural sentences that convey several facts at once.
# ---------------------------------------------------------------------------


def _multi_op_sample():
    # Each tuple: (template, edge_rel, property_key)
    # "battery" group
    templates_battery = [
        (
            "{robot} is one of our robots and it's in the {loc} with {bat}% battery.",
            "in",
        ),
        (
            "Our robot {robot} is currently in the {loc}. Its battery reads {bat}.",
            "in",
        ),
        (
            "Hey, {robot} just arrived in the {loc} and the battery is at {bat}.",
            "in",
        ),
        (
            "{robot} is working in the {loc} and has {bat} percent power.",
            "in",
        ),
        (
            "I deployed {robot} to the {loc}. Battery is sitting at {bat}.",
            "at",
        ),
    ]

    # "status" group
    templates_status = [
        (
            "{robot} is over at the {loc} and it's {status} right now.",
            "at",
        ),
        (
            "I can confirm {robot} is at the {loc} and currently {status}.",
            "at",
        ),
        (
            "Our robot {robot} is stationed near the {loc} and it's {status}.",
            "near",
        ),
        (
            "{robot} is near the {loc}. It was last seen {status}.",
            "near",
        ),
        (
            "{robot} reached the {loc} and went {status}.",
            "at",
        ),
    ]

    # "speed" group
    templates_speed = [
        (
            "{robot} is heading to the {loc} at {speed} metres per second.",
            "at",
        ),
        (
            "I put {robot} near the {loc}. It's running at {speed} m/s.",
            "near",
        ),
        (
            "{robot} is in the {loc}, travelling at {speed} m/s.",
            "in",
        ),
        (
            "Our robot {robot} is at the {loc} and its speed is {speed}.",
            "at",
        ),
        (
            "{robot} is near the {loc} and moving at {speed} m/s.",
            "near",
        ),
    ]

    # "carrying" group — use "holds" as edge_type because
    # "holding", "has … with it", "in its grip", "picked up" all map to "holds"
    templates_carrying = [
        (
            "{robot} is at the {loc} and it holds a {obj}.",
            "at",
        ),
        (
            "I see {robot} at the {loc} holding a {obj}.",
            "at",
        ),
        (
            "{robot} is at the {loc} and it's holding a {obj}.",
            "at",
        ),
        (
            "{robot} is at the {loc} with a {obj} in its grip.",
            "at",
        ),
        (
            "{robot} is at the {loc} and has a {obj} in hand.",
            "at",
        ),
    ]

    robot = _capitalize_name(_pick(ROBOT_NAMES))
    loc = _capitalize_name(_pick(LOCATIONS))
    obj = _capitalize_name(_pick(OBJECT_NAMES))
    bat = _pick(PROPERTIES["battery"])
    status = _pick(PROPERTIES["status"])
    speed = _pick(PROPERTIES["speed"])

    group = random.choice(["battery", "status", "speed", "carrying"])

    if group == "battery":
        tmpl, edge_rel = _pick(templates_battery)
        nl = tmpl.format(robot=robot, loc=loc, bat=bat)
        ops = [
            {"op": "create_node", "name": robot, "node_type": "robot"},
            {"op": "create_node", "name": loc, "node_type": "location"},
            {"op": "create_edge", "edge_type": edge_rel, "source": robot, "target": loc},
            {"op": "set_property", "name": robot, "key": "battery", "value": bat},
        ]
    elif group == "status":
        tmpl, edge_rel = _pick(templates_status)
        nl = tmpl.format(robot=robot, loc=loc, status=status)
        ops = [
            {"op": "create_node", "name": robot, "node_type": "robot"},
            {"op": "create_node", "name": loc, "node_type": "location"},
            {"op": "create_edge", "edge_type": edge_rel, "source": robot, "target": loc},
            {"op": "set_property", "name": robot, "key": "status", "value": status},
        ]
    elif group == "speed":
        tmpl, edge_rel = _pick(templates_speed)
        nl = tmpl.format(robot=robot, loc=loc, speed=speed)
        ops = [
            {"op": "create_node", "name": robot, "node_type": "robot"},
            {"op": "create_node", "name": loc, "node_type": "location"},
            {"op": "create_edge", "edge_type": edge_rel, "source": robot, "target": loc},
            {"op": "set_property", "name": robot, "key": "speed", "value": speed},
        ]
    else:  # carrying
        tmpl, edge_rel = _pick(templates_carrying)
        nl = tmpl.format(robot=robot, loc=loc, obj=obj)
        ops = [
            {"op": "create_node", "name": robot, "node_type": "robot"},
            {"op": "create_node", "name": loc, "node_type": "location"},
            {"op": "create_edge", "edge_type": edge_rel, "source": robot, "target": loc},
            {"op": "create_node", "name": obj, "node_type": obj},
            {"op": "create_edge", "edge_type": "holds", "source": robot, "target": obj},
        ]

    return {
        "nl_input": nl,
        "expected": {
            "intent": "assert",
            "operations": ops,
            "response": f"Got it, I've recorded all the information about {robot}.",
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
    parser = argparse.ArgumentParser(
        description="Generate the KG-Dialogue-500 benchmark dataset"
    )
    parser.add_argument("-n", type=int, default=500, help="Number of samples")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="dataset/kg_dialogue_500.json",
        help="Output file path",
    )
    args = parser.parse_args()

    dataset = generate_dataset(n=args.n, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))

    # Print category distribution
    from collections import Counter

    cats = Counter(s["category"] for s in dataset)
    print(f"Generated {len(dataset)} samples → {out_path}")
    for cat, cnt in sorted(cats.items()):
        print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()
