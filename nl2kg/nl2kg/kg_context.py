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


from knowledge_graph import KnowledgeGraph


def serialize_graph(graph: KnowledgeGraph) -> str:
    """Convert the full KG state into a human-readable text block.

    Used as context in the LLM prompt so the model is aware of the current
    graph state when generating operations.
    """

    lines: list[str] = []

    nodes = graph.get_nodes()
    edges = graph.get_edges()

    if not nodes and not edges:
        return "(empty graph)"

    # Nodes
    if nodes:
        lines.append("Nodes:")
        for n in nodes:
            props = _format_properties(n)
            line = f"  - {n.get_name()} (type: {n.get_type()})"
            if props:
                line += f"  [{props}]"
            lines.append(line)

    # Edges
    if edges:
        lines.append("Edges:")
        for e in edges:
            props = _format_properties(e)
            line = (
                f"  - ({e.get_source_node()}) "
                f"--[{e.get_type()}]--> "
                f"({e.get_target_node()})"
            )
            if props:
                line += f"  [{props}]"
            lines.append(line)

    return "\n".join(lines)


def _format_properties(element) -> str:
    """Format properties of a Node or Edge into a compact string."""
    parts: list[str] = []
    for prop_msg in element.properties_to_msg():
        val = _extract_content_value(prop_msg.value)
        parts.append(f"{prop_msg.key}={val}")
    return ", ".join(parts)


def _extract_content_value(content) -> str:
    """Extract a human-readable value from a knowledge_graph_msgs/Content msg."""
    from knowledge_graph_msgs.msg import Content

    _ACCESSORS = {
        Content.BOOL: "bool_value",
        Content.INT: "int_value",
        Content.FLOAT: "float_value",
        Content.DOUBLE: "double_value",
        Content.STRING: "string_value",
        Content.VBOOL: "bool_vector",
        Content.VINT: "int_vector",
        Content.VFLOAT: "float_vector",
        Content.VDOUBLE: "double_vector",
        Content.VSTRING: "string_vector",
    }
    attr = _ACCESSORS.get(content.type, "string_value")
    return str(getattr(content, attr))
