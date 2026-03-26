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


import json
from typing import List
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup

from ament_index_python.packages import get_package_share_directory

from knowledge_graph import KnowledgeGraph
from llama_ros.langchain import ChatLlamaROS, LlamaROSEmbeddings, LlamaROSReranker

from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_chroma import Chroma

from nl2kg_msgs.action import NL2KG
from nl2kg_msgs.msg import KGOperation as KGOperationMsg
from nl2kg.models import KGResponse, KGOperation
from nl2kg.kg_context import serialize_graph
from nl2kg.utils import cast_value


SYSTEM_PROMPT = """\
You are a Knowledge-Graph assistant for a robotic system.
Your job is to translate natural-language sentences into structured \
Knowledge Graph (KG) operations AND to answer questions about the current \
graph state.

## Intent definitions
- "assert": The user is adding NEW facts to the graph (creating nodes, \
creating edges, or doing multiple operations that introduce new information). \
Use this for ANY sentence that adds nodes or edges, even if it also sets \
properties.
- "query": The user is asking a question about the graph. Do NOT generate \
operations; put the answer in "response".
- "remove": The user wants to delete nodes or edges from the graph.
- "modify": The user wants to change a property on an EXISTING node or edge \
(set_property only, without creating new nodes or edges).
- "unclear": The request is ambiguous; ask for clarification.

## Available operations
| op             | Required fields                             |
|----------------|---------------------------------------------|
| create_node    | name, node_type                             |
| create_edge    | edge_type, source, target                   |
| remove_node    | name                                        |
| remove_edge    | edge_type, source, target                   |
| set_property   | name, key, value                            |
| query          | (none — answer in "response")               |

## Domain vocabulary (use ONLY these values)
- node_type: "robot", "location", "person", "cup", "book", "box", "bottle", \
"tool", "plate", "charger", "tray"
- edge_type: "at", "in", "near", "holds", "carries", "sees", "faces"
- set_property key — use ONLY these canonical names regardless of how the \
user phrases it:
  - "battery"  (e.g. "battery level", "power", "charge")
  - "speed"    (e.g. "pace", "velocity", "rate")
  - "status"   (e.g. "state", "mode", "condition")
  - "color"    (e.g. "colour", "finish", "painted", "looks like")
  - "weight"   (e.g. "mass", "kilograms", "kg")

## set_property rules
- For node properties, always set the "name" field to the node name.
- Only use "source" and "target" when setting a property on an edge.
- ALWAYS use the canonical key name from the vocabulary above (e.g. \
write "color", never "colour").

## Output format
Always reply with a single JSON object:
{{
  "intent": "<assert|query|remove|modify|unclear>",
  "operations": [ ... ],
  "response": "<natural language answer>"
}}

## Rules
1. Before creating an edge, always check the current graph state. \
If the source or target node does NOT already exist in the graph, \
add a create_node operation for it in the same response.
2. For "modify" (set_property), do NOT create nodes or edges — only \
set the property. The node already exists in the graph.
3. For "remove", do NOT create nodes or edges — only remove what is asked. \
The entity already exists in the graph.
4. If the user request is ambiguous, set intent to "unclear" and ask for \
clarification in "response".
5. For queries, set intent to "query", operations to [] and put the answer \
in "response".
6. Keep "response" concise and informative.
7. When the user says a robot "is at/in/near" a location, or "went to" / \
"moved to" a location, create an edge using the EXACT preposition from the \
sentence as edge_type (at, in, or near). If no preposition is given, \
default to "at".
8. For edge removal, use the EXACT edge_type mentioned in the sentence. \
If not explicitly stated, default to "at".
9. Do NOT add set_property operations unless the user explicitly states a \
property value (battery level, speed, status, color, weight, etc.).
10. Do NOT remove edges that are not explicitly mentioned by the user.
11. For node creation, the "name" field must be the entity's proper name (e.g. \
"tiago", "kitchen"). Do NOT use positional words like "here", "nearby", \
"area", "there" as the name.

## Examples

User: "Just so you know, tiago is one of our robots."
{{
  "intent": "assert",
  "operations": [{{"op": "create_node", "name": "tiago", "node_type": "robot"}}],
  "response": "Got it, I'll remember that tiago is a robot."
}}

User: "I just spotted robot1 in the kitchen."
{{
  "intent": "assert",
  "operations": [
    {{"op": "create_node", "name": "robot1", "node_type": "robot"}},
    {{"op": "create_node", "name": "kitchen", "node_type": "location"}},
    {{"op": "create_edge", "edge_type": "in", "source": "robot1", "target": "kitchen"}}
  ],
  "response": "Understood, robot1 is now in the kitchen."
}}

User: "robot1 is at the kitchen."
{{
  "intent": "assert",
  "operations": [
    {{"op": "create_node", "name": "robot1", "node_type": "robot"}},
    {{"op": "create_node", "name": "kitchen", "node_type": "location"}},
    {{"op": "create_edge", "edge_type": "at", "source": "robot1", "target": "kitchen"}}
  ],
  "response": "Understood, robot1 is now at the kitchen."
}}

User: "robot1's color is green."
{{
  "intent": "modify",
  "operations": [{{"op": "set_property", "name": "robot1", "key": "color", "value": "green"}}],
  "response": "Got it, I've updated robot1's color to green."
}}

User: "robot1's battery is at 80 percent."
{{
  "intent": "modify",
  "operations": [{{"op": "set_property", "name": "robot1", "key": "battery", "value": "80"}}],
  "response": "Got it, I've updated robot1's battery to 80."
}}

User: "Where is robot1?"
{{
  "intent": "query",
  "operations": [],
  "response": "robot1 is at the kitchen."
}}

User: "robot1 is gone. You can forget about it."
{{
  "intent": "remove",
  "operations": [{{"op": "remove_node", "name": "robot1"}}],
  "response": "Understood, I'll forget about robot1."
}}

User: "robot1 left the kitchen."
{{
  "intent": "remove",
  "operations": [{{"op": "remove_edge", "edge_type": "at", "source": "robot1", "target": "kitchen"}}],
  "response": "Got it, I'll note that robot1 is no longer at the kitchen."
}}

User: "spot is one of our robots and it's in the lab with 50% battery."
{{
  "intent": "assert",
  "operations": [
    {{"op": "create_node", "name": "spot", "node_type": "robot"}},
    {{"op": "create_node", "name": "lab", "node_type": "location"}},
    {{"op": "create_edge", "edge_type": "in", "source": "spot", "target": "lab"}},
    {{"op": "set_property", "name": "spot", "key": "battery", "value": "50"}}
  ],
  "response": "Got it, I've recorded all the information about spot."
}}

## Current Knowledge Graph state
{kg_context}
"""


class NL2KGNode(Node):
    """ROS 2 node that bridges natural language and a Knowledge Graph."""

    def __init__(self) -> None:
        super().__init__("nl2kg_node")

        # Parameters
        self.declare_parameter("temperature", 0.0)
        self.declare_parameter("use_gbnf", True)
        self.declare_parameter("enable_rag", False)

        temp = self.get_parameter("temperature").value
        use_gbnf = self.get_parameter("use_gbnf").value
        enable_rag = self.get_parameter("enable_rag").value

        # Knowledge Graph
        self.graph = KnowledgeGraph.get_instance()

        # LLM — optionally with GBNF grammar for constrained decoding
        grammar = ""
        if use_gbnf:
            grammar = self._load_grammar()
            self.get_logger().info("Using GBNF grammar for constrained output")

        self.llm = ChatLlamaROS(temp=temp, grammar=grammar)

        # Structured output chain (Pydantic parser fallback when GBNF off)
        if not use_gbnf:
            self.structured_llm = self.llm.with_structured_output(
                KGResponse, method="function_calling"
            )
        else:
            self.structured_llm = None

        # Optional RAG with reranker
        self.rag_retriever = None
        if enable_rag:
            self._setup_rag()

        # ROS 2 action server
        self._action_server = ActionServer(
            self,
            NL2KG,
            "~/nl2kg",
            execute_callback=self._execute_action,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info("NL2KG node ready")

    # ------------------------------------------------------------------
    # Action callbacks
    # ------------------------------------------------------------------
    def _goal_callback(self, goal_request) -> GoalResponse:
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_action(self, goal_handle) -> NL2KG.Result:
        user_text = goal_handle.request.input_text
        self.get_logger().info(f"Input: {user_text}")

        result = NL2KG.Result()

        try:
            # Publish feedback: processing
            feedback = NL2KG.Feedback()
            feedback.status = "processing"
            goal_handle.publish_feedback(feedback)

            kg_response = self._process(user_text)

            # Publish feedback: executing operations
            feedback.status = "executing_operations"
            goal_handle.publish_feedback(feedback)

            self._execute_operations(kg_response.operations)

            result.success = True
            result.intent = kg_response.intent
            result.operations = self._to_operation_msgs(kg_response.operations)
            result.response = kg_response.response

            self.get_logger().info(f"Response: {kg_response.response}")

        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")
            result.success = False
            result.intent = "unclear"
            result.operations = []
            result.response = f"Error: {e}"

        goal_handle.succeed()
        return result

    @staticmethod
    def _to_operation_msgs(operations: List[KGOperation]) -> List[KGOperationMsg]:
        msgs = []
        for op in operations:
            msg = KGOperationMsg()
            msg.op = op.op
            msg.name = op.name or ""
            msg.node_type = op.node_type or ""
            msg.edge_type = op.edge_type or ""
            msg.source = op.source or ""
            msg.target = op.target or ""
            msg.key = op.key or ""
            msg.value = op.value or ""
            msgs.append(msg)
        return msgs

    # ------------------------------------------------------------------
    # Grammar loading
    # ------------------------------------------------------------------
    def _load_grammar(self) -> str:
        pkg_share = get_package_share_directory("nl2kg")
        grammar_path = Path(pkg_share) / "grammars" / "nl2kg.gbnf"
        return grammar_path.read_text()

    # ------------------------------------------------------------------
    # RAG setup with reranker
    # ------------------------------------------------------------------
    def _setup_rag(self) -> None:
        self.embeddings = LlamaROSEmbeddings()
        self.vectorstore = Chroma(
            collection_name="nl2kg_context",
            embedding_function=self.embeddings,
        )
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

        # Reranker for contextual compression
        reranker = LlamaROSReranker(top_n=5)
        self.rag_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever,
        )
        self.get_logger().info(
            "RAG enabled with Chroma + LlamaROSEmbeddings + LlamaROSReranker"
        )

    def _update_rag_index(self) -> None:
        """Re-index the current KG state into the vector store."""
        if self.rag_retriever is None:
            return

        docs: List[Document] = []
        for n in self.graph.get_nodes():
            docs.append(
                Document(
                    page_content=f"Node '{n.get_name()}' of type '{n.get_type()}'",
                    metadata={"kind": "node", "name": n.get_name()},
                )
            )
        for e in self.graph.get_edges():
            docs.append(
                Document(
                    page_content=(
                        f"Edge '{e.get_type()}' from '{e.get_source_node()}' "
                        f"to '{e.get_target_node()}'"
                    ),
                    metadata={"kind": "edge", "type": e.get_type()},
                )
            )
        if docs:
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma.from_documents(
                docs,
                self.embeddings,
                collection_name="nl2kg_context",
            )
            # Rebuild retriever with reranker
            base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
            self.rag_retriever = ContextualCompressionRetriever(
                base_compressor=self.rag_retriever.base_compressor,
                base_retriever=base_retriever,
            )

    # ------------------------------------------------------------------
    # NL -> KG pipeline
    # ------------------------------------------------------------------
    def _process(self, user_text: str) -> KGResponse:
        """Run the NL -> KG pipeline and return a structured response."""

        # Build context
        if self.rag_retriever is not None:
            self._update_rag_index()
            rag_docs = self.rag_retriever.invoke(user_text)
            kg_context = "\n".join(d.page_content for d in rag_docs)
            if not kg_context:
                kg_context = serialize_graph(self.graph)
        else:
            kg_context = serialize_graph(self.graph)

        system_msg = SYSTEM_PROMPT.format(kg_context=kg_context)

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_text),
        ]

        # Invoke LLM
        if self.structured_llm is not None:
            return self.structured_llm.invoke(messages)
        else:
            ai_msg = self.llm.invoke(messages)
            raw = ai_msg.content.strip()
            data = json.loads(raw)
            # Remove reasoning field if present (not part of KGResponse)
            data.pop("reasoning", None)
            # Normalize operations: convert empty strings to None
            for op in data.get("operations", []):
                for k in list(op.keys()):
                    if op[k] == "":
                        op[k] = None
            return KGResponse(**data)

    # ------------------------------------------------------------------
    # Operation execution
    # ------------------------------------------------------------------
    def _execute_operations(self, operations: List[KGOperation]) -> None:
        for op in operations:
            try:
                self._execute_one(op)
            except Exception as e:
                self.get_logger().warn(f"Op {op.op} failed: {e}")

    def _execute_one(self, op: KGOperation) -> None:
        g = self.graph

        if op.op == "create_node":
            if not g.has_node(op.name):
                g.create_node(op.name, op.node_type or "unknown")
                self.get_logger().info(f"Created node: {op.name}")
            else:
                self.get_logger().info(f"Node already exists: {op.name}")

        elif op.op == "create_edge":
            if not g.has_node(op.source):
                g.create_node(op.source, "unknown")
            if not g.has_node(op.target):
                g.create_node(op.target, "unknown")
            if not g.has_edge(op.edge_type, op.source, op.target):
                g.create_edge(op.edge_type, op.source, op.target)
                self.get_logger().info(
                    f"Created edge: {op.source} --[{op.edge_type}]--> {op.target}"
                )

        elif op.op == "remove_node":
            if g.has_node(op.name):
                node = g.get_node(op.name)
                g.remove_node(node)
                self.get_logger().info(f"Removed node: {op.name}")

        elif op.op == "remove_edge":
            if g.has_edge(op.edge_type, op.source, op.target):
                edge = g.get_edge(op.edge_type, op.source, op.target)
                g.remove_edge(edge)
                self.get_logger().info(
                    f"Removed edge: {op.source} --[{op.edge_type}]--> {op.target}"
                )

        elif op.op == "set_property":
            if op.name and g.has_node(op.name):
                node = g.get_node(op.name)
                node.set_property(op.key, cast_value(op.value))
                g.update_node(node)
                self.get_logger().info(
                    f"Set property {op.key}={op.value} on node {op.name}"
                )
            elif op.source and op.target:
                edge_type = op.edge_type or ""
                if g.has_edge(edge_type, op.source, op.target):
                    edge = g.get_edge(edge_type, op.source, op.target)
                    edge.set_property(op.key, cast_value(op.value))
                    g.update_edge(edge)

        elif op.op == "query":
            pass

        else:
            self.get_logger().warn(f"Unknown operation: {op.op}")


def main(args=None):
    rclpy.init(args=args)
    node = NL2KGNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
