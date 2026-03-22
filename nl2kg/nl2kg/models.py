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

from typing import List, Optional
from pydantic import BaseModel, Field


class KGOperation(BaseModel):
    """A single Knowledge Graph operation produced by the LLM."""

    op: str = Field(
        description=(
            "Operation type: create_node, create_edge, "
            "remove_node, remove_edge, set_property, query"
        )
    )
    name: Optional[str] = Field(
        None, description="Node name (for create_node, remove_node, set_property)"
    )
    node_type: Optional[str] = Field(None, description="Node type (for create_node)")
    edge_type: Optional[str] = Field(
        None, description="Edge type (for create_edge, remove_edge)"
    )
    source: Optional[str] = Field(
        None, description="Source node name (for edges and set_property on edges)"
    )
    target: Optional[str] = Field(
        None, description="Target node name (for edges and set_property on edges)"
    )
    key: Optional[str] = Field(None, description="Property key (for set_property)")
    value: Optional[str] = Field(None, description="Property value (for set_property)")


class KGResponse(BaseModel):
    """Structured LLM response containing intent, operations, and NL reply."""

    intent: str = Field(
        description=(
            "User intent: assert (add facts), query (ask questions), "
            "remove (delete facts), modify (update facts), unclear"
        )
    )
    operations: List[KGOperation] = Field(
        default_factory=list,
        description="List of KG operations to execute",
    )
    response: str = Field(description="Natural language response to the user")
