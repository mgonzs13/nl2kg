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

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory


def generate_launch_description():

    bringup_share = get_package_share_directory("nl2kg_bringup")
    llama_bringup_share = get_package_share_directory("llama_bringup")
    base_launch = os.path.join(llama_bringup_share, "launch", "base.launch.py")

    # Launch arguments
    model_params_arg = DeclareLaunchArgument(
        "model_params",
        default_value=os.path.join(bringup_share, "params", "llm.yaml"),
        description="Path to the LLM model YAML config",
    )

    embedding_params_arg = DeclareLaunchArgument(
        "embedding_params",
        default_value=os.path.join(bringup_share, "params", "bge-base-en-v1.5.yaml"),
        description="Path to the embedding model YAML config",
    )

    reranker_params_arg = DeclareLaunchArgument(
        "reranker_params",
        default_value=os.path.join(bringup_share, "params", "bge-reranker-v2-m3.yaml"),
        description="Path to the reranker model YAML config",
    )

    enable_hri_arg = DeclareLaunchArgument(
        "enable_hri",
        default_value="false",
        description="Enable voice-based HRI (STT + TTS)",
    )

    system_prompt_file_arg = DeclareLaunchArgument(
        "system_prompt_file",
        default_value="",
        description="Override path for the nl2kg_node system prompt file.",
    )

    grammar_file_arg = DeclareLaunchArgument(
        "grammar_file",
        default_value="",
        description="Override path for the nl2kg_node GBNF grammar file.",
    )

    enable_embedding_arg = DeclareLaunchArgument(
        "enable_embedding",
        default_value="false",
        description="Enable the embedding model nodes",
    )

    # Chat LLM
    llama_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch),
        launch_arguments={
            "params_file": LaunchConfiguration("model_params"),
            "node_name": "llama_node",
        }.items(),
    )

    # Embedding model
    embedding_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch),
        launch_arguments={
            "params_file": LaunchConfiguration("embedding_params"),
            "node_name": "embedding_node",
        }.items(),
        condition=IfCondition(LaunchConfiguration("enable_embedding")),
    )

    # Reranker model
    reranker_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(base_launch),
        launch_arguments={
            "params_file": LaunchConfiguration("reranker_params"),
            "node_name": "reranker_node",
        }.items(),
        condition=IfCondition(LaunchConfiguration("enable_embedding")),
    )

    # NL2KG node
    nl2kg_node = Node(
        package="nl2kg",
        executable="nl2kg_node",
        name="nl2kg_node",
        parameters=[
            os.path.join(bringup_share, "params", "nl2kg.yaml"),
            {
                "system_prompt_file": LaunchConfiguration("system_prompt_file"),
                "grammar_file": LaunchConfiguration("grammar_file"),
            },
        ],
        output="screen",
    )

    # Whisper STT (conditional)
    whisper_bringup_share = get_package_share_directory("whisper_bringup")
    whisper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(whisper_bringup_share, "launch", "whisper.launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("enable_hri")),
    )

    # Piper TTS (conditional)
    piper_bringup_share = get_package_share_directory("piper_bringup")
    piper_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(piper_bringup_share, "launch", "piper.launch.py")
        ),
        condition=IfCondition(LaunchConfiguration("enable_hri")),
    )

    # NL2KG HRI node (conditional)
    nl2kg_hri_node = Node(
        package="nl2kg",
        executable="nl2kg_hri_node",
        name="nl2kg_hri_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("enable_hri")),
    )

    ld = LaunchDescription()
    ld.add_action(model_params_arg)
    ld.add_action(embedding_params_arg)
    ld.add_action(reranker_params_arg)
    ld.add_action(enable_hri_arg)
    ld.add_action(system_prompt_file_arg)
    ld.add_action(grammar_file_arg)
    ld.add_action(enable_embedding_arg)
    ld.add_action(llama_launch)
    ld.add_action(embedding_launch)
    ld.add_action(reranker_launch)
    ld.add_action(nl2kg_node)
    ld.add_action(whisper_launch)
    ld.add_action(piper_launch)
    ld.add_action(nl2kg_hri_node)
    return ld
