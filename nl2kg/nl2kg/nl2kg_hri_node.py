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


import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from nl2kg_msgs.action import NL2KG
from whisper_msgs.action import STT
from audio_common_msgs.action import TTS


class NL2KGHRINode(Node):
    """ROS 2 node that provides voice-based HRI for the Knowledge Graph.

    Orchestrates a spoken dialogue loop:
      1. Listen to the user via whisper_ros (STT)
      2. Send transcribed text to the NL2KG action server
      3. Speak the response via piper_ros (TTS)
    """

    def __init__(self) -> None:
        super().__init__("nl2kg_hri_node")

        # Parameters
        self.declare_parameter("continuous_listening", True)
        self._continuous = self.get_parameter("continuous_listening").value

        cb_group = ReentrantCallbackGroup()

        # NL2KG action client
        self._nl2kg_client = ActionClient(
            self, NL2KG, "/nl2kg_node/nl2kg", callback_group=cb_group
        )

        # STT action client (whisper_ros)
        self._stt_client = ActionClient(
            self, STT, "/whisper/listen", callback_group=cb_group
        )

        # TTS action client (piper_ros)
        self._tts_client = ActionClient(self, TTS, "/say", callback_group=cb_group)

        # Start the interaction loop via a timer
        if self._continuous:
            self._timer = self.create_timer(0.5, self._interaction_loop)
            self._busy = False

        self.get_logger().info("NL2KG HRI node ready")

    # ------------------------------------------------------------------
    # Interaction loop (timer entry point)
    # ------------------------------------------------------------------
    def _interaction_loop(self) -> None:
        """Start one listen-process-speak cycle (non-blocking)."""
        if self._busy:
            return
        self._busy = True

        goal = STT.Goal()
        self._stt_client.send_goal_async(goal).add_done_callback(
            self._on_stt_goal_response
        )

    # ------------------------------------------------------------------
    # STT callbacks
    # ------------------------------------------------------------------
    def _on_stt_goal_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._busy = False
            return
        goal_handle.get_result_async().add_done_callback(self._on_stt_result)

    def _on_stt_result(self, future) -> None:
        text = future.result().result.text.strip()
        if not text:
            self._busy = False
            return

        self.get_logger().info(f"User said: {text}")

        goal = NL2KG.Goal()
        goal.input_text = text
        self._nl2kg_client.send_goal_async(goal).add_done_callback(
            self._on_nl2kg_goal_response
        )

    # ------------------------------------------------------------------
    # NL2KG callbacks
    # ------------------------------------------------------------------
    def _on_nl2kg_goal_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._busy = False
            return
        goal_handle.get_result_async().add_done_callback(self._on_nl2kg_result)

    def _on_nl2kg_result(self, future) -> None:
        result = future.result().result
        if not result.success or not result.response:
            self._busy = False
            return

        self.get_logger().info(f"Response: {result.response}")

        goal = TTS.Goal()
        goal.text = result.response
        self._tts_client.send_goal_async(goal).add_done_callback(
            self._on_tts_goal_response
        )

    # ------------------------------------------------------------------
    # TTS callbacks
    # ------------------------------------------------------------------
    def _on_tts_goal_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._busy = False
            return
        goal_handle.get_result_async().add_done_callback(self._on_tts_result)

    def _on_tts_result(self, future) -> None:
        self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = NL2KGHRINode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
