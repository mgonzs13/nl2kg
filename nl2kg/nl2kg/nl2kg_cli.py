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

from nl2kg_msgs.action import NL2KG


class NL2KGCLI(Node):
    def __init__(self) -> None:
        super().__init__("nl2kg_cli")
        self._action_client = ActionClient(self, NL2KG, "/nl2kg_node/nl2kg")

    def _send_goal(self, text: str) -> None:
        goal = NL2KG.Goal()
        goal.input_text = text

        self._action_client.wait_for_server()
        future = self._action_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback
        )
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            print("\nGoal rejected.")
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self._print_result(result)

    def _feedback_callback(self, feedback_msg) -> None:
        status = feedback_msg.feedback.status
        self.get_logger().info(f"Status: {status}")

    @staticmethod
    def _print_result(result: NL2KG.Result) -> None:
        print(f"\n[{result.intent}] {result.response}")
        if result.operations:
            print(f"  Operations executed: {len(result.operations)}")
            for op_msg in result.operations:
                print(
                    f"    - {op_msg.op}: name={op_msg.name} "
                    f"source={op_msg.source} target={op_msg.target}"
                )

    def run(self) -> None:
        print("NL2KG Interactive CLI")
        print("Type your message and press Enter. Ctrl+C to exit.\n")

        print("Waiting for NL2KG action server...", end=" ", flush=True)
        self._action_client.wait_for_server()
        print("connected.\n")

        try:
            while rclpy.ok():
                user_input = input("You: ")
                if not user_input.strip():
                    continue
                self._send_goal(user_input.strip())
                print()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")


def main(args=None):
    rclpy.init(args=args)
    cli = NL2KGCLI()
    try:
        cli.run()
    finally:
        cli.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
