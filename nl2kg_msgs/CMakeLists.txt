cmake_minimum_required(VERSION 3.8)
project(nl2kg_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/KGOperation.msg"
  "action/NL2KG.action"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
