cmake_minimum_required(VERSION 3.8)
project(esn_path_prediction)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)

find_package(rcl_interfaces REQUIRED)

find_package(OpenCV REQUIRED)

# C++ section
#add_executable(esn_path_prediction_node 
#    src/esn_path_prediction_node.cpp
#)

#ament_target_dependencies(esn_path_prediction_node
#    rclcpp
#    rcl_interfaces
#)

# Python section
ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  ${PROJECT_NAME}/esn_path_prediction.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
