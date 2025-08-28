cmake_minimum_required(VERSION 3.0.2)
project(pathfinding_simulation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  nav_msgs
  geometry_msgs
  message_generation
)

## Declare your service files
add_service_files(
  FILES
    StartGoalPath.srv
)

## Generate added messages and services
generate_messages(
  DEPENDENCIES
    std_msgs
    geometry_msgs
    nav_msgs
)

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS
    message_runtime
    rospy
    std_msgs
    nav_msgs
    geometry_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Install your ROS-executable scripts
catkin_install_python(PROGRAMS
  src/pathfinding_simulation/grid/make_occupancy.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install your Python modules so you can import them
install(
  DIRECTORY src/pathfinding_simulation
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
