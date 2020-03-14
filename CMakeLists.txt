cmake_minimum_required(VERSION 2.8.3)
project(gym_cpp)


find_package(Eigen3 REQUIRED)


include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
)
link_directories(
  ${catkin_LIBRARY_DIRS}
)

add_executable(env_test
  test/env_test.cpp
)
