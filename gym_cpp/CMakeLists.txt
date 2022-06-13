cmake_minimum_required(VERSION 2.8.3)
project(gym_cpp)


find_package(catkin REQUIRED COMPONENTS)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(OpenMP)
find_package(Boost REQUIRED system thread program_options)

# TODO : Fix the libtorch path
set(CMAKE_MODULE_PATH
${CMAKE_MODULE_PATH} /home/efe/installation/libtorch/share/cmake)
set(CMAKE_PREFIX_PATH /home/efe/installation/libtorch )
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

catkin_package(
  INCLUDE_DIRS
    include
	  ${boost_INCLUDE_DIR}
  LIBRARIES
    ${OpenCV_LIBRARIES}
  CATKIN_DEPENDS
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)
link_directories(
  ${catkin_LIBRARY_DIRS}
  ${Boost_LIBRARIES}
)


install(DIRECTORY include/${PROJECT_NAME}
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        PATTERN ".svn" EXCLUDE
)
