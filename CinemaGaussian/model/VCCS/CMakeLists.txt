cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(pcl_vccs)

find_package(PCL 1.12 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pcl_vccs pcl_vccs.cpp)
target_link_libraries (pcl_vccs ${PCL_LIBRARIES})