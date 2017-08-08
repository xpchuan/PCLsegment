cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

project(openni_grabber)
SET(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++11)
set(PCL_DIR "/home/xpc/pcl/build/install/share/pcl-1.8")
find_package(PCL 1.8 REQUIRED)


include_directories(${PCL_INCLUDE_DIRS})
include_directories("/usr/lib/gperf/include")
message(STATUS "+++++++++++++++++"  ${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})
link_directories("/usr/lib/gperf")
add_definitions(${PCL_DEFINITIONS})


add_executable (openni_grabber main3.cpp)
target_link_libraries (openni_grabber ${PCL_LIBRARIES})
target_link_libraries (openni_grabber "/usr/lib/gperf/lib/libprofiler.so")

