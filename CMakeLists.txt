cmake_minimum_required(VERSION 3.7)
project(opencv_test)
find_package( OpenCV REQUIRED )

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp)
add_executable(opencv_test ${SOURCE_FILES})
target_link_libraries( opencv_test ${OpenCV_LIBS} )