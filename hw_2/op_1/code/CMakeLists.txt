cmake_minimum_required(VERSION 3.15)
project(op1_image_compression CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

get_filename_component(PROJECT_SOURCE_DIR_ABS ${CMAKE_SOURCE_DIR} ABSOLUTE)

set(FIGS_DIR "${CMAKE_SOURCE_DIR}/../figs")
set(DEFAULT_IMAGE_PATH "${FIGS_DIR}/original.png")

add_executable(op1_template text.cpp)
target_link_libraries(op1_template PRIVATE ${OpenCV_LIBS})
