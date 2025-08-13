cmake_minimum_required(VERSION 3.16)
project(rosalind)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(src/io)
add_subdirectory(src/utils)
add_subdirectory(src/exercises)
add_subdirectory(src)
add_subdirectory(tests)
