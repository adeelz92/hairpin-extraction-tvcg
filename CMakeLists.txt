cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(hairpinMain)

add_subdirectory("src/contour_splitting")
add_subdirectory("src/region_growing")
add_subdirectory("src/extract_hairpins")
add_subdirectory("src/eval_hairpins")
