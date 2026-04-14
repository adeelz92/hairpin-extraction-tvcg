# Created by the script cgal_create_cmake_script
# This is the CMake script for compiling a CGAL application.

cmake_minimum_required(VERSION 3.1...3.23)

project(Extract_Hairpins)

find_package(VTK COMPONENTS 
  vtkCommonColor
  vtkFiltersGeometry
  vtkFiltersFlowPaths
  vtkFiltersModeling
  vtkFiltersParallelDIY2
  vtkFiltersPoints
  vtkFiltersSources
  vtkFiltersExtraction
  vtkInteractionStyle
  vtkIOImage
  vtkIOLegacy
  vtkIOGeometry
  vtkjsoncpp
  vtkRenderingAnnotation
  vtkRenderingContextOpenGL2
  vtkRenderingCore
  vtkRenderingFreeType
  vtkRenderingGL2PSOpenGL2
  vtkRenderingOpenGL2
  QUIET
)

if(WIN32)
  # do nothing
else()
  set(EIGEN3_INCLUDE_DIR "$ENV{EIGEN_HOME}")
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()

find_package(CGAL REQUIRED)

find_package(TTKVTK REQUIRED)

if (NOT VTK_FOUND)
  message("Skipping ${PROJECT_NAME}: ${VTK_NOT_FOUND_MESSAGE}")
  return()
endif()
message (STATUS "VTK_VERSION: ${VTK_VERSION}")
if (VTK_VERSION VERSION_LESS "8.90.0")
  # old system
  include(${VTK_USE_FILE})
  # add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${PROJECT_NAME}.cpp SkeletonExtractor.cpp)
  add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${PROJECT_NAME}.cpp)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${VTK_LIBRARIES} CGAL::CGAL ttk::vtk::ttkAll)
else()
  # Prevent a "command line is too long" failure in Windows.
  set(CMAKE_NINJA_FORCE_RESPONSE_FILE "ON" CACHE BOOL "Force Ninja to use response files.")
  # add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${PROJECT_NAME}.cpp SkeletonExtractor.cpp)
  add_executable(${PROJECT_NAME} MACOSX_BUNDLE ${PROJECT_NAME}.cpp)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${VTK_LIBRARIES} CGAL::CGAL ttk::vtk::ttkAll)
  # vtk_module_autoinit is needed
  vtk_module_autoinit(
    TARGETS ${PROJECT_NAME}
    MODULES ${VTK_LIBRARIES}
    )
endif()

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

if(CGAL_Qt5_FOUND)
  target_link_libraries(draw_triangulation_2 PUBLIC CGAL::CGAL_Basic_viewer)
    target_link_libraries(star_conflict_zone PUBLIC CGAL::CGAL_Basic_viewer)
else()
  message(
    STATUS
      "NOTICE: The example draw_triangulation_2 requires Qt and will not be compiled."
  )
endif()
