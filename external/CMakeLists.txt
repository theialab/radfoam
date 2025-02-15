if(PIP_GLFW)
  set(USE_PIP_GLFW
    True
    PARENT_SCOPE)
  set(GLFW_LIBRARY
      ""
      PARENT_SCOPE)
  set(GLFW_INCLUDES
      ${CMAKE_SOURCE_DIR}/external/submodules/glfw/include
      PARENT_SCOPE)
else()
  set(USE_PIP_GLFW
    False
    PARENT_SCOPE)
  set(GLFW_BUILD_EXAMPLES
      OFF
      CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS
      OFF
      CACHE BOOL "" FORCE)
  set(GLFW_BUILD_DOCS
      OFF
      CACHE BOOL "" FORCE)
  add_subdirectory(submodules/glfw)
  set(GLFW_LIBRARY
      glfw
      PARENT_SCOPE)
  set(GLFW_INCLUDES
      ""
      PARENT_SCOPE)
  message(STATUS "GLFW not found from pip, building from source")
endif()

add_library(gl3w STATIC gl3w/gl3w.c)
target_include_directories(gl3w PUBLIC "include")

add_library(
  imgui STATIC
  submodules/imgui/imgui.cpp
  submodules/imgui/imgui_draw.cpp
  submodules/imgui/imgui_demo.cpp
  submodules/imgui/imgui_widgets.cpp
  submodules/imgui/imgui_tables.cpp
  submodules/imgui/backends/imgui_impl_glfw.cpp
  submodules/imgui/backends/imgui_impl_opengl3.cpp)
target_include_directories(
  imgui PUBLIC "submodules/imgui" "submodules/glfw/include"
               "submodules/mesa/include")

find_package(TBB GLOBAL)
if(NOT TBB_FOUND)
    add_subdirectory(submodules/tbb)
endif()

set(RADFOAM_EXTERNAL_INCLUDES
    "${CMAKE_SOURCE_DIR}/external/include"
    "${CMAKE_SOURCE_DIR}/external/submodules/imgui"
    "${CMAKE_SOURCE_DIR}/external/submodules/imgui/backends"
    PARENT_SCOPE)
