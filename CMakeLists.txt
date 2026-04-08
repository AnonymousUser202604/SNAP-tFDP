# 1.
# need set CMAKE_CUDA_COMPILER to the path of nvcc like:
# set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)

cmake_minimum_required(VERSION 3.20)

option(VERBOSE "Enable verbose mode" OFF)
option(ENABLE_PARALLEL "Enable SNAP-tFDP-GPU" OFF)
option(ENABLE_CUDA "Enable SNAP-tFDP-GPU" OFF)
option(METRICS "Build metrics tools(for metrics:np)" OFF)
option(OGDF "Build OGDF" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

project(snap-tfdp)
if (VERBOSE)
    add_compile_definitions(VERBOSE)
endif ()
if (ENABLE_PARALLEL)
    find_package(OpenMP REQUIRED)
    add_compile_definitions(ENABLE_PARALLEL)
endif ()
if (ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 14)
    add_compile_definitions(ENABLE_CUDA)
endif ()
if (METRICS)
    find_package(OpenMP REQUIRED)
    find_package(CGAL REQUIRED)
endif ()
if (OGDF)
    add_subdirectory(third_party/OGDF)
endif ()

include_directories(include)
include_directories(lib/pcg-cpp/include)
include_directories(lib/CLI)

### libs
## graph
file(GLOB_RECURSE LIB_GRAPH_SRC src/graph/*.cpp)
add_library(graph STATIC ${LIB_GRAPH_SRC})
## layout
file(GLOB_RECURSE LIB_LAYOUT_SRC src/layout/*.cpp)
if (ENABLE_CUDA)
    file(GLOB_RECURSE LIB_LAYOUT_CUDA_SRC src/layout/*.cu)
    list(APPEND LIB_LAYOUT_SRC ${LIB_LAYOUT_CUDA_SRC})
endif ()
add_library(layout STATIC ${LIB_LAYOUT_SRC})
target_link_libraries(layout PUBLIC graph)
if (ENABLE_PARALLEL)
    target_link_libraries(layout PRIVATE OpenMP::OpenMP_CXX)
endif ()
if (ENABLE_CUDA)
    target_link_libraries(layout PUBLIC CUDA::cudart)
endif ()
## metrics
if (METRICS)
    file(GLOB_RECURSE LIB_METRICS_SRC src/metrics/*.cpp)
    add_library(metrics STATIC ${LIB_METRICS_SRC})
    target_link_libraries(metrics PUBLIC graph CGAL::CGAL OpenMP::OpenMP_CXX)
endif ()


### applications
if (VERBOSE)
    set(main_target snap-tfdp-v)
else ()
    set(main_target snap-tfdp)
endif ()
add_executable(${main_target} apps/snap_tfdp.cpp)
target_link_libraries(${main_target} layout)
set_target_properties(${main_target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
)

add_executable(txt2mtx apps/txt2mtx.cpp)
set_target_properties(txt2mtx PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools
)

add_executable(draw apps/draw.cpp)
target_link_libraries(draw graph)
set_target_properties(draw PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools
)

if (METRICS)
    add_executable(np apps/metrics_np.cpp)
    set_target_properties(np PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools/metrics
    )
    target_link_libraries(np metrics OpenMP::OpenMP_CXX)
endif ()

if (OGDF)
    add_executable(pmds apps/run_pmds.cpp)
    target_link_libraries(pmds PRIVATE OGDF)
    set_target_properties(pmds PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
    add_executable(fr apps/run_fr.cpp)
    target_link_libraries(fr PRIVATE OGDF)
    set_target_properties(fr PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
endif ()
