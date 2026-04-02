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

project(snap-tfdp)
if (VERBOSE)
    add_compile_definitions(VERBOSE)
endif ()
if (ENABLE_PARALLEL)
    find_package(OpenMP REQUIRED)
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

add_executable(preprocess_snap apps/preprocess_snap.cpp)
set_target_properties(preprocess_snap PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools
)

add_executable(preprocess_mtx apps/preprocess_mtx.cpp)
set_target_properties(preprocess_mtx PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools
)

add_executable(txt2mtx apps/txt2mtx.cpp)
set_target_properties(txt2mtx PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools
)

add_executable(dataset_info apps/dataset_info.cpp)
set_target_properties(dataset_info PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools
)

add_executable(cluster_inter_com_edge apps/cluster_inter_com_edge.cpp)
set_target_properties(cluster_inter_com_edge PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools
)

if (METRICS)
    add_executable(qgg apps/metrics_qgg.cpp)
    target_link_libraries(qgg metrics)
    set_target_properties(qgg PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools/metrics
    )

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


## tests
if (ENABLE_PARALLEL)
    add_executable(test_omp tests/test_omp.cpp)
    target_link_libraries(test_omp OpenMP::OpenMP_CXX)

    add_executable(test_parallel_sort tests/test_parallel_sort.cpp)
    target_link_libraries(test_parallel_sort OpenMP::OpenMP_CXX)

    add_executable(test_ours_cpu_parallel tests/test_ours_cpu_parallel.cpp)
    target_link_libraries(test_ours_cpu_parallel layout OpenMP::OpenMP_CXX)
endif ()
add_executable(test_graph_info tests/test_graph_info.cpp)

add_executable(test_fs tests/test_fs.cpp)


if (ENABLE_CUDA)
    add_executable(test_gpu_1b1t tests/test_gpu_1b1t.cu)

    add_executable(test_cuda tests/test_cuda.cu)

    add_executable(test_sgd tests/test_sgd.cu)

    add_executable(test_ours_gpu tests/test_ours_gpu.cpp)
    target_link_libraries(test_ours_gpu layout OpenMP::OpenMP_CXX CUDA::cudart)
endif ()
if (OGDF)
    add_executable(test_ogdf tests/test_ogdf.cpp)
    target_include_directories(test_ogdf PRIVATE third_party/ogdf/include)
    target_link_directories(test_ogdf PRIVATE third_party/ogdf/lib)
    target_link_libraries(test_ogdf OGDF)
endif ()
if (METRICS)
    add_executable(test_cgal tests/test_cgal.cpp)
    target_link_libraries(test_cgal CGAL::CGAL)

    add_executable(test_qgg tests/test_qgg.cpp)
    target_link_libraries(test_qgg metrics)

    add_executable(test_np tests/test_np.cpp)
    target_link_libraries(test_np metrics OpenMP::OpenMP_CXX)
endif ()