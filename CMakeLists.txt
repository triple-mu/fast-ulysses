cmake_minimum_required(VERSION 3.18)
project(fast_ulysses LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
if(NOT TORCH_CMAKE_PREFIX_PATH)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_PREFIX_RESULT)
    if(NOT TORCH_PREFIX_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Cannot locate PyTorch. Pass -DTORCH_CMAKE_PREFIX_PATH=/path/to/torch/share/cmake")
    endif()
endif()
list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
find_library(MLX5_LIBRARY mlx5 REQUIRED)
# cuMemGetHandleForAddressRange is driver-only: registering torch's virtual-memory-backed
# symmetric memory with the NIC goes through a dma-buf descriptor, and the runtime API cannot
# produce one.
find_package(CUDAToolkit REQUIRED)

add_library(_C SHARED csrc/fast_ulysses.cpp csrc/rdma.cpp csrc/transfer.cu)
set_target_properties(_C PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}")
target_include_directories(_C PRIVATE csrc)
find_library(TORCH_PYTHON torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON} Python::Module
                                 ${IBVERBS_LIBRARY} ${MLX5_LIBRARY} CUDA::cuda_driver)
target_compile_options(_C PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
