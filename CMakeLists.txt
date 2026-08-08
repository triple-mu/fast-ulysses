cmake_minimum_required(VERSION 3.18)
# Fall back to the conventional CUDA location when the caller did not pick a compiler
# (override via -DCMAKE_CUDA_COMPILER=... or the CUDACXX env var).
if (NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif ()
project(fast_ulysses LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "80;90;100;120")
endif ()
# Snapshot it HERE. find_package(Torch) below sets CMAKE_CUDA_ARCHITECTURES to OFF and drives the
# arch selection from TORCH_CUDA_ARCH_LIST via its own -gencode flags instead, so anything read
# after that point gets OFF rather than what was asked for.
set(FAST_ULYSSES_ARCH_LIST "${CMAKE_CUDA_ARCHITECTURES}")
# Torch consumes a dotted form (90 -> 9.0). setup.py already exports this; set it for a plain
# cmake invocation too, otherwise -DCMAKE_CUDA_ARCHITECTURES is silently ignored.
if (NOT DEFINED ENV{TORCH_CUDA_ARCH_LIST})
    set(_arch_dotted "")
    foreach (a IN LISTS FAST_ULYSSES_ARCH_LIST)
        string(REGEX REPLACE "^(.+)(.)$" "\\1.\\2" _dotted "${a}")
        list(APPEND _arch_dotted "${_dotted}")
    endforeach ()
    string(REPLACE ";" " " _arch_dotted "${_arch_dotted}")
    set(ENV{TORCH_CUDA_ARCH_LIST} "${_arch_dotted}")
endif ()

# ccache: wrap CXX/CUDA compilers to cache objects and skip recompiling unchanged TUs (skipped if not installed)
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache enabled: ${CCACHE_PROGRAM}")
endif ()

# Development.Module, not Development: the target below links Python::Module, which needs only the
# headers. Plain Development also demands libpython, which a manylinux CPython does not ship at all,
# so asking for it fails everywhere the release wheels are actually built.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}" CACHE STRING "Torch prefix" FORCE)
find_package(Torch REQUIRED)

# ---- Does this libtorch have c10d::register_work? ----
# It is what binds an async result's completion event to torch's work registry, which is how
# the caller gets an AsyncCollectiveTensor (wait_tensor finds the event) instead of a handle
# nothing forces them to wait on. Probe the LIBRARY, not the header: a declaration says
# nothing about what libtorch_cpu exports, and the substring below is the mangled
# c10d::register_work(const at::Tensor&, const c10::intrusive_ptr<c10d::Work>&).
# Unquoted ${CMAKE_NM}: if it is unset, the command fails and RESULT_VARIABLE says so rather
# than CMake erroring on an empty COMMAND. Missing it is not fatal -- csrc/work.cpp then
# orders the result eagerly and comm.py hands back a CompletedHandle instead.
find_library(TORCH_CPU_LIBRARY torch_cpu PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
execute_process(COMMAND ${CMAKE_NM} -D --defined-only "${TORCH_CPU_LIBRARY}"
        OUTPUT_VARIABLE TORCH_CPU_SYMBOLS RESULT_VARIABLE NM_STATUS ERROR_QUIET)
set(HAVE_C10D_REGISTER_WORK OFF)
if (NM_STATUS EQUAL 0)
    string(FIND "${TORCH_CPU_SYMBOLS}" "_ZN4c10d13register_workERKN2at6Tensor" REGISTER_WORK_AT)
    if (NOT REGISTER_WORK_AT EQUAL -1)
        set(HAVE_C10D_REGISTER_WORK ON)
    endif ()
endif ()
message(STATUS "c10d::register_work available: ${HAVE_C10D_REGISTER_WORK}")

# setup.py resolves NVSHMEM and passes both paths, because the install we most want to use
# -- the nvidia-nvshmem-cu13 wheel that torch already depends on -- ships headers and
# libnvshmem_host.so.3 but NO lib/cmake/nvshmem, so find_package(NVSHMEM) cannot see it.
# Linking the host library by path works for every layout and needs no package config.
if (NOT DEFINED NVSHMEM_HOME OR NOT EXISTS "${NVSHMEM_HOME}/include/nvshmem.h")
    message(FATAL_ERROR "NVSHMEM_HOME missing: setup.py must pass -DNVSHMEM_HOME=<absolute>.")
endif ()
if (NOT DEFINED NVSHMEM_HOST_LIB OR NOT EXISTS "${NVSHMEM_HOST_LIB}")
    message(FATAL_ERROR "NVSHMEM_HOST_LIB missing: setup.py must pass -DNVSHMEM_HOST_LIB=<absolute>.")
endif ()

if (NOT DEFINED EXT_SUFFIX)
    set(EXT_SUFFIX ".so")
endif ()

# One version literal, in ./VERSION. Read it here so a bare `cmake -S . -B build` bakes the same
# string the Python side reports; setup.py overrides it with the full version including any local
# tag.
if (NOT DEFINED FAST_ULYSSES_VERSION)
    file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" FAST_ULYSSES_VERSION LIMIT_COUNT 1)
endif ()
# Re-run configure when the literal is bumped, so the baked macro cannot go stale.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")

# Wheel builds bake only $ORIGIN-relative RPATH entries; developer builds also keep the absolute
# directory this build found NVSHMEM in.
option(FAST_ULYSSES_RELOCATABLE "Emit an $ORIGIN-only RPATH (wheel builds)" OFF)

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu")
add_library(_C SHARED ${SOURCES})
set_target_properties(_C PROPERTIES PREFIX "" SUFFIX "${EXT_SUFFIX}"
        CUDA_SEPARABLE_COMPILATION OFF CUDA_ARCHITECTURES "${FAST_ULYSSES_ARCH_LIST}")
# What build_info() reports. The arch list is worth carrying in the binary: a .so built without
# this GPU's arch fails in a way that does not mention arch.
#
# Comma-separated, because a CMake list expands with semicolons and target_compile_definitions
# would read those as separate definitions.
string(REPLACE ";" "," _arch_csv "${FAST_ULYSSES_ARCH_LIST}")
target_compile_definitions(_C PRIVATE
        FAST_ULYSSES_VERSION="${FAST_ULYSSES_VERSION}"
        FAST_ULYSSES_CUDA_ARCH_LIST="${_arch_csv}"
        FAST_ULYSSES_NVSHMEM_HOME="${NVSHMEM_HOME}")
# CUDA 13 moved the CCCL headers (cuda/std/*) into include/cccl, where nvshmem.h no longer
# finds them -- and nvshmem.h is included by bindings.cpp, so the HOST compiler needs the
# path too, not just nvcc. Adding it here removes what INSTALL.md used to ask users to pass
# by hand.
find_package(CUDAToolkit REQUIRED)
set(CCCL_INCLUDE_DIRS "")
foreach (dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if (EXISTS "${dir}/cccl")
        list(APPEND CCCL_INCLUDE_DIRS "${dir}/cccl")
    endif ()
endforeach ()
target_include_directories(_C PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include" "${NVSHMEM_HOME}/include"
        ${CCCL_INCLUDE_DIRS})
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
# Only the HOST library. No device-side nvshmem_* call exists in these kernels, so there is
# no relocatable-device-code link step and libnvshmem_device.a stays out -- which also keeps
# nvshmemi_init_thread out, the symbol that clashes with torch's libtorch_nvshmem.so (see the
# DEVIATION notes in ulysses_group.cu).
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python::Module
        "${NVSHMEM_HOST_LIB}")
# The wheel layout is not on any default search path, so bake it in rather than making every
# caller set LD_LIBRARY_PATH. Relative to $ORIGIN (= <site-packages>/fast_ulysses) so the wheel
# stays relocatable; four nvidia directories because the cudart wheel layout differs across torch
# minors. These stay DT_RUNPATH -- no --disable-new-dtags -- so LD_LIBRARY_PATH can still override
# the bundled NVSHMEM with a site build.
set(FAST_ULYSSES_RPATH
        "$ORIGIN/../nvidia/nvshmem/lib"
        "$ORIGIN/../nvidia/cu13/lib"
        "$ORIGIN/../nvidia/cu12/lib"
        "$ORIGIN/../nvidia/cuda_runtime/lib"
        "$ORIGIN/../torch/lib")
if (NOT FAST_ULYSSES_RELOCATABLE)
    # Developer build: nothing of that layout exists above $ORIGIN, so fall back to where this
    # build found NVSHMEM -- last, so an installed copy still wins.
    get_filename_component(NVSHMEM_LIB_DIR "${NVSHMEM_HOST_LIB}" DIRECTORY)
    list(APPEND FAST_ULYSSES_RPATH "${NVSHMEM_LIB_DIR}")
endif ()
set_target_properties(_C PROPERTIES BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "${FAST_ULYSSES_RPATH}")
if (HAVE_C10D_REGISTER_WORK)
    target_compile_definitions(_C PRIVATE FAST_ULYSSES_HAS_WORK_REGISTRY=1)
else ()
    message(WARNING "c10d::register_work not found in ${TORCH_CPU_LIBRARY}: the async calls will "
            "return a CompletedHandle instead of an AsyncCollectiveTensor, with the caller's stream "
            "made to wait at launch -- correct, but no overlap.")
    target_compile_definitions(_C PRIVATE FAST_ULYSSES_HAS_WORK_REGISTRY=0)
endif ()
# -t0: nvcc compiles the per-arch device passes in parallel (one thread per arch)
target_compile_options(_C PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 -t0 --expt-relaxed-constexpr>)
