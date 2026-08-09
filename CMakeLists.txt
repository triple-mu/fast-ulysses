cmake_minimum_required(VERSION 3.18)
# Fall back to the conventional CUDA location when the caller did not pick a compiler
# (override via -DCMAKE_CUDA_COMPILER=... or the CUDACXX env var).
if (NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX} AND EXISTS /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif ()
# Snapshot the caller's request BEFORE project(). project(LANGUAGES CUDA) always leaves
# CMAKE_CUDA_ARCHITECTURES defined -- it falls back to the compiler's own default, which is 52 --
# so a test placed after it cannot tell "nothing was asked for" from "52 was asked for", and the
# default below never applied to a bare `cmake -S . -B build`: that produced a .so holding
# sm_52 alone, which no supported device can run. Left unset here on purpose, so project()'s
# compiler check still compiles for the default arch: seeding it with the Blackwell entries would
# make an old toolkit fail inside that check, with nvcc's own message, before the guard below can
# name the toolkit and the way out.
set(FAST_ULYSSES_ARCH_REQUESTED "${CMAKE_CUDA_ARCHITECTURES}")
project(fast_ulysses LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if ("${FAST_ULYSSES_ARCH_REQUESTED}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "80;90;100;120")
else ()
    set(CMAKE_CUDA_ARCHITECTURES "${FAST_ULYSSES_ARCH_REQUESTED}")
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

# ---- Is this toolkit new enough for the architectures that were asked for? ----
# Only Blackwell forces a floor: CUDA 12.6 and below cannot emit sm_100/sm_120 and nvcc says so
# with "unsupported gpu architecture 'compute_100'", which names neither the toolkit version nor
# the way out. Gated on the list, because FAST_ULYSSES_CUDA_ARCH=80;90 on CUDA 12.4 builds and
# runs -- docs/install.md states 12.8 as the supported floor, and this refuses only what nvcc
# provably cannot emit. Runs before the torch discovery, so it costs nothing.
#
# CMAKE_CUDA_COMPILER_VERSION, not CUDAToolkit_VERSION: project(LANGUAGES CUDA) above already set
# it, and it is by construction the version of the nvcc named in the message. Reading it needs no
# find_package, so the guard cannot itself fail to resolve, and an unset version cannot make
# VERSION_LESS compare an empty string and abort a good toolkit.
set(_blackwell "")
foreach (a IN LISTS FAST_ULYSSES_ARCH_LIST)
    string(REGEX MATCH "^[0-9]+" _sm "${a}")  # 100a, 120-real -> 100, 120
    if (_sm AND _sm GREATER_EQUAL 100)
        list(APPEND _blackwell "sm_${_sm}")
    endif ()
endforeach ()
if (_blackwell AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
    string(REPLACE ";" ", " _blackwell "${_blackwell}")
    message(FATAL_ERROR
            "CUDA ${CMAKE_CUDA_COMPILER_VERSION} (nvcc: ${CMAKE_CUDA_COMPILER}) cannot compile "
            "for ${_blackwell}: Blackwell needs CUDA 12.8 or newer, and any CUDA 13 also "
            "qualifies.\n"
            "If the GPU you will run on is Blackwell, a newer toolkit is the only fix -- a .so "
            "without its arch loads and then fails at the first launch with 'no kernel image is "
            "available for execution on the device'.\n"
            "If it is not, drop the architectures this toolkit cannot emit:\n"
            "    FAST_ULYSSES_CUDA_ARCH=\"80;90\" pip install -e . --no-build-isolation\n"
            "Requested architectures: ${FAST_ULYSSES_ARCH_LIST}")
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
# than CMake erroring on an empty COMMAND. Missing it is not fatal -- src/work.cc then orders
# the result eagerly and group.py hands back a CompletedHandle instead.
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
        FAST_ULYSSES_CUDA_ARCH_LIST="${_arch_csv}")
target_include_directories(_C PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH REQUIRED)
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python::Module)
# torch's own lib directory is the only thing this links that is not on a default search path.
# Relative to $ORIGIN (= <site-packages>/fast_ulysses) so the wheel stays relocatable, and left as
# DT_RUNPATH -- no --disable-new-dtags -- so LD_LIBRARY_PATH can still override it.
set_target_properties(_C PROPERTIES BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "$ORIGIN/../torch/lib")
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
