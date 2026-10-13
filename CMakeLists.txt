cmake_minimum_required(VERSION 3.20)
project(TinyDL LANGUAGES CXX)

set(TINYDL_DCU_DEFAULT OFF)
if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    set(TINYDL_DCU_DEFAULT ON)
endif()
option(TINYDL_ENABLE_DCU "Build the DTK/HIP DCU backend" ${TINYDL_DCU_DEFAULT})
option(TINYDL_BUILD_TESTS "Build TinyDL tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/sentencepiece/CMakeLists.txt")
    # Vendored for reproducible builds on compute nodes without Internet access.
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    add_subdirectory(extern/sentencepiece)
else()
    find_package(sentencepiece CONFIG REQUIRED)
endif()

if(TARGET sentencepiece::sentencepiece)
    set(TINYDL_SENTENCEPIECE_TARGET sentencepiece::sentencepiece)
elseif(TARGET sentencepiece-static)
    set(TINYDL_SENTENCEPIECE_TARGET sentencepiece-static)
elseif(TARGET sentencepiece)
    set(TINYDL_SENTENCEPIECE_TARGET sentencepiece)
else()
    message(FATAL_ERROR "No usable SentencePiece CMake target was found")
endif()

add_library(tinydl
    src/tensor.cpp
    src/backend.cpp
    src/autograd.cpp
    src/nn.cpp
    src/optimizer.cpp
    src/tokenizer.cpp
    src/checkpoint.cpp
    src/cpu/cpu_backend.cpp
)
target_include_directories(tinydl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extern/sentencepiece/src")
    target_include_directories(tinydl PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/sentencepiece/src)
endif()
target_link_libraries(tinydl PUBLIC ${TINYDL_SENTENCEPIECE_TARGET})
if(MSVC)
    target_compile_options(tinydl PUBLIC /utf-8)
endif()

if(TINYDL_ENABLE_DCU)
    if(NOT DEFINED ENV{ROCM_PATH} OR "$ENV{ROCM_PATH}" STREQUAL "")
        message(FATAL_ERROR
            "ROCM_PATH is not set. Load the DTK environment before configuring TinyDL.")
    endif()
    set(TINYDL_DTK_ROOT "$ENV{ROCM_PATH}")

    if(NOT IS_DIRECTORY "${TINYDL_DTK_ROOT}")
        message(FATAL_ERROR
            "ROCM_PATH does not point to a directory: ${TINYDL_DTK_ROOT}. Load the DTK environment again.")
    endif()
    if(NOT EXISTS "${TINYDL_DTK_ROOT}/llvm/bin/clang++")
        message(FATAL_ERROR
            "ROCM_PATH does not contain llvm/bin/clang++: ${TINYDL_DTK_ROOT}")
    endif()

    message(STATUS "TinyDL DTK root from ROCM_PATH: ${TINYDL_DTK_ROOT}")
    list(PREPEND CMAKE_PREFIX_PATH "${TINYDL_DTK_ROOT}" "${TINYDL_DTK_ROOT}/hip")
    if(NOT CMAKE_HIP_COMPILER)
        set(CMAKE_HIP_COMPILER "${TINYDL_DTK_ROOT}/llvm/bin/clang++"
            CACHE FILEPATH "HIP compiler" FORCE)
    endif()
    if(NOT CMAKE_HIP_COMPILER_ROCM_ROOT)
        set(CMAKE_HIP_COMPILER_ROCM_ROOT "${TINYDL_DTK_ROOT}"
            CACHE PATH "HIP compiler ROCm root" FORCE)
    endif()

    if(NOT CMAKE_HIP_ARCHITECTURES)
        find_program(TINYDL_ROCMINFO rocminfo HINTS "${TINYDL_DTK_ROOT}/bin")
        if(TINYDL_ROCMINFO)
            execute_process(
                COMMAND ${TINYDL_ROCMINFO}
                OUTPUT_VARIABLE TINYDL_ROCMINFO_OUTPUT
                RESULT_VARIABLE TINYDL_ROCMINFO_RESULT
                ERROR_QUIET)
            if(TINYDL_ROCMINFO_RESULT EQUAL 0)
                string(REGEX MATCHALL "Name:[ \t]+gfx[0-9A-Za-z]+"
                    TINYDL_ARCH_LINES "${TINYDL_ROCMINFO_OUTPUT}")
                foreach(TINYDL_ARCH_LINE IN LISTS TINYDL_ARCH_LINES)
                    string(REGEX REPLACE ".*(gfx[0-9A-Za-z]+).*" "\\1"
                        TINYDL_ARCH "${TINYDL_ARCH_LINE}")
                    if(NOT TINYDL_ARCH STREQUAL "gfx000")
                        list(APPEND TINYDL_DETECTED_ARCHITECTURES "${TINYDL_ARCH}")
                    endif()
                endforeach()
                list(REMOVE_DUPLICATES TINYDL_DETECTED_ARCHITECTURES)
                if(TINYDL_DETECTED_ARCHITECTURES)
                    set(CMAKE_HIP_ARCHITECTURES "${TINYDL_DETECTED_ARCHITECTURES}"
                        CACHE STRING "HIP GPU architectures" FORCE)
                    message(STATUS "TinyDL detected HIP architecture: ${CMAKE_HIP_ARCHITECTURES}")
                endif()
            endif()
        endif()
    endif()

    include(CheckLanguage)
    check_language(HIP)
    if(NOT CMAKE_HIP_COMPILER)
        message(FATAL_ERROR "DTK/HIP compiler was not found; load the DTK environment first")
    endif()
    enable_language(HIP)
    find_package(hip REQUIRED)
    find_package(hipblas REQUIRED)
    target_sources(tinydl PRIVATE src/dcu/dcu_backend.hip)
    set_property(TARGET tinydl PROPERTY HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    target_compile_definitions(tinydl PUBLIC TINYDL_ENABLE_DCU=1)
    target_link_libraries(tinydl PUBLIC hip::host roc::hipblas)
    set_source_files_properties(src/dcu/dcu_backend.hip PROPERTIES LANGUAGE HIP)
endif()

add_executable(tinydl_demo apps/demo.cpp)
target_link_libraries(tinydl_demo PRIVATE tinydl)

add_executable(tinydl_train apps/train.cpp)
target_link_libraries(tinydl_train PRIVATE tinydl)

add_executable(tinydl_generate apps/generate.cpp)
target_link_libraries(tinydl_generate PRIVATE tinydl)

if(TINYDL_BUILD_TESTS)
    enable_testing()
    add_executable(tinydl_tests tests/test_cpu.cpp)
    target_link_libraries(tinydl_tests PRIVATE tinydl)
    add_test(NAME tinydl_cpu_tests COMMAND tinydl_tests)

    add_executable(tinydl_autograd_tests tests/test_autograd.cpp)
    target_link_libraries(tinydl_autograd_tests PRIVATE tinydl)
    add_test(NAME tinydl_autograd_tests COMMAND tinydl_autograd_tests)
endif()
