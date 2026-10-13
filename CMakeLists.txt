cmake_minimum_required(VERSION 3.20)
project(TinyDL LANGUAGES CXX)

option(TINYDL_ENABLE_DCU "Build the DTK/HIP DCU backend" OFF)
option(TINYDL_BUILD_TESTS "Build TinyDL tests" ON)
option(TINYDL_FETCH_SENTENCEPIECE "Download SentencePiece when it is not installed" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(sentencepiece CONFIG QUIET)
if(NOT sentencepiece_FOUND)
    if(NOT TINYDL_FETCH_SENTENCEPIECE)
        message(FATAL_ERROR "SentencePiece was not found; install it or enable TINYDL_FETCH_SENTENCEPIECE")
    endif()
    include(FetchContent)
    # SentencePiece 0.2.0 predates CMake 4's removal of compatibility below 3.5.
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(sentencepiece
        URL https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz)
    FetchContent_MakeAvailable(sentencepiece)
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
if(DEFINED sentencepiece_SOURCE_DIR)
    target_include_directories(tinydl PUBLIC ${sentencepiece_SOURCE_DIR}/src)
endif()
target_link_libraries(tinydl PUBLIC ${TINYDL_SENTENCEPIECE_TARGET})
if(MSVC)
    target_compile_options(tinydl PUBLIC /utf-8)
endif()

if(TINYDL_ENABLE_DCU)
    include(CheckLanguage)
    check_language(HIP)
    if(NOT CMAKE_HIP_COMPILER)
        message(FATAL_ERROR "DTK/HIP compiler was not found; load the DTK environment first")
    endif()
    enable_language(HIP)
    find_package(hip REQUIRED)
    find_package(hipblas REQUIRED)
    target_sources(tinydl PRIVATE src/dcu/dcu_backend.hip)
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
