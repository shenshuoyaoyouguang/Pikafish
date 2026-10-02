# CMakeLists.txt - Optional CMake build for Pikafish
#
# This is an ALTERNATIVE build system to src/Makefile. The Makefile remains the
# canonical build; CMake is provided for IDE integration (compile_commands.json)
# and users who prefer CMake.
#
# Usage:
#   cmake -B build -DARCH=x86-64-avx2
#   cmake --build build -j
#   ./build/pikafish bench     # run benchmark
#
# Supported ARCH values (see `make help` in src/ for the full list):
#   x86-64-modern      (default, alias of x86-64-sse41-popcnt)
#   x86-64-sse41-popcnt
#   x86-64-avx2
#   x86-64-avxvnni
#   x86-64-bmi2
#   x86-64-avx512
#   x86-64-vnni512
#   x86-64-avx512icl
#   x86-64-ssse3
#   x86-64
#   x86-64-native       (let the compiler pick the host CPU features)
#
# This file does NOT modify src/Makefile.

cmake_minimum_required(VERSION 3.16)

project(Pikafish LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

# ARCH selection. Mirrors the Makefile ARCH=... flag.
set(ARCH "x86-64-modern" CACHE STRING "Target architecture (see src/Makefile for the full list)")

# Extra knobs
option(PIKAFISH_LTO "Enable link-time optimization" ON)
option(PIKAFISH_USE_PTHREAD "Link pthread" ON)

# Generate compile_commands.json for clangd / IDE integration.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Architecture flag mapping (mirrors src/Makefile Section 2.2 / 3.x)
# ---------------------------------------------------------------------------
#
# Each architecture sets:
#   ARCH_DEFS    - preprocessor defines (-D...)
#   ARCH_MFLAGS  - machine flags (-m...)
#
# The mapping intentionally follows the Makefile so that `cmake -DARCH=X`
# produces the same code paths as `make build ARCH=X`.

set(ARCH_DEFS "")
set(ARCH_MFLAGS "")

# Helper: 64-bit x86 baseline (sse + sse2 + prefetch)
macro(_x86_64_baseline)
  list(APPEND ARCH_DEFS  IS_64BIT USE_SSE2)
  list(APPEND ARCH_MFLAGS -msse -msse2)
endmacro()

if(ARCH STREQUAL "x86-64-native")
  # Let the compiler detect host features; keep defines minimal so the
  # runtime dispatch paths match a plain `make ARCH=native` best-effort.
  _x86_64_baseline()
  list(APPEND ARCH_MFLAGS -march=native)

elseif(ARCH STREQUAL "x86-64")
  _x86_64_baseline()

elseif(ARCH STREQUAL "x86-64-ssse3")
  _x86_64_baseline()
  list(APPEND ARCH_DEFS USE_SSSE3)
  list(APPEND ARCH_MFLAGS -mssse3)

elseif(ARCH STREQUAL "x86-64-sse41-popcnt" OR ARCH STREQUAL "x86-64-modern")
  # x86-64-modern is deprecated in the Makefile and maps to sse41-popcnt.
  _x86_64_baseline()
  list(APPEND ARCH_DEFS USE_SSSE3 USE_SSE41 USE_POPCNT)
  list(APPEND ARCH_MFLAGS -mssse3 -msse4.1 -msse3 -mpopcnt)

elseif(ARCH STREQUAL "x86-64-avx2")
  _x86_64_baseline()
  list(APPEND ARCH_DEFS USE_SSSE3 USE_SSE41 USE_POPCNT USE_AVX2)
  list(APPEND ARCH_MFLAGS -mssse3 -msse4.1 -msse3 -mpopcnt -mavx2 -mbmi)

elseif(ARCH STREQUAL "x86-64-avxvnni")
  _x86_64_baseline()
  list(APPEND ARCH_DEFS USE_SSSE3 USE_SSE41 USE_POPCNT USE_AVX2 USE_VNNI USE_AVXVNNI USE_PEXT)
  list(APPEND ARCH_MFLAGS -mssse3 -msse4.1 -msse3 -mpopcnt -mavx2 -mbmi -mavxvnni -mbmi2)

elseif(ARCH STREQUAL "x86-64-bmi2")
  _x86_64_baseline()
  list(APPEND ARCH_DEFS USE_SSSE3 USE_SSE41 USE_POPCNT USE_AVX2 USE_PEXT)
  list(APPEND ARCH_MFLAGS -mssse3 -msse4.1 -msse3 -mpopcnt -mavx2 -mbmi -mbmi2)

elseif(ARCH STREQUAL "x86-64-avx512")
  _x86_64_baseline()
  list(APPEND ARCH_DEFS USE_SSSE3 USE_SSE41 USE_POPCNT USE_AVX2 USE_PEXT USE_AVX512)
  list(APPEND ARCH_MFLAGS -mssse3 -msse4.1 -msse3 -mpopcnt -mavx2 -mbmi -mbmi2
                           -mavx512f -mavx512bw -mavx512dq -mavx512vl)

elseif(ARCH STREQUAL "x86-64-vnni512")
  _x86_64_baseline()
  list(APPEND ARCH_DEFS USE_SSSE3 USE_SSE41 USE_POPCNT USE_AVX2 USE_PEXT USE_AVX512 USE_VNNI)
  list(APPEND ARCH_MFLAGS -mssse3 -msse4.1 -msse3 -mpopcnt -mavx2 -mbmi -mbmi2
                           -mavx512f -mavx512bw -mavx512vnni -mavx512dq -mavx512vl)

elseif(ARCH STREQUAL "x86-64-avx512icl")
  _x86_64_baseline()
  list(APPEND ARCH_DEFS USE_SSSE3 USE_SSE41 USE_POPCNT USE_AVX2 USE_PEXT
                        USE_AVX512 USE_VNNI USE_AVX512ICL)
  list(APPEND ARCH_MFLAGS -mssse3 -msse4.1 -msse3 -mpopcnt -mavx2 -mbmi -mbmi2
                           -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw
                           -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vpopcntdq
                           -mavx512bitalg -mavx512vnni -mvpclmulqdq -mgfni -mvaes)

else()
  message(WARNING
    "ARCH='${ARCH}' is not explicitly mapped in CMakeLists.txt; falling back to "
    "x86-64-modern. For the full architecture matrix use src/Makefile.")
  _x86_64_baseline()
  list(APPEND ARCH_DEFS USE_SSSE3 USE_SSE41 USE_POPCNT)
  list(APPEND ARCH_MFLAGS -mssse3 -msse4.1 -msse3 -mpopcnt)
  set(ARCH "x86-64-modern")
endif()

# Pass the selected ARCH as a define (Makefile does -DARCH=$(ARCH)).
list(APPEND ARCH_DEFS ARCH=${ARCH})

message(STATUS "Pikafish ARCH: ${ARCH}")
message(STATUS "  defines: ${ARCH_DEFS}")
message(STATUS "  machine flags: ${ARCH_MFLAGS}")

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
#
# Mirrors src/Makefile:
#   SRCS = find . \( -path ./universal -o -path ./temp_builds \) -prune \
#          -o \( -name '*.cpp' -o -name '*.S' \) -print
#
# i.e. everything under src/ EXCEPT src/universal/, including src/external/
# (zstd) and src/nnue/. The Python binding (tools/nnue_pybind.cpp) lives
# outside src/ and is intentionally excluded.

file(GLOB PIKAFISH_SRC_TOP       CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB PIKAFISH_SRC_NNUE      CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/nnue/*.cpp")
file(GLOB PIKAFISH_SRC_NNUE_FEAT CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/nnue/features/*.cpp")
file(GLOB PIKAFISH_SRC_EXT_COMMON   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/external/common/*.cpp")
file(GLOB PIKAFISH_SRC_EXT_DECOMP   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/external/decompress/*.cpp")

set(PIKAFISH_SOURCES
    ${PIKAFISH_SRC_TOP}
    ${PIKAFISH_SRC_NNUE}
    ${PIKAFISH_SRC_NNUE_FEAT}

    ${PIKAFISH_SRC_EXT_COMMON}
    ${PIKAFISH_SRC_EXT_DECOMP}
)

# The amd64 assembly file is only relevant for x86-64 targets and uses GNU
# assembler syntax, so it is only enabled for GCC/Clang/MinGW. MSVC users fall
# back to the pure-C++ zstd decompression path.
if(ARCH MATCHES "^x86-64")
  set(_AMD64_ASM "${CMAKE_CURRENT_SOURCE_DIR}/src/external/decompress/huf_decompress_amd64.S")
  if(EXISTS "${_AMD64_ASM}")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
      list(APPEND PIKAFISH_SOURCES "${_AMD64_ASM}")
      enable_language(ASM)
      set_source_files_properties("${_AMD64_ASM}" PROPERTIES LANGUAGE ASM)
    else()
      message(STATUS "Skipping huf_decompress_amd64.S (incompatible with "
                      "${CMAKE_CXX_COMPILER_ID}); using C fallback.")
    endif()
  endif()
endif()

# Sanity check: make sure we actually found sources.
if(NOT PIKAFISH_SOURCES)
  message(FATAL_ERROR "No Pikafish sources found under src/. Expected files like src/main.cpp.")
endif()

list(LENGTH PIKAFISH_SOURCES _src_count)
message(STATUS "Pikafish sources: ${_src_count} translation units")

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------

# Base flags shared with the Makefile: -Wall -Wcast-qual -fno-exceptions -std=c++17
add_compile_options(
    -Wall
    -Wcast-qual
    -fno-exceptions
)

# Release / optimization flags.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops")

# Architecture-specific flags (applied to the target below).
# Convert the define list into -D arguments.
set(_ARCH_DEFINE_FLAGS "")
foreach(_d IN LISTS ARCH_DEFS)
  list(APPEND _ARCH_DEFINE_FLAGS "-D${_d}")
endforeach()

# ---------------------------------------------------------------------------
# Executable target
# ---------------------------------------------------------------------------

add_executable(pikafish ${PIKAFISH_SOURCES})

target_include_directories(pikafish PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/external"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/external/common"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/external/decompress"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/nnue"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/nnue/features"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/nnue/layers"
)

target_compile_options(pikafish PRIVATE
    ${_ARCH_DEFINE_FLAGS}
    ${ARCH_MFLAGS}
)

# Threading. The Makefile links -lpthread on Linux/Haiku (and relies on the
# pthread API inside the engine). Windows (MSVC/MinGW) does not need the lib.
if(PIKAFISH_USE_PTHREAD)
  if(UNIX AND NOT APPLE)
    target_link_libraries(pikafish PRIVATE pthread rt)
  elseif(APPLE)
    target_link_libraries(pikafish PRIVATE pthread)
  endif()
  # MinGW / MSVC: pthread is part of the runtime; no extra link line needed.
endif()

# Link-time optimization (mirrors Makefile -flto). Only enable for Release
# builds with a known compiler.
if(PIKAFISH_LTO AND CMAKE_BUILD_TYPE STREQUAL "Release")
  include(CheckIPOption)
  check_ipo_support(RESULT _lto_supported OUTPUT _lto_err)
  if(_lto_supported)
    set_property(TARGET pikafish PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "LTO: enabled")
  else()
    message(STATUS "LTO: not available (${_lto_err})")
  endif()
endif()

# ---------------------------------------------------------------------------
# Convenience targets
# ---------------------------------------------------------------------------

# `make bench` / `cmake --build build --target bench` runs the built-in
# benchmark, matching `make bench` in src/Makefile.
add_custom_target(bench
    COMMAND $<TARGET_FILE:pikafish> bench
    DEPENDS pikafish
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running Pikafish built-in benchmark"
    VERBATIM
)

# `make net` is a no-op here; the nets are expected to be present under src/.
# Users should run `make net` from src/ to fetch them.

# ---------------------------------------------------------------------------
# Install (optional, mirrors `make install`)
# ---------------------------------------------------------------------------

include(GNUInstallDirs)
install(TARGETS pikafish RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})