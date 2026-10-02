# ===========================================================================
# Pikafish 单元测试构建配置
# ===========================================================================
# 引入 GoogleTest（FetchContent），编译 Pikafish 核心（不含 main.cpp）为静态库，
# 链接测试目标。LMR θ₀ 等价性测试通过子进程运行完整 pikafish（bench runner）。
#
# 用法：
#   cd tests && bash run_tests.sh
# 或手动：
#   cmake -B build -S tests -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   (cd build && ctest --output-on-failure)
# ===========================================================================

cmake_minimum_required(VERSION 3.14)
project(PikafishTests CXX C ASM)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------------------
# Pikafish 源码目录
# ---------------------------------------------------------------------------
set(PK_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src CACHE PATH "Pikafish src directory")

# ---------------------------------------------------------------------------
# 编译选项（模拟 src/Makefile 的 x86-64-avx2 默认配置）
#   - PUBLIC 选项传递给链接本库的测试目标（inline 函数需要相同的 SIMD 特性）
#   - PRIVATE 选项仅作用于 pikafish_core 自身
# ---------------------------------------------------------------------------
set(PK_PUBLIC_FLAGS
    -DNDEBUG -DIS_64BIT
    -msse -msse2 -mssse3 -msse4.1 -mavx2 -mbmi -mbmi2 -mpopcnt
    -DUSE_SSE2 -DUSE_SSSE3 -DUSE_SSE41 -DUSE_AVX2 -DUSE_POPCNT -DUSE_PEXT
)
set(PK_PRIVATE_FLAGS -fno-exceptions -w)

# ---------------------------------------------------------------------------
# Pikafish 源文件（排除 main.cpp 避免与 gtest_main 冲突，排除 universal/ 多架构入口）
# 包含 .cpp 和 .S（汇编，如 zstd 的 huf_decompress_amd64.S）
# ---------------------------------------------------------------------------
file(GLOB PK_CORE_SRCS CONFIGURE_DEPENDS
    ${PK_SRC_DIR}/*.cpp
    ${PK_SRC_DIR}/*.S
    ${PK_SRC_DIR}/nnue/*.cpp
    ${PK_SRC_DIR}/nnue/features/*.cpp
    ${PK_SRC_DIR}/external/common/*.cpp
    ${PK_SRC_DIR}/external/decompress/*.cpp
    ${PK_SRC_DIR}/external/decompress/*.S
)
list(REMOVE_ITEM PK_CORE_SRCS "${PK_SRC_DIR}/main.cpp")

# ---------------------------------------------------------------------------
# pikafish_core 静态库
# ---------------------------------------------------------------------------
add_library(pikafish_core STATIC ${PK_CORE_SRCS})
target_include_directories(pikafish_core PUBLIC ${PK_SRC_DIR})
target_compile_options(pikafish_core PRIVATE ${PK_PRIVATE_FLAGS} PUBLIC ${PK_PUBLIC_FLAGS})

find_package(Threads REQUIRED)
target_link_libraries(pikafish_core PUBLIC Threads::Threads)

# ---------------------------------------------------------------------------
# pikafish_bench_runner —— 完整 pikafish 可执行文件（复用 src/main.cpp，不修改源文件）
# 用于 LMR θ₀ 等价性测试的子进程调用
# ---------------------------------------------------------------------------
add_executable(pikafish_bench_runner ${PK_SRC_DIR}/main.cpp)
target_link_libraries(pikafish_bench_runner PRIVATE pikafish_core)

# 复制 NNUE 网络文件到构建目录（bench runner 运行时需在同目录找到 pikafish.nnue）
if(EXISTS ${PK_SRC_DIR}/pikafish.nnue)
    configure_file(${PK_SRC_DIR}/pikafish.nnue ${CMAKE_BINARY_DIR}/pikafish.nnue COPYONLY)
else()
    message(WARNING "pikafish.nnue not found in ${PK_SRC_DIR}; bench runner may fail to load NNUE network.")
endif()

# ---------------------------------------------------------------------------
# GoogleTest（FetchContent，需 CMake >= 3.14，避免 git submodule）
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE
)
# 避免在 MSVC 上覆盖父项目的 CRT 设置
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# 抑制 gtest 自身的编译警告
set(GOOGLETEST_DISABLE_WARNINGS ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---------------------------------------------------------------------------
# 测试辅助函数
# ---------------------------------------------------------------------------
function(pk_add_test name src)
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE pikafish_core GTest::gtest_main)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ---------------------------------------------------------------------------
# 1. position.cpp FEN round-trip 测试
# ---------------------------------------------------------------------------
pk_add_test(test_position test_position.cpp)

# ---------------------------------------------------------------------------
# 2. movegen.cpp perft 测试
# ---------------------------------------------------------------------------
pk_add_test(test_movegen test_movegen.cpp)

# ---------------------------------------------------------------------------
# 3. search.cpp LMR θ₀ 等价性测试（子进程运行 bench runner）
# ---------------------------------------------------------------------------
add_executable(test_search test_search.cpp)
target_link_libraries(test_search PRIVATE GTest::gtest_main)
# 通过 configure_file 生成头文件传递 bench runner 绝对路径（避免命令行引号问题）
set(BENCH_RUNNER_PATH_VALUE "${CMAKE_BINARY_DIR}/pikafish_bench_runner${CMAKE_EXECUTABLE_SUFFIX}")
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_runner_path.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/bench_runner_path.h
    @ONLY)
target_include_directories(test_search PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME search COMMAND test_search)
# bench depth 13 可能需要数十秒，设置 10 分钟超时
set_tests_properties(search PROPERTIES TIMEOUT 600)