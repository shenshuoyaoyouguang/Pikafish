#!/bin/bash
# ===========================================================================
# Pikafish 单元测试构建运行脚本
# ===========================================================================
# 用法：cd tests && bash run_tests.sh
#
# 功能：
#   1. 用 CMake 配置构建目录（FetchContent 下载 GoogleTest）
#   2. 编译 pikafish_core 静态库 + bench runner + 3 个测试可执行文件
#   3. 用 ctest 运行所有测试
#
# 环境要求：
#   - CMake >= 3.14
#   - C++17 编译器（g++ >= 9.3 / clang++ >= /msvc >= 19.14）
#   - 网络连接（首次运行需下载 GoogleTest）
#   - Pikafish 源码在 ../src
# ===========================================================================

set -e

# 切换到脚本所在目录
cd "$(dirname "$0")"

BUILD_DIR=${BUILD_DIR:-build}
JOBS=$(nproc 2>/dev/null || echo 2)

echo "============================================================"
echo " Pikafish 单元测试"
echo "   构建目录: $BUILD_DIR"
echo "   并行任务: $JOBS"
echo "============================================================"

# 检查 cmake
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake not found. Please install CMake >= 3.14."
    exit 1
fi

# 检查 Pikafish 源码
if [ ! -f ../src/position.cpp ]; then
    echo "ERROR: Pikafish source not found at ../src/position.cpp"
    exit 1
fi

echo ""
echo "=== [1/3] CMake Configure ==="
# 在 Windows/MSYS2 下，CMake 默认可能选 MSVC（无法编译 Pikafish 的 GCC 扩展），
# 因此显式指定 MSYS Makefiles generator + g++ 编译器。
# 在 Linux/macOS 下使用默认 generator。
if [ "$(uname -s 2>/dev/null | cut -c1-5)" = "MINGW" ] || [ "$(uname -s 2>/dev/null | cut -c1-4)" = "MSYS" ]; then
    cmake -B "$BUILD_DIR" -S . -G "MSYS Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++ \
        -DCMAKE_C_COMPILER=gcc
else
    cmake -B "$BUILD_DIR" -S . -DCMAKE_BUILD_TYPE=Release
fi

echo ""
echo "=== [2/3] Build ==="
cmake --build "$BUILD_DIR" -j "$JOBS"

echo ""
echo "=== [3/3] Test ==="
(cd "$BUILD_DIR" && ctest --output-on-failure)

echo ""
echo "============================================================"
echo " 测试完成"
echo "============================================================"