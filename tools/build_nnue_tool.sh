#!/bin/bash
# Build script for nnue_tool - run inside WSL
# Must be run AFTER pikafish has been built (reuses its .o files)
set -e
cd /mnt/e/xiaoxiao/pikayu/Pikafish/src

# Architecture flags matching the pikafish binary (x86-64-avxvnni)
ARCH_FLAGS="-m64 -msse -msse2 -mssse3 -msse4.1 -mavx2 -mbmi -mbmi2 -mpopcnt -msse3 -mavxvnni"
DEF_FLAGS="-DUSE_SSE2 -DUSE_SSSE3 -DUSE_SSE41 -DUSE_AVX2 -DUSE_POPCNT -DUSE_PEXT -DUSE_VNNI -DUSE_AVXVNNI -DIS_64BIT -DNDEBUG -DTRAINING_TOOL -DARCH=x86-64-avxvnni"
CXXFLAGS="-std=c++17 -O2 $ARCH_FLAGS $DEF_FLAGS -fno-exceptions -Wno-unused-command-line-argument -I."

echo "=== Compiling nnue_tool.cpp ==="
g++ $CXXFLAGS -flto -c ../tools/nnue_tool.cpp -o nnue_tool.o
echo "Compile OK"

echo "=== Linking nnue_tool ==="
# Collect all .o files except main.o and nnue_tool.o
OBJS=""
for f in *.o; do
    if [ "$f" != "main.o" ] && [ "$f" != "nnue_tool.o" ]; then
        OBJS="$OBJS $f"
    fi
done
echo "Linking with $(echo $OBJS | wc -w) object files"

# Use -flto to properly link the LTO-compiled .o files
g++ -flto -O2 -m64 -mavxvnni -o nnue_tool nnue_tool.o $OBJS -lpthread -lrt -Wl,--no-as-needed
echo "Link OK"
ls -la nnue_tool