#!/bin/bash
# Build script for nnue_pybind (Python extension module) - run inside WSL
#
# Prerequisites:
#   1. pikafish has been built in src/ (produces the .o files we link against)
#   2. pip install pybind11
#
# Usage:
#   bash tools/build_nnue_pybind.sh
#
# Output:
#   src/nnue_pybind<extension-suffix>  (importable as `import nnue_pybind`)
set -e
cd /mnt/e/xiaoxiao/pikayu/Pikafish/src

# --- Architecture flags matching the pikafish binary (x86-64-avxvnni) ---
ARCH_FLAGS="-m64 -msse -msse2 -mssse3 -msse4.1 -mavx2 -mbmi -mbmi2 -mpopcnt -msse3 -mavxvnni"
DEF_FLAGS="-DUSE_SSE2 -DUSE_SSSE3 -DUSE_SSE41 -DUSE_AVX2 -DUSE_POPCNT -DUSE_PEXT -DUSE_VNNI -DUSE_AVXVNNI -DIS_64BIT -DNDEBUG -DTRAINING_TOOL -DARCH=x86-64-avxvnni"
CXXFLAGS="-std=c++17 -O2 $ARCH_FLAGS $DEF_FLAGS -fno-exceptions -Wno-unused-command-line-argument -I."

# --- Python / pybind11 include paths ---
PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INC=$(python3 -c "import pybind11; print(pybind11.get_include())")
CXXFLAGS="$CXXFLAGS -I$PYTHON_INC -I$PYBIND_INC -fPIC"

# --- Determine the correct extension suffix (e.g. .cpython-312-x86_64-linux-gnu.so) ---
EXT_SUFFIX=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
if [ -z "$EXT_SUFFIX" ]; then
    EXT_SUFFIX=".so"
fi
OUTPUT="nnue_pybind${EXT_SUFFIX}"

echo "=== Build configuration ==="
echo "Python include: $PYTHON_INC"
echo "pybind11 include: $PYBIND_INC"
echo "Output module:   $OUTPUT"
echo ""

# --- Sanity check: pikafish .o files must exist ---
if ! ls *.o >/dev/null 2>&1; then
    echo "ERROR: No .o files found in src/. Build pikafish first (make pikafish)."
    exit 1
fi

echo "=== Compiling nnue_pybind.cpp ==="
# pybind11 relies on C++ exceptions (throw/catch), so we must compile this
# translation unit with -fexceptions even though the engine .o files were
# built with -fno-exceptions.  Exception handling is per-TU, so mixing is safe
# (the engine code never throws or catches).
PYBIND_CXXFLAGS="${CXXFLAGS//-fno-exceptions/-fexceptions}"
g++ $PYBIND_CXXFLAGS -c ../tools/nnue_pybind.cpp -o nnue_pybind.o
echo "Compile OK"

echo "=== Linking $OUTPUT ==="
# Collect all engine .o files except the ones with their own main():
#   main.o        (engine entry point)
#   nnue_tool.o   (weight extraction tool)
#   nnue_pybind.o (this module, linked explicitly)
OBJS=""
for f in *.o; do
    if [ "$f" != "main.o" ] && [ "$f" != "nnue_tool.o" ] && [ "$f" != "nnue_pybind.o" ]; then
        OBJS="$OBJS $f"
    fi
done
echo "Linking with $(echo $OBJS | wc -w) engine object files"

# Produce a Python extension module (position-independent shared object).
g++ -shared -fPIC -O2 -m64 -mavxvnni \
    -o "$OUTPUT" nnue_pybind.o $OBJS \
    -lpthread -lrt -Wl,--no-as-needed
echo "Link OK"
ls -la "$OUTPUT"

echo ""
echo "=== Done ==="
echo "Import from Python with:"
echo "    import sys; sys.path.insert(0, '$(pwd)')"
echo "    import nnue_pybind"