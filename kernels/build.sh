#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
OUT_DIR="$SCRIPT_DIR/../src/main/scala/scalation/mathstat/libC"

mkdir -p "$OUT_DIR"

echo "==> Compiling check_cuda.c -> libcudacheck.so"
gcc -shared -fPIC -o "$OUT_DIR/libcudacheck.so" "$SRC_DIR/check_cuda.c" -ldl

echo "==> Compiling libcudakernels.cu -> libcudakernels.so"
nvcc --shared -Xcompiler -fPIC -o "$OUT_DIR/libcudakernels.so" "$SRC_DIR/libcudakernels.cu" -lcublas

echo "==> Done. Libraries written to $OUT_DIR"
