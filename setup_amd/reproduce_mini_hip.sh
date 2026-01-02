#!/bin/bash

# Script to reproduce mini HIP example with ASAN support

# Set environment variables for ASAN and ROCm
export HSA_XNACK=1
export LD_LIBRARY_PATH=/opt/rocm/lib/asan:$LD_LIBRARY_PATH
export PATH=/opt/rocm/llvm/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm/llvm/lib/clang/20/lib/linux:$LD_LIBRARY_PATH

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Compile mini.hip with ASAN
hipcc -g --offload-arch=gfx942:xnack+ -fsanitize=address -shared-libsan "$SCRIPT_DIR/mini.hip" -o "$SCRIPT_DIR/mini"

# Run the compiled binary
"$SCRIPT_DIR/mini" 100 11 10 100
