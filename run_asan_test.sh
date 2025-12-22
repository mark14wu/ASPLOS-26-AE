#!/bin/bash

ulimit -s 1024

export PATH=$(find ~/.triton/llvm -name llvm-symbolizer -printf '%h\n'):$PATH

TORCH_PATH=$(find /opt -name libcaffe2_nvrtc.so -printf '%h\n')
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TORCH_PATH

# Ensure libamdhip64.so is restored on exit/interrupt
cleanup() {
    if [ -f "$TORCH_PATH/libamdhip64_bck.so" ]; then
        mv $TORCH_PATH/libamdhip64_bck.so $TORCH_PATH/libamdhip64.so
    fi
}
trap cleanup EXIT

mv $TORCH_PATH/libamdhip64.so $TORCH_PATH/libamdhip64_bck.so

export LD_LIBRARY_PATH=$(find /opt -name libclang_rt.asan-x86_64.so -printf '%h\n'):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(find /opt -type d -wholename *lib/llvm/lib/asan):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(find /opt -wholename *lib/asan/libamdhip64.so -printf '%h\n'):$LD_LIBRARY_PATH

export CLANG_ASAN_LIB=$(find /opt -name libclang_rt.asan-x86_64.so)
export HIP_ASAN_LIB=$(find /opt -wholename *lib/asan/libamdhip64.so)

export HSA_DISABLE_FRAGMENT_ALLOCATOR=1
export AMD_PYTORCH_NO_CUDA_MEMORY_CACHING=1
export PYTORCH_NO_HIP_MEMORY_CACHING=1
export TRITON_ENABLE_ASAN=1

# HSA_XNACK here is required to set the xnack+ setting for the GPU at runtime.
# If it is not set and the default xnack setting of the system is xnack-
# a runtime error something like "No kernel image found" will occur. The system
# xnack setting can be found through rocminfo. xnack+ is required for ASAN.
# More information about xnack in general can be found here:
# https://llvm.org/docs/AMDGPUUsage.html#target-features
# https://rocm.docs.amd.com/en/docs-6.1.0/conceptual/gpu-memory.html
export HSA_XNACK=1

# Disable buffer ops given it has builtin support for out of bound access.
export AMDGCN_USE_BUFFER_OPS=0

ASAN_OPTIONS=detect_leaks=0,alloc_dealloc_mismatch=0 \
LD_PRELOAD=$CLANG_ASAN_LIB:$HIP_ASAN_LIB python test_asan.py
