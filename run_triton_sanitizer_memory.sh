#!/bin/bash
# Run triton-sanitizer experiments for all repositories

echo "========================================"
echo "Running Triton-Sanitizer Experiments"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Check if triton-sanitizer is available
if ! command -v triton-sanitizer &> /dev/null; then
    echo "Warning: triton-sanitizer not found in PATH"
    echo "Make sure triton-sanitizer is installed and available in PATH"
    echo "Continue? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Check for whitelists
echo "Checking for whitelists..."
if [ -f "liger_kernel_whitelist.txt" ]; then
    echo "  ✓ Liger-Kernel whitelist detected (27 tests)"
fi
if [ -f "flag_gems_whitelist.txt" ]; then
    echo "  ✓ FlagGems whitelist detected (20 tests)"
fi
if [ -f "tritonbench_whitelist.txt" ]; then
    echo "  ✓ TritonBench whitelist detected (64 files)"
fi
if [ ! -f "liger_kernel_whitelist.txt" ] && [ ! -f "flag_gems_whitelist.txt" ] && [ ! -f "tritonbench_whitelist.txt" ]; then
    echo "  No whitelists found, will run all tests"
fi
echo ""

# Run triton-sanitizer tests for all repositories
echo "Starting triton-sanitizer tests..."
echo "Tool: Triton Sanitizer for Triton kernel verification"
echo "Configuration: 4 triton-sanitizer configurations with TRITON_ALWAYS_COMPILE and PYTORCH_NO_CUDA_MEMORY_CACHING variations"
echo "Repositories: liger_kernel, flag_gems, tritonbench"
echo ""

/usr/bin/time -v python3 runner.py \
    --repos all \
    --config-groups triton_sanitizer \
    --output-dir test_outputs_triton_sanitizer_${TIMESTAMP}

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Triton-sanitizer experiments completed successfully!"
    echo "Results saved in: test_outputs_triton_sanitizer_${TIMESTAMP}/"
    echo "CSV results: test_outputs_triton_sanitizer_${TIMESTAMP}/results_*.csv"
    echo ""
    echo "To check for sanitizer findings:"
    echo "  grep -i 'sanitizer\|error\|warning' test_outputs_triton_sanitizer_${TIMESTAMP}/triton_sanitizer/*/*.log"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Triton-sanitizer experiments failed or were interrupted"
    echo "Partial results may be in: test_outputs_triton_sanitizer_${TIMESTAMP}/"
    echo "========================================"
    exit 1
fi
