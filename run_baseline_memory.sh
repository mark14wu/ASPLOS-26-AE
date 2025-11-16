#!/bin/bash
# Run baseline experiments for all repositories

echo "========================================"
echo "Running Baseline Experiments"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed or not in PATH"
    exit 1
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

# Run baseline tests for all repositories
echo "Starting baseline tests..."
echo "Configuration: 4 baseline configurations with TRITON_ALWAYS_COMPILE and PYTORCH_NO_CUDA_MEMORY_CACHING variations"
echo "Repositories: liger_kernel, flag_gems, tritonbench"
echo ""

/usr/bin/time -v python3 runner.py \
    --repos all \
    --config-groups baseline \
    --output-dir test_outputs_baseline_${TIMESTAMP}

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Baseline experiments completed successfully!"
    echo "Results saved in: test_outputs_baseline_${TIMESTAMP}/"
    echo "CSV results: test_outputs_baseline_${TIMESTAMP}/results_*.csv"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "Baseline experiments failed or were interrupted"
    echo "Partial results may be in: test_outputs_baseline_${TIMESTAMP}/"
    echo "========================================"
    exit 1
fi
