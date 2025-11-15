#!/bin/bash
# Run ablation study with kernel timing across all repositories
# This script measures kernel execution time with 5 different cache configurations

echo "========================================"
echo "Running Ablation Study with Kernel Timing"
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

# Base output directory
OUTPUT_BASE="test_outputs_ablation_kernel_time_${TIMESTAMP}"

# Create base output directory
mkdir -p "${OUTPUT_BASE}"

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

# Configuration explanation
echo "Ablation Study Cache Configurations:"
echo "  All using triton-sanitizer with ENABLE_TIMING=1"
echo "  Testing cache configurations (symbol, loop, grid, kernel):"
echo "    1. no_cache:          (0, 0, 0, 0)"
echo "    2. symbol_only:       (1, 0, 0, 0)"
echo "    3. symbol_loop:       (1, 1, 0, 0)"
echo "    4. symbol_loop_grid:  (1, 1, 1, 0)"
echo "    5. all_cache:         (1, 1, 1, 1)"
echo ""
echo "Repositories: Liger-Kernel, FlagGems, TritonBench"
echo ""

# Run all tests
echo "========================================"
echo "Starting Tests"
echo "========================================"
echo ""

python3 runner.py \
    --repos all \
    --config-groups ablation_studies \
    --output-dir "${OUTPUT_BASE}"

TEST_EXIT_CODE=$?

echo ""
echo "========================================"
echo "Test Runs Complete"
echo "========================================"
echo ""
echo "Test execution completed. Check the summary below for individual test results."
echo ""
echo "Results saved in: ${OUTPUT_BASE}/"
echo ""

# Run analysis script and generate CSV
echo "========================================"
echo "Analyzing Results and Generating CSV"
echo "========================================"
echo ""

python3 analyze_ablation_kernel_time.py "${OUTPUT_BASE}" --csv "ablation_kernel_timing_results.csv"

ANALYSIS_EXIT_CODE=$?

if [ ${ANALYSIS_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Analysis Complete!"
    echo "========================================"
    echo ""
    echo "Results:"
    echo "  CSV File:  ${OUTPUT_BASE}/ablation_kernel_timing_results.csv (ordered by test number)"
    echo "  Log files: ${OUTPUT_BASE}/ablation_studies/*/"
    echo ""
    echo "To view CSV:"
    echo "  cat ${OUTPUT_BASE}/ablation_kernel_timing_results.csv"
    echo "  or"
    echo "  column -t -s, ${OUTPUT_BASE}/ablation_kernel_timing_results.csv | less -S"
    echo ""
else
    echo ""
    echo "⚠ Analysis script failed, but test logs are still available in ${OUTPUT_BASE}/"
    echo ""
fi

echo "========================================"

# Exit with analysis status (tests always succeed in runner.py)
exit ${ANALYSIS_EXIT_CODE}
