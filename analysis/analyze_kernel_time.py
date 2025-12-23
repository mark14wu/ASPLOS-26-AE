#!/usr/bin/env python3
"""
Analyze kernel timing results from Liger-Kernel tests.
Parses output files and calculates total GPU time for:
1. Baseline and compute-sanitizer: Sum of gpu_time_ms
2. Triton-sanitizer: Sum of execution time in ms
"""

import os
import re
import sys
import csv
from pathlib import Path
import argparse
from collections import defaultdict


def parse_baseline_compute_sanitizer(log_file):
    """
    Parse baseline or compute-sanitizer log file for gpu_time_ms.

    Supports two formats:
    1. [liger][triton] kernel=_jsd_kernel cpu_launch_ms=3.136 gpu_time_ms=119.417
    2. [triton-profiler] kernel=softmax_kernel_online_v2 cpu_launch_ms=0.037 gpu_time_ms=0.029

    Returns:
        dict: {test_name: [list of gpu_time_ms values], ...}
    """
    # Support both [liger][triton] and [triton-profiler] formats
    # Pattern1 handles [liger][triton] format
    pattern1 = r'\[liger\]\[triton\]\s+kernel=(\S+)\s+cpu_launch_ms=[\d.]+\s+gpu_time_ms=([\d.]+)'
    # Pattern2 handles [triton-profiler] format with optional kernel parameters like [M=256, N=256, K=128]
    pattern2 = r'\[triton-profiler\]\s+kernel=(\S+)(?:\s+\[.*?\])?\s+cpu_launch_ms=[\d.]+\s+gpu_time_ms=([\d.]+)'

    results = defaultdict(list)
    test_name = None
    test_number = None

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Extract test number from header
                # Example: Test Number: 01
                number_match = re.search(r'Test Number:\s+(\d+)', line)
                if number_match:
                    test_number = number_match.group(1)

                # Extract test name from header
                # Example: Test: tritonbench/softmax_optimize
                name_match = re.search(r'Test:\s+(.+)', line)
                if name_match:
                    test_name = name_match.group(1).strip()
                    # If we have both number and name, create a combined key
                    if test_number and test_name:
                        test_name = f"{test_number}_{test_name}"

                # Also support pytest format
                # Example: test_fused_linear_jsd.py::test_correctness_functional[...]
                pytest_match = re.search(r'(test_\w+\.py::\S+)', line)
                if pytest_match:
                    test_name = pytest_match.group(1)

                # Extract gpu_time_ms - try both patterns
                match = re.search(pattern1, line)
                if not match:
                    match = re.search(pattern2, line)

                if match and test_name:
                    kernel_name = match.group(1)
                    gpu_time = float(match.group(2))
                    results[test_name].append({
                        'kernel': kernel_name,
                        'gpu_time_ms': gpu_time
                    })
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file}")
        return {}
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return {}

    return results


def parse_triton_sanitizer(log_file):
    """
    Parse triton-sanitizer log file for execution time.

    Example line:
    Triton-Viz: execution time for _jsd_kernel: 3.326 ms

    Returns:
        dict: {test_name: [list of execution_time values], ...}
    """
    pattern = r'Triton-Viz:\s+execution time for\s+(\S+):\s+([\d.]+)\s+ms'

    results = defaultdict(list)
    test_name = None
    test_number = None

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Extract test number from header
                # Example: Test Number: 01
                number_match = re.search(r'Test Number:\s+(\d+)', line)
                if number_match:
                    test_number = number_match.group(1)

                # Extract test name from header
                # Example: Test: tritonbench/softmax_optimize
                name_match = re.search(r'Test:\s+(.+)', line)
                if name_match:
                    test_name = name_match.group(1).strip()
                    # If we have both number and name, create a combined key
                    if test_number and test_name:
                        test_name = f"{test_number}_{test_name}"

                # Also support pytest format
                # Example: test_fused_linear_jsd.py::test_correctness_functional[...]
                pytest_match = re.search(r'(test_\w+\.py::\S+)', line)
                if pytest_match:
                    test_name = pytest_match.group(1)

                # Extract execution time
                match = re.search(pattern, line)
                if match and test_name:
                    kernel_name = match.group(1)
                    exec_time = float(match.group(2))
                    results[test_name].append({
                        'kernel': kernel_name,
                        'exec_time_ms': exec_time
                    })
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file}")
        return {}
    except Exception as e:
        print(f"Error parsing {log_file}: {e}")
        return {}

    return results


def calculate_totals(results, time_key='gpu_time_ms'):
    """
    Calculate total time for each test.

    Args:
        results: dict from parse functions
        time_key: key to sum ('gpu_time_ms' or 'exec_time_ms')

    Returns:
        dict: {test_name: total_time_ms, ...}
    """
    totals = {}
    for test_name, measurements in results.items():
        total = sum(m[time_key] for m in measurements)
        totals[test_name] = {
            'total_ms': total,
            'count': len(measurements)
        }
    return totals


def find_log_files(output_dir, config_name):
    """
    Find all log files for a given configuration.

    Args:
        output_dir: base output directory
        config_name: 'baseline', 'compute-sanitizer', or 'triton-sanitizer'

    Returns:
        list of Path objects sorted by file number
    """
    output_path = Path(output_dir)

    # Try multiple possible directory structures
    possible_paths = [
        # Direct structure: output_dir/baseline/
        output_path / config_name,
        # Nested structure from runner.py: output_dir/kernel_time_liger_kernel/baseline/
        output_path / "kernel_time_liger_kernel" / config_name,
        # Nested structure for TritonBench: output_dir/kernel_time_tritonbench/baseline/
        output_path / "kernel_time_tritonbench" / config_name,
    ]

    config_path = None
    for path in possible_paths:
        if path.exists():
            config_path = path
            break

    if config_path is None:
        return []

    # Look for log files in the configuration directory
    log_files = []

    # Check for run.log
    run_log = config_path / "run.log"
    if run_log.exists():
        log_files.append(run_log)

    # Check for other log files (numbered logs from runner.py)
    for pattern in ['*.log', '**/*.log']:
        log_files.extend(config_path.glob(pattern))

    # Remove duplicates
    log_files = list(set(log_files))

    # Sort by file number (extract number from filename like "01_test.log")
    def extract_number(path):
        match = re.search(r'(\d+)_', path.name)
        if match:
            return int(match.group(1))
        return float('inf')  # Put files without numbers at the end

    log_files.sort(key=extract_number)

    return log_files


def analyze_configuration(output_dir, config_name, is_triton_sanitizer=False):
    """
    Analyze all log files for a configuration.

    Args:
        output_dir: base output directory
        config_name: configuration name
        is_triton_sanitizer: True if this is triton-sanitizer

    Returns:
        dict: analysis results
    """
    log_files = find_log_files(output_dir, config_name)

    if not log_files:
        print(f"  No log files found for {config_name}")
        return None

    print(f"  Found {len(log_files)} log file(s)")

    # Parse all log files
    all_results = defaultdict(list)

    for log_file in log_files:
        print(f"    Parsing: {log_file.name}")

        if is_triton_sanitizer:
            results = parse_triton_sanitizer(log_file)
            time_key = 'exec_time_ms'
        else:
            results = parse_baseline_compute_sanitizer(log_file)
            time_key = 'gpu_time_ms'

        # Merge results
        for test_name, measurements in results.items():
            all_results[test_name].extend(measurements)

    # Calculate totals
    totals = {}
    for test_name, measurements in all_results.items():
        total = sum(m[time_key] for m in measurements)
        totals[test_name] = {
            'total_ms': total,
            'count': len(measurements),
            'measurements': measurements
        }

    return totals


def print_results(config_name, totals):
    """
    Print analysis results for a configuration.
    """
    if not totals:
        print(f"  No results to display")
        return

    print(f"\n  Results by test:")
    print(f"  {'─' * 80}")

    # Sort by test name
    sorted_tests = sorted(totals.items())

    grand_total = 0
    total_measurements = 0

    for test_name, data in sorted_tests:
        total_ms = data['total_ms']
        count = data['count']
        grand_total += total_ms
        total_measurements += count

        # Shorten test name if too long
        display_name = test_name
        if len(display_name) > 60:
            display_name = display_name[:57] + "..."

        print(f"    {display_name}")
        print(f"      Total: {total_ms:>10.3f} ms  (from {count} kernel calls)")

    print(f"  {'─' * 80}")
    print(f"  Grand Total: {grand_total:>10.3f} ms")
    print(f"  Total Tests: {len(totals)}")
    print(f"  Total Kernel Calls: {total_measurements}")
    print(f"  Average per Test: {grand_total / len(totals):.3f} ms" if totals else "")


def normalize_test_name(test_name):
    """
    Normalize test name by removing parametrization details and file number prefix.

    Examples:
        test_cross_entropy.py::test_correctness[0.1-dtype0-...] -> test_cross_entropy.py::test_correctness
        01_tritonbench/softmax_optimize -> tritonbench/softmax_optimize

    Args:
        test_name: full test name with parameters

    Returns:
        normalized test name without parameters and number prefix
    """
    # Remove parametrization: test_name[params] -> test_name
    if '[' in test_name:
        test_name = test_name[:test_name.index('[')]

    # Remove number prefix: 01_test_name -> test_name
    test_name = re.sub(r'^\d+_', '', test_name)

    return test_name


def aggregate_by_function(totals_dict):
    """
    Aggregate parametrized test results by function name.

    Args:
        totals_dict: dict of {test_name: {total_ms, count, measurements}}

    Returns:
        dict of aggregated results by function name
    """
    if not totals_dict:
        return {}

    aggregated = defaultdict(lambda: {'total_ms': 0.0, 'count': 0, 'test_variants': 0})

    for test_name, data in totals_dict.items():
        # Normalize the test name (remove parametrization)
        normalized_name = normalize_test_name(test_name)

        # Aggregate times
        aggregated[normalized_name]['total_ms'] += data['total_ms']
        aggregated[normalized_name]['count'] += data['count']
        aggregated[normalized_name]['test_variants'] += 1

    return dict(aggregated)


def export_to_csv(output_dir, baseline_totals, compute_totals, triton_totals, csv_filename):
    """
    Export results to CSV file with format:
    Test_Name, baseline_kernel_time, compute_sanitizer_kernel_time, triton_sanitizer_kernel_time

    This function aggregates all parametrized test variants into a single entry per test function.
    Results are sorted by test number if available.

    Args:
        output_dir: base output directory
        baseline_totals: baseline results dict
        compute_totals: compute-sanitizer results dict
        triton_totals: triton-sanitizer results dict
        csv_filename: output CSV filename
    """
    # First, collect all original test names with their numbers
    all_original_tests = {}  # {normalized_name: original_name_with_number}

    for totals_dict in [baseline_totals, compute_totals, triton_totals]:
        if totals_dict:
            for test_name in totals_dict.keys():
                normalized = normalize_test_name(test_name)
                # Keep the first occurrence (which should have the number)
                if normalized not in all_original_tests:
                    all_original_tests[normalized] = test_name

    # Aggregate parametrized tests by function name
    baseline_agg = aggregate_by_function(baseline_totals)
    compute_agg = aggregate_by_function(compute_totals)
    triton_agg = aggregate_by_function(triton_totals)

    # Collect all unique test function names
    all_tests = set()
    if baseline_agg:
        all_tests.update(baseline_agg.keys())
    if compute_agg:
        all_tests.update(compute_agg.keys())
    if triton_agg:
        all_tests.update(triton_agg.keys())

    if not all_tests:
        print("No test results to export")
        return None

    # Sort test names by file number
    def extract_test_number(test_name):
        # Try to get the original name with number
        original = all_original_tests.get(test_name, test_name)
        match = re.search(r'^(\d+)_', original)
        if match:
            return int(match.group(1))
        return float('inf')

    sorted_tests = sorted(all_tests, key=extract_test_number)

    # Prepare CSV data
    csv_data = []
    for test_name in sorted_tests:
        baseline_time = baseline_agg.get(test_name, {}).get('total_ms', 0.0)
        compute_time = compute_agg.get(test_name, {}).get('total_ms', 0.0)
        triton_time = triton_agg.get(test_name, {}).get('total_ms', 0.0)

        csv_data.append({
            'Test_Name': test_name,
            'baseline_kernel_time': f"{baseline_time:.3f}",
            'compute_sanitizer_kernel_time': f"{compute_time:.3f}",
            'triton_sanitizer_kernel_time': f"{triton_time:.3f}"
        })

    # Write CSV file
    csv_path = Path(output_dir) / csv_filename
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Test_Name', 'baseline_kernel_time', 'compute_sanitizer_kernel_time', 'triton_sanitizer_kernel_time']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

        print(f"\n✓ CSV exported to: {csv_path}")
        print(f"  Total test functions: {len(csv_data)}")

        # Show aggregation info
        if baseline_agg:
            total_variants = sum(v['test_variants'] for v in baseline_agg.values())
            print(f"  Aggregated {total_variants} parametrized test variants into {len(csv_data)} functions")

        return csv_path
    except Exception as e:
        print(f"\n✗ Error exporting CSV: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze kernel timing results from Liger-Kernel tests"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory containing test results (e.g., test_outputs_kernel_time_20231113_120000)"
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show detailed breakdown of each kernel call"
    )
    parser.add_argument(
        "--csv",
        default="kernel_timing_results.csv",
        help="Export results to CSV file (default: kernel_timing_results.csv)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    print("=" * 80)
    print("Kernel Timing Analysis for Liger-Kernel Tests")
    print("=" * 80)
    print(f"\nAnalyzing: {output_dir}")
    print()

    # Analyze baseline
    print("━" * 80)
    print("BASELINE")
    print("━" * 80)
    baseline_totals = analyze_configuration(output_dir, "baseline", is_triton_sanitizer=False)
    if baseline_totals:
        print_results("baseline", baseline_totals)
    else:
        print("  No baseline results found")

    print()

    # Analyze compute-sanitizer
    print("━" * 80)
    print("COMPUTE-SANITIZER")
    print("━" * 80)
    compute_totals = analyze_configuration(output_dir, "compute-sanitizer", is_triton_sanitizer=False)
    if compute_totals:
        print_results("compute-sanitizer", compute_totals)
    else:
        print("  No compute-sanitizer results found")

    print()

    # Analyze triton-sanitizer
    print("━" * 80)
    print("TRITON-SANITIZER")
    print("━" * 80)
    triton_totals = analyze_configuration(output_dir, "triton-sanitizer", is_triton_sanitizer=True)
    if triton_totals:
        print_results("triton-sanitizer", triton_totals)
    else:
        print("  No triton-sanitizer results found")

    print()
    print("=" * 80)

    # Summary comparison
    if baseline_totals or compute_totals or triton_totals:
        print("\nSUMMARY")
        print("=" * 80)

        if baseline_totals:
            baseline_total = sum(t['total_ms'] for t in baseline_totals.values())
            print(f"  Baseline Total:           {baseline_total:>12.3f} ms")

        if compute_totals:
            compute_total = sum(t['total_ms'] for t in compute_totals.values())
            print(f"  Compute-Sanitizer Total:  {compute_total:>12.3f} ms")
            if baseline_totals:
                overhead = ((compute_total / baseline_total) - 1) * 100 if baseline_total > 0 else 0
                print(f"    Overhead vs Baseline:   {overhead:>12.2f} %")

        if triton_totals:
            triton_total = sum(t['total_ms'] for t in triton_totals.values())
            print(f"  Triton-Sanitizer Total:   {triton_total:>12.3f} ms")
            if baseline_totals:
                overhead = ((triton_total / baseline_total) - 1) * 100 if baseline_total > 0 else 0
                print(f"    Overhead vs Baseline:   {overhead:>12.2f} %")

        print("=" * 80)

    # Export to CSV
    if args.csv:
        print()
        print("=" * 80)
        print("EXPORTING TO CSV")
        print("=" * 80)
        csv_path = export_to_csv(output_dir, baseline_totals, compute_totals, triton_totals, args.csv)
        if csv_path:
            print()
            print("CSV file contains kernel timing data for all tests across all configurations")
            print("=" * 80)


if __name__ == "__main__":
    main()
