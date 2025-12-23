#!/usr/bin/env python3
"""
Analyze ablation study kernel timing results.
Parses output files and calculates total execution time for 5 cache configurations:
1. no_cache: (0, 0, 0, 0)
2. symbol_only: (1, 0, 0, 0)
3. symbol_loop: (1, 1, 0, 0)
4. symbol_loop_grid: (1, 1, 1, 0)
5. all_cache: (1, 1, 1, 1)

All configurations use triton-sanitizer with ENABLE_TIMING=1 and parse Triton-Viz execution time.
"""

import os
import re
import sys
import csv
from pathlib import Path
import argparse
from collections import defaultdict


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


def find_log_files(output_dir, config_name):
    """
    Find all log files for a given configuration.

    Args:
        output_dir: base output directory
        config_name: configuration name (no_cache, symbol_only, etc.)

    Returns:
        list of Path objects sorted by file number
    """
    output_path = Path(output_dir)

    # Try multiple possible directory structures
    possible_paths = [
        # Direct structure: output_dir/no_cache/
        output_path / config_name,
        # Nested structure from runner.py: output_dir/ablation_studies/no_cache/
        output_path / "ablation_studies" / config_name,
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


def analyze_configuration(output_dir, config_name):
    """
    Analyze all log files for a configuration.
    Returns the total execution time for each log file.

    Args:
        output_dir: base output directory
        config_name: configuration name

    Returns:
        dict: {file_number: {'file_name': str, 'total_ms': float, 'count': int}}
    """
    log_files = find_log_files(output_dir, config_name)

    if not log_files:
        print(f"  No log files found for {config_name}")
        return None

    print(f"  Found {len(log_files)} log file(s)")

    # Parse each log file and calculate total time per file
    file_totals = {}

    for log_file in log_files:
        print(f"    Parsing: {log_file.name}")

        # Extract file number from filename
        match = re.search(r'^(\d+)_(.+)\.log$', log_file.name)
        if not match:
            continue

        file_number = int(match.group(1))
        base_name = match.group(2)

        # Parse all execution times in this file
        results = parse_triton_sanitizer(log_file)

        # Sum all execution times from this file
        total_ms = 0.0
        total_count = 0
        for test_name, measurements in results.items():
            for m in measurements:
                total_ms += m['exec_time_ms']
                total_count += 1

        file_totals[file_number] = {
            'file_name': base_name,
            'total_ms': total_ms,
            'count': total_count
        }

    return file_totals


def print_results(config_name, file_totals):
    """
    Print analysis results for a configuration.

    Args:
        config_name: configuration name
        file_totals: dict of {file_number: {'file_name': str, 'total_ms': float, 'count': int}}
    """
    if not file_totals:
        print(f"  No results to display")
        return

    print(f"\n  Results by file:")
    print(f"  {'─' * 80}")

    # Sort by file number
    sorted_files = sorted(file_totals.items())

    grand_total = 0
    total_measurements = 0

    for file_number, data in sorted_files:
        total_ms = data['total_ms']
        count = data['count']
        file_name = data['file_name']
        grand_total += total_ms
        total_measurements += count

        # Shorten file name if too long
        display_name = f"{file_number:03d}_{file_name}"
        if len(display_name) > 60:
            display_name = display_name[:57] + "..."

        print(f"    {display_name}")
        print(f"      Total: {total_ms:>10.3f} ms  (from {count} kernel calls)")

    print(f"  {'─' * 80}")
    print(f"  Grand Total: {grand_total:>10.3f} ms")
    print(f"  Total Files: {len(file_totals)}")
    print(f"  Total Kernel Calls: {total_measurements}")
    print(f"  Average per File: {grand_total / len(file_totals):.3f} ms" if file_totals else "")


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


def format_test_name(file_name):
    """
    Convert file name to formatted test name with slashes.

    Examples:
        liger_kernel_test_fused_linear_jsd_test_correctness_functional
        -> liger_kernel/test_fused_linear_jsd/test_correctness_functional

        flag_gems_test_libentry_test_threadsafety
        -> flag_gems/test_libentry/test_threadsafety

        tritonbench_softmax_optimize
        -> tritonbench/softmax_optimize

    Args:
        file_name: base file name without number prefix

    Returns:
        formatted test name with slashes
    """
    # Known repository prefixes
    repos = ['liger_kernel', 'flag_gems', 'tritonbench']

    # Find which repo this file belongs to
    repo = None
    for r in repos:
        if file_name.startswith(r + '_'):
            repo = r
            remainder = file_name[len(r) + 1:]  # Remove "repo_"
            break

    if repo is None:
        # If no known repo found, return as-is
        return file_name

    # For tritonbench, it's just repo/test_name
    if repo == 'tritonbench':
        return f"{repo}/{remainder}"

    # For liger_kernel and flag_gems, split by test_ patterns
    # Find all positions where 'test_' starts
    test_positions = []
    i = 0
    while i < len(remainder):
        if remainder[i:].startswith('test_'):
            test_positions.append(i)
            i += 5  # Skip past 'test_'
        else:
            i += 1

    if len(test_positions) == 0:
        # No test_ found, just return repo/remainder
        return f"{repo}/{remainder}"
    elif len(test_positions) == 1:
        # Only one test_ found, format as repo/test_xxx
        return f"{repo}/{remainder}"
    else:
        # Multiple test_ found, split at the last test_
        # First part is the test file, second part is the test function
        split_pos = test_positions[-1]
        test_file = remainder[:split_pos - 1]  # -1 to remove the underscore before test_
        test_func = remainder[split_pos:]
        return f"{repo}/{test_file}/{test_func}"


def export_to_csv(output_dir, config_results, csv_filename):
    """
    Export results to CSV file with format:
    Test_Name, ablation_kernel_time_no_cache, ablation_kernel_time_symbol_only,
    ablation_kernel_time_symbol_loop, ablation_kernel_time_symbol_loop_grid,
    ablation_kernel_time_all_cache

    Each log file gets its own row (111 rows total), sorted by file number.
    Each cell contains the sum of all kernel execution times in that log file.

    Args:
        output_dir: base output directory
        config_results: dict mapping config names to their file_totals
        csv_filename: output CSV filename
    """
    # Configuration names in order
    config_names = ['no_cache', 'symbol_only', 'symbol_loop', 'symbol_loop_grid', 'all_cache']

    # Collect all unique file numbers from all configurations
    all_file_numbers = set()
    for config_name in config_names:
        file_totals = config_results.get(config_name)
        if file_totals:
            all_file_numbers.update(file_totals.keys())

    if not all_file_numbers:
        print("No test results to export")
        return None

    # Sort by file number
    sorted_file_numbers = sorted(all_file_numbers)

    # Prepare CSV data - one row per log file
    csv_data = []
    for file_number in sorted_file_numbers:
        # Get the file name from the first available config
        file_name = None
        for config_name in config_names:
            file_totals = config_results.get(config_name, {})
            if file_number in file_totals:
                file_name = file_totals[file_number]['file_name']
                break

        if file_name is None:
            continue

        # Format the test name with slashes
        formatted_name = format_test_name(file_name)
        row = {'Test_Name': formatted_name}

        # Add timing data for each configuration
        for config_name in config_names:
            file_totals = config_results.get(config_name, {})
            time_ms = file_totals.get(file_number, {}).get('total_ms', 0.0)
            row[f'ablation_kernel_time_{config_name}'] = f"{time_ms:.3f}"

        csv_data.append(row)

    # Write CSV file
    csv_path = Path(output_dir) / csv_filename
    try:
        fieldnames = ['Test_Name'] + [f'ablation_kernel_time_{name}' for name in config_names]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

        print(f"\n✓ CSV exported to: {csv_path}")
        print(f"  Total log files: {len(csv_data)}")

        return csv_path
    except Exception as e:
        print(f"\n✗ Error exporting CSV: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ablation study kernel timing results"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory containing test results (e.g., test_outputs_ablation_kernel_time_20231113_120000)"
    )
    parser.add_argument(
        "--csv",
        default="ablation_kernel_timing_results.csv",
        help="Export results to CSV file (default: ablation_kernel_timing_results.csv)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    print("=" * 80)
    print("Ablation Study Kernel Timing Analysis")
    print("=" * 80)
    print(f"\nAnalyzing: {output_dir}")
    print()

    # Configuration names and descriptions
    configs = [
        ('no_cache', 'No Cache (0,0,0,0)'),
        ('symbol_only', 'Symbol Only (1,0,0,0)'),
        ('symbol_loop', 'Symbol + Loop (1,1,0,0)'),
        ('symbol_loop_grid', 'Symbol + Loop + Grid (1,1,1,0)'),
        ('all_cache', 'All Cache (1,1,1,1)'),
    ]

    # Analyze each configuration
    config_results = {}

    for config_name, config_desc in configs:
        print("━" * 80)
        print(f"{config_desc.upper()}")
        print("━" * 80)
        totals = analyze_configuration(output_dir, config_name)
        config_results[config_name] = totals

        if totals:
            print_results(config_name, totals)
        else:
            print(f"  No {config_name} results found")

        print()

    print("=" * 80)

    # Summary comparison
    print("\nSUMMARY")
    print("=" * 80)

    summary_data = []
    for config_name, config_desc in configs:
        totals = config_results.get(config_name)
        if totals:
            total_ms = sum(t['total_ms'] for t in totals.values())
            summary_data.append((config_desc, total_ms))
            print(f"  {config_desc:30s} {total_ms:>12.3f} ms")

    # Calculate speedups relative to no_cache
    if summary_data and summary_data[0][0] == 'No Cache (0,0,0,0)':
        baseline = summary_data[0][1]
        print()
        print("  Speedup vs No Cache:")
        for config_desc, total_ms in summary_data[1:]:
            if total_ms > 0:
                speedup = baseline / total_ms
                reduction = (1 - total_ms / baseline) * 100
                print(f"    {config_desc:28s} {speedup:>6.2f}x  ({reduction:>5.1f}% reduction)")

    print("=" * 80)

    # Export to CSV
    if args.csv:
        print()
        print("=" * 80)
        print("EXPORTING TO CSV")
        print("=" * 80)
        csv_path = export_to_csv(output_dir, config_results, args.csv)
        if csv_path:
            print()
            print("CSV file contains kernel timing data for all tests across all ablation configurations")
            print("=" * 80)


if __name__ == "__main__":
    main()
