#!/usr/bin/env python3
"""
Analyze memory usage from Triton Sanitizer log files.
Extracts and calculates average "Maximum resident set size (kbytes)" for each category.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def extract_memory_usage(log_file_path):
    """
    Extract the maximum resident set size from a log file.

    Args:
        log_file_path: Path to the log file

    Returns:
        Memory size in kbytes, or None if not found
    """
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
            match = re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)', content)
            if match:
                return int(match.group(1))
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
    return None


def analyze_memory_usage(base_dir, subdir_name=None):
    """
    Analyze memory usage across all categories.

    Args:
        base_dir: Base directory containing the subdirectory with logs
        subdir_name: Name of the subdirectory (e.g., 'triton_sanitizer', 'baseline').
                     If None, will auto-detect.
    """
    # Define the categories
    categories = [
        'compile_no_cache',
        'compile_with_cache',
        'no_compile_no_cache',
        'no_compile_with_cache'
    ]

    # Auto-detect subdirectory if not specified
    if subdir_name is None:
        # Look for common subdirectory names
        for possible_name in ['triton_sanitizer', 'baseline']:
            possible_dir = os.path.join(base_dir, possible_name)
            if os.path.exists(possible_dir) and os.path.isdir(possible_dir):
                subdir_name = possible_name
                break

        if subdir_name is None:
            print(f"Error: Could not auto-detect subdirectory in {base_dir}")
            return

    logs_dir = os.path.join(base_dir, subdir_name)

    if not os.path.exists(logs_dir):
        print(f"Error: Directory {logs_dir} not found!")
        return

    # Store results for each category
    results = {}

    print("=" * 80)
    print(f"Memory Usage Analysis - {subdir_name.replace('_', ' ').title()} Logs")
    print(f"Directory: {base_dir}")
    print("=" * 80)
    print()

    for category in categories:
        category_dir = os.path.join(logs_dir, category)

        if not os.path.exists(category_dir):
            print(f"Warning: Category directory not found: {category_dir}")
            continue

        # Get all .log files
        log_files = sorted([f for f in os.listdir(category_dir) if f.endswith('.log')])

        memory_values = []

        for log_file in log_files:
            log_path = os.path.join(category_dir, log_file)
            memory = extract_memory_usage(log_path)

            if memory is not None:
                memory_values.append(memory)

        # Calculate statistics
        if memory_values:
            avg_memory = sum(memory_values) / len(memory_values)
            min_memory = min(memory_values)
            max_memory = max(memory_values)

            results[category] = {
                'count': len(memory_values),
                'average': avg_memory,
                'min': min_memory,
                'max': max_memory,
                'total_files': len(log_files)
            }

            print(f"Category: {category}")
            print(f"  Total log files: {len(log_files)}")
            print(f"  Files with memory data: {len(memory_values)}")
            print(f"  Average memory: {avg_memory:,.2f} kbytes ({avg_memory/1024:,.2f} MB)")
            print(f"  Min memory: {min_memory:,} kbytes ({min_memory/1024:.2f} MB)")
            print(f"  Max memory: {max_memory:,} kbytes ({max_memory/1024:.2f} MB)")
            print()
        else:
            print(f"Category: {category}")
            print(f"  No memory data found in {len(log_files)} log files")
            print()

    # Summary comparison
    if results:
        print("=" * 80)
        print("Summary Comparison (Average Memory Usage)")
        print("=" * 80)
        print()

        # Sort by average memory
        sorted_results = sorted(results.items(), key=lambda x: x[1]['average'])

        for category, stats in sorted_results:
            print(f"{category:30s}: {stats['average']:>12,.2f} kbytes ({stats['average']/1024:>8,.2f} MB)")

        print()
        print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze memory usage from test log files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_memory_usage.py test_outputs_baseline_20251114_200201
  python analyze_memory_usage.py test_outputs_triton_sanitizer_20251114_203204
  python analyze_memory_usage.py test_outputs_baseline_20251114_200201 --subdir baseline
        """
    )

    parser.add_argument(
        'directory',
        nargs='?',
        default='test_outputs_triton_sanitizer_20251114_203204',
        help='Base directory containing the test outputs (default: test_outputs_triton_sanitizer_20251114_203204)'
    )

    parser.add_argument(
        '--subdir',
        help='Subdirectory name (e.g., triton_sanitizer, baseline). If not specified, will auto-detect.'
    )

    args = parser.parse_args()

    if not os.path.exists(args.directory):
        print(f"Error: Directory {args.directory} not found!")
        sys.exit(1)

    analyze_memory_usage(args.directory, args.subdir)
