#!/usr/bin/env python3
"""
Calculate ablation kernel time speedup statistics

Input: test_results.csv
Output: average, median, max, min for each speedup metric
"""

import pandas as pd
import numpy as np

def calculate_speedup_stats(csv_file='test_results.csv'):
    """
    Read data from CSV file and calculate speedup statistics

    Speedup calculations:
    - speedup1: ablation_kernel_time_no_cache / ablation_kernel_time_symbol_only
    - speedup2: ablation_kernel_time_no_cache / ablation_kernel_time_symbol_loop
    - speedup3: ablation_kernel_time_no_cache / ablation_kernel_time_symbol_loop_grid
    - speedup4: ablation_kernel_time_no_cache / ablation_kernel_time_all_cache
    """

    # Read CSV file
    df = pd.read_csv(csv_file)

    # Get relevant columns
    no_cache = df['ablation_kernel_time_no_cache']
    symbol_only = df['ablation_kernel_time_symbol_only']
    symbol_loop = df['ablation_kernel_time_symbol_loop']
    symbol_loop_grid = df['ablation_kernel_time_symbol_loop_grid']
    all_cache = df['ablation_kernel_time_all_cache']

    # Calculate speedup (avoid division by zero)
    # Filter out rows where no_cache is 0 or any value is 0
    valid_mask = (no_cache > 0) & (symbol_only > 0) & (symbol_loop > 0) & (symbol_loop_grid > 0) & (all_cache > 0)

    speedup1 = no_cache[valid_mask] / symbol_only[valid_mask]
    speedup2 = no_cache[valid_mask] / symbol_loop[valid_mask]
    speedup3 = no_cache[valid_mask] / symbol_loop_grid[valid_mask]
    speedup4 = no_cache[valid_mask] / all_cache[valid_mask]

    # Create results dictionary
    results = {
        'Speedup Metric': [
            'no_cache/symbol_only',
            'no_cache/symbol_loop',
            'no_cache/symbol_loop_grid',
            'no_cache/all_cache'
        ],
        'Average': [
            speedup1.mean(),
            speedup2.mean(),
            speedup3.mean(),
            speedup4.mean()
        ],
        'Median': [
            speedup1.median(),
            speedup2.median(),
            speedup3.median(),
            speedup4.median()
        ],
        'Max': [
            speedup1.max(),
            speedup2.max(),
            speedup3.max(),
            speedup4.max()
        ],
        'Min': [
            speedup1.min(),
            speedup2.min(),
            speedup3.min(),
            speedup4.min()
        ],
        'Valid Samples': [
            len(speedup1),
            len(speedup2),
            len(speedup3),
            len(speedup4)
        ]
    }

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Print results
    print("=" * 80)
    print("Ablation Kernel Time Speedup Statistics")
    print("=" * 80)
    print(f"\nTotal tests: {len(df)}")
    print(f"Valid samples: {valid_mask.sum()}")
    print(f"Invalid samples: {(~valid_mask).sum()}")
    print("\n" + "=" * 80)
    print("\nSpeedup Statistics:")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("\n" + "=" * 80)

    # Save detailed speedup data to CSV
    speedup_details = pd.DataFrame({
        'Test_Name': df['Test_Name'][valid_mask].values,
        'speedup_symbol_only': speedup1.values,
        'speedup_symbol_loop': speedup2.values,
        'speedup_symbol_loop_grid': speedup3.values,
        'speedup_all_cache': speedup4.values,
        'baseline_time_no_cache': no_cache[valid_mask].values,
        'time_symbol_only': symbol_only[valid_mask].values,
        'time_symbol_loop': symbol_loop[valid_mask].values,
        'time_symbol_loop_grid': symbol_loop_grid[valid_mask].values,
        'time_all_cache': all_cache[valid_mask].values
    })

    output_file = 'ablation_speedup_details.csv'
    speedup_details.to_csv(output_file, index=False)
    print(f"\nDetailed speedup data saved to: {output_file}")

    # Save summary statistics
    summary_file = 'ablation_speedup_summary.csv'
    results_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")

    return results_df, speedup_details

if __name__ == '__main__':
    results_df, speedup_details = calculate_speedup_stats()
