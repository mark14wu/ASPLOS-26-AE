import csv
import statistics

# Read the CSV file
data = []
with open('test_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# Function to calculate kernel-only overhead ratios
def calculate_kernel_overhead(data, sanitizer_col, baseline_col):
    """
    Calculate kernel-only overhead as sanitizer_kernel_time/baseline_kernel_time ratio
    Returns list of valid ratios (excluding FAILED, NaN, and invalid values)
    """
    ratios = []
    for row in data:
        sanitizer_val = row[sanitizer_col].strip()
        baseline_val = row[baseline_col].strip()

        # Skip if either value is FAILED, empty, or invalid
        if (not sanitizer_val or not baseline_val or
            sanitizer_val == 'FAILED' or baseline_val == 'FAILED' or
            'Block Tensor Not Supported' in sanitizer_val or 'Block Tensor Not Supported' in baseline_val):
            continue

        try:
            sanitizer_val = float(sanitizer_val)
            baseline_val = float(baseline_val)
            if baseline_val > 0:  # Avoid division by zero
                ratio = sanitizer_val / baseline_val
                ratios.append(ratio)
        except (ValueError, TypeError):
            continue

    return ratios

# Function to compute statistics
def compute_stats(ratios):
    """
    Compute avg, median, lower (min), upper (max) for a list of ratios
    """
    if len(ratios) == 0:
        return {'avg': 'N/A', 'median': 'N/A', 'lower': 'N/A', 'upper': 'N/A'}

    return {
        'avg': statistics.mean(ratios),
        'median': statistics.median(ratios),
        'lower': min(ratios),
        'upper': max(ratios)
    }

print("=" * 100)
print("KERNEL-ONLY OVERHEAD ANALYSIS")
print("=" * 100)

# Calculate Compute-Sanitizer kernel-only overhead
print("\nCompute-Sanitizer Kernel-Only Overhead")
print("-" * 80)
cs_ratios = calculate_kernel_overhead(data,
                                       'kernel_time_compute_sanitizer (ms)',
                                       'kernel_time_baseline (ms)')
cs_stats = compute_stats(cs_ratios)
print(f"Valid samples: {len(cs_ratios)}")
print(f"  avg:    {cs_stats['avg']:.4f}x" if isinstance(cs_stats['avg'], float) else f"  avg:    {cs_stats['avg']}")
print(f"  median: {cs_stats['median']:.4f}x" if isinstance(cs_stats['median'], float) else f"  median: {cs_stats['median']}")
print(f"  lower:  {cs_stats['lower']:.4f}x" if isinstance(cs_stats['lower'], float) else f"  lower:  {cs_stats['lower']}")
print(f"  upper:  {cs_stats['upper']:.4f}x" if isinstance(cs_stats['upper'], float) else f"  upper:  {cs_stats['upper']}")

# Calculate Triton-Sanitizer kernel-only overhead
print("\nTriton-Sanitizer Kernel-Only Overhead")
print("-" * 80)
ts_ratios = calculate_kernel_overhead(data,
                                       'kernel_time_triton_sanitizer (ms)',
                                       'kernel_time_baseline (ms)')
ts_stats = compute_stats(ts_ratios)
print(f"Valid samples: {len(ts_ratios)}")
print(f"  avg:    {ts_stats['avg']:.4f}x" if isinstance(ts_stats['avg'], float) else f"  avg:    {ts_stats['avg']}")
print(f"  median: {ts_stats['median']:.4f}x" if isinstance(ts_stats['median'], float) else f"  median: {ts_stats['median']}")
print(f"  lower:  {ts_stats['lower']:.4f}x" if isinstance(ts_stats['lower'], float) else f"  lower:  {ts_stats['lower']}")
print(f"  upper:  {ts_stats['upper']:.4f}x" if isinstance(ts_stats['upper'], float) else f"  upper:  {ts_stats['upper']}")

# Print summary table
print("\n" + "=" * 100)
print("SUMMARY TABLE - KERNEL-ONLY OVERHEAD")
print("=" * 100)
print()
print("| Sanitizer         | avg     | median  | lower   | upper   | samples |")
print("|-------------------|---------|---------|---------|---------|---------|")

if isinstance(cs_stats['avg'], float):
    print(f"| Compute-Sanitizer | {cs_stats['avg']:7.2f} | {cs_stats['median']:7.2f} | {cs_stats['lower']:7.2f} | {cs_stats['upper']:7.2f} | {len(cs_ratios):7} |")
else:
    print(f"| Compute-Sanitizer | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {len(cs_ratios):7} |")

if isinstance(ts_stats['avg'], float):
    print(f"| Triton-Sanitizer  | {ts_stats['avg']:7.2f} | {ts_stats['median']:7.2f} | {ts_stats['lower']:7.2f} | {ts_stats['upper']:7.2f} | {len(ts_ratios):7} |")
else:
    print(f"| Triton-Sanitizer  | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | {len(ts_ratios):7} |")

print()

# Additional analysis: breakdown by test suite
print("=" * 100)
print("BREAKDOWN BY TEST SUITE")
print("=" * 100)

test_suites = {}
for row in data:
    test_name = row['Test_Name']
    suite = test_name.split('/')[0]  # Extract suite name (e.g., 'liger_kernel', 'tritonbench')

    if suite not in test_suites:
        test_suites[suite] = {'cs_ratios': [], 'ts_ratios': []}

    # Compute-Sanitizer
    cs_val = row['kernel_time_compute_sanitizer (ms)'].strip()
    baseline_val = row['kernel_time_baseline (ms)'].strip()
    if (cs_val and baseline_val and cs_val != 'FAILED' and baseline_val != 'FAILED'
        and 'Block Tensor Not Supported' not in cs_val and 'Block Tensor Not Supported' not in baseline_val):
        try:
            cs_f = float(cs_val)
            bl_f = float(baseline_val)
            if bl_f > 0:
                test_suites[suite]['cs_ratios'].append(cs_f / bl_f)
        except (ValueError, TypeError):
            pass

    # Triton-Sanitizer
    ts_val = row['kernel_time_triton_sanitizer (ms)'].strip()
    if (ts_val and baseline_val and ts_val != 'FAILED' and baseline_val != 'FAILED'
        and 'Block Tensor Not Supported' not in ts_val and 'Block Tensor Not Supported' not in baseline_val):
        try:
            ts_f = float(ts_val)
            bl_f = float(baseline_val)
            if bl_f > 0:
                test_suites[suite]['ts_ratios'].append(ts_f / bl_f)
        except (ValueError, TypeError):
            pass

for suite in sorted(test_suites.keys()):
    print(f"\n{suite}")
    print("-" * 80)

    cs_ratios = test_suites[suite]['cs_ratios']
    ts_ratios = test_suites[suite]['ts_ratios']

    cs_stats = compute_stats(cs_ratios)
    ts_stats = compute_stats(ts_ratios)

    print(f"  Compute-Sanitizer ({len(cs_ratios)} samples):")
    if isinstance(cs_stats['avg'], float):
        print(f"    avg: {cs_stats['avg']:.2f}x, median: {cs_stats['median']:.2f}x, range: [{cs_stats['lower']:.2f}, {cs_stats['upper']:.2f}]")
    else:
        print(f"    No valid data")

    print(f"  Triton-Sanitizer ({len(ts_ratios)} samples):")
    if isinstance(ts_stats['avg'], float):
        print(f"    avg: {ts_stats['avg']:.2f}x, median: {ts_stats['median']:.2f}x, range: [{ts_stats['lower']:.2f}, {ts_stats['upper']:.2f}]")
    else:
        print(f"    No valid data")

print()
