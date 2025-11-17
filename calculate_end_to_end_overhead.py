import csv
import statistics

# Read the CSV file
data = []
with open('test_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# Function to calculate overhead ratios
def calculate_overhead(data, sanitizer_col, baseline_col):
    """
    Calculate overhead as sanitizer/baseline ratio
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

# Define the configurations
configs = [
    {
        'name': 'Compilation Cache On, Torch Cuda Caching Allocator On',
        'baseline': 'baseline_compile_with_cache',
        'compute_sanitizer': 'compute_sanitizer_compile_with_cache',
        'triton_sanitizer': 'triton_sanitizer_compile_with_cache'
    },
    {
        'name': 'Compilation Cache On, Torch Cuda Caching Allocator Off',
        'baseline': 'baseline_compile_no_cache',
        'compute_sanitizer': 'compute_sanitizer_compile_no_cache',
        'triton_sanitizer': 'triton_sanitizer_compile_no_cache'
    },
    {
        'name': 'Compilation Cache Off, Torch Cuda Caching Allocator On',
        'baseline': 'baseline_no_compile_with_cache',
        'compute_sanitizer': 'compute_sanitizer_no_compile_with_cache',
        'triton_sanitizer': 'triton_sanitizer_no_compile_with_cache'
    },
    {
        'name': 'Compilation Cache Off, Torch Cuda Caching Allocator Off',
        'baseline': 'baseline_no_compile_no_cache',
        'compute_sanitizer': 'compute_sanitizer_no_compile_no_cache',
        'triton_sanitizer': 'triton_sanitizer_no_compile_no_cache'
    }
]

# Calculate overheads for each configuration
results = []
for config in configs:
    print(f"\n{config['name']}")
    print("=" * 80)

    # Compute-Sanitizer overhead
    cs_ratios = calculate_overhead(data, config['compute_sanitizer'], config['baseline'])
    cs_stats = compute_stats(cs_ratios)
    print(f"\nCompute-Sanitizer (n={len(cs_ratios)} valid samples):")
    print(f"  avg:    {cs_stats['avg']:.4f}x" if isinstance(cs_stats['avg'], float) else f"  avg:    {cs_stats['avg']}")
    print(f"  median: {cs_stats['median']:.4f}x" if isinstance(cs_stats['median'], float) else f"  median: {cs_stats['median']}")
    print(f"  lower:  {cs_stats['lower']:.4f}x" if isinstance(cs_stats['lower'], float) else f"  lower:  {cs_stats['lower']}")
    print(f"  upper:  {cs_stats['upper']:.4f}x" if isinstance(cs_stats['upper'], float) else f"  upper:  {cs_stats['upper']}")

    # Triton-Sanitizer overhead
    ts_ratios = calculate_overhead(data, config['triton_sanitizer'], config['baseline'])
    ts_stats = compute_stats(ts_ratios)
    print(f"\nTriton-Sanitizer (n={len(ts_ratios)} valid samples):")
    print(f"  avg:    {ts_stats['avg']:.4f}x" if isinstance(ts_stats['avg'], float) else f"  avg:    {ts_stats['avg']}")
    print(f"  median: {ts_stats['median']:.4f}x" if isinstance(ts_stats['median'], float) else f"  median: {ts_stats['median']}")
    print(f"  lower:  {ts_stats['lower']:.4f}x" if isinstance(ts_stats['lower'], float) else f"  lower:  {ts_stats['lower']}")
    print(f"  upper:  {ts_stats['upper']:.4f}x" if isinstance(ts_stats['upper'], float) else f"  upper:  {ts_stats['upper']}")

    results.append({
        'config': config['name'],
        'cs_stats': cs_stats,
        'ts_stats': ts_stats
    })

# Print the formatted table
print("\n\n" + "=" * 100)
print("SUMMARY TABLE")
print("=" * 100)
print()
print("| Compilation Cache | Torch Cuda Caching Allocator | Triton-Sanitizer                                                      | Compute-Sanitizer                                                     |")
print("|-------------------|------------------------------|-----------------------------------------------------------------------|-----------------------------------------------------------------------|")

for i, result in enumerate(results):
    # Parse cache settings from config name
    if i == 0:
        cache, allocator = "On", "On"
    elif i == 1:
        cache, allocator = "On", "Off"
    elif i == 2:
        cache, allocator = "Off", "On"
    else:
        cache, allocator = "Off", "Off"

    ts = result['ts_stats']
    cs = result['cs_stats']

    if isinstance(ts['avg'], float):
        ts_str = f"avg: {ts['avg']:.2f}  median: {ts['median']:.2f}  lower: {ts['lower']:.2f}  upper: {ts['upper']:.2f}"
    else:
        ts_str = f"avg: {ts['avg']}  median: {ts['median']}  lower: {ts['lower']}  upper: {ts['upper']}"

    if isinstance(cs['avg'], float):
        cs_str = f"avg: {cs['avg']:.2f}  median: {cs['median']:.2f}  lower: {cs['lower']:.2f}  upper: {cs['upper']:.2f}"
    else:
        cs_str = f"avg: {cs['avg']}  median: {cs['median']}  lower: {cs['lower']}  upper: {cs['upper']}"

    print(f"| {cache:17} | {allocator:28} | {ts_str:69} | {cs_str:69} |")

print()
