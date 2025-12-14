import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Set font to Times New Roman (install ttf-mscorefonts-installer if not available)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'

def load_and_clean_data(filepath, source_name):
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('\t')
        
        if len(parts) == 3:
            baseline_str = parts[0]
            cs_str = parts[1]
            z3_str = parts[2]
            
            try:
                baseline = float(baseline_str)
            except:
                continue
                
            if cs_str.lower() in ['oom', 'failed'] or z3_str.lower() in ['oom', 'failed']:
                continue
                
            try:
                cs = float(cs_str)
                z3 = float(z3_str)
                
                if cs > 0 and z3 > 0 and baseline > 0:
                    data.append({
                        'baseline_time': baseline * 1000,  # Convert to ms
                        'cs_time': cs * 1000,  # Convert to ms
                        'z3_time': z3 * 1000,  # Convert to ms
                        'source': source_name
                    })
            except:
                continue
                
    return pd.DataFrame(data)

data_dir = Path('/home/hwu27/workspace/triton-viz-figures/overhead_data')
all_data = []

for file in data_dir.glob('*.txt'):
    source_name = file.stem
    print(f"Processing {file.name}...")
    
    if file.name == 'flaggems.txt':
        df_temp = load_and_clean_data(file, source_name)
    else:
        with open(file, 'r') as f:
            lines = f.readlines()
            
        if lines and ('baseline' in lines[0].lower() or '_4090' in lines[0]):
            lines = lines[1:]
            
        temp_file = '/tmp/temp_data.txt'
        with open(temp_file, 'w') as f:
            f.writelines(lines)
            
        df_temp = load_and_clean_data(temp_file, source_name)
        os.remove(temp_file)
    
    print(f"  Loaded {len(df_temp)} valid data points from {source_name}")
    all_data.append(df_temp)

df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal data points after cleaning: {len(df)}")

df['speedup'] = df['cs_time'] / df['z3_time']

# Save cleaned data to TSV file
output_file = 'cleaned_data.tsv'
df.to_csv(output_file, sep='\t', index=False)
print(f"\nCleaned data saved to: {output_file}")

print("\nData summary:")
print(f"Baseline time range: {df['baseline_time'].min():.4f} - {df['baseline_time'].max():.4f}")
print(f"Speedup range: {df['speedup'].min():.4f} - {df['speedup'].max():.4f}")
print(f"Mean speedup: {df['speedup'].mean():.4f}")
print(f"Median speedup: {df['speedup'].median():.4f}")

fig, ax = plt.subplots(figsize=(10, 6))

# Use a single color for all points
point_color = '#4A90E2'  # Nice blue color

# Calculate point sizes proportional to baseline_time
# Normalize sizes between 20 and 300 for better visualization
min_baseline = df['baseline_time'].min()
max_baseline = df['baseline_time'].max()

# Plot all points with the same color
# Scale point sizes based on baseline_time
sizes = 20 + 280 * (df['baseline_time'] - min_baseline) / (max_baseline - min_baseline)
ax.scatter(df['cs_time'], df['speedup'], 
          alpha=0.6, s=sizes, 
          color=point_color)

ax.set_xscale('log')

ax.set_xlabel('Compute-Sanitizer Time (ms, log)', fontsize=24)
ax.set_ylabel('Speedup (x)', fontsize=24)

# Set tick label font sizes
ax.tick_params(axis='both', which='major', labelsize=24)
ax.tick_params(axis='both', which='minor', labelsize=24)

ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)

ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('/home/hwu27/workspace/triton-viz-figures/speedup_scatter.pdf', dpi=600, bbox_inches='tight')
plt.show()

print(f"\nPlot saved as speedup_scatter.pdf and speedup_scatter.png")
print(f"Points below the red line (y=1) indicate z3 is faster than compute-sanitizer")
print(f"Points above the red line indicate z3 is slower than compute-sanitizer")
print(f"Point size is proportional to baseline execution time")