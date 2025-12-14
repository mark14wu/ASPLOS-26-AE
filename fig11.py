import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects

# Set up the style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['grid.alpha'] = 0.2

# Data for ablation study (removed Autotune Cache)
configurations = [
    'Baseline',
    '+ Symbol\nCache',
    '+ Loop\nCache',
    '+ Grid\nCache',
    '+ Kernel\nCache'
]
times = [48.375, 41.859, 30.437, 26.861, 5.594]

# Create figure with better proportions
fig, ax = plt.subplots(figsize=(14, 9), facecolor='white')

# Create bar positions
x_pos = np.arange(len(configurations))

# Define a gradient color scheme
colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.8, len(configurations)))

# Create bars with gradient colors
bars = ax.bar(x_pos, times, color=colors, edgecolor='#2c3e50',
               linewidth=2.5, alpha=0.85, width=0.75)

# Add subtle gradient effect to each bar
for bar, color in zip(bars, colors):
    bar.set_facecolor(color)
    bar.set_alpha(0.9)

# Enhanced value labels on top of bars
for i, (bar, time) in enumerate(zip(bars, times)):
    height = bar.get_height()
    time_text = ax.text(bar.get_x() + bar.get_width()/2., height + 0.8,
                        f'{time:.2f}s', ha='center', va='bottom',
                        fontsize=30, fontweight='bold', color='#2c3e50')
    time_text.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

    # Add speedup label inside bars (except baseline)
    if i > 0:
        speedup = times[0] / time
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{speedup:.1f}×', ha='center', va='center',
                fontsize=32, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='#34495e', alpha=0.8))

# Customize x-axis
ax.set_xticks(x_pos)
ax.set_xticklabels(configurations, fontsize=28, fontweight='bold', color='#2c3e50')

# Customize y-axis
ax.set_ylabel('Execution Time (seconds)', fontsize=32, fontweight='bold', color='#2c3e50')
ax.set_ylim(0, max(times) * 1.12)
ax.tick_params(axis='y', labelsize=28, colors='#2c3e50')
ax.tick_params(axis='x', length=0)

# Enhanced grid
ax.grid(True, axis='y', alpha=0.25, linestyle='-', linewidth=0.5, color='gray')
ax.set_axisbelow(True)

# Add baseline reference
ax.axhline(y=times[0], color='#e74c3c', linestyle='--',
           alpha=0.4, linewidth=2.5, zorder=1)
ax.annotate('Baseline', xy=(len(configurations)-0.5, times[0]),
           xytext=(len(configurations)-0.5, times[0]+2.5),
           fontsize=26, color='#e74c3c', ha='right', fontweight='bold',
           arrowprops=dict(arrowstyle='->', color='#e74c3c', alpha=0.6, lw=2))

# Style the spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(2.5)
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['left'].set_color('#2c3e50')
ax.spines['bottom'].set_color('#2c3e50')

# Add performance improvement arrow
if len(times) > 1:
    arrow_props = dict(arrowstyle='->', lw=4, color='#27ae60', alpha=0.7)
    ax.annotate('', xy=(len(configurations)-1, times[-1]*0.5),
                xytext=(0, times[0]*0.9),
                arrowprops=arrow_props)
    ax.text(len(configurations)/2, times[0]*0.65, 'Performance Improvement',
           rotation=-25, fontsize=28, color='#27ae60',
           fontweight='bold', ha='center', alpha=0.8)

plt.tight_layout()

# Save figures
plt.savefig('ablation_study_enhanced.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('ablation_study_enhanced.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()

# Print enhanced summary
print("\n" + "="*60)
print(" ABLATION STUDY RESULTS SUMMARY ".center(60, "="))
print("="*60)
print(f"\n{'Configuration':<25} {'Time (s)':<12} {'Speedup':<12}")
print("-"*60)

for i, (config, time) in enumerate(zip(configurations, times)):
    config_clean = config.replace('\n', ' ')
    if i == 0:
        print(f"{config_clean:<25} {time:>10.3f}   {'—':^10}")
    else:
        speedup = times[0] / time
        print(f"{config_clean:<25} {time:>10.3f}   {speedup:>9.2f}×")

print("-"*60)
print(f"\n{'TOTAL IMPROVEMENT:':<25} {times[-1]:>10.3f}   "
      f"{times[0]/times[-1]:>9.2f}×")
print("="*60)

