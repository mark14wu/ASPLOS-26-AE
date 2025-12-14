import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from collections import defaultdict
import re


# --- Global look & feel (one-time) ---
def set_global_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "axes.linewidth": 2.2,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
    })

def _outlined_text(ax, x, y, text, **kw):
    t = ax.text(x, y, text, **kw)
    t.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])
    return t

def parse_test_name(test_name):
    """Parse test name to extract kernel name and data shape."""
    match = re.match(r'(.+?)\[(.*?)\]', test_name)
    if match:
        kernel_name = match.group(1).split('::')[-1]
        data_shape = match.group(2)
        return kernel_name, data_shape
    return None, None

def parse_compile_file(filepath):
    """Parse compile.txt file to extract and sum compilation stage timings."""
    data = defaultdict(lambda: defaultdict(float))

    with open(filepath, 'r') as f:
        lines = f.readlines()

    current_test = None
    for line in lines:
        line = line.strip()

        # Check for test name lines
        if '::' in line and '[' in line and ']' in line:
            test_name = line.split()[0]
            kernel, shape = parse_test_name(test_name)
            if kernel and shape:
                current_test = (kernel, shape)

                # Parse the rest of the line for timing info
                rest_of_line = ' '.join(line.split()[1:])
                if 'AST parsing took' in rest_of_line:
                    time_val = float(rest_of_line.split('took')[1].split('seconds')[0].strip())
                    data[current_test]['AST parsing'] += time_val

        # Check for test case lines (for different format like tritonbench)
        elif 'test case' in line:
            # Handle both "test case 1: torch.Size([2, 3, 4, 5])" and "test case 1"
            if ':' in line and 'torch.Size' in line:
                shape_part = line.split(':')[1].strip()
                shape = shape_part.replace('torch.Size([', '').replace('])', '').replace(', ', '-')
                current_test = ('test_case', shape)
            else:
                # Simple test case format like "test case 1"
                test_num = line.replace('test case', '').strip()
                current_test = ('test_case', test_num)

        # Parse compilation timing lines
        elif 'took' in line and 'seconds' in line:
            if current_test:
                if 'AST parsing' in line:
                    time_val = float(line.split('took')[1].split('seconds')[0].strip())
                    data[current_test]['AST parsing'] += time_val
                elif 'ttir compilation' in line:
                    time_val = float(line.split('took')[1].split('seconds')[0].strip())
                    data[current_test]['TTIR'] += time_val
                elif 'ttgir compilation' in line:
                    time_val = float(line.split('took')[1].split('seconds')[0].strip())
                    data[current_test]['TTGIR'] += time_val
                elif 'llir compilation' in line:
                    time_val = float(line.split('took')[1].split('seconds')[0].strip())
                    data[current_test]['LLIR'] += time_val
                elif 'ptx compilation' in line:
                    time_val = float(line.split('took')[1].split('seconds')[0].strip())
                    data[current_test]['PTX'] += time_val
                elif 'cubin compilation' in line:
                    time_val = float(line.split('took')[1].split('seconds')[0].strip())
                    data[current_test]['CUBIN'] += time_val

    return data

def parse_z3_file(filepath):
    """Parse z3.txt file to extract and sum all kernel elapsed times."""
    total_time_ms = 0

    if not filepath.exists():
        return 0

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if 'kernel elapsed time:' in line:
            # Parse the time value - can be in ms or seconds
            time_part = line.split(':')[1].strip()

            if 'ms' in time_part:
                # Time in milliseconds
                time_str = time_part.replace('ms', '').strip()
                if time_str:
                    try:
                        time_val = float(time_str)
                        total_time_ms += time_val
                    except ValueError:
                        continue
            elif 'second' in time_part:
                # Time in seconds - convert to milliseconds
                time_str = time_part.replace('seconds', '').replace('second', '').strip()
                if time_str:
                    try:
                        time_val = float(time_str) * 1000  # Convert to ms
                        total_time_ms += time_val
                    except ValueError:
                        continue

    return total_time_ms

def parse_execution_file(filepath):
    """Parse execution.txt file to extract and sum all execution timings."""
    data = defaultdict(float)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    current_test = None
    for line in lines:
        line = line.strip()

        # Check for test name lines
        if '::' in line and '[' in line and ']' in line:
            test_name = line.split()[0]
            kernel, shape = parse_test_name(test_name)
            if kernel and shape:
                current_test = (kernel, shape)

                # Parse the rest of the line for timing info
                rest_of_line = ' '.join(line.split()[1:])
                if 'Forward kernel elapsed time:' in rest_of_line:
                    time_val = float(rest_of_line.split(':')[1].split('ms')[0].strip())
                    data[current_test] += time_val
                elif 'Backward kernel elapsed time:' in rest_of_line:
                    time_val = float(rest_of_line.split(':')[1].split('ms')[0].strip())
                    data[current_test] += time_val

        # Check for test case lines (for different format like tritonbench)
        elif 'test case' in line:
            # Handle both "test case 1: torch.Size([2, 3, 4, 5])" and "test case 1"
            if ':' in line and 'torch.Size' in line:
                shape_part = line.split(':')[1].strip()
                shape = shape_part.replace('torch.Size([', '').replace('])', '').replace(', ', '-')
                current_test = ('test_case', shape)
            else:
                # Simple test case format like "test case 1"
                test_num = line.replace('test case', '').strip()
                current_test = ('test_case', test_num)

        # Parse execution timing lines
        elif 'elapsed time:' in line:
            if current_test:
                # Parse the time value and unit
                time_part = line.split(':')[1].strip()
                if 'ms' in time_part:
                    # Time in milliseconds
                    time_val = float(time_part.split('ms')[0].strip())
                    data[current_test] += time_val
                elif 'seconds' in time_part or 'second' in time_part:
                    # Time in seconds - convert to milliseconds
                    time_val = float(time_part.split('second')[0].strip())
                    data[current_test] += time_val * 1000  # Convert to ms

    return data

def parse_end_to_end_file(filepath):
    """Parse end_to_end.txt file to extract compute-sanitizer and z3 total times."""
    times = {'compute-sanitizer': 0, 'z3': 0}

    if not filepath.exists():
        return times

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if 'compute-sanitizer:' in line:
            # Parse time, handling both "seconds" and "s" formats
            time_part = line.split(':')[1].strip()
            if 'second' in time_part:
                time_str = time_part.replace('seconds', '').replace('second', '').strip()
            else:
                time_str = time_part.replace('s', '').strip()
            times['compute-sanitizer'] = float(time_str)
        elif 'z3:' in line:
            # Parse time, handling both "seconds" and "s" formats
            time_part = line.split(':')[1].strip()
            if 'second' in time_part:
                time_str = time_part.replace('seconds', '').replace('second', '').strip()
            else:
                time_str = time_part.replace('s', '').strip()
            times['z3'] = float(time_str)

    return times

def create_breakdown_plots(breakdown_dir='breakdown_data_'):
    set_global_style()

    # --- font sizes (tweak here if you want bigger/smaller) ---
    LABEL_FS  = 30   # axis label
    TICK_FS   = 26   # tick labels
    YTICK_FS  = 26   # left category names
    ANNOT_FS  = 26   # numbers on bars / “CS: …s”
    LEGEND_FS = 24

    breakdown_path = Path(breakdown_dir)
    all_data = []

    for subdir in breakdown_path.iterdir():
        if not subdir.is_dir():
            continue

        compile_file = subdir / 'compile.txt'
        execution_file = subdir / 'execution.txt'
        z3_file = subdir / 'z3.txt'
        if not (compile_file.exists() and execution_file.exists()):
            continue

        print(f"\nProcessing {subdir.name}...")

        compile_data = parse_compile_file(compile_file)
        execution_data = parse_execution_file(execution_file)
        z3_time_ms = parse_z3_file(z3_file)

        end_to_end_file = subdir / 'end_to_end.txt'
        end_to_end_times = parse_end_to_end_file(end_to_end_file)

        folder_totals = {
            'AST parsing': 0, 'TTIR': 0, 'TTGIR': 0,
            'LLIR': 0, 'PTX': 0, 'CUBIN': 0, 'Execution': 0
        }

        for tk in compile_data:
            folder_totals['AST parsing'] += compile_data[tk].get('AST parsing', 0)
            folder_totals['TTIR']        += compile_data[tk].get('TTIR', 0)
            folder_totals['TTGIR']       += compile_data[tk].get('TTGIR', 0)
            folder_totals['LLIR']        += compile_data[tk].get('LLIR', 0)
            folder_totals['PTX']         += compile_data[tk].get('PTX', 0)
            folder_totals['CUBIN']       += compile_data[tk].get('CUBIN', 0)

        for tk in execution_data:
            folder_totals['Execution'] += execution_data[tk] / 1000.0  # ms -> s

        all_data.append({
            'source': subdir.name,
            'z3_time': z3_time_ms / 1000.0,
            'end_to_end_compute_sanitizer': end_to_end_times['compute-sanitizer'],
            'end_to_end_z3': end_to_end_times['z3'],
            **folder_totals
        })

    if not all_data:
        print("No valid data found!")
        return

    # --- Figure setup (white background) ---
    fig, ax = plt.subplots(figsize=(16, 10.5))  # no facecolor override
    # removed: ax.set_facecolor('#f8f9fa')
    ax.grid(True, axis='x', linestyle='-', linewidth=0.8, alpha=0.25)
    ax.set_axisbelow(True)

    colors = {
        'AST parsing': '#e76f51',
        'TTIR':        '#2a9d8f',
        'TTGIR':       '#0278A7',
        'LLIR':        '#6abf69',
        'PTX':         '#f4d35e',
        'CUBIN':       '#c084fc',
        'Execution':   '#ff9566',
        'Others':      '#bdbdbd'
    }
    z3_color = '#7cc0ff'
    stages = ['AST parsing', 'TTIR', 'TTGIR', 'LLIR', 'PTX', 'CUBIN', 'Execution']

    # Geometry & spacing
    bar_h   = 0.30
    row_gap = 0.80
    pair_gap = 0.18

    folder_names, y_main, y_z3 = [], [], []

    x_right = 102.8
    xlim_max = 103.0

    all_data = [{"source": "ligerkernel_fuse_linear_jsd", "z3_time": 0.45166999999999996, "end_to_end_compute_sanitizer": 33.33, "end_to_end_z3": 4.45, "AST parsing": 0.05, "TTIR": 0.0, "TTGIR": 0.05, "LLIR": 1.53, "PTX": 0.86, "CUBIN": 9.6, "Execution": 16.193939999999998}, {"source": "tritonbench_chunk_delta_fwd", "z3_time": 0.11, "end_to_end_compute_sanitizer": 25.252, "end_to_end_z3": 4.335, "AST parsing": 0.33, "TTIR": 0.09, "TTGIR": 0.29000000000000004, "LLIR": 6.72, "PTX": 2.24, "CUBIN": 9.280000000000001, "Execution": 1.8900000000000001}, {"source": "ligerkernel_geglu", "z3_time": 0.020309999999999998, "end_to_end_compute_sanitizer": 23.56, "end_to_end_z3": 9.22, "AST parsing": 0.15000000000000002, "TTIR": 0.0, "TTGIR": 0.01, "LLIR": 0.33, "PTX": 0.08, "CUBIN": 0.16, "Execution": 0.0142}, {"source": "tritonbench_reversed_cumsum", "z3_time": 0.10462, "end_to_end_compute_sanitizer": 8.66, "end_to_end_z3": 4.364, "AST parsing": 0.3899999999999999, "TTIR": 0.0, "TTGIR": 0.32999999999999996, "LLIR": 3.03, "PTX": 0.8400000000000001, "CUBIN": 1.6600000000000001, "Execution": 0}, {"source": "ligerkernel_kldiv", "z3_time": 0.05204, "end_to_end_compute_sanitizer": 7.75, "end_to_end_z3": 3.92, "AST parsing": 0.19999999999999998, "TTIR": 0.0, "TTGIR": 0.0, "LLIR": 0.43000000000000005, "PTX": 0.12000000000000001, "CUBIN": 0.23000000000000004, "Execution": 0.47458}, {"source": "tritonbench_rotary_emb_nopad", "z3_time": 0.05256, "end_to_end_compute_sanitizer": 5.541, "end_to_end_z3": 4.396, "AST parsing": 0, "TTIR": 0, "TTGIR": 0, "LLIR": 0, "PTX": 0, "CUBIN": 0, "Execution": 1.87}]


    for idx, dp in enumerate(reversed(all_data)):
        # --- name cleanup ---
        fname = dp['source']
        if '_' in fname:
            fname = fname.split('_', 1)[1]
        folder_names.append(fname)

        y_cs = idx * row_gap * 2
        y_ts = y_cs - bar_h - pair_gap
        y_main.append(y_cs); y_z3.append(y_ts)

        # --- CS stacked bar ---
        values = [dp[s] for s in stages]
        measured_cs = sum(values)
        e2e_cs = dp.get('end_to_end_compute_sanitizer', measured_cs)
        others_cs = max(0.0, e2e_cs - measured_cs)
        vals_cs = values + [others_cs]
        names_cs = stages + ['Others']

        pct_cs = [(v / e2e_cs) * 100.0 if e2e_cs > 0 else 0.0 for v in vals_cs]

        left = 0.0
        for i, stage in enumerate(names_cs):
            width = pct_cs[i]
            ax.barh(
                y_cs, width, left=left, height=bar_h,
                color=colors.get(stage, '#bdbdbd'),
                edgecolor='#2c3e50', linewidth=1.2,
                label=stage if idx == 0 else None
            )
            if width > 5:
                _outlined_text(
                    ax, left + width/2, y_cs, f'{width:.1f}%',
                    ha='center', va='center',
                    fontsize=ANNOT_FS, color='#2c3e50', fontweight='bold',
                ).set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
            left += width

        # --- Compact CS total text (no background patch) ---
        cs_w = 2.6
        cs_x = x_right - cs_w
        ax.text(
            cs_x + cs_w/2 + 4.1, y_cs, f'CS: {e2e_cs:.1f}s',
            ha='center', va='center', fontsize=ANNOT_FS, color='#7a5b14', fontweight='bold'
        )

        # --- Triton-Sanitizer (Z3) bar ---
        z3_meas = dp.get('z3_time', 0.0)
        e2e_z3  = dp.get('end_to_end_z3', z3_meas)
        if e2e_cs > 0 and e2e_z3 > 0:
            total_w = (e2e_z3 / e2e_cs) * 100.0
            exec_w  = (z3_meas / e2e_z3) * total_w if e2e_z3 > 0 else 0.0
            other_w = max(0.0, total_w - exec_w)

            ax.barh(
                y_ts, exec_w, height=bar_h, color=z3_color,
                edgecolor='#2c3e50', linewidth=1.2,
                label='Triton-Sanitizer' if idx == 0 else None
            )
            if exec_w > 5:
                _outlined_text(
                    ax, exec_w/2, y_ts,
                    f'{(z3_meas/e2e_z3)*100:.1f}%',
                    ha='center', va='center', fontsize=ANNOT_FS, color='#0e3b66', fontweight='bold'
                ).set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

            if other_w > 0:
                ax.barh(
                    y_ts, other_w, left=exec_w, height=bar_h,
                    color=colors['Others'], edgecolor='#2c3e50', linewidth=1.2
                )
                if other_w > 5:
                    _outlined_text(
                        ax, exec_w + other_w/2, y_ts,
                        f'{(1 - z3_meas/e2e_z3)*100:.1f}%',
                        ha='center', va='center', fontsize=ANNOT_FS, color='#2c3e50', fontweight='bold'
                    ).set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

            # small Z3 text (no background patch)
            z3_w = 2.6
            z3_x = x_right - z3_w
            ax.text(
                z3_x + z3_w/2 + 4, y_ts, f'TS: {e2e_z3:.1f}s',
                ha='center', va='center', fontsize=ANNOT_FS, color='#0e3b66', fontweight='bold'
            )

    # y-tick labels centered between the row pair
    y_mid = [(a + b) / 2 for a, b in zip(y_main, y_z3)]
    ax.set_yticks(y_mid)
    ax.set_yticklabels(folder_names, fontsize=YTICK_FS, rotation=0, ha='right', color='#2c3e50')

    # axes labels/limits
    ax.set_xlabel('Percentage of End-to-End Time (%)', fontsize=LABEL_FS, fontweight='bold', color='#2c3e50')
    ax.set_xlim(0, xlim_max)
    ax.set_ylim(min(y_z3) - 0.5, max(y_main) + 0.5)

    # spines & ticks
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_linewidth(2.0)
        ax.spines[s].set_color('#2c3e50')
    ax.tick_params(axis='x', labelsize=TICK_FS, colors='#2c3e50')
    ax.tick_params(axis='y', length=0, labelsize=TICK_FS)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    desired_order = ['AST parsing','TTIR','TTGIR','LLIR','PTX','CUBIN','Execution','Others','Triton-Sanitizer']
    h2, l2 = [], []
    for name in desired_order:
        if name in labels:
            i = labels.index(name)
            h2.append(handles[i]); l2.append(labels[i])
    leg = ax.legend(
        h2, l2, loc='upper center', bbox_to_anchor=(0.5, 1.16),
        ncol=5, fontsize=LEGEND_FS, frameon=True, fancybox=True, shadow=False,
        columnspacing=1.2, handletextpad=0.6
    )
    leg.get_frame().set_edgecolor('#d0d0d0')
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_alpha(0.97)

    for out in ('performance_breakdown_beautified.pdf', 'performance_breakdown_beautified.png'):
        plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Figure saved as {out}")

    plt.show()


create_breakdown_plots()

