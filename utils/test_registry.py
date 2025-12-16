#!/usr/bin/env python3
"""
Test registry module containing environment configurations and repository settings.
"""

from collections import OrderedDict
from pathlib import Path

# Define the environment variable combinations with meaningful names
ENV_CONFIGS = OrderedDict([
    # Baseline configurations (without sanitizers)
    ("baseline_compile_no_cache", {
        "group": "baseline",
        "name": "compile_no_cache",
        "description": "Always compile Triton kernels, disable CUDA memory caching",
        "env": {
            "TRITON_ALWAYS_COMPILE": "1",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"
        },
        "command_prefix": "/usr/bin/time -v"
    }),
    ("baseline_no_compile_with_cache", {
        "group": "baseline",
        "name": "no_compile_with_cache",
        "description": "Use cached Triton kernels, enable CUDA memory caching",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0"
        },
        "command_prefix": "/usr/bin/time -v"
    }),
    ("baseline_compile_with_cache", {
        "group": "baseline",
        "name": "compile_with_cache",
        "description": "Always compile Triton kernels, enable CUDA memory caching",
        "env": {
            "TRITON_ALWAYS_COMPILE": "1",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0"
        },
        "command_prefix": "/usr/bin/time -v"
    }),
    ("baseline_no_compile_no_cache", {
        "group": "baseline",
        "name": "no_compile_no_cache",
        "description": "Use cached Triton kernels, disable CUDA memory caching",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"
        },
        "command_prefix": "/usr/bin/time -v"
    }),
    # Compute-sanitizer configurations
    ("compute_sanitizer_compile_no_cache", {
        "group": "compute_sanitizer",
        "name": "compile_no_cache",
        "description": "Compute-sanitizer with always compile, disable CUDA caching",
        "env": {
            "TRITON_ALWAYS_COMPILE": "1",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"
        },
        "command_prefix": "compute-sanitizer"
    }),
    ("compute_sanitizer_no_compile_with_cache", {
        "group": "compute_sanitizer",
        "name": "no_compile_with_cache",
        "description": "Compute-sanitizer with cached kernels, enable CUDA caching",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0"
        },
        "command_prefix": "compute-sanitizer"
    }),
    ("compute_sanitizer_compile_with_cache", {
        "group": "compute_sanitizer",
        "name": "compile_with_cache",
        "description": "Compute-sanitizer with always compile, enable CUDA caching",
        "env": {
            "TRITON_ALWAYS_COMPILE": "1",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0"
        },
        "command_prefix": "compute-sanitizer"
    }),
    ("compute_sanitizer_no_compile_no_cache", {
        "group": "compute_sanitizer",
        "name": "no_compile_no_cache",
        "description": "Compute-sanitizer with cached kernels, disable CUDA caching",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"
        },
        "command_prefix": "compute-sanitizer"
    }),
    # Triton-sanitizer configurations
    ("triton_sanitizer_compile_no_cache", {
        "group": "triton_sanitizer",
        "name": "compile_no_cache",
        "description": "Triton-sanitizer with always compile, disable CUDA caching",
        "env": {
            "TRITON_ALWAYS_COMPILE": "1",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"
        },
        "command_prefix": "/usr/bin/time -v triton-sanitizer"
    }),
    ("triton_sanitizer_no_compile_with_cache", {
        "group": "triton_sanitizer",
        "name": "no_compile_with_cache",
        "description": "Triton-sanitizer with cached kernels, enable CUDA caching",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0"
        },
        "command_prefix": "/usr/bin/time -v triton-sanitizer"
    }),
    ("triton_sanitizer_compile_with_cache", {
        "group": "triton_sanitizer",
        "name": "compile_with_cache",
        "description": "Triton-sanitizer with always compile, enable CUDA caching",
        "env": {
            "TRITON_ALWAYS_COMPILE": "1",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0"
        },
        "command_prefix": "/usr/bin/time -v triton-sanitizer"
    }),
    ("triton_sanitizer_no_compile_no_cache", {
        "group": "triton_sanitizer",
        "name": "no_compile_no_cache",
        "description": "Triton-sanitizer with cached kernels, disable CUDA caching",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"
        },
        "command_prefix": "/usr/bin/time -v triton-sanitizer"
    }),
    # Kernel timing configurations for pytest-based repos (Liger-Kernel, FlagGems)
    ("kernel_time_baseline", {
        "group": "kernel_time_liger_kernel",
        "name": "baseline",
        "description": "Kernel timing: Baseline (no compile, with cache)",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
            "ENABLE_TRITON_PROFILER": "1"
        },
        "command_prefix": ""
    }),
    ("kernel_time_compute_sanitizer", {
        "group": "kernel_time_liger_kernel",
        "name": "compute-sanitizer",
        "description": "Kernel timing: Compute-sanitizer (no compile, with cache)",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
            "ENABLE_TRITON_PROFILER": "1"
        },
        "command_prefix": "compute-sanitizer"
    }),
    ("kernel_time_triton_sanitizer", {
        "group": "kernel_time_liger_kernel",
        "name": "triton-sanitizer",
        "description": "Kernel timing: Triton-sanitizer (no compile, with cache)",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
            "ENABLE_TIMING": "1"
        },
        "command_prefix": "triton-sanitizer"
    }),
    # Generic kernel timing configurations for pytest-based repos
    ("pytest_kernel_time_baseline", {
        "group": "kernel_time",
        "name": "baseline",
        "description": "Kernel timing: Baseline (pytest, profiling enabled)",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
            "ENABLE_TRITON_PROFILER": "1"
        },
        "command_prefix": ""
    }),
    ("pytest_kernel_time_compute_sanitizer", {
        "group": "kernel_time",
        "name": "compute-sanitizer",
        "description": "Kernel timing: Compute-sanitizer (pytest, profiling enabled)",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
            "ENABLE_TRITON_PROFILER": "1"
        },
        "command_prefix": "compute-sanitizer"
    }),
    ("pytest_kernel_time_triton_sanitizer", {
        "group": "kernel_time",
        "name": "triton-sanitizer",
        "description": "Kernel timing: Triton-sanitizer (pytest, ENABLE_TIMING)",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
            "ENABLE_TIMING": "1"
        },
        "command_prefix": "triton-sanitizer"
    }),
    # Kernel timing configurations for TritonBench
    ("kernel_time_tritonbench_baseline", {
        "group": "kernel_time_tritonbench",
        "name": "baseline",
        "description": "TritonBench kernel timing: Baseline (no compile, with cache, profiling enabled)",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
            "ENABLE_TRITON_PROFILER": "1"
        },
        "command_prefix": ""
    }),
    ("kernel_time_tritonbench_compute_sanitizer", {
        "group": "kernel_time_tritonbench",
        "name": "compute-sanitizer",
        "description": "TritonBench kernel timing: Compute-sanitizer (no compile, with cache, profiling enabled)",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
            "ENABLE_TRITON_PROFILER": "1"
        },
        "command_prefix": "compute-sanitizer"
    }),
    ("kernel_time_tritonbench_triton_sanitizer", {
        "group": "kernel_time_tritonbench",
        "name": "triton-sanitizer",
        "description": "TritonBench kernel timing: Triton-sanitizer (no compile, with cache, profiling disabled)",
        "env": {
            "TRITON_ALWAYS_COMPILE": "0",
            "PYTORCH_NO_CUDA_MEMORY_CACHING": "0",
            "ENABLE_TRITON_PROFILER": "0",
            "ENABLE_TIMING": "1"
        },
        "command_prefix": "triton-sanitizer"
    }),
    # Ablation study configurations
    ("ablation_no_cache", {
        "group": "ablation_studies",
        "name": "no_cache",
        "description": "Ablation study: No cache enabled (0,0,0,0)",
        "env": {
            "SANITIZER_ENABLE_SYMBOL_CACHE": "0",
            "SANITIZER_ENABLE_LOOP_CACHE": "0",
            "SANITIZER_ENABLE_GRID_CACHE": "0",
            "SANITIZER_ENABLE_KERNEL_CACHE": "0",
            "ENABLE_TIMING": "1"
        },
        "command_prefix": "triton-sanitizer"
    }),
    ("ablation_symbol_only", {
        "group": "ablation_studies",
        "name": "symbol_only",
        "description": "Ablation study: Symbol cache only (1,0,0,0)",
        "env": {
            "SANITIZER_ENABLE_SYMBOL_CACHE": "1",
            "SANITIZER_ENABLE_LOOP_CACHE": "0",
            "SANITIZER_ENABLE_GRID_CACHE": "0",
            "SANITIZER_ENABLE_KERNEL_CACHE": "0",
            "ENABLE_TIMING": "1"
        },
        "command_prefix": "triton-sanitizer"
    }),
    ("ablation_symbol_loop", {
        "group": "ablation_studies",
        "name": "symbol_loop",
        "description": "Ablation study: Symbol and loop cache (1,1,0,0)",
        "env": {
            "SANITIZER_ENABLE_SYMBOL_CACHE": "1",
            "SANITIZER_ENABLE_LOOP_CACHE": "1",
            "SANITIZER_ENABLE_GRID_CACHE": "0",
            "SANITIZER_ENABLE_KERNEL_CACHE": "0",
            "ENABLE_TIMING": "1"
        },
        "command_prefix": "triton-sanitizer"
    }),
    ("ablation_symbol_loop_grid", {
        "group": "ablation_studies",
        "name": "symbol_loop_grid",
        "description": "Ablation study: Symbol, loop and grid cache (1,1,1,0)",
        "env": {
            "SANITIZER_ENABLE_SYMBOL_CACHE": "1",
            "SANITIZER_ENABLE_LOOP_CACHE": "1",
            "SANITIZER_ENABLE_GRID_CACHE": "1",
            "SANITIZER_ENABLE_KERNEL_CACHE": "0",
            "ENABLE_TIMING": "1"
        },
        "command_prefix": "triton-sanitizer"
    }),
    ("ablation_all_cache", {
        "group": "ablation_studies",
        "name": "all_cache",
        "description": "Ablation study: All cache enabled (1,1,1,1)",
        "env": {
            "SANITIZER_ENABLE_SYMBOL_CACHE": "1",
            "SANITIZER_ENABLE_LOOP_CACHE": "1",
            "SANITIZER_ENABLE_GRID_CACHE": "1",
            "SANITIZER_ENABLE_KERNEL_CACHE": "1",
            "ENABLE_TIMING": "1"
        },
        "command_prefix": "triton-sanitizer"
    })
])

# Repository configurations
REPO_CONFIGS = {
    "liger_kernel": {
        "test_dir": "submodules/Liger-Kernel/test/transformers/",
        "test_pattern": "test_*.py",
        "test_command": "pytest -s --assert=plain",
        "skip_tests": [],
        "whitelist_file": "utils/liger_kernel_whitelist.txt"
    },
    "flag_gems": {
        "test_dir": "submodules/FlagGems/tests/",
        "test_pattern": "test_*.py",
        "test_command": "pytest -s --assert=plain",
        "skip_tests": [],
        "whitelist_file": "utils/flag_gems_whitelist.txt"
    },
    "tritonbench": {
        "test_dir": "submodules/TritonBench/",
        "test_pattern": "*.py",
        "test_command": "python",
        "skip_tests": [],
        "special_handling": True,
        "whitelist_file": "utils/tritonbench_whitelist.txt"
    }
}

# Available configuration groups
CONFIG_GROUPS = [
    "baseline",
    "compute_sanitizer",
    "triton_sanitizer",
    "kernel_time",
    "kernel_time_liger_kernel",
    "kernel_time_tritonbench",
    "ablation_studies"
]


def get_configs_by_group(group_name):
    """Get all configurations belonging to a specific group.

    Args:
        group_name: Name of the configuration group

    Returns:
        OrderedDict of configurations belonging to the group
    """
    result = OrderedDict()
    for key, config in ENV_CONFIGS.items():
        if config["group"] == group_name:
            result[key] = config
    return result


def get_whitelist_path(repo_name, project_root=None):
    """Get the path to the whitelist file for a repository.

    Args:
        repo_name: Name of the repository
        project_root: Project root directory (Path object or string)

    Returns:
        Path to the whitelist file, or None if not configured
    """
    if repo_name not in REPO_CONFIGS:
        return None

    whitelist_file = REPO_CONFIGS[repo_name].get("whitelist_file")
    if not whitelist_file:
        return None

    if project_root:
        return Path(project_root) / whitelist_file
    return Path(whitelist_file)


def get_all_repos():
    """Get list of all repository names."""
    return list(REPO_CONFIGS.keys())
