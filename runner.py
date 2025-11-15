#!/usr/bin/env python3
"""
Test runner for Liger-Kernel, FlagGems, and TritonBench repositories.
Version 3: Added whitelist support for selective test execution.
"""

import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import sys
import argparse
import csv
import ast
from collections import OrderedDict

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
        "skip_tests": []
    },
    "flag_gems": {
        "test_dir": "submodules/FlagGems/tests/",
        "test_pattern": "test_*.py",
        "test_command": "pytest -s --assert=plain",
        "skip_tests": []
    },
    "tritonbench": {
        "test_dir": "submodules/TritonBench/",
        "test_pattern": "*.py",
        "test_command": "python",
        "skip_tests": [],
        "special_handling": True
    }
}

class TestRunner:
    def __init__(self, output_base_dir="test_outputs"):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.global_test_counter = 0
        self.total_tests = 0
        self.test_results = OrderedDict()
        self.test_list = []

    def load_whitelist(self, whitelist_file, repo_name):
        """Load test whitelist from file."""
        whitelist = {}

        if not Path(whitelist_file).exists():
            return None

        with open(whitelist_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                if repo_name == "tritonbench":
                    # TritonBench whitelist is just file names
                    file_name = Path(line).stem  # Remove .py extension
                    whitelist[file_name] = []  # Empty list means run the whole file
                elif "::" in line:
                    test_file, test_function = line.split("::")
                    test_file = Path(test_file).stem  # Remove .py extension
                    if test_file not in whitelist:
                        whitelist[test_file] = []
                    whitelist[test_file].append(test_function)

        return whitelist if whitelist else None

    def discover_test_functions(self, test_file):
        """Discover individual test functions in a pytest file."""
        test_functions = []

        try:
            with open(test_file, 'r') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_functions.append(node.name)
        except Exception as e:
            print(f"Warning: Could not parse {test_file} to find test functions: {e}")
            return None

        return test_functions if test_functions else None

    def discover_tests(self, repo_name):
        """Discover test files in a repository."""
        config = REPO_CONFIGS[repo_name]
        test_dir = Path(config["test_dir"])

        if not test_dir.exists():
            print(f"Warning: Test directory {test_dir} does not exist for {repo_name}")
            return []

        test_files = []

        if repo_name == "tritonbench":
            benchmark_dirs = [
                test_dir / "data" / "TritonBench_G_v1",
                test_dir / "LLM_generated",
                test_dir / "EVAL"
            ]

            for bench_dir in benchmark_dirs:
                if bench_dir.exists():
                    for py_file in bench_dir.glob("**/*.py"):
                        if "__pycache__" not in str(py_file) and not py_file.name.startswith("__"):
                            test_files.append(py_file)
        else:
            pattern = config["test_pattern"]
            test_files = list(test_dir.glob(pattern))

        test_files = [f for f in test_files if f.name not in config.get("skip_tests", [])]

        return test_files

    def prepare_test_list(self, repositories, whitelists=None):
        """Prepare a complete list of all tests to run."""
        self.test_list = []

        for repo in repositories:
            test_files = self.discover_tests(repo)
            whitelist = whitelists.get(repo) if whitelists else None

            if whitelist:
                print(f"Using whitelist for {repo}: {len(whitelist)} files with specific tests")

            for test_file in test_files:
                test_file_stem = test_file.stem

                # Check if this file is in whitelist
                if whitelist and test_file_stem in whitelist:
                    if repo == "tritonbench":
                        # TritonBench whitelist - run whole file
                        self.test_list.append({
                            "repository": repo,
                            "test_file": test_file,
                            "test_function": None,
                            "test_name": f"{repo}/{test_file_stem}"
                        })
                    else:
                        # pytest-based whitelist - run specific functions
                        for test_function in whitelist[test_file_stem]:
                            self.test_list.append({
                                "repository": repo,
                                "test_file": test_file,
                                "test_function": test_function,
                                "test_name": f"{repo}/{test_file_stem}/{test_function}"
                            })
                elif whitelist and repo == "tritonbench":
                    # Skip files not in whitelist for TritonBench
                    continue
                elif not whitelist and repo in ["liger_kernel", "flag_gems"]:
                    # No whitelist, discover all functions
                    test_functions = self.discover_test_functions(test_file)

                    if test_functions:
                        for test_function in test_functions:
                            self.test_list.append({
                                "repository": repo,
                                "test_file": test_file,
                                "test_function": test_function,
                                "test_name": f"{repo}/{test_file_stem}/{test_function}"
                            })
                    else:
                        self.test_list.append({
                            "repository": repo,
                            "test_file": test_file,
                            "test_function": None,
                            "test_name": f"{repo}/{test_file_stem}"
                        })
                elif not whitelist:
                    # TritonBench or no whitelist - run whole files
                    self.test_list.append({
                        "repository": repo,
                        "test_file": test_file,
                        "test_function": None,
                        "test_name": f"{repo}/{test_file_stem}"
                    })

        self.total_tests = len(self.test_list)
        print(f"Discovered {self.total_tests} tests across {len(repositories)} repositories")
        return self.test_list

    def run_single_test(self, test_info, env_config_key):
        """Run a single test with specific environment configuration."""
        env_config = ENV_CONFIGS[env_config_key]
        repo_name = test_info["repository"]
        test_file = test_info["test_file"]
        test_function = test_info["test_function"]
        config = REPO_CONFIGS[repo_name]

        self.global_test_counter += 1
        test_number = str(self.global_test_counter).zfill(len(str(self.total_tests)))

        output_dir = self.output_base_dir / env_config["group"] / env_config["name"]
        output_dir.mkdir(parents=True, exist_ok=True)

        if test_function:
            output_filename = f"{test_number}_{repo_name}_{test_file.stem}_{test_function}.log"
        else:
            output_filename = f"{test_number}_{repo_name}_{test_file.stem}.log"

        output_file = output_dir / output_filename

        env = os.environ.copy()
        env.update(env_config["env"])

        # Add base directory to PYTHONPATH for pytest plugin
        base_dir = str(Path(__file__).parent.absolute())
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{base_dir}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = base_dir

        if repo_name == "tritonbench" and config.get("special_handling"):
            # Use relative path from test_dir for tritonbench
            relative_path = test_file.relative_to(Path(config["test_dir"]))
            # Check if profiler should be enabled based on environment variable
            if env.get("ENABLE_TRITON_PROFILER") == "1":
                # Use wrapper script to enable profiling
                wrapper_script = Path(__file__).parent / "tritonbench_profiler_wrapper.py"
                cmd = ["python", str(wrapper_script), str(relative_path)]
            else:
                cmd = ["python", str(relative_path)]
        else:
            # For pytest-based tests (Liger-Kernel, FlagGems)
            if test_function:
                cmd = ["pytest", "-s", "--assert=plain", f"{test_file.name}::{test_function}"]
            else:
                cmd = config["test_command"].split() + [test_file.name]

            # Add pytest plugin for Triton profiling if enabled
            if env.get("ENABLE_TRITON_PROFILER") == "1" and "pytest" in cmd[0]:
                # Insert the plugin option after pytest command
                plugin_path = Path(__file__).parent / "pytest_triton_profiler.py"
                cmd.insert(1, "-p")
                cmd.insert(2, "pytest_triton_profiler")

        command_prefix = env_config.get("command_prefix", "")
        if command_prefix:
            cmd = command_prefix.split() + cmd

        if test_function:
            test_display = f"{test_file.name}::{test_function}"
        else:
            test_display = test_file.name

        print(f"  [{self.global_test_counter}/{self.total_tests}] [{repo_name}] Running: {test_display}")
        if command_prefix:
            print(f"    Prefix: {command_prefix}")

        start_time = time.time()

        try:
            with open(output_file, "w") as log_file:
                log_file.write(f"Test Number: {test_number}\n")
                log_file.write(f"Test: {test_info['test_name']}\n")
                log_file.write(f"Environment: {env_config_key}\n")
                log_file.write(f"Command: {' '.join(cmd)}\n")
                log_file.write(f"Start Time: {datetime.now().isoformat()}\n")
                log_file.write("=" * 80 + "\n")
                log_file.flush()

                result = subprocess.run(
                    cmd,
                    env=env,
                    cwd=config["test_dir"],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=300,
                    text=True
                )

            elapsed_time = time.time() - start_time
            success = result.returncode == 0
            status = "PASSED" if success else "FAILED"
            error_msg = "" if success else f"Return code: {result.returncode}"

        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            status = "TIMEOUT"
            error_msg = "Test exceeded 5 minute timeout"
            success = False

        except Exception as e:
            elapsed_time = time.time() - start_time
            status = "ERROR"
            error_msg = str(e)
            success = False

        with open(output_file, "a") as log_file:
            log_file.write("\n" + "=" * 80 + "\n")
            log_file.write(f"End Time: {datetime.now().isoformat()}\n")
            log_file.write(f"Elapsed Time: {elapsed_time:.4f} seconds\n")
            log_file.write(f"Status: {status}\n")
            if error_msg:
                log_file.write(f"Error: {error_msg}\n")

        print(f"    Status: {status} ({elapsed_time:.2f}s)")
        if error_msg:
            print(f"    Error: {error_msg}")

        return {
            "test_number": test_number,
            "status": status,
            "elapsed_time": elapsed_time,
            "error_message": error_msg,
            "output_file": str(output_file)
        }

    def run_all_tests(self, repositories, config_groups, whitelists=None):
        """Run all tests with selected environment configurations."""
        self.prepare_test_list(repositories, whitelists)

        if not self.test_list:
            print("No tests found to run")
            return

        for test_info in self.test_list:
            self.test_results[test_info["test_name"]] = {
                "test_number": None,
                "repository": test_info["repository"],
                "test_file": str(test_info["test_file"]),
                "test_function": test_info["test_function"] or ""
            }

        selected_configs = OrderedDict()
        for group in config_groups:
            if group == "all":
                selected_configs = ENV_CONFIGS
                break
            else:
                for key, config in ENV_CONFIGS.items():
                    if config["group"] == group:
                        selected_configs[key] = config

        print(f"\nRunning tests with {len(selected_configs)} configurations")
        print("=" * 60)

        for env_key, env_config in selected_configs.items():
            print(f"\nConfiguration: [{env_key}] {env_config['description']}")
            print("-" * 50)

            self.global_test_counter = 0

            for test_info in self.test_list:
                result = self.run_single_test(test_info, env_key)

                test_name = test_info["test_name"]
                if self.test_results[test_name]["test_number"] is None:
                    self.test_results[test_name]["test_number"] = result["test_number"]

                if result["status"] == "PASSED":
                    self.test_results[test_name][env_key] = result["elapsed_time"]
                else:
                    self.test_results[test_name][env_key] = result["status"]

    def save_results_csv(self):
        """Save test results to CSV file."""
        csv_file = self.output_base_dir / f"results_{self.timestamp}.csv"

        with open(csv_file, "w", newline="") as f:
            header = ["Test_Number", "Test_Name"]
            for env_key in ENV_CONFIGS.keys():
                header.append(env_key)

            writer = csv.writer(f)
            writer.writerow(header)

            for test_name, data in self.test_results.items():
                row = [data.get("test_number", ""), test_name]

                for env_key in ENV_CONFIGS.keys():
                    value = data.get(env_key, "N/A")
                    if isinstance(value, float):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(value)

                writer.writerow(row)

        print(f"\nResults saved to: {csv_file}")
        return csv_file

    def print_summary(self):
        """Print a summary of test results."""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        for env_key in ENV_CONFIGS.keys():
            passed = 0
            failed = 0
            timeout = 0
            error = 0
            total_time = 0

            for test_name, data in self.test_results.items():
                value = data.get(env_key, "N/A")
                if isinstance(value, float):
                    passed += 1
                    total_time += value
                elif value == "FAILED":
                    failed += 1
                elif value == "TIMEOUT":
                    timeout += 1
                elif value == "ERROR":
                    error += 1

            total = passed + failed + timeout + error
            if total > 0:
                print(f"\n{env_key}:")
                print(f"  Total: {total}")
                print(f"  Passed: {passed} ({passed*100/total:.1f}%)")
                if failed > 0:
                    print(f"  Failed: {failed}")
                if timeout > 0:
                    print(f"  Timeout: {timeout}")
                if error > 0:
                    print(f"  Error: {error}")
                print(f"  Total Time: {total_time:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Run tests for Triton repositories")
    parser.add_argument(
        "--repos",
        nargs="+",
        choices=["liger_kernel", "flag_gems", "tritonbench", "all"],
        default=["all"],
        help="Repositories to test"
    )
    parser.add_argument(
        "--output-dir",
        default="test_outputs",
        help="Base directory for test outputs"
    )
    parser.add_argument(
        "--config-groups",
        nargs="+",
        choices=["baseline", "compute_sanitizer", "triton_sanitizer", "kernel_time", "kernel_time_liger_kernel", "kernel_time_tritonbench", "ablation_studies", "all"],
        default=["baseline"],
        help="Configuration groups to run"
    )
    parser.add_argument(
        "--whitelist",
        help="Whitelist file for specific tests (e.g., flag_gems_whitelist.txt)"
    )
    parser.add_argument(
        "--whitelist-repo",
        help="Repository to apply whitelist to (e.g., flag_gems)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean all generated test output directories"
    )
    parser.add_argument(
        "--clean-pattern",
        default="test_outputs*",
        help="Pattern for directories to clean (default: test_outputs*)"
    )

    args = parser.parse_args()

    # Handle cleanup first
    if args.clean:
        import glob
        import shutil

        print("Cleaning generated files...")
        print("=" * 60)

        # Find all directories matching the pattern
        dirs_to_remove = glob.glob(args.clean_pattern)

        if not dirs_to_remove:
            print(f"No directories found matching pattern: {args.clean_pattern}")
        else:
            print(f"Found {len(dirs_to_remove)} directories to remove:")
            for dir_path in sorted(dirs_to_remove):
                print(f"  - {dir_path}")

            print("\nAre you sure you want to delete these directories? (y/n)")
            response = input().strip().lower()

            if response == 'y':
                removed_count = 0
                for dir_path in dirs_to_remove:
                    try:
                        if Path(dir_path).is_dir():
                            shutil.rmtree(dir_path)
                            print(f"  ✓ Removed: {dir_path}")
                            removed_count += 1
                    except Exception as e:
                        print(f"  ✗ Error removing {dir_path}: {e}")

                print(f"\n✓ Cleanup complete! Removed {removed_count} directories.")
            else:
                print("Cleanup cancelled.")

        print("=" * 60)
        sys.exit(0)

    if "all" in args.repos:
        repos = list(REPO_CONFIGS.keys())
    else:
        repos = args.repos

    if "all" in args.config_groups:
        config_groups = ["baseline", "compute_sanitizer", "triton_sanitizer"]
    else:
        config_groups = args.config_groups

    # Load whitelist if specified
    whitelists = {}
    runner_temp = TestRunner()

    if args.whitelist and args.whitelist_repo:
        whitelist = runner_temp.load_whitelist(args.whitelist, args.whitelist_repo)
        if whitelist:
            whitelists[args.whitelist_repo] = whitelist
            print(f"Loaded whitelist for {args.whitelist_repo} from {args.whitelist}")

    # Auto-load whitelists if they exist
    if Path("liger_kernel_whitelist.txt").exists() and "liger_kernel" in repos:
        whitelist = runner_temp.load_whitelist("liger_kernel_whitelist.txt", "liger_kernel")
        if whitelist and "liger_kernel" not in whitelists:
            whitelists["liger_kernel"] = whitelist
            print(f"Auto-loaded whitelist for liger_kernel (27 tests)")

    if Path("flag_gems_whitelist.txt").exists() and "flag_gems" in repos:
        whitelist = runner_temp.load_whitelist("flag_gems_whitelist.txt", "flag_gems")
        if whitelist and "flag_gems" not in whitelists:
            whitelists["flag_gems"] = whitelist
            print(f"Auto-loaded whitelist for flag_gems (20 tests)")

    if Path("tritonbench_whitelist.txt").exists() and "tritonbench" in repos:
        whitelist = runner_temp.load_whitelist("tritonbench_whitelist.txt", "tritonbench")
        if whitelist and "tritonbench" not in whitelists:
            whitelists["tritonbench"] = whitelist
            print(f"Auto-loaded whitelist for tritonbench (64 files)")

    runner = TestRunner(output_base_dir=args.output_dir)

    print(f"Starting test run")
    print(f"Output directory: {args.output_dir}")
    print(f"Repositories: {', '.join(repos)}")
    print(f"Configuration groups: {', '.join(config_groups)}")
    if whitelists:
        print(f"Using whitelist for: {', '.join(whitelists.keys())}")

    runner.run_all_tests(repos, config_groups, whitelists)
    runner.save_results_csv()
    runner.print_summary()

if __name__ == "__main__":
    main()