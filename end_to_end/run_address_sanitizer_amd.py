#!/usr/bin/env python3
"""
Run address sanitizer end-to-end experiments for all repositories.
Results are saved to end_to_end/results/address_sanitizer/

This script runs the same configurations as baseline but with TRITON_ENABLE_ASAN=1
to enable Address Sanitizer for detecting memory errors.
"""

import os
import subprocess
import time
import ast
import csv
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

# Add project root to path for imports
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.test_registry import REPO_CONFIGS, get_configs_by_group
from utils.misc import EASTERN_TZ

# Get baseline configurations from registry
ENV_CONFIGS = get_configs_by_group("baseline")

# Memory profiling prefix
MEMORY_PROFILE_PREFIX = "/usr/bin/time -v"


class AddressSanitizerRunner:
    def __init__(self, enable_memory=False):
        self.script_dir = Path(__file__).parent.absolute()
        self.project_root = self.script_dir.parent
        self.output_base_dir = self.script_dir / "results" / "address_sanitizer"
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now(EASTERN_TZ).strftime("%Y%m%d_%H%M%S")
        self.global_test_counter = 0
        self.total_tests = 0
        self.test_results = OrderedDict()
        self.test_list = []
        self.enable_memory = enable_memory

    def load_whitelist(self, whitelist_file, repo_name):
        """Load test whitelist from file."""
        whitelist = {}
        whitelist_path = self.project_root / whitelist_file

        if not whitelist_path.exists():
            return None

        with open(whitelist_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                if repo_name == "tritonbench":
                    file_name = Path(line).stem
                    whitelist[file_name] = []
                elif "::" in line:
                    test_file, test_function = line.split("::")
                    test_file = Path(test_file).stem
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
        test_dir = self.project_root / config["test_dir"]

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

                if whitelist and test_file_stem in whitelist:
                    if repo == "tritonbench":
                        self.test_list.append({
                            "repository": repo,
                            "test_file": test_file,
                            "test_function": None,
                            "test_name": f"{repo}/{test_file_stem}"
                        })
                    else:
                        for test_function in whitelist[test_file_stem]:
                            self.test_list.append({
                                "repository": repo,
                                "test_file": test_file,
                                "test_function": test_function,
                                "test_name": f"{repo}/{test_file_stem}/{test_function}"
                            })
                elif whitelist and repo == "tritonbench":
                    continue
                elif not whitelist and repo in ["liger_kernel", "flag_gems"]:
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

        output_dir = self.output_base_dir / env_config["name"]
        output_dir.mkdir(parents=True, exist_ok=True)

        if test_function:
            output_filename = f"{test_number}_{repo_name}_{test_file.stem}_{test_function}.log"
        else:
            output_filename = f"{test_number}_{repo_name}_{test_file.stem}.log"

        output_file = output_dir / output_filename

        env = os.environ.copy()
        env.update(env_config["env"])

        # Enable Address Sanitizer
        env["TRITON_ENABLE_ASAN"] = "1"
        env["HSA_XNACK"] = "1"

        # Convert CUDA to HIP memory caching env vars for AMD
        if env.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1":
            env["PYTORCH_NO_HIP_MEMORY_CACHING"] = "1"
            env["HSA_DISABLE_FRAGMENT_ALLOCATOR"] = "1"
            env["AMD_PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
            env["AMDGCN_USE_BUFFER_OPS"] = "0"
        env.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)

        # Add project root to PYTHONPATH
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{self.project_root}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = str(self.project_root)

        test_dir = self.project_root / config["test_dir"]

        if repo_name == "tritonbench" and config.get("special_handling"):
            relative_path = test_file.relative_to(self.project_root / config["test_dir"])
            cmd = ["python", str(relative_path)]
        else:
            if test_function:
                cmd = ["pytest", "-s", "--assert=plain", f"{test_file.name}::{test_function}"]
            else:
                cmd = config["test_command"].split() + [test_file.name]

        # Only use memory profiling prefix if --memory flag is set
        if self.enable_memory:
            cmd = MEMORY_PROFILE_PREFIX.split() + cmd

        if test_function:
            test_display = f"{test_file.name}::{test_function}"
        else:
            test_display = test_file.name

        print(f"  [{self.global_test_counter}/{self.total_tests}] [{repo_name}] Running: {test_display}")

        start_time = time.time()

        try:
            with open(output_file, "w") as log_file:
                log_file.write(f"Test Number: {test_number}\n")
                log_file.write(f"Test: {test_info['test_name']}\n")
                log_file.write(f"Environment: {env_config_key}\n")
                log_file.write(f"TRITON_ENABLE_ASAN: 1\n")
                log_file.write(f"HSA_XNACK: 1\n")
                log_file.write(f"Command: {' '.join(cmd)}\n")
                log_file.write(f"Start Time: {datetime.now(EASTERN_TZ).isoformat()}\n")
                log_file.write("=" * 80 + "\n")
                log_file.flush()

                result = subprocess.run(
                    cmd,
                    env=env,
                    cwd=test_dir,
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

        except Exception as e:
            elapsed_time = time.time() - start_time
            status = "ERROR"
            error_msg = str(e)

        with open(output_file, "a") as log_file:
            log_file.write("\n" + "=" * 80 + "\n")
            log_file.write(f"End Time: {datetime.now(EASTERN_TZ).isoformat()}\n")
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

    def run_all_tests(self, repositories, whitelists=None):
        """Run all tests with address sanitizer configurations."""
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

        print(f"\nRunning tests with {len(ENV_CONFIGS)} configurations (TRITON_ENABLE_ASAN=1)")
        print("=" * 60)

        for env_key, env_config in ENV_CONFIGS.items():
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
        print("ADDRESS SANITIZER TEST SUMMARY")
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
    parser = argparse.ArgumentParser(description="Run address sanitizer end-to-end experiments")
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Enable memory profiling with /usr/bin/time -v"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Running Address Sanitizer End-to-End Experiments")
    print("TRITON_ENABLE_ASAN: 1")
    print("HSA_XNACK: 1")
    if args.memory:
        print("Memory profiling: ENABLED")
    print("=" * 60)
    print()

    runner = AddressSanitizerRunner(enable_memory=args.memory)

    # Auto-load whitelists
    whitelists = {}
    repos = list(REPO_CONFIGS.keys())

    for repo in repos:
        whitelist_file = f"utils/{repo}_whitelist.txt"
        whitelist = runner.load_whitelist(whitelist_file, repo)
        if whitelist:
            whitelists[repo] = whitelist
            print(f"Loaded whitelist for {repo}")

    print(f"\nOutput directory: {runner.output_base_dir}")
    print(f"Repositories: {', '.join(repos)}")
    print()

    runner.run_all_tests(repos, whitelists)
    runner.save_results_csv()
    runner.print_summary()

    print("\n" + "=" * 60)
    print("Address sanitizer experiments completed!")
    print(f"Results saved in: {runner.output_base_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
