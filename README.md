# ASPLOS-26-AE

## Setup

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone submodules

```bash
git submodule init && git submodule update
```

### AMD Docker Environment

Start the Docker container:

```bash
./start_docker_rocm_asan.sh
docker exec -it rocm_asan bash
```

To set up the virtual environment using system torch/triton in AMD Docker:

```bash
# 1. Create virtual environment
uv venv .venv

# 2. Link system torch/triton (no activation needed)
./amd_docker_link_torch.sh

# 3. Sync other dependencies (skip torch and triton)
uv sync --no-install-package torch --extra rocm

# 4. Activate the environment
source .venv/bin/activate
```

### Test Address Sanitizer

To verify the address sanitizer is correctly installed, run:

```bash
python test_asan.py
```

This test intentionally triggers an out-of-bounds access. If ASAN is working correctly, it will detect and report the memory violation.

### CUDA Environment

```bash
uv venv .venv
uv sync --extra cuda
source .venv/bin/activate
```
