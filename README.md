# ASPLOS-26-AE

## Setup

### AMD Docker Environment

To set up the virtual environment using system torch/triton in AMD Docker:

```bash
# 1. Create virtual environment
uv venv .venv

# 2. Link system torch/triton (no activation needed)
./amd_docker_link_torch.sh

# 3. Sync other dependencies (skip torch and triton)
uv sync --no-install-package torch --no-install-package triton --extra rocm

# 4. Activate the environment
source .venv/bin/activate
```

### CUDA Environment

```bash
uv venv .venv
uv sync --extra cuda
source .venv/bin/activate
```
