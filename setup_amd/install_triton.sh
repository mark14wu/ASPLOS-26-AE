#!/bin/bash
set -e

pip uninstall -y triton pytorch-triton-rocm

cd ../submodules/triton/
python -m pip install -r python/requirements.txt
python -m pip install -r python/test-requirements.txt

python -m pip install -e . --no-build-isolation -v
