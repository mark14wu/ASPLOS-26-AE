#!/bin/bash

docker run -d -t \
    --name rocm_asan \
    --restart unless-stopped \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --ipc=host \
    --shm-size 8G \
    -v "$PWD":/workspace \
    -w /workspace \
    pmylonamd/rocm7.0_ubuntu22.04_py3.10_pytorch_2.8.0_asan \
    sleep infinity
