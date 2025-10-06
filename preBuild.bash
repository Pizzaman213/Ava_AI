#!/bin/bash
set -e

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential \
    ninja-build \
    cuda-toolkit-12-0
    
sudo rm -rf /var/lib/apt/lists/*