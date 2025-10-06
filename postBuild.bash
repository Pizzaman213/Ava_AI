#!/bin/bash
# This file contains bash commands that will be executed at the end of the container build process,
# after all system packages and programming language specific package have been installed.
#
# Note: This file may be removed if you don't need to use it


# Download and install nvm:
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
# in lieu of restarting the shell
\. "$HOME/.nvm/nvm.sh"
# Download and install Node.js:
nvm install 24
# Verify the Node.js version:
node -v # Should print "v24.8.0".
# Verify npm version:
npm -v # Should print "11.6.0

npm install -g @anthropic-ai/claude-code

sudo apt-get update && sudo apt-get install -y build-essential

set -e

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install transformer_engine_torch with proper CUDA settings
pip install --user --no-cache-dir transformer_engine_torch || {
    echo "Failed to install transformer_engine_torch from source, trying pre-built wheel..."
    pip install --user --only-binary=:all: transformer_engine_torch || {
        echo "Could not install transformer_engine_torch, skipping..."
    }
}
