# Installation Guide

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Environment Setup](#environment-setup)
- [Dependency Management](#dependency-management)
- [GPU Configuration](#gpu-configuration)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ / Windows 10+ / macOS 12+
- **Python**: 3.8 or higher
- **RAM**: 32GB minimum
- **GPU**: NVIDIA GPU with 16GB+ VRAM (for training)
- **Storage**: 100GB free space (more for large datasets)
- **CUDA**: 11.8 or higher (for GPU support)

### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.10
- **RAM**: 64GB or more
- **GPU**: 8x NVIDIA A100 80GB (for large models)
- **Storage**: 1TB NVMe SSD
- **CUDA**: 12.1 with cuDNN 8.9

## Installation Methods

### Method 1: pip install (Recommended)

```bash
# Create and activate virtual environment
python -m venv moe_env
source moe_env/bin/activate  # On Windows: moe_env\Scripts\activate

# Install the package
pip install -e .

# Install optional dependencies
pip install -e ".[dev]"  # Development tools
pip install -e ".[rlhf]"  # RLHF components
pip install -e ".[distributed]"  # Multi-GPU training
```

### Method 2: conda install

```bash
# Create conda environment
conda create -n moe_llm python=3.10
conda activate moe_llm

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install the package
pip install -e .
```

### Method 3: Docker

```bash
# Build Docker image
docker build -t moe-llm:latest .

# Run container with GPU support
docker run --gpus all -it -v $(pwd):/workspace moe-llm:latest

# Inside container
cd /workspace
pip install -e .
```

### Method 4: Development Installation

```bash
# Clone repository
git clone https://github.com/your-username/moe-llm.git
cd moe-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

## Environment Setup

### Setting up CUDA

```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Set CUDA paths (add to ~/.bashrc)
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### Flash Attention Setup

```bash
# Install Flash Attention 2
pip install flash-attn --no-build-isolation

# Alternative: install from source
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

### DeepSpeed Setup

```bash
# Install DeepSpeed with all ops
pip install deepspeed

# Build DeepSpeed ops
ds_report  # Check installation

# For NVMe support
sudo apt-get install libaio-dev
pip install deepspeed[nvme]
```

### Apex Installation (Optional)

```bash
# For mixed precision training
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

## Dependency Management

### Core Dependencies

```bash
# requirements.txt
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
deepspeed>=0.12.0
flash-attn>=2.3.0
einops>=0.7.0
scipy>=1.10.0
numpy<2.0.0
tqdm>=4.65.0
wandb>=0.16.0
```

### RLHF Dependencies

```bash
# requirements-rlhf.txt
trlx>=0.7.0
trl>=0.7.0
peft>=0.7.0
bitsandbytes>=0.41.0
```

### RAG Dependencies

```bash
# requirements-rag.txt
faiss-cpu>=1.7.4  # or faiss-gpu
chromadb>=0.4.0
langchain>=0.0.340
sentence-transformers>=2.2.0
```

### Installing Specific Versions

```bash
# Pin specific versions for reproducibility
pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.35.2
pip install deepspeed==0.12.3
```

## GPU Configuration

### Multi-GPU Setup

```bash
# Check available GPUs
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"

# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# NCCL settings for distributed training
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
```

### Memory Management

```bash
# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable TF32 for Ampere GPUs
export NVIDIA_TF32_OVERRIDE=1

# Set memory fraction
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,backend:cudaMallocAsync
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Solutions:
# - Reduce batch size
# - Enable gradient checkpointing
# - Use DeepSpeed ZeRO-3
# - Enable CPU offloading

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

#### 2. Flash Attention Build Errors

```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install -y build-essential ninja-build

# Use pre-built wheels
pip install flash-attn --no-build-isolation --force-reinstall
```

#### 3. DeepSpeed Compatibility

```bash
# Check compatibility
ds_report

# Reinstall with specific CUDA version
DS_BUILD_CUDA_EXT=1 pip install deepspeed --force-reinstall
```

#### 4. Import Errors

```python
# Test imports
python -c "
import torch
import transformers
import deepspeed
import flash_attn
print('All imports successful!')
"
```

### Performance Optimization

```bash
# Enable persistent workers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optimize NCCL
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_DISABLE=0
export NCCL_TREE_THRESHOLD=0

# Use faster dataloader
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
```

### Verification Script

```python
# verify_installation.py
import sys
import torch
import transformers
import deepspeed

def check_installation():
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"DeepSpeed: {deepspeed.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test basic operations
    if torch.cuda.is_available():
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = torch.matmul(x, y)
        print("GPU computation test: PASSED")
    
    print("\nInstallation verified successfully!")

if __name__ == "__main__":
    check_installation()
```

## Next Steps

After successful installation:

1. **Configure your environment**: Set up configuration files in `configs/`
2. **Prepare your data**: Follow the [Data Preparation Guide](data_preparation.md)
3. **Start training**: See the [Training Guide](training.md)
4. **Monitor progress**: Use TensorBoard or Weights & Biases

For more detailed information on specific components:
- [Architecture Overview](architecture.md)
- [Training Guide](training.md)
- [Inference Guide](inference.md)
- [RLHF Pipeline](rlhf.md)