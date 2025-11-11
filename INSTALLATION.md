# Installation Guide for Ava AI (MoE-LLM Advanced)

## Quick Start

To install all dependencies in one command:

```bash
cd /project
./install_dependencies.sh
```

This script will automatically:
- Check system prerequisites (Python 3.8+, CUDA if available)
- Install system packages (build tools, MPI, etc.)
- Install all Python dependencies
- Set up DeepSpeed for distributed training
- Configure monitoring tools (W&B, TensorBoard)
- Verify the installation
- Create necessary directories

## What Gets Installed

### System Packages (via apt)
- **Build Tools**: build-essential, git, wget, curl
- **Development**: python3-dev, python3-pip
- **Distributed Computing**: libopenmpi-dev, openmpi-bin, libaio-dev
- **Utilities**: vim, htop, tmux

### Core ML Frameworks
- **PyTorch** (>=2.0.0) - Deep learning framework
- **Transformers** (>=4.30.0) - Hugging Face transformers
- **Datasets** (>=2.0.0) - Dataset loading and processing
- **Accelerate** (>=0.20.0) - Distributed training utilities

### Training Infrastructure
- **DeepSpeed** (>=0.9.0) - Distributed training optimization
- **mpi4py** - MPI bindings for distributed computing
- **Weights & Biases** (>=0.15.0) - Experiment tracking
- **TensorBoard** (>=2.13.0) - Training visualization

### Optimization Libraries
- **torchao** - PyTorch architecture optimizations
- **bitsandbytes** - Quantization and optimization
- **torch-lr-finder** - Learning rate finder

### Data Processing
- **JupyterLab** (>3.0) - Interactive development
- **pandas**, **numpy**, **pyarrow** - Data manipulation
- **xxhash**, **langdetect**, **chardet**, **ftfy** - Text processing

### Development Tools (Optional)
- **black**, **isort** - Code formatting
- **flake8**, **mypy** - Linting and type checking
- **pytest**, **pytest-cov** - Testing

## Manual Installation Steps

If you prefer to install manually or the script fails:

### 1. Update System Packages
```bash
sudo apt-get update
sudo apt-get install -y build-essential git python3-dev python3-pip \
    libopenmpi-dev openmpi-bin libaio-dev
```

### 2. Install Python Dependencies
```bash
cd /project
pip install -r requirements.txt
```

### 3. Install DeepSpeed (Optional but Recommended)
```bash
pip install deepspeed>=0.9.0
```

### 4. Install Development Tools (Optional)
```bash
pip install black isort flake8 mypy pytest pytest-cov
```

## Verification

After installation, verify everything works:

```bash
# Check Python packages
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import datasets; print(f'Datasets: {datasets.__version__}')"

# Check CUDA availability (if applicable)
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Check DeepSpeed
python3 -c "import deepspeed; print(f'DeepSpeed: {deepspeed.__version__}')"
```

## Post-Installation Setup

### 1. Configure Weights & Biases
```bash
wandb login
```

### 2. Download Training Data
```bash
python code/scripts/1_data_download/unified_download.py
```

### 3. Setup Training Data
```bash
./setup_training_data.sh
```

### 4. Start Training
```bash
./run_training.sh
```

## Troubleshooting

### CUDA Not Available
If PyTorch is installed but CUDA is not available:
```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### DeepSpeed Installation Fails
DeepSpeed requires specific system dependencies:
```bash
sudo apt-get install -y libaio-dev
pip install deepspeed --no-cache-dir
```

### MPI4PY Installation Fails
Ensure OpenMPI is properly installed:
```bash
sudo apt-get install -y libopenmpi-dev openmpi-bin
pip install mpi4py --no-cache-dir
```

### Memory Issues During Installation
If you encounter memory issues:
```bash
# Install packages one at a time
pip install torch
pip install transformers
pip install deepspeed
# etc.
```

### Permission Errors
If you encounter permission errors:
```bash
# Don't run the entire script with sudo, but you may need sudo for apt commands
sudo apt-get install <package>
pip install --user <python-package>
```

## Environment Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (optional but recommended for GPU training)
- **RAM**: 16GB minimum, 32GB+ recommended
- **Disk Space**: 50GB+ for dependencies and datasets
- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended

## Directory Structure

The installation script creates these directories:
```
/project/
├── code/
│   ├── data/
│   │   ├── raw/              # Raw downloaded datasets
│   │   └── processed/        # Processed training data
│   ├── models/
│   │   └── tokenizer/        # Tokenizer models
│   └── outputs/
│       ├── runs/             # Training runs
│       └── checkpoints/      # Model checkpoints
```

## Additional Resources

- **Documentation**: `/project/docs/README.md`
- **Configuration Examples**: `/project/code/configs/examples/`
- **Training Scripts**: `/project/code/scripts/5_training/`
- **Data Download**: `/project/code/scripts/1_data_download/`

## Support

If you encounter issues:
1. Check the error messages from the installation script
2. Verify system requirements are met
3. Check the troubleshooting section above
4. Review logs in `/tmp/` or installation output

## Quick Reference Commands

```bash
# Full installation
./install_dependencies.sh

# Verify installation
python3 -c "import torch, transformers, datasets; print('OK')"

# Check GPU
nvidia-smi

# Login to W&B
wandb login

# Download data
python code/scripts/1_data_download/unified_download.py

# Setup training
./setup_training_data.sh

# Run training
./run_training.sh

# Monitor training
tensorboard --logdir outputs/
```
