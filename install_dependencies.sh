#!/bin/bash
################################################################################
# Dependency Installation Script for Ava AI (MoE-LLM Advanced)
################################################################################
# This script installs all required dependencies for the project including:
# - System packages (apt)
# - Python packages (pip)
# - Development tools
# - ML/AI libraries (PyTorch, Transformers, DeepSpeed, etc.)
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

print_step() {
    echo -e "${BLUE}▸${NC} $1"
}

################################################################################
# Check Prerequisites
################################################################################

print_header "Dependency Installation for Ava AI"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run as root. Use sudo when needed."
    exit 1
fi

# Check Python version
print_step "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
    print_success "Python $PYTHON_VERSION detected (>= 3.8 required)"
else
    print_error "Python 3.8+ required, found $PYTHON_VERSION"
    exit 1
fi

# Check if CUDA is available
print_step "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "unknown")
    print_success "CUDA detected: $CUDA_VERSION"
    HAS_CUDA=true
else
    print_info "CUDA not detected - will install CPU-only versions"
    HAS_CUDA=false
fi

################################################################################
# System Dependencies (APT)
################################################################################

print_header "Installing System Dependencies"

print_step "Updating package lists..."
if sudo apt-get update -qq; then
    print_success "Package lists updated"
else
    print_error "Failed to update package lists"
    exit 1
fi

print_step "Installing system packages..."
SYSTEM_PACKAGES=(
    "build-essential"           # Compilation tools
    "git"                       # Version control
    "wget"                      # File downloads
    "curl"                      # HTTP requests
    "vim"                       # Text editor
    "htop"                      # System monitoring
    "tmux"                      # Terminal multiplexer
    "python3-dev"               # Python development headers
    "python3-pip"               # Python package manager
    "libaio-dev"                # Async I/O (for DeepSpeed)
    "libopenmpi-dev"            # MPI development (for distributed training)
    "openmpi-bin"               # MPI binaries
)

for package in "${SYSTEM_PACKAGES[@]}"; do
    if dpkg -l | grep -q "^ii  $package "; then
        print_success "$package already installed"
    else
        print_step "Installing $package..."
        if sudo apt-get install -y -qq "$package" 2>&1 | grep -v "^debconf:"; then
            print_success "$package installed"
        else
            print_error "Failed to install $package"
        fi
    fi
done

################################################################################
# Python Package Upgrades
################################################################################

print_header "Upgrading Python Package Managers"

print_step "Upgrading pip, setuptools, and wheel..."
if python3 -m pip install --upgrade pip setuptools wheel --quiet; then
    print_success "Package managers upgraded"
else
    print_error "Failed to upgrade package managers"
fi

################################################################################
# Core Python Dependencies
################################################################################

print_header "Installing Core Python Dependencies"

print_step "Installing from requirements.txt..."
if [ -f "/project/requirements.txt" ]; then
    # Install from requirements.txt
    if python3 -m pip install -r /project/requirements.txt; then
        print_success "Requirements.txt dependencies installed"
    else
        print_error "Failed to install some dependencies from requirements.txt"
        print_info "Continuing with manual installation..."
    fi
else
    print_info "No requirements.txt found, will install manually"
fi

################################################################################
# ML/AI Framework Dependencies
################################################################################

print_header "Installing ML/AI Frameworks"

# Core ML packages
ML_PACKAGES=(
    "torch>=2.0.0"              # PyTorch
    "transformers>=4.30.0"      # Hugging Face Transformers
    "datasets>=2.0.0"           # Hugging Face Datasets
    "accelerate>=0.20.0"        # Hugging Face Accelerate
    "sentencepiece>=0.1.99"     # Tokenization
    "tokenizers>=0.13.0"        # Fast tokenizers
    "safetensors>=0.3.0"        # Safe model serialization
)

print_step "Installing core ML packages..."
for package in "${ML_PACKAGES[@]}"; do
    package_name=$(echo $package | cut -d'>' -f1 | cut -d'=' -f1 | cut -d'<' -f1)
    print_step "Installing $package_name..."
    if python3 -m pip install "$package" --quiet; then
        print_success "$package_name installed"
    else
        print_error "Failed to install $package_name (may need to continue manually)"
    fi
done

################################################################################
# DeepSpeed for Distributed Training
################################################################################

print_header "Installing DeepSpeed"

print_step "Installing DeepSpeed (may take a few minutes)..."
if python3 -m pip install deepspeed>=0.9.0 --quiet --no-cache-dir; then
    print_success "DeepSpeed installed"
else
    print_error "DeepSpeed installation failed"
    print_info "This is optional - you can continue without it"
fi

################################################################################
# Optimization & Quantization Libraries
################################################################################

print_header "Installing Optimization Libraries"

OPT_PACKAGES=(
    "torchao"                   # PyTorch optimizations
    "bitsandbytes>=0.41.0"      # Quantization
    "scipy>=1.7.0"              # Scientific computing
    "scikit-learn>=1.0.0"       # ML utilities
)

for package in "${OPT_PACKAGES[@]}"; do
    package_name=$(echo $package | cut -d'>' -f1 | cut -d'=' -f1 | cut -d'<' -f1)
    print_step "Installing $package_name..."
    if python3 -m pip install "$package" --quiet 2>&1 | grep -v "WARNING"; then
        print_success "$package_name installed"
    else
        print_info "$package_name skipped (may not be available for this platform)"
    fi
done

################################################################################
# Training & Monitoring Tools
################################################################################

print_header "Installing Training & Monitoring Tools"

TRAIN_PACKAGES=(
    "wandb>=0.15.0"             # Weights & Biases
    "tensorboard>=2.13.0"       # TensorBoard
    "torch-lr-finder>=0.2.1"    # Learning rate finder
    "matplotlib>=3.5.0"         # Plotting
    "tqdm>=4.65.0"              # Progress bars
)

for package in "${TRAIN_PACKAGES[@]}"; do
    package_name=$(echo $package | cut -d'>' -f1 | cut -d'=' -f1 | cut -d'<' -f1)
    print_step "Installing $package_name..."
    if python3 -m pip install "$package" --quiet; then
        print_success "$package_name installed"
    else
        print_error "Failed to install $package_name"
    fi
done

################################################################################
# MPI for Distributed Training
################################################################################

print_header "Installing MPI4PY for Distributed Training"

print_step "Installing mpi4py..."
if python3 -m pip install mpi4py --quiet 2>&1 | grep -v "WARNING"; then
    print_success "mpi4py installed"
else
    print_info "mpi4py installation failed (optional)"
fi

################################################################################
# Data Processing Libraries
################################################################################

print_header "Installing Data Processing Libraries"

DATA_PACKAGES=(
    "jupyterlab>3.0"            # JupyterLab
    "xxhash"                    # Fast hashing
    "langdetect"                # Language detection
    "chardet"                   # Character encoding detection
    "ftfy"                      # Text fixing
    "pandas>=1.5.0"             # Data manipulation
    "numpy>=1.23.0"             # Numerical computing
    "pyarrow>=10.0.0"           # Arrow format
)

for package in "${DATA_PACKAGES[@]}"; do
    package_name=$(echo $package | cut -d'>' -f1 | cut -d'=' -f1 | cut -d'<' -f1)
    print_step "Installing $package_name..."
    if python3 -m pip install "$package" --quiet; then
        print_success "$package_name installed"
    else
        print_error "Failed to install $package_name"
    fi
done

################################################################################
# Development Tools (Optional)
################################################################################

print_header "Installing Development Tools (Optional)"

DEV_PACKAGES=(
    "black>=23.0.0"             # Code formatter
    "isort>=5.12.0"             # Import sorter
    "flake8>=6.1.0"             # Linter
    "mypy>=1.7.0"               # Type checker
    "pytest>=7.4.0"             # Testing framework
    "pytest-cov>=4.1.0"         # Coverage
)

print_info "Installing development tools (can be skipped)..."
for package in "${DEV_PACKAGES[@]}"; do
    package_name=$(echo $package | cut -d'>' -f1 | cut -d'=' -f1 | cut -d'<' -f1)
    if python3 -m pip install "$package" --quiet 2>&1 > /dev/null; then
        print_success "$package_name installed"
    else
        print_info "$package_name skipped"
    fi
done

################################################################################
# Verification
################################################################################

print_header "Verifying Installation"

print_step "Checking critical packages..."

# Function to check package
check_package() {
    local package=$1
    local import_name=${2:-$package}

    if python3 -c "import $import_name" 2>/dev/null; then
        local version=$(python3 -c "import $import_name; print(getattr($import_name, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
        print_success "$package ($version)"
        return 0
    else
        print_error "$package not found"
        return 1
    fi
}

CRITICAL_PACKAGES=(
    "torch"
    "transformers"
    "datasets"
    "wandb"
    "tqdm"
)

FAILURES=0
for package in "${CRITICAL_PACKAGES[@]}"; do
    if ! check_package "$package"; then
        FAILURES=$((FAILURES + 1))
    fi
done

# Check CUDA availability in PyTorch
if [ "$HAS_CUDA" = true ]; then
    print_step "Checking PyTorch CUDA support..."
    if python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in PyTorch'" 2>/dev/null; then
        CUDA_DEVICES=$(python3 -c "import torch; print(torch.cuda.device_count())")
        print_success "PyTorch CUDA enabled ($CUDA_DEVICES device(s) available)"
    else
        print_error "PyTorch installed but CUDA support not available"
        print_info "You may need to reinstall PyTorch with CUDA support"
    fi
fi

################################################################################
# Post-Installation Setup
################################################################################

print_header "Post-Installation Setup"

print_step "Creating necessary directories..."
mkdir -p /project/code/data/raw
mkdir -p /project/code/data/processed
mkdir -p /project/code/models/tokenizer
mkdir -p /project/code/outputs/runs
mkdir -p /project/code/outputs/checkpoints
print_success "Directories created"

# Check if W&B needs login
print_step "Checking Weights & Biases setup..."
if command -v wandb &> /dev/null; then
    if wandb verify 2>&1 | grep -q "logged in"; then
        print_success "W&B already configured"
    else
        print_info "W&B login required - run 'wandb login' when ready"
    fi
fi

################################################################################
# Summary
################################################################################

print_header "Installation Summary"

if [ $FAILURES -eq 0 ]; then
    print_success "All critical dependencies installed successfully!"
    echo ""
    print_info "Next steps:"
    echo "  1. Login to Weights & Biases: wandb login"
    echo "  2. Download training data: python code/scripts/1_data_download/unified_download.py"
    echo "  3. Setup training data: ./setup_training_data.sh"
    echo "  4. Start training: ./run_training.sh"
    echo ""
    print_success "Installation complete!"
else
    print_error "$FAILURES critical package(s) failed to install"
    print_info "Please check the errors above and install manually if needed"
    exit 1
fi

echo ""
print_info "For help, see: /project/docs/README.md"
echo ""

################################################################################
# Optional: Display system info
################################################################################

print_header "System Information"

echo "Python Version: $PYTHON_VERSION"
echo "Working Directory: $(pwd)"
if [ "$HAS_CUDA" = true ]; then
    echo "CUDA Version: $CUDA_VERSION"
    echo "GPU(s):"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | nl
fi

echo ""
print_success "Setup script completed!"
echo ""
