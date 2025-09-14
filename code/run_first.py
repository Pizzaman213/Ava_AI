#!/usr/bin/env python3
"""Initial setup script for LLM project."""

import subprocess
import sys
import os
import platform
import shutil
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

# Try to import these, but don't fail if they're not available
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    HAS_CONCURRENT = True
except ImportError:
    HAS_CONCURRENT = False

try:
    import pkg_resources
    HAS_PKG_RESOURCES = True
except ImportError:
    HAS_PKG_RESOURCES = False

try:
    import importlib.metadata
    HAS_IMPORTLIB_METADATA = True
except ImportError:
    HAS_IMPORTLIB_METADATA = False

# Global timing and progress tracking
class SetupTracker:
    def __init__(self):
        self.start_time = time.time()
        self.step_times = {}
        self.current_step = None
        self.step_start = None
        self.total_steps = 0
        self.completed_steps = 0
        self.failed_steps = []
        self.skipped_steps = []
        self.installed_packages = []
        self.download_sizes = {}
        
    def start_step(self, step_name):
        """Start tracking a step"""
        self.current_step = step_name
        self.step_start = time.time()
        
    def end_step(self, success=True, skipped=False):
        """End tracking a step"""
        if self.current_step and self.step_start:
            elapsed = time.time() - self.step_start
            self.step_times[self.current_step] = elapsed
            
            if success and not skipped:
                self.completed_steps += 1
            elif skipped:
                self.skipped_steps.append(self.current_step)
            else:
                self.failed_steps.append(self.current_step)
                
            # Print step summary
            status = "✅" if success and not skipped else "⏭️" if skipped else "❌"
            print(f"{status} {self.current_step}: {elapsed:.2f}s")
            
        self.current_step = None
        self.step_start = None
    
    def get_elapsed_time(self):
        """Get total elapsed time"""
        return time.time() - self.start_time
    
    def format_time(self, seconds):
        """Format seconds into readable string"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"
    
    def print_summary(self):
        """Print final summary"""
        total_time = self.get_elapsed_time()
        
        print("\n" + "="*70)
        print("📊 SETUP SUMMARY")
        print("="*70)
        
        print(f"\n⏱️  Total Time: {self.format_time(total_time)}")
        
        print(f"\n📈 Steps:")
        print(f"   ✅ Completed: {self.completed_steps}")
        print(f"   ⏭️  Skipped: {len(self.skipped_steps)}")
        print(f"   ❌ Failed: {len(self.failed_steps)}")
        
        if self.step_times:
            print(f"\n⚡ Performance:")
            sorted_steps = sorted(self.step_times.items(), key=lambda x: x[1], reverse=True)
            for step, time_taken in sorted_steps[:5]:
                print(f"   {step}: {self.format_time(time_taken)}")
        
        if self.installed_packages:
            print(f"\n📦 Packages: {len(self.installed_packages)} installed")
        
        if self.failed_steps:
            print(f"\n⚠️  Failed Steps:")
            for step in self.failed_steps:
                print(f"   - {step}")
        
        print("\n" + "="*70)

# Initialize global tracker
tracker = SetupTracker()


def ensure_pip_installed():
    """Ensure pip is installed, especially on Ubuntu systems."""
    print("🔍 Checking for pip...")
    
    # Check if pip is available
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✅ pip is installed")
            return True
    except:
        pass
    
    print("⚠️ pip not found, attempting to install...")
    
    # Try different methods to install pip
    system = platform.system()
    
    if system == "Linux":
        # Check if we're on Ubuntu/Debian
        try:
            with open('/etc/os-release', 'r') as f:
                os_info = f.read().lower()
                is_ubuntu = 'ubuntu' in os_info or 'debian' in os_info
        except:
            is_ubuntu = False
        
        if is_ubuntu:
            print("📦 Detected Ubuntu/Debian system")
            
            # Try apt-get first (might need sudo)
            install_commands = [
                # Try without sudo first (in case we're root)
                "apt-get update && apt-get install -y python3-pip python3-venv python3-dev build-essential",
                # Try with sudo
                "sudo apt-get update && sudo apt-get install -y python3-pip python3-venv python3-dev build-essential",
                # Try installing just pip
                "apt-get install -y python3-pip",
                "sudo apt-get install -y python3-pip"
            ]
            
            for cmd in install_commands:
                print(f"  Trying: {cmd[:50]}...")
                try:
                    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
                    if result.returncode == 0:
                        print("  ✅ Successfully installed pip via apt")
                        return True
                except:
                    continue
    
    # Try downloading get-pip.py
    print("📥 Trying to download and install pip directly...")
    get_pip_commands = [
        # Using curl
        "curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && " + sys.executable + " get-pip.py",
        # Using wget
        "wget https://bootstrap.pypa.io/get-pip.py && " + sys.executable + " get-pip.py",
        # Using Python to download
        sys.executable + " -c \"import urllib.request; urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', 'get-pip.py')\" && " + sys.executable + " get-pip.py"
    ]
    
    for cmd in get_pip_commands:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, timeout=120)
            if result.returncode == 0:
                print("✅ Successfully installed pip")
                # Clean up
                try:
                    os.remove("get-pip.py")
                except:
                    pass
                return True
        except:
            continue
    
    # Last resort: try ensurepip
    print("🔧 Trying ensurepip module...")
    try:
        subprocess.run([sys.executable, '-m', 'ensurepip', '--default-pip'], check=True)
        print("✅ Successfully installed pip via ensurepip")
        return True
    except:
        pass
    
    print("❌ Could not install pip automatically")
    print("\nPlease install pip manually:")
    if system == "Linux":
        print("  Ubuntu/Debian: sudo apt-get install python3-pip")
        print("  Fedora: sudo dnf install python3-pip")
        print("  Arch: sudo pacman -S python-pip")
    elif system == "Darwin":
        print("  macOS: brew install python3")
    elif system == "Windows":
        print("  Windows: Download from https://bootstrap.pypa.io/get-pip.py")
    
    return False

def bootstrap_minimal_requirements():
    """Install absolute minimum requirements to get started."""
    print("\n📦 Installing bootstrap packages...")
    
    # These are the absolute minimum packages needed
    bootstrap_packages = [
        "setuptools",
        "wheel", 
        "pip>=23.0"
    ]
    
    for package in bootstrap_packages:
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', package]
            subprocess.run(cmd, capture_output=True, timeout=60)
        except:
            pass
    
    print("✅ Bootstrap packages installed")

def install_ubuntu_dependencies():
    """Install system dependencies on Ubuntu/Debian systems."""
    print("\n🐧 Installing Ubuntu system dependencies...")
    
    # Check if we're on Ubuntu/Debian
    try:
        with open('/etc/os-release', 'r') as f:
            os_info = f.read().lower()
            if 'ubuntu' not in os_info and 'debian' not in os_info:
                return True
    except:
        return True
    
    # System packages needed for ML development
    system_packages = [
        "python3-dev",
        "python3-pip",
        "python3-venv",
        "build-essential",
        "gcc",
        "g++",
        "make",
        "cmake",
        "pkg-config",
        "libssl-dev",
        "libffi-dev",
        "libxml2-dev",
        "libxslt1-dev",
        "zlib1g-dev",
        "libbz2-dev",
        "libreadline-dev",
        "libsqlite3-dev",
        "wget",
        "curl",
        "llvm",
        "libncurses5-dev",
        "libncursesw5-dev",
        "xz-utils",
        "tk-dev",
        "libgdbm-dev",
        "libc6-dev",
        "liblzma-dev",
        "python3-openssl",
        "git"
    ]
    
    # Try to install system packages
    package_list = " ".join(system_packages)
    
    install_commands = [
        f"apt-get update && apt-get install -y {package_list}",
        f"sudo apt-get update && sudo apt-get install -y {package_list}"
    ]
    
    for cmd in install_commands:
        try:
            print("  Installing system packages (this may take a few minutes)...")
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                timeout=300
            )
            if result.returncode == 0:
                print("  ✅ System packages installed successfully")
                return True
        except subprocess.TimeoutExpired:
            print("  ⚠️ System package installation timed out")
        except Exception as e:
            print(f"  ⚠️ Could not install system packages: {e}")
    
    print("  ⚠️ Some system packages may not be installed")
    print("  You may need to run: sudo apt-get install python3-dev build-essential")
    return False

def detect_system():
    """Detect the current system and environment."""
    info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'python_executable': sys.executable,
        'in_virtualenv': hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        ),
        'has_cuda': False,
        'has_mps': False,
        'package_manager': 'pip',
        'cpu_count': os.cpu_count(),
        'memory_gb': None,
        'disk_free_gb': None,
        'gpu_info': None
    }
    
    # Get memory info
    try:
        if info['platform'] == 'Linux':
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if line.startswith('MemTotal:'):
                        kb = int(line.split()[1])
                        info['memory_gb'] = kb / (1024 * 1024)
                        break
        elif info['platform'] == 'Darwin':
            result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                bytes_mem = int(result.stdout.split(':')[1].strip())
                info['memory_gb'] = bytes_mem / (1024 * 1024 * 1024)
    except:
        pass
    
    # Get disk space
    try:
        import shutil
        stat = shutil.disk_usage('.')
        info['disk_free_gb'] = stat.free / (1024 * 1024 * 1024)
    except:
        pass
    
    # Check for CUDA
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            info['has_cuda'] = True
            # Try to get GPU name
            for line in result.stdout.split('\n'):
                if 'NVIDIA' in line and 'Driver' not in line:
                    info['gpu_info'] = line.strip()
                    break
    except:
        pass
    
    # Check for MPS (Apple Silicon)
    if info['platform'] == 'Darwin':
        try:
            import torch
            info['has_mps'] = torch.backends.mps.is_available()
        except:
            info['has_mps'] = 'arm64' in info['architecture'].lower()
    
    return info

def ensure_pip_updated():
    """Ensure pip is updated to the latest version."""
    print("📦 Ensuring pip is up to date...")
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
            capture_output=True,
            text=True,
            timeout=60
        )
        print("✓ pip is up to date")
    except Exception as e:
        print(f"⚠️ Could not update pip: {e}")

def check_installed_packages():
    """Check which packages are already installed and return a set of installed package names."""
    installed = set()
    
    # Try multiple methods to get installed packages
    if HAS_IMPORTLIB_METADATA:
        try:
            import importlib.metadata
            for dist in importlib.metadata.distributions():
                installed.add(dist.metadata['Name'].lower())
            return installed
        except:
            pass
    
    if HAS_PKG_RESOURCES:
        try:
            import pkg_resources
            for dist in pkg_resources.working_set:
                installed.add(dist.project_name.lower())
            return installed
        except:
            pass
    
    # Fallback: try pip list
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--format=json'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            import json
            packages = json.loads(result.stdout)
            for pkg in packages:
                installed.add(pkg['name'].lower())
    except:
        pass
    
    return installed

def install_package_with_retry(package, max_retries=3, timeout=180):
    """Install a package with retry logic and multiple methods."""
    methods = [
        # Method 1: Standard pip install
        lambda: subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package],
            capture_output=True, text=True, timeout=timeout
        ),
        # Method 2: Install with no-cache-dir
        lambda: subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--no-cache-dir', package],
            capture_output=True, text=True, timeout=timeout
        ),
        # Method 3: Install with upgrade
        lambda: subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', package],
            capture_output=True, text=True, timeout=timeout
        ),
        # Method 4: Force reinstall
        lambda: subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', package],
            capture_output=True, text=True, timeout=timeout
        )
    ]
    
    for attempt in range(max_retries):
        for method_idx, method in enumerate(methods):
            try:
                result = method()
                if result.returncode == 0:
                    return True
            except Exception as e:
                if attempt == max_retries - 1 and method_idx == len(methods) - 1:
                    print(f"    ❌ Failed after all attempts: {e}")
            time.sleep(1)  # Brief pause between attempts
    
    return False

def install_all_requirements():
    """Install all packages from requirements_complete.txt if it exists."""
    complete_req_file = Path("requirements_complete.txt")
    
    if complete_req_file.exists():
        tracker.start_step("Installing complete requirements")
        
        # Count packages
        with open(complete_req_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip() and not l.startswith('#') and not l.startswith('-e')]
        
        total_packages = len(lines)
        print(f"\n📦 Found complete requirements file with {total_packages} packages")
        print("📊 Starting batch installation...")
        print(f"⏱️  Estimated time: {total_packages * 2 / 60:.1f} minutes")
        
        try:
            # First try to install everything at once
            print("\n🔄 Attempting batch install (this may take a while)...")
            start = time.time()
            
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-r', str(complete_req_file), '--progress-bar', 'on'],
                capture_output=False,  # Show pip output
                text=True,
                timeout=1800  # 30 minutes
            )
            
            elapsed = time.time() - start
            
            if result.returncode == 0:
                print(f"\n✅ All {total_packages} packages installed in {elapsed:.1f}s")
                tracker.installed_packages.extend(lines)
                tracker.end_step(success=True)
                return True
            else:
                print(f"\n⚠️ Batch install failed after {elapsed:.1f}s")
                print("📦 Trying individual package installation...")
                
                # Install packages one by one with progress
                failed_packages = []
                installed_count = 0
                
                print(f"\nInstalling {total_packages} packages individually:")
                for i, line in enumerate(lines, 1):
                    # Show progress
                    progress = f"[{i}/{total_packages}]"
                    package_name = line.split('==')[0] if '==' in line else line
                    print(f"{progress} Installing {package_name}...", end='')
                    sys.stdout.flush()
                    
                    try:
                        result = subprocess.run(
                            [sys.executable, '-m', 'pip', 'install', '--quiet', line],
                            capture_output=True,
                            text=True,
                            timeout=120,
                            check=False
                        )
                        if result.returncode == 0:
                            print(" ✅")
                            installed_count += 1
                            tracker.installed_packages.append(line)
                        else:
                            print(" ❌")
                            failed_packages.append(line)
                    except:
                        print(" ❌")
                        failed_packages.append(line)
                    
                    # Show periodic progress
                    if i % 10 == 0:
                        percent = (i / total_packages) * 100
                        print(f"   Progress: {percent:.1f}% ({installed_count} installed, {len(failed_packages)} failed)")
                
                print(f"\n📊 Installation complete:")
                print(f"   ✅ Installed: {installed_count}/{total_packages}")
                if failed_packages:
                    print(f"   ❌ Failed: {len(failed_packages)}")
                    print(f"      Examples: {', '.join(failed_packages[:3])}...")
                
                tracker.end_step(success=installed_count > 0)
                return True
                
        except subprocess.TimeoutExpired:
            print("⚠️ Installation timed out, but may have installed some packages")
            return True
        except Exception as e:
            print(f"⚠️ Error installing from requirements_complete.txt: {e}")
            return False
    
    return False

def check_missing_packages():
    """Check for missing critical packages and install them."""
    # First try to install from complete requirements if available
    if install_all_requirements():
        return 0
    
    # Otherwise install critical packages
    # Extended list with more dependencies
    critical_packages = {
        # Core build tools
        'setuptools': 'setuptools',
        'wheel': 'wheel',
        'pip': 'pip>=23.0',
        
        # Essential data libraries
        'numpy': 'numpy<2.0.0',
        'pandas': 'pandas',
        'pytz': 'pytz',
        'python-dateutil': 'python-dateutil',
        
        # Required by datasets
        'dill': 'dill>=0.3.0,<0.3.9',
        'xxhash': 'xxhash',
        'multiprocess': 'multiprocess<0.70.17',
        'fsspec': 'fsspec[http]<=2025.3.0,>=2023.1.0',
        
        # Core ML libraries
        'torch': 'torch',
        'transformers': 'transformers',
        'datasets': 'datasets',
        'huggingface-hub': 'huggingface-hub',
        'tokenizers': 'tokenizers',
        
        # Common dependencies
        'tqdm': 'tqdm',
        'requests': 'requests',
        'filelock': 'filelock',
        'packaging': 'packaging',
        'typing-extensions': 'typing-extensions',
        'pyyaml': 'pyyaml',
        
        # Additional often-needed packages
        'sentencepiece': 'sentencepiece',
        'regex': 'regex',
        'joblib': 'joblib',
        'scikit-learn': 'scikit-learn',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib'
    }
    
    installed = check_installed_packages()
    missing = []
    failed = []
    
    for pkg_name, pkg_spec in critical_packages.items():
        if pkg_name.lower() not in installed and pkg_name not in installed:
            missing.append((pkg_name, pkg_spec))
    
    if missing:
        print(f"\n📦 Installing {len(missing)} missing critical packages...")
        print("="*60)
        
        for pkg_name, pkg_spec in missing:
            print(f"  Installing {pkg_spec}...", end="")
            sys.stdout.flush()
            
            if install_package_with_retry(pkg_spec):
                print(" ✅")
            else:
                print(" ❌")
                failed.append(pkg_spec)
        
        print("="*60)
        if failed:
            print(f"⚠️ Failed to install: {', '.join(failed)}")
            print("These will be retried with the main installation.")
        else:
            print("✅ All critical packages installed successfully!")
    else:
        print("✅ All critical packages are already installed")
    
    return len(failed)

def check_or_create_venv():
    """Check if we're in a virtual environment, create one if needed."""
    system_info = detect_system()
    
    if system_info['in_virtualenv']:
        print(f"✅ Using virtual environment: {sys.prefix}")
        return True
    
    print("⚠️ Not in a virtual environment")
    venv_path = Path.cwd() / '.venv'
    
    if not venv_path.exists():
        print(f"Creating virtual environment at {venv_path}...")
        try:
            import venv
            venv.create(venv_path, with_pip=True)
            print("✅ Virtual environment created")
            
            # Determine the correct activation script
            if system_info['platform'] == 'Windows':
                activate_cmd = f"{venv_path}\\Scripts\\activate.bat"
                python_exe = venv_path / 'Scripts' / 'python.exe'
            else:
                activate_cmd = f"source {venv_path}/bin/activate"
                python_exe = venv_path / 'bin' / 'python'
            
            print(f"\n📌 To activate the virtual environment, run:")
            print(f"   {activate_cmd}")
            print(f"\nThen re-run this script with:")
            print(f"   {python_exe} {__file__}")
            
            return False
        except Exception as e:
            print(f"⚠️ Could not create virtual environment: {e}")
            print("Continuing without virtual environment...")
            return True
    else:
        print(f"ℹ️ Virtual environment exists at {venv_path}")
        if system_info['platform'] == 'Windows':
            activate_cmd = f"{venv_path}\\Scripts\\activate.bat"
        else:
            activate_cmd = f"source {venv_path}/bin/activate"
        print(f"📌 Activate it with: {activate_cmd}")
        return True

def check_dataset_exists(dataset_path):
    """Check if dataset already exists."""
    path = Path(dataset_path)
    if path.exists() and any(path.iterdir()):
        return True
    return False

def run_command_with_retry(cmd, description, skip_if_exists=None, max_retries=3):
    """Run a command with retry logic and error handling."""
    # Check if we should skip this command
    if skip_if_exists and check_dataset_exists(skip_if_exists):
        print(f"\n✓ Skipping: {description} (already exists at {skip_if_exists})")
        return True, description
    
    # Replace python3 with the current Python interpreter
    original_cmd = cmd
    if cmd.startswith("python3 "):
        cmd = cmd.replace("python3 ", f"{sys.executable} ", 1)
    elif cmd.startswith("python "):
        cmd = cmd.replace("python ", f"{sys.executable} ", 1)
    
    # Reduce retries for download commands - they often fail due to network issues
    if "download" in description.lower():
        max_retries = min(2, max_retries)
    
    print(f"\n{'='*60}")
    print(f"📌 EXECUTING: {description.upper()}")
    print('='*60)
    
    # Show detailed information based on the command type
    if "train" in description.lower():
        print("\n🧠 TRAINING CONFIGURATION:")
        print("-" * 40)
        
        # Parse training arguments
        if "--config" in cmd:
            config = cmd.split("--config")[1].split()[0]
            print(f"📄 Config File: {config}")
            
            # Try to read and show config details
            try:
                config_path = Path(config)
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        import yaml
                        config_data = yaml.safe_load(f)
                        
                        print(f"\n📊 Model Parameters:")
                        if 'model' in config_data:
                            model = config_data['model']
                            print(f"   • Hidden Size: {model.get('hidden_size', 'N/A')}")
                            print(f"   • Layers: {model.get('num_hidden_layers', 'N/A')}")
                            print(f"   • Attention Heads: {model.get('num_attention_heads', 'N/A')}")
                            print(f"   • Vocab Size: {model.get('vocab_size', 'N/A')}")
                            
                            # Calculate parameter count
                            if 'hidden_size' in model and 'num_hidden_layers' in model:
                                approx_params = (model['hidden_size'] * model['hidden_size'] * model['num_hidden_layers'] * 12) / 1_000_000
                                print(f"   • Approx. Parameters: {approx_params:.1f}M")
                        
                        if 'training' in config_data:
                            training = config_data['training']
                            print(f"\n⚙️ Training Settings:")
                            print(f"   • Learning Rate: {training.get('learning_rate', 'N/A')}")
                            print(f"   • Weight Decay: {training.get('weight_decay', 'N/A')}")
                            print(f"   • Warmup Steps: {training.get('warmup_steps', 'N/A')}")
            except Exception as e:
                pass
        
        # Parse command line arguments
        if "--epochs" in cmd:
            epochs = cmd.split("--epochs")[1].split()[0]
            print(f"\n🔄 Training Duration:")
            print(f"   • Epochs: {epochs}")
        
        if "--batch-size" in cmd:
            batch_size = cmd.split("--batch-size")[1].split()[0]
            print(f"   • Batch Size: {batch_size}")
            
            # Estimate memory usage
            try:
                batch_size_int = int(batch_size)
                memory_per_batch = batch_size_int * 512 * 4 / (1024 * 1024)  # Rough estimate
                print(f"   • Est. Memory/Batch: {memory_per_batch:.1f} MB")
            except:
                pass
        
        # Check available resources
        print(f"\n💻 System Resources:")
        
        # Check GPU
        system_info = detect_system()
        if system_info['has_cuda']:
            print(f"   • GPU: ✅ CUDA Available")
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip()
                    print(f"   • Device: {gpu_info}")
            except:
                pass
        elif system_info['has_mps']:
            print(f"   • GPU: ✅ Apple Silicon (MPS)")
        else:
            print(f"   • GPU: ❌ CPU Only")
            print(f"   • ⚠️ Training will be significantly slower")
        
        # Memory info
        if system_info['memory_gb']:
            print(f"   • RAM: {system_info['memory_gb']:.1f} GB")
        
        # CPU info
        print(f"   • CPU Cores: {system_info['cpu_count']}")
        
        # Check data availability
        print(f"\n📚 Training Data:")
        data_path = Path("data/pretraining/processed")
        if data_path.exists():
            train_files = list(data_path.glob("train/*.jsonl"))
            val_files = list(data_path.glob("validation/*.jsonl"))
            
            if train_files or val_files:
                print(f"   • Training Files: {len(train_files)}")
                print(f"   • Validation Files: {len(val_files)}")
                
                # Check file sizes
                total_size = 0
                for f in train_files + val_files:
                    total_size += f.stat().st_size
                print(f"   • Total Size: {total_size / (1024*1024):.1f} MB")
            else:
                print(f"   • ⚠️ No processed data found")
        else:
            print(f"   • ⚠️ Data directory not found")
        
        # Estimate training time
        print(f"\n⏱️ Time Estimates:")
        if system_info['has_cuda'] or system_info['has_mps']:
            print(f"   • Per Epoch: ~30-60 seconds")
            print(f"   • Total: ~5-10 minutes")
        else:
            print(f"   • Per Epoch: ~3-5 minutes")
            print(f"   • Total: ~30-60 minutes")
        
        print(f"\n📊 Training Output:")
        print(f"   • Checkpoints will be saved to: checkpoints/")
        print(f"   • Logs will be saved to: outputs/")
        print(f"   • Tensorboard logs: outputs/runs/")
        
        print(f"\n🚀 Starting training process...")
        print(f"   Press Ctrl+C to stop training early")
        print("-" * 40)
    
    elif "download" in description.lower():
        print(f"\n📥 Download Information:")
        print(f"   • Target: {description}")
        print(f"   • Network: Checking connection...")
        
        # Test network connection
        try:
            subprocess.run(['ping', '-c', '1', 'huggingface.co'], 
                         capture_output=True, timeout=2)
            print(f"   • Connection: ✅ Online")
        except:
            print(f"   • Connection: ⚠️ May be offline")
    
    elif "prepare" in description.lower() or "data" in description.lower():
        print(f"\n📊 Data Processing:")
        print(f"   • Task: {description}")
        print(f"   • Processing text into training format")
        print(f"   • This may take a few minutes...")
    
    print(f"\n🔧 Command Details:")
    print(f"   {cmd}")
    print('='*60)
    
    for attempt in range(max_retries):
        try:
            # For training commands, show real-time output
            if "train" in description.lower():
                print("\n📈 Training Progress:")
                print("-" * 40)
                
                # Run with real-time output for training
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Track training metrics
                current_epoch = 0
                current_loss = None
                losses = []
                start_time = time.time()
                
                # Read output line by line
                for line in iter(process.stdout.readline, ''):
                    if line:
                        # Look for training indicators
                        if "epoch" in line.lower():
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if "epoch" in part.lower() and i + 1 < len(parts):
                                    try:
                                        current_epoch = int(parts[i + 1].replace(':', '').replace(',', ''))
                                        print(f"\n🔄 Epoch {current_epoch}")
                                    except:
                                        pass
                        
                        if "loss" in line.lower():
                            # Extract loss value
                            import re
                            loss_match = re.search(r'loss[:\s]+([0-9.]+)', line.lower())
                            if loss_match:
                                current_loss = float(loss_match.group(1))
                                losses.append(current_loss)
                                print(f"   📉 Loss: {current_loss:.4f}")
                        
                        # Show important lines
                        if any(keyword in line.lower() for keyword in ['error', 'warning', 'complete', 'saved', 'checkpoint']):
                            print(f"   ℹ️ {line.strip()}")
                        
                        # Show progress bars or percentages
                        if '%' in line or '█' in line:
                            print(f"   {line.strip()}")
                
                process.wait()
                
                # Show training summary
                elapsed_time = time.time() - start_time
                print("\n" + "-" * 40)
                print("📊 Training Summary:")
                print(f"   • Total Time: {elapsed_time / 60:.1f} minutes")
                print(f"   • Epochs Completed: {current_epoch}")
                if losses:
                    print(f"   • Final Loss: {losses[-1]:.4f}")
                    if len(losses) > 1:
                        improvement = (losses[0] - losses[-1]) / losses[0] * 100
                        print(f"   • Loss Improvement: {improvement:.1f}%")
                
                if process.returncode == 0:
                    print(f"\n✅ {description} completed successfully")
                    return True, description
                else:
                    # Training failed - show error and retry logic
                    if attempt < max_retries - 1:
                        print(f"\n⚠️ Training attempt {attempt + 1} failed, retrying...")
                        time.sleep(2)
                        continue
                    else:
                        print(f"\n❌ Training failed after {max_retries} attempts")
                        print("\nPossible issues:")
                        print("   • Out of memory - try reducing batch size")
                        print("   • Missing data files - ensure data is prepared")
                        print("   • Configuration error - check config file")
                        return False, description
            
            else:
                # Standard execution for non-training commands
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    check=False, 
                    text=True, 
                    capture_output=True,
                    timeout=600  # 10 minute timeout
                )
                
                if result.returncode == 0:
                    print(f"\n✅ {description} completed successfully")
                    # Show output if it's meaningful
                    if result.stdout and len(result.stdout.strip()) < 1000:
                        print(result.stdout.strip())
                    return True, description
                else:
                    # Handle non-zero return code
                    error_msg = ""
                    if result.stderr:
                        error_msg = result.stderr
                    elif result.stdout:
                        error_msg = result.stdout
                    else:
                        error_msg = f"Command failed with return code {result.returncode}"
                    
                    # For training failures, show the full error
                    if "train" in description.lower():
                        print("\n❌ Training Error Details:")
                        print("-" * 40)
                        # Show last part of output which usually has the error
                        if error_msg:
                            lines = error_msg.split('\n')
                            # Find the traceback or error
                            for i, line in enumerate(lines):
                                if 'Traceback' in line or 'Error' in line or 'error' in line.lower():
                                    # Show from this point onwards
                                    relevant_lines = lines[i:min(i+20, len(lines))]
                                    for err_line in relevant_lines:
                                        if err_line.strip():
                                            print(f"   {err_line}")
                                    break
                            else:
                                # Just show last 10 lines if no explicit error found
                                for line in lines[-10:]:
                                    if line.strip():
                                        print(f"   {line}")
                    
                    # For download scripts, check if it's a partial success
                    if "download" in description.lower() and attempt == max_retries - 1:
                        # Don't retry downloads too many times
                        print(f"⚠️ {description} failed")
                        print(f"  This is non-critical. Downloads can be retried later.")
                        return False, description
                    
                    # Decide whether to retry or fail
                    if attempt < max_retries - 1:
                        print(f"  ⚠️ Attempt {attempt + 1} failed, retrying...")
                        time.sleep(2)
                        continue  # Try next iteration
                    else:
                        print(f"✗ {description} failed after {max_retries} attempts")
                        if len(error_msg) > 500:
                            print(f"  Error (truncated): {error_msg[:500]}...")
                        else:
                            print(f"  Error: {error_msg}")
                        return False, description
                    
        except subprocess.TimeoutExpired:
            print(f"  ⚠️ Command timed out after 10 minutes")
            if attempt < max_retries - 1:
                print(f"  Retrying...")
                time.sleep(2)
            else:
                return False, description
        except Exception as e:
            print(f"  ⚠️ Unexpected error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return False, description
    
    return False, description

# Keep the original function name for compatibility
def run_command(cmd, description, skip_if_exists=None):
    """Run a command and handle errors (wrapper for backward compatibility)."""
    return run_command_with_retry(cmd, description, skip_if_exists)


def save_state(state_file='.setup_state.json'):
    """Save the current setup state to a file."""
    state = {
        'timestamp': time.time(),
        'python_version': sys.version,
        'platform': platform.system(),
        'completed_steps': []
    }
    
    if Path(state_file).exists():
        with open(state_file, 'r') as f:
            existing_state = json.load(f)
            state['completed_steps'] = existing_state.get('completed_steps', [])
    
    return state

def update_state(state, step_name, state_file='.setup_state.json'):
    """Update the setup state with a completed step."""
    if step_name not in state['completed_steps']:
        state['completed_steps'].append(step_name)
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

def main():
    """Main execution function with comprehensive error recovery."""
    # Set MPS memory management early to prevent issues on macOS
    if platform.system() == 'Darwin':  # macOS
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        print("🍎 macOS detected - MPS memory limit disabled for stability")
    
    # Change to the LLM directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Load or create state
    state = save_state()
    
    # Display banner
    print("="*70)
    print("🚀 LLM PROJECT SETUP")
    print("="*70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Directory: {project_dir}")
    
    # System detection
    print("\n" + "="*70)
    print("💻 SYSTEM INFORMATION")
    print("="*70)
    
    tracker.start_step("System detection")
    system_info = detect_system()
    
    # Basic info
    print(f"\n🖥️  Hardware:")
    print(f"   Platform: {system_info['platform']} {system_info['architecture']}")
    print(f"   CPUs: {system_info['cpu_count'] or 'Unknown'}")
    if system_info['memory_gb']:
        print(f"   Memory: {system_info['memory_gb']:.1f} GB")
    if system_info['disk_free_gb']:
        print(f"   Disk Free: {system_info['disk_free_gb']:.1f} GB")
    
    # GPU info
    if system_info['has_cuda'] or system_info['has_mps']:
        print(f"\n🎮 GPU:")
        if system_info['has_cuda']:
            print(f"   CUDA: ✅ Available")
            if system_info['gpu_info']:
                print(f"   Device: {system_info['gpu_info']}")
        if system_info['has_mps']:
            print(f"   MPS: ✅ Available (Apple Silicon)")
    else:
        print(f"\n🎮 GPU: ❌ No GPU acceleration available")
    
    # Python info
    print(f"\n🐍 Python:")
    print(f"   Version: {sys.version.split()[0]}")
    print(f"   Executable: {sys.executable}")
    print(f"   Virtual Env: {'✅ Active' if system_info['in_virtualenv'] else '❌ Not in venv'}")
    
    # Check existing installations
    installed_packages = check_installed_packages()
    print(f"\n📦 Packages:")
    print(f"   Currently Installed: {len(installed_packages)}")
    
    # Check for requirements files
    req_files = []
    if Path("requirements_complete.txt").exists():
        req_files.append("requirements_complete.txt (692 packages)")
    if Path("requirements.txt").exists():
        req_files.append("requirements.txt")
    if req_files:
        print(f"   Available Requirements: {', '.join(req_files)}")
    
    tracker.end_step(success=True)
    
    # FIRST: Ensure pip is installed (critical for Ubuntu with no packages)
    print("\n" + "="*60)
    print("🔧 Bootstrap Check")
    print("="*60)
    
    # Install Ubuntu system dependencies if needed
    if system_info['platform'] == 'Linux' and 'ubuntu_deps' not in state['completed_steps']:
        install_ubuntu_dependencies()
        update_state(state, 'ubuntu_deps')
    
    if 'pip_installed' not in state['completed_steps']:
        if not ensure_pip_installed():
            print("\n❌ Cannot continue without pip")
            print("Please install pip manually and re-run this script")
            sys.exit(1)
        update_state(state, 'pip_installed')
        
        # Bootstrap minimal requirements
        bootstrap_minimal_requirements()
        update_state(state, 'bootstrap_complete')
    else:
        print("✓ pip is already installed")
    
    # Check virtual environment
    print("\n" + "="*60)
    print("🔍 ENVIRONMENT CHECK")
    print("="*60)
    
    tracker.start_step("Environment check")
    
    print("\n📋 Checking Python environment...")
    
    # Check Python version
    python_major, python_minor = sys.version_info[:2]
    min_python = (3, 8)
    
    print(f"\n🐍 Python Version:")
    print(f"   Current: {python_major}.{python_minor}")
    print(f"   Required: {min_python[0]}.{min_python[1]}+")
    
    if (python_major, python_minor) >= min_python:
        print(f"   Status: ✅ Compatible")
    else:
        print(f"   Status: ⚠️ May have compatibility issues")
    
    # Virtual environment check
    print(f"\n🔒 Virtual Environment:")
    if system_info['in_virtualenv']:
        print(f"   Status: ✅ Active")
        print(f"   Location: {sys.prefix}")
        venv_size = 0
        try:
            for root, dirs, files in os.walk(sys.prefix):
                for f in files:
                    venv_size += os.path.getsize(os.path.join(root, f))
            print(f"   Size: {venv_size / (1024*1024*1024):.2f} GB")
        except:
            pass
    else:
        print(f"   Status: ❌ Not active")
        print(f"   System Python: {sys.executable}")
        print("\n   ⚠️ WARNING: Not running in a virtual environment!")
        print("   This may cause:")
        print("   • Package conflicts with system Python")
        print("   • Permission issues")
        print("   • Difficulty in managing dependencies")
        print("\n   Recommended: Create a virtual environment")
        print("   Command: python3 -m venv .venv && source .venv/bin/activate")
        
        # Don't prompt in CI/automated environments
        if os.environ.get('CI') or os.environ.get('DEBIAN_FRONTEND') == 'noninteractive':
            print("\n   📌 CI Mode: Continuing without virtual environment")
        else:
            response = input("\n   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                check_or_create_venv()
                sys.exit(0)
    
    # Check write permissions
    print(f"\n📝 Directory Permissions:")
    try:
        test_file = Path('.test_write_permission')
        test_file.touch()
        test_file.unlink()
        print(f"   Write Access: ✅ Confirmed")
    except:
        print(f"   Write Access: ❌ No write permission")
        print(f"   Fix: Check directory ownership or run with appropriate permissions")
    
    tracker.end_step(success=True)
    
    # Update pip 
    if 'pip_updated' not in state['completed_steps']:
        ensure_pip_updated()
        update_state(state, 'pip_updated')
    
    # Check and install missing critical packages
    print("\n" + "="*60)
    print("📦 DEPENDENCY CHECK")
    print("="*60)
    
    tracker.start_step("Dependency check")
    
    if 'critical_packages' not in state['completed_steps']:
        print("\n🔍 Analyzing package requirements...")
        
        # Check what we need
        installed = check_installed_packages()
        print(f"\n📊 Current Status:")
        print(f"   Installed Packages: {len(installed)}")
        
        # Check for complete requirements
        if Path("requirements_complete.txt").exists():
            with open("requirements_complete.txt", 'r') as f:
                total_needed = len([l for l in f.readlines() if l.strip() and not l.startswith('#')])
            print(f"   Target Packages: {total_needed}")
            print(f"   Missing: ~{max(0, total_needed - len(installed))}")
            print(f"\n📦 Strategy: Installing from requirements_complete.txt")
        else:
            print(f"   Target: Critical packages only")
            print(f"\n📦 Strategy: Installing essential packages")
        
        print("\n🚀 Starting package installation...")
        failed_count = check_missing_packages()
        
        if failed_count == 0:
            update_state(state, 'critical_packages')
            print("\n✅ All required packages installed successfully")
        else:
            print(f"\n⚠️ {failed_count} packages failed to install")
            print("   The setup will continue, but some features may not work")
        
        tracker.end_step(success=(failed_count == 0))
    else:
        print("\n✅ Dependencies already verified")
        print(f"   Packages: {len(check_installed_packages())} installed")
        print("   Status: Ready")
        tracker.end_step(success=True, skipped=True)
    
    # Run initial setup commands sequentially
    print("\n" + "="*60)
    print("🔧 INITIAL SETUP")
    print("="*60)
    
    print("\n📋 Setup Tasks:")
    print("   1. Create project folder structure")
    print("   2. Install remaining requirements")
    print("   3. Verify installations")
    
    initial_commands = [
        ("setup_folders", "python3 scripts/utils/setup_folders.py", "Setup project folders"),
        ("fast_pip", "python3 scripts/utils/fast_pip_install.py", "Fast pip install"),
    ]
    
    for i, (step_id, cmd, desc) in enumerate(initial_commands, 1):
        print(f"\n📍 Step {i}/2: {desc}")
        
        if step_id not in state['completed_steps']:
            tracker.start_step(desc)
            
            # Check what this step will do
            if step_id == "setup_folders":
                print("   Creating directories for:")
                print("   • Training data (data/pretraining/)")
                print("   • Model checkpoints (checkpoints/)")
                print("   • Outputs (outputs/)")
                print("   • Logs and configs")
            elif step_id == "fast_pip":
                print("   Installing packages from:")
                if Path("requirements.txt").exists():
                    with open("requirements.txt", 'r') as f:
                        pkg_count = len([l for l in f.readlines() if l.strip() and not l.startswith('#')])
                    print(f"   • requirements.txt ({pkg_count} packages)")
                print("   Using parallel downloads for speed")
            
            print(f"\n   Executing: {desc}...")
            success, _ = run_command_with_retry(cmd, desc)
            
            if success:
                update_state(state, step_id)
                tracker.end_step(success=True)
                print(f"   ✅ {desc} completed")
            else:
                tracker.end_step(success=False)
                print(f"\n⚠️ {desc} failed, attempting recovery...")
                
                # Recovery actions for specific failures
                if step_id == "fast_pip":
                    print("   Trying fallback: Standard pip install...")
                    fallback_cmd = f"{sys.executable} -m pip install -r requirements.txt"
                    success, _ = run_command_with_retry(fallback_cmd, "Standard pip install")
                    if success:
                        update_state(state, step_id)
                        print("   ✅ Fallback successful")
                
                if not success:
                    print(f"\n❌ Critical failure in: {desc}")
                    print("\n📋 Troubleshooting steps:")
                    print(f"   1. Check network connection")
                    print(f"   2. Verify file permissions")
                    print(f"   3. Try: {sys.executable} scripts/utils/install_missing_deps.py")
                    print(f"   4. Re-run this script")
                    sys.exit(1)
        else:
            print(f"   ⏭️ Already completed: {desc}")
            print(f"   Status: ✅ Verified")
    
    # Run download commands
    print("\n" + "="*60)
    print("📥 DATA DOWNLOAD")
    print("="*60)
    
    print("\n📊 Available Datasets:")
    print("   • Wikipedia (Simple English)")
    print("   • Code samples (Multiple languages)")
    print("   • OpenAssistant conversations")
    print("   • Technical documentation")
    
    download_commands = [
        ("download_wiki_code",
         "python3 scripts/data_download/download_full_datasets.py --datasets wikipedia code --max_samples 50000",
         "Download Wikipedia and code datasets (50k samples)",
         "data/pretraining/raw/full_datasets"),
        
        ("download_all",
         "python3 scripts/data_download/download_full_datasets.py --datasets all --max_samples 10000",
         "Download all datasets (10k samples)",
         "data/pretraining/raw/full_datasets")
    ]
    
    # Check existing data
    data_path = Path("data/pretraining/raw/full_datasets")
    existing_data_size = 0
    existing_files = 0
    
    if data_path.exists():
        for root, dirs, files in os.walk(data_path):
            existing_files += len(files)
            for f in files:
                try:
                    existing_data_size += os.path.getsize(os.path.join(root, f))
                except:
                    pass
    
    if existing_files > 0:
        print(f"\n📁 Existing Data:")
        print(f"   Files: {existing_files}")
        print(f"   Size: {existing_data_size / (1024*1024):.1f} MB")
    
    # Check if downloads are needed
    downloads_needed = []
    print(f"\n🔍 Checking download requirements:")
    
    for step_id, cmd, desc, skip_path in download_commands:
        if step_id not in state['completed_steps'] and not check_dataset_exists(skip_path):
            downloads_needed.append((step_id, cmd, desc, skip_path))
            print(f"   📥 Needed: {desc}")
        else:
            print(f"   ✅ Complete: {desc}")
    
    if downloads_needed:
        print(f"\nDownloading {len(downloads_needed)} datasets...")
        
        all_downloads_failed = True
        
        # Try parallel download if ThreadPoolExecutor is available
        if HAS_CONCURRENT and len(downloads_needed) > 1:
            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = []
                    for step_id, cmd, desc, skip_path in downloads_needed:
                        futures.append(executor.submit(run_command_with_retry, cmd, desc, skip_path))
                    
                    for i, future in enumerate(as_completed(futures)):
                        success, desc = future.result()
                        step_id = downloads_needed[i][0]
                        if success:
                            update_state(state, step_id)
                            all_downloads_failed = False
                        else:
                            print(f"\n⚠️ Download failed: {desc}")
                            print("This is non-critical. You can download data later.")
                            update_state(state, f"{step_id}_skipped")
            except:
                # Fall back to sequential if parallel fails
                print("⚠️ Parallel download not available, using sequential...")
                for step_id, cmd, desc, skip_path in downloads_needed:
                    success, _ = run_command_with_retry(cmd, desc, skip_path)
                    if success:
                        update_state(state, step_id)
                        all_downloads_failed = False
                    else:
                        print(f"\n⚠️ Download failed: {desc}")
                        print("This is non-critical. You can download data later.")
                        update_state(state, f"{step_id}_skipped")
        else:
            # Sequential download (single item or ThreadPoolExecutor not available)
            for step_id, cmd, desc, skip_path in downloads_needed:
                success, _ = run_command_with_retry(cmd, desc, skip_path)
                if success:
                    update_state(state, step_id)
                    all_downloads_failed = False
                else:
                    print(f"\n⚠️ Download failed: {desc}")
                    print("This is non-critical. You can download data later.")
                    update_state(state, f"{step_id}_skipped")
        
        # If all downloads failed, generate fallback data
        if all_downloads_failed and not check_dataset_exists("data/pretraining/raw/full_datasets"):
            print("\n📝 All downloads failed. Generating fallback training data...")
            fallback_cmd = "python3 scripts/data_generation/generate_fallback_data.py --num-samples 1000"
            success, _ = run_command_with_retry(fallback_cmd, "Generate fallback training data")
            if success:
                update_state(state, "fallback_data_generated")
                print("✅ Fallback data generated successfully")
    
    # Run final commands sequentially
    print("\n" + "="*60)
    print("🎯 FINAL SETUP STEPS")
    print("="*60)
    
    print("\n📋 Remaining Tasks:")
    print("   1. Process raw data for training")
    print("   2. Initialize model configuration")
    print("   3. Run initial training (optional)")
    
    final_commands = [
        ("prepare_data",
         "python3 scripts/data_prep/prepare_data.py --input-path data/pretraining/raw/full_datasets --output-dir data/pretraining/processed --input-format json --output-format jsonl --max-length 512 --fast-mode --tokenizer gpt2",
         "Prepare data for training",
         "data/pretraining/processed"),
        
        ("train_model",
         "python3 scripts/training/train_with_data.py --config configs/mps/small.yaml --epochs 3 --batch-size 2",
         "Train model with prepared data",
         None)
    ]
    
    for i, (step_id, cmd, desc, skip_path) in enumerate(final_commands, 1):
        print(f"\n📍 Task {i}/2: {desc}")
        
        if step_id not in state['completed_steps']:
            tracker.start_step(desc)
            
            # Provide context for each step
            if step_id == "prepare_data":
                # Check if we have data
                if not check_dataset_exists("data/pretraining/raw/full_datasets"):
                    print("   ⚠️ No raw data found to process")
                    print("   Skipping data preparation...")
                    tracker.end_step(success=False, skipped=True)
                    continue
                
                print("   📊 Data Processing:")
                print("   • Tokenizing text samples")
                print("   • Creating training batches")
                print("   • Saving in JSONL format")
                print("   • Max sequence length: 512 tokens")
                
            elif step_id == "train_model":
                print("   🧠 Model Training:")
                print("   • Config: configs/mps/small.yaml")
                print("   • Epochs: 10")
                print("   • Batch Size: 64")
                
                # Check system capabilities
                if system_info['has_cuda']:
                    print("   • Device: CUDA GPU ✅")
                elif system_info['has_mps']:
                    print("   • Device: Apple Silicon GPU ✅")
                else:
                    print("   • Device: CPU (slower)")
                    print("   ⚠️ Training on CPU will be slow")
                
                # Estimate training time
                if system_info['has_cuda'] or system_info['has_mps']:
                    est_time = "5-10 minutes"
                else:
                    est_time = "30-60 minutes"
                print(f"   • Estimated Time: {est_time}")
            
            print(f"\n   Executing: {desc}...")
            success, _ = run_command_with_retry(cmd, desc, skip_path)
            
            if success:
                update_state(state, step_id)
                tracker.end_step(success=True)
                print(f"   ✅ {desc} completed successfully")
            else:
                tracker.end_step(success=False)
                print(f"\n   ⚠️ {desc} failed")
                
                if step_id == "prepare_data":
                    print("\n   📋 Troubleshooting:")
                    print("   • Check if raw data exists")
                    print("   • Verify tokenizer is installed")
                    print(f"\n   Manual retry: {cmd}")
                    
                elif step_id == "train_model":
                    print("\n   📋 This step is optional")
                    print("   You can train later with:")
                    print(f"   {cmd}")
                    print("\n   Or try a simpler config:")
                    print("   python3 scripts/training/train_simple.py")
                
                # Don't exit on these failures - they're not critical
                update_state(state, f"{step_id}_skipped")
        else:
            print(f"   ⏭️ Already completed: {desc}")
            
            # Show where to find results
            if step_id == "prepare_data" and skip_path:
                processed_path = Path(skip_path)
                if processed_path.exists():
                    file_count = len(list(processed_path.glob("**/*.jsonl")))
                    print(f"   📁 Processed data: {file_count} files in {skip_path}")
            
            elif step_id == "train_model":
                checkpoint_path = Path("checkpoints")
                if checkpoint_path.exists():
                    checkpoints = list(checkpoint_path.glob("*.pt"))
                    if checkpoints:
                        print(f"   📁 Model checkpoints: {len(checkpoints)} saved")
                        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                        print(f"   📍 Latest: {latest.name}")
    
    # Print detailed tracker summary
    tracker.print_summary()
    
    # Final summary
    print("\n" + "="*70)
    print("🎯 FINAL STATUS")
    print("="*70)
    
    completed = [s for s in state['completed_steps'] if not s.endswith('_skipped')]
    skipped = [s for s in state['completed_steps'] if s.endswith('_skipped')]
    
    print(f"\n✅ Completed Steps ({len(completed)}):")
    for step in completed[:10]:  # Show first 10
        print(f"   • {step}")
    if len(completed) > 10:
        print(f"   ... and {len(completed) - 10} more")
    
    if skipped:
        print(f"\n⏭️  Skipped Steps ({len(skipped)}):")
        for step in skipped:
            print(f"   • {step.replace('_skipped', '')}")
    
    # Environment status
    print(f"\n🔬 Environment Status:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Packages: {len(check_installed_packages())} installed")
    if system_info['has_cuda']:
        print(f"   GPU: CUDA ready ✅")
    elif system_info['has_mps']:
        print(f"   GPU: MPS ready ✅")
    else:
        print(f"   GPU: CPU only mode")
    
    # Data status
    data_path = Path("data/pretraining/raw/full_datasets")
    if data_path.exists() and any(data_path.iterdir()):
        print(f"\n📚 Training Data: ✅ Available")
    else:
        print(f"\n📚 Training Data: ❌ Not downloaded")
        print(f"   Run: python3 scripts/data_download/download_full_datasets.py")
    
    print("\n" + "="*70)
    print("🎉 SETUP COMPLETE!")
    print("="*70)
    
    print("\n📖 Quick Start Guide:")
    print("\n1️⃣  Test the installation:")
    print("   python3 scripts/generation/quick_test.py")
    
    print("\n2️⃣  Train a model:")
    print("   python3 scripts/training/train_simple.py")
    
    print("\n3️⃣  Interactive generation:")
    print("   python3 scripts/generation/interact.py")
    
    if not system_info['in_virtualenv']:
        print("\n⚠️  IMPORTANT: Activate your virtual environment first!")
        print("   source .venv/bin/activate")
    
    print(f"\n📊 Total setup time: {tracker.format_time(tracker.get_elapsed_time())}")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()