#!/usr/bin/env python3
"""Simplified setup script for LLM project."""

import subprocess
import sys
import os
import platform
import json
import time
import threading
from pathlib import Path
from datetime import datetime

class TeeLogger:
    """Logger that writes to both stdout and a file."""
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def setup_logging():
    """Setup logging to file and console."""
    # Create logs directory
    log_dir = Path("/project/code/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"setup_{timestamp}.log"

    # Set up tee logging
    logger = TeeLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger

    print(f"Logging to: {log_file}")
    return logger

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """Print a progress bar to the console."""
    if total == 0:
        return
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    # Print New Line on Complete
    if iteration == total:
        print()

def run_command_with_progress(cmd, description):
    """Run a command with real-time progress monitoring for downloads."""
    print(f"\n→ {description}")

    # Replace python3 with current interpreter
    if cmd.startswith("python3 "):
        cmd = cmd.replace("python3 ", f"{sys.executable} ", 1)

    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Monitor output for progress indicators
        current_dataset = ""
        total_samples = 0
        downloaded_samples = 0

        def read_output(pipe, prefix=""):
            nonlocal current_dataset, total_samples, downloaded_samples
            for line in iter(pipe.readline, ''):
                if line:
                    line = line.strip()

                    # Look for dataset download indicators
                    if "Downloading" in line and "dataset" in line:
                        current_dataset = line.split("Downloading")[1].split("dataset")[0].strip()
                        print(f"\n  📥 Downloading {current_dataset}")

                    # Look for sample counts
                    if "samples" in line.lower():
                        try:
                            # Try to extract numbers from lines like "Downloaded 1000/5000 samples"
                            import re
                            numbers = re.findall(r'\d+', line)
                            if len(numbers) >= 2:
                                downloaded_samples = int(numbers[0])
                                total_samples = int(numbers[1])
                                print_progress_bar(downloaded_samples, total_samples,
                                                 prefix='  Progress:',
                                                 suffix=f'{downloaded_samples}/{total_samples} samples')
                        except:
                            pass

                    # Look for percentage indicators
                    if "%" in line:
                        try:
                            import re
                            match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
                            if match:
                                percent = float(match.group(1))
                                print_progress_bar(int(percent), 100,
                                                 prefix='  Progress:',
                                                 suffix=f'{percent:.1f}% Complete')
                        except:
                            pass

                    # Show important messages
                    if any(keyword in line.lower() for keyword in ['error', 'failed', 'success', 'complete', 'saved']):
                        print(f"  {prefix}{line}")

        # Create threads to read stdout and stderr
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, ""))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, "⚠️ "))

        stdout_thread.start()
        stderr_thread.start()

        # Wait for process to complete
        return_code = process.wait(timeout=600)

        # Wait for threads to finish
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

        if return_code == 0:
            print(f"\n  ✓ {description} completed successfully")
            return True
        else:
            print(f"\n  ✗ {description} failed (return code: {return_code})")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n  ✗ Timeout after 600 seconds")
        process.kill()
        return False
    except Exception as e:
        print(f"\n  ✗ Exception: {e}")
        return False

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n→ {description}")
    print(f"  Command: {cmd[:100]}..." if len(cmd) > 100 else f"  Command: {cmd}")

    # Replace python3 with current interpreter
    if cmd.startswith("python3 "):
        cmd = cmd.replace("python3 ", f"{sys.executable} ", 1)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=600
        )

        # Log the output
        if result.stdout:
            print(f"  Output: {result.stdout[:500]}")

        if result.returncode == 0:
            print(f"  ✓ Success")
            return True
        else:
            print(f"  ✗ Failed (return code: {result.returncode})")
            if result.stderr:
                print(f"  Error: {result.stderr[:500]}")
            elif result.stdout:
                # Sometimes errors go to stdout
                print(f"  Output: {result.stdout[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout after 600 seconds")
        return False
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        return False

def ensure_pip():
    """Ensure pip is installed."""
    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', '--version'],
            capture_output=True,
            timeout=10
        )
        return True
    except:
        print("Installing pip...")

        # Try to install pip
        try:
            subprocess.run(
                "curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && " + sys.executable + " get-pip.py",
                shell=True,
                capture_output=True,
                timeout=120
            )
            os.remove("get-pip.py")
            return True
        except:
            pass

        try:
            subprocess.run([sys.executable, '-m', 'ensurepip'], capture_output=True)
            return True
        except:
            pass

        print("❌ Could not install pip. Please install manually.")
        return False

def install_packages():
    """Install required packages."""
    print("\nInstalling packages...")

    # Essential packages
    packages = [
        'setuptools',
        'wheel',
        'numpy<2.0.0',
        'torch',
        'transformers',
        'datasets',
        'tqdm',
        'pyyaml',
        'requests',
        'huggingface-hub',
    ]

    # Try to install from requirements_complete.txt first
    if Path("requirements_complete.txt").exists():
        print("Found requirements_complete.txt, installing all packages...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements_complete.txt'],
            capture_output=True,
            text=True,
            timeout=1800
        )
        if result.returncode == 0:
            print("✓ All packages installed")
            return True
        else:
            print("⚠️ Batch install failed, installing essential packages...")

    # Install essential packages individually
    failed = []
    for pkg in packages:
        print(f"  Installing {pkg}...", end="")
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', pkg],
                capture_output=True,
                text=True,
                timeout=180
            )
            if result.returncode == 0:
                print(" ✓")
            else:
                print(" ✗")
                failed.append(pkg)
        except:
            print(" ✗")
            failed.append(pkg)

    if failed:
        print(f"\n⚠️ Failed to install: {', '.join(failed)}")

    return len(failed) == 0

def detect_gpu():
    """Detect GPU availability."""
    gpu_type = None

    # Check for CUDA
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        if result.returncode == 0:
            gpu_type = "CUDA"
    except:
        pass

    # Check for Apple Silicon
    if not gpu_type and platform.system() == 'Darwin':
        if 'arm64' in platform.machine().lower():
            gpu_type = "MPS"

    return gpu_type

def check_data_exists(path):
    """Check if data directory exists and has files."""
    data_path = Path(path)
    return data_path.exists() and any(data_path.iterdir())

def main():
    """Main setup function."""

    # Set MPS memory for macOS
    if platform.system() == 'Darwin':
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    # Change to project directory
    os.chdir(Path(__file__).parent)

    # Setup logging
    logger = setup_logging()

    print("="*60)
    print("LLM PROJECT SETUP (Simplified)")
    print("="*60)

    # System info
    print(f"\nSystem: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")

    gpu = detect_gpu()
    if gpu:
        print(f"GPU: {gpu} available")
    else:
        print("GPU: CPU only")

    # Check if in virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    if not in_venv:
        print("\n⚠️ Not in virtual environment. Consider using one.")

    # State tracking
    state_file = '.setup_state.json'
    state = {'completed': []}

    if Path(state_file).exists():
        try:
            with open(state_file, 'r') as f:
                loaded_state = json.load(f)
                # Ensure 'completed' key exists
                if 'completed' in loaded_state:
                    state = loaded_state
                else:
                    # Migrate old format
                    state['completed'] = loaded_state.get('completed_steps', [])
        except:
            pass

    def save_state():
        with open(state_file, 'w') as f:
            json.dump(state, f)

    # Step 1: Ensure pip
    if 'pip' not in state['completed']:
        print("\n[1/6] Checking pip...")
        if ensure_pip():
            state['completed'].append('pip')
            save_state()
        else:
            sys.exit(1)

    # Step 2: Install packages
    if 'packages' not in state['completed']:
        print("\n[2/6] Installing packages...")
        if install_packages():
            state['completed'].append('packages')
            save_state()

    # Step 3: Setup folders
    if 'folders' not in state['completed']:
        print("\n[3/6] Setting up folders...")
        if run_command("python3 scripts/utils/setup_folders.py", "Creating project structure"):
            state['completed'].append('folders')
            save_state()

    # Step 4: Download data (optional, don't fail if it doesn't work)
    if 'data' not in state['completed']:
        print("\n[4/6] Downloading data...")

        if not check_data_exists("data/pretraining/raw/full_datasets"):
            # Try to download some data with progress monitoring
            cmd = "python3 scripts/data_download/download_datasets.py"
            if run_command_with_progress(cmd, "Downloading datasets"):
                state['completed'].append('data')
                save_state()
            else:
                print("  ⚠️ Data download failed (non-critical)")
                # Generate fallback data
                if run_command("python3 scripts/data_generation/generate_fallback_data.py --num-samples 1000",
                             "Generating fallback data"):
                    state['completed'].append('data')
                    save_state()

    # Step 5: Prepare data
    if 'prepare' not in state['completed']:
        print("\n[5/6] Preparing data...")

        if check_data_exists("data/pretraining/raw/full_datasets"):
            cmd = ("python3 scripts/data_prep/prepare_data.py "
                   "--input-path data/pretraining/raw/full_datasets "
                   "--output-dir data/pretraining/processed "
                   "--input-format json --output-format jsonl "
                   "--max-length 512 --fast-mode --tokenizer gpt2")

            if run_command(cmd, "Processing training data"):
                state['completed'].append('prepare')
                save_state()
        else:
            print("  ⚠️ No data to prepare")

    # Step 6: Optional training
    if 'train' not in state['completed']:
        print("\n[6/6] Initial training (optional)...")

        if check_data_exists("data/pretraining/processed"):
            # Use appropriate config based on GPU
            if gpu == "CUDA":
                config = "configs/cuda/small.yaml"
            elif gpu == "MPS":
                config = "configs/mps/small.yaml"
            else:
                config = "configs/cpu/small.yaml"

            cmd = f"python3 scripts/training/train_with_data.py --config {config} --epochs 3 --batch-size 2"

            if run_command(cmd, "Training model"):
                state['completed'].append('train')
                save_state()
            else:
                print("  ⚠️ Training failed (optional step)")
        else:
            print("  ⚠️ No data for training")

    # Summary
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)

    print(f"\nCompleted steps: {len(state['completed'])}/6")

    if check_data_exists("data/pretraining/raw/full_datasets"):
        print("✓ Training data available")
    else:
        print("✗ No training data (run download scripts)")

    if Path("checkpoints").exists() and any(Path("checkpoints").glob("*.pt")):
        print("✓ Model checkpoints found")
    else:
        print("✗ No model checkpoints yet")

    print("\nNext steps:")
    print("1. Test: python3 scripts/generation/quick_test.py")
    print("2. Train: python3 scripts/training/train_simple.py")
    print("3. Chat: python3 scripts/generation/interact.py")

    print("\n" + "="*60)

    # Close logger
    print(f"\nLog saved to: outputs/logs/")
    if hasattr(sys.stdout, 'terminal'):
        sys.stdout = sys.stdout.terminal
        sys.stderr = sys.stderr.terminal
        logger.close()

if __name__ == "__main__":
    main()