# Auto-Install Requirements Feature

All Python scripts in this project now have automatic requirements installation built-in. If a script fails due to missing dependencies, it will automatically install the required packages and retry.

## How It Works

Each script includes an auto-install block at the top that:

1. Attempts to import required packages
2. If an `ImportError` or `ModuleNotFoundError` occurs:
   - Automatically runs `pip install -r requirements.txt --upgrade`
   - Retries the imports
3. If still failing, provides helpful error messages

## Files

- **requirements.txt** - Python package dependencies
- **apt_requirements.txt** - System package dependencies (for manual installation)
- **code/src/Ava/utils/auto_install_requirements.py** - Reusable auto-install utility

## Updated Scripts

The following scripts now have auto-install functionality:

### Training Scripts
- `code/scripts/5_training/train_100m_full.py`
- `code/scripts/5_training/finetune.py`
- `code/scripts/6_rhlf_Finetuning/train_rlhf.py`

### Data Download Scripts
- `code/scripts/1_data_download/unified_download.py`

### Test Scripts
- `code/scripts/test_auto_install.py` - Test script to verify functionality

## Usage Examples

### Running Training Scripts

```bash
# If dependencies are missing, they'll be auto-installed
python code/scripts/5_training/train_100m_full.py --config configs/gpu/small.yaml

# The script will:
# 1. Try to import required packages
# 2. If missing: Install from requirements.txt
# 3. Retry and continue execution
```

### Running Fine-tuning

```bash
python code/scripts/5_training/finetune.py

# Auto-installs dependencies if needed
```

### Running Data Downloads

```bash
python code/scripts/1_data_download/unified_download.py

# Auto-installs huggingface_hub and other dependencies if needed
```

## System Dependencies

For system-level packages (apt), you'll need to install manually:

```bash
# Install system dependencies
sudo apt-get update
while read package; do
    [[ -z "$package" || "$package" =~ ^#.* ]] && continue
    sudo apt-get install -y "$package"
done < apt_requirements.txt
```

Or install individually:
```bash
sudo apt-get update
sudo apt-get install -y build-essential git libopenmpi-dev openmpi-bin
```

## Adding Auto-Install to New Scripts

To add auto-install functionality to a new script, add this code at the top (before other imports):

```python
#!/usr/bin/env python3
"""Your script description"""

# Auto-install requirements if needed (must be before other imports)
import subprocess
import sys
from pathlib import Path

def auto_install_requirements():
    """Auto-install requirements if imports fail"""
    project_root = Path(__file__).resolve().parents[2]  # Adjust depth as needed
    requirements_file = project_root / "requirements.txt"

    print("🔧 Installing Python requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "-r", str(requirements_file), "--upgrade", "-q"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

# Try imports with auto-install
try:
    import torch
    import transformers
    # ... other imports
except (ImportError, ModuleNotFoundError) as e:
    print(f"❌ Import error: {e}")
    print("🔧 Attempting to install requirements...")
    if auto_install_requirements():
        print("🔄 Retrying imports...")
        import torch
        import transformers
        # ... other imports
    else:
        print("❌ Failed to install requirements. Please run:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

# Rest of your script...
```

## Benefits

✅ **User-friendly**: Scripts automatically handle missing dependencies
✅ **Resilient**: Reduces setup friction for new users
✅ **Development-friendly**: Makes it easy to keep dependencies up to date
✅ **Production-ready**: Graceful error handling with clear messages

## Testing

Test the auto-install functionality:

```bash
python code/scripts/test_auto_install.py
```

Expected output:
```
✅ All imports successful!
   PyTorch version: 2.9.0+cu128
   Transformers version: 4.57.1
   NumPy version: 2.2.6
   Pandas version: 2.2.3

✅ Test passed! Auto-install functionality is working correctly.
```

## Notes

- Auto-install runs quietly (`-q` flag) to avoid verbose output
- Uses `--upgrade` flag to ensure latest compatible versions
- Only triggers on import errors (minimal performance impact)
- System packages (apt) require manual installation with sudo privileges
