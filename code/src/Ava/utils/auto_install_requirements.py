"""
Auto-install requirements helper for Python scripts.
Add this to the top of any script that needs automatic dependency installation.
"""

import subprocess
import sys
from pathlib import Path


def install_requirements_if_needed():
    """
    Automatically install requirements.txt and apt requirements if imports fail.
    This should be called at the top of scripts before imports.
    """
    project_root = Path(__file__).resolve().parents[3]  # Go up to /project
    requirements_file = project_root / "requirements.txt"
    apt_requirements_file = project_root / "apt_requirements.txt"

    def install_python_requirements():
        """Install Python requirements from requirements.txt"""
        print("\n" + "="*80)
        print("🔧 Installing Python requirements...")
        print("="*80)

        if not requirements_file.exists():
            print(f"❌ Error: {requirements_file} not found!")
            return False

        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "-r", str(requirements_file),
                "--upgrade", "-q"
            ])
            print("✅ Python requirements installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Python requirements: {e}")
            return False

    def install_apt_requirements():
        """Install system packages from apt_requirements.txt"""
        print("\n" + "="*80)
        print("🔧 Installing system packages...")
        print("="*80)

        if not apt_requirements_file.exists():
            print(f"⚠️  Warning: {apt_requirements_file} not found, skipping apt installs")
            return True

        try:
            with open(apt_requirements_file, 'r') as f:
                packages = [
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith('#')
                ]

            if not packages:
                print("No system packages to install")
                return True

            # Update apt cache
            print("Updating apt cache...")
            subprocess.check_call(["sudo", "apt-get", "update", "-qq"])

            # Install each package
            for package in packages:
                print(f"Installing: {package}")
                subprocess.check_call([
                    "sudo", "apt-get", "install", "-y", package
                ])

            print("✅ System packages installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install system packages: {e}")
            return False
        except Exception as e:
            print(f"⚠️  Warning: Could not process apt requirements: {e}")
            return True  # Don't fail if apt requirements are optional

    # First install apt requirements (system dependencies)
    install_apt_requirements()

    # Then install Python requirements
    return install_python_requirements()


def try_import_with_install(func):
    """
    Decorator that wraps the entire script execution.
    If any ImportError occurs, install requirements and retry.

    Usage:
        from auto_install_requirements import try_import_with_install

        @try_import_with_install
        def main():
            import torch
            import transformers
            # ... rest of script

        if __name__ == "__main__":
            main()
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ImportError, ModuleNotFoundError) as e:
            print("\n" + "="*80)
            print(f"❌ Import Error: {e}")
            print("="*80)
            print("🔧 Attempting to install requirements and retry...")

            if install_requirements_if_needed():
                print("\n" + "="*80)
                print("🔄 Retrying script execution...")
                print("="*80 + "\n")

                # Retry the function
                try:
                    return func(*args, **kwargs)
                except (ImportError, ModuleNotFoundError) as e2:
                    print("\n" + "="*80)
                    print(f"❌ Still failing after installing requirements: {e2}")
                    print("="*80)
                    print("Please check:")
                    print("  1. All required packages are in requirements.txt")
                    print("  2. System dependencies are installed")
                    print("  3. Virtual environment is activated")
                    sys.exit(1)
            else:
                print("❌ Failed to install requirements. Please install manually:")
                print(f"   pip install -r {Path(__file__).resolve().parents[2]}/requirements.txt")
                sys.exit(1)

    return wrapper
