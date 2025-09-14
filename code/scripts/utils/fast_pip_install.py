#!/usr/bin/env python3
"""
Multi-core pip downloader for faster requirements installation.
Uses concurrent downloads and parallel processing to speed up package installation.
"""

import argparse
import concurrent.futures
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Set

def get_installed_packages() -> Set[str]:
    """Get a set of installed package names."""
    installed = set()
    try:
        import importlib.metadata
        for dist in importlib.metadata.distributions():
            installed.add(dist.metadata['Name'].lower())
    except:
        import pkg_resources
        for dist in pkg_resources.working_set:
            installed.add(dist.project_name.lower())
    return installed

def parse_requirements(requirements_file: str) -> List[str]:
    """Parse requirements.txt file and return list of packages."""
    packages = []
    
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found")
        sys.exit(1)
    
    with open(requirements_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#') and not line.startswith('-'):
                # Handle git+https urls and commented lines
                if line.startswith('# ') and ('git+https' in line or 'http' in line):
                    continue
                # Remove inline comments (everything after # on the same line)
                if '#' in line:
                    line = line.split('#')[0].strip()
                if line:  # Only add if there's something left after removing comments
                    packages.append(line)
    
    return packages

def download_package(package: str, download_dir: str) -> tuple:
    """Download a single package using pip download."""
    try:
        cmd = [
            sys.executable, '-m', 'pip', 'download',
            '--no-deps',  # Don't download dependencies separately
            '--dest', download_dir,
            package
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per package
        )
        
        if result.returncode == 0:
            return package, True, ""
        else:
            return package, False, result.stderr
            
    except subprocess.TimeoutExpired:
        return package, False, "Timeout downloading package"
    except Exception as e:
        return package, False, str(e)

def install_package(package: str) -> tuple:
    """Install a single package using pip install."""
    try:
        cmd = [
            sys.executable, '-m', 'pip', 'install',
            '--no-deps',  # Dependencies should already be downloaded
            package
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per install
        )
        
        if result.returncode == 0:
            return package, True, ""
        else:
            return package, False, result.stderr
            
    except subprocess.TimeoutExpired:
        return package, False, "Timeout installing package"
    except Exception as e:
        return package, False, str(e)

def fast_pip_install(requirements_file: str, max_workers: int = None, download_only: bool = False):
    """Fast pip installation using parallel processing."""
    
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)
    
    print(f"Starting fast pip install with {max_workers} workers")
    print(f"Requirements file: {requirements_file}")
    
    # First, install essential dependencies that are often missing
    print("Installing essential dependencies...")
    essential_packages = [
        "pytz",
        "python-dateutil",
        "numpy<2.0.0",  # Pin numpy to avoid compatibility issues
        "setuptools",
        "wheel",
        "pip>=23.0",
        "dill>=0.3.0,<0.3.9",  # Required by datasets
        "xxhash",  # Required by datasets
        "multiprocess<0.70.17",  # Required by datasets with version constraint
        "fsspec",  # Required for file system specs
        "filelock",  # Common dependency
        "requests",  # Common HTTP library
        "pandas"  # Data manipulation library
    ]
    
    for package in essential_packages:
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--quiet', package],
                capture_output=True,
                text=True,
                timeout=60
            )
        except:
            pass  # Continue even if some essentials fail
    
    # Parse requirements
    packages = parse_requirements(requirements_file)
    
    # Get already installed packages
    installed_packages = get_installed_packages()
    print(f"📋 Found {len(installed_packages)} packages already installed")
    
    # Filter out known problematic packages based on platform
    filtered_packages = []
    platform_specific_replacements = {
        'faiss-gpu>=1.7.2': 'faiss-cpu>=1.7.2',  # Use CPU version if GPU not available
        'py3nvml>=0.2.7': 'pynvml>=11.5.0',  # Use alternative package
        'ftfy>=6.1.0': 'ftfy',  # Remove version constraint
        'tensorrt>=8.6.0': None,  # Skip on non-NVIDIA systems
        'flash-attn>=2.3.0': None,  # Skip - requires CUDA
    }
    
    skipped_installed = 0
    for package in packages:
        # Extract package name (before any version specifier)
        pkg_name = package.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].split('[')[0].lower()
        
        # Check if package needs replacement
        skip_package = False
        for problematic, replacement in platform_specific_replacements.items():
            if package == problematic or package.startswith(problematic.split('>=')[0]):
                if replacement:
                    # Check if replacement is already installed
                    repl_name = replacement.split('>=')[0].split('==')[0].split('<')[0].split('>')[0].lower()
                    if repl_name not in installed_packages:
                        filtered_packages.append(replacement)
                        print(f"📝 Replaced {package} with {replacement}")
                    else:
                        print(f"✓ {replacement} (replacement for {package}) already installed")
                        skipped_installed += 1
                else:
                    print(f"⚠️  Skipping platform-specific package: {package}")
                skip_package = True
                break
        
        if not skip_package:
            # Check if package is already installed
            if pkg_name in installed_packages:
                print(f"✓ {package} already installed")
                skipped_installed += 1
            else:
                filtered_packages.append(package)
    
    packages = filtered_packages
    print(f"📦 Will process {len(packages)} new packages ({skipped_installed} already installed)")
    
    # If all packages are already installed, we're done
    if len(packages) == 0:
        print("\n✅ All required packages are already installed!")
        return
    
    # Create download directory
    download_dir = "pip_downloads"
    os.makedirs(download_dir, exist_ok=True)
    
    # Phase 1: Concurrent downloads
    print("\n📥 Phase 1: Downloading packages...")
    start_time = time.time()
    
    failed_downloads = []
    successful_downloads = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_package = {
            executor.submit(download_package, package, download_dir): package 
            for package in packages
        }
        
        for future in concurrent.futures.as_completed(future_to_package):
            package, success, error = future.result()
            if success:
                successful_downloads.append(package)
                print(f"✅ Downloaded: {package}")
            else:
                failed_downloads.append((package, error))
                print(f"❌ Failed to download: {package} - {error[:100]}")
    
    download_time = time.time() - start_time
    print(f"\n📊 Download phase completed in {download_time:.2f}s")
    print(f"✅ Successful downloads: {len(successful_downloads)}")
    print(f"❌ Failed downloads: {len(failed_downloads)}")
    
    if download_only:
        print(f"Download-only mode. Packages saved to {download_dir}")
        return
    
    # Phase 2: Install downloaded packages (if not download-only)
    print("\n🔧 Phase 2: Installing packages...")
    start_time = time.time()
    
    # First, try to install all packages with dependencies
    print("Installing with dependencies resolution...")
    try:
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', requirements_file]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            install_time = time.time() - start_time
            print(f"✅ All packages installed successfully in {install_time:.2f}s")
            print(f"📊 Total time: {download_time + install_time:.2f}s")
        else:
            print(f"❌ Installation failed: {result.stderr[:200]}")
            print("Falling back to individual package installation...")
            
            # Fallback: install successful downloads individually
            failed_installs = []
            successful_installs = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers//2) as executor:
                future_to_package = {
                    executor.submit(install_package, package): package 
                    for package in successful_downloads
                }
                
                for future in concurrent.futures.as_completed(future_to_package):
                    package, success, error = future.result()
                    if success:
                        successful_installs.append(package)
                        print(f"✅ Installed: {package}")
                    else:
                        failed_installs.append((package, error))
                        print(f"❌ Failed to install: {package} - {error[:100]}")
            
            install_time = time.time() - start_time
            print(f"\n📊 Individual installation completed in {install_time:.2f}s")
            print(f"✅ Successful installs: {len(successful_installs)}")
            print(f"❌ Failed installs: {len(failed_installs)}")
    
    except subprocess.TimeoutExpired:
        print("❌ Installation timeout. Some packages may not be installed.")
    
    total_time = time.time() - (start_time - download_time)
    print(f"\n🎉 Total processing time: {total_time:.2f}s")
    
    # Clean up download directory
    if not download_only:
        import shutil
        try:
            shutil.rmtree(download_dir)
            print(f"🧹 Cleaned up download directory: {download_dir}")
        except:
            print(f"⚠️  Could not clean up download directory: {download_dir}")

def main():
    parser = argparse.ArgumentParser(description="Fast multi-core pip installer")
    parser.add_argument(
        'requirements_file',
        nargs='?',
        default='requirements.txt',
        help='Requirements file to install (default: requirements.txt)'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=None,
        help='Number of worker threads (default: auto-detect)'
    )
    parser.add_argument(
        '-d', '--download-only',
        action='store_true',
        help='Only download packages, do not install'
    )
    
    args = parser.parse_args()
    
    try:
        fast_pip_install(
            args.requirements_file,
            max_workers=args.workers,
            download_only=args.download_only
        )
    except KeyboardInterrupt:
        print("\n⚠️  Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()