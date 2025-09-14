"""
Setup script for MoE++ LLM with advanced features

This setup.py is kept for backward compatibility and dynamic requirements handling.
The main configuration is now in pyproject.toml.
"""
from setuptools import setup
import os
import platform

def get_install_requires():
    """Dynamically determine requirements based on platform"""
    # Check if running on macOS
    if platform.system() == 'Darwin' and os.path.exists('requirements-macos.txt'):
        req_file = 'requirements-macos.txt'
    else:
        req_file = 'requirements.txt'
    
    with open(req_file) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Only specify dynamic dependencies here
# All other configuration is in pyproject.toml
if __name__ == '__main__':
    setup(
        install_requires=get_install_requires()
    )