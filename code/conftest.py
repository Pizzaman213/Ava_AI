"""
PyTest configuration for the project
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Ensure all source packages can be imported
sys.path.insert(0, str(project_root / "src"))