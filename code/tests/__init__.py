"""
Comprehensive Testing Suite for Advanced LLM Training Improvements

This package contains comprehensive tests for all advanced training optimizations including:
- Advanced optimizers (Lion, Sophia, AdaFactor)
- Progressive training framework
- Enhanced learning rate schedulers
- FP8 training support
- Integration and performance tests

Test Structure:
- Unit tests for individual components
- Integration tests for combined functionality
- Performance benchmarks
- Stress and edge case tests
"""

import os
import sys
import torch
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test configuration
TEST_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "precision": torch.float32,
    "seed": 42,
    "temp_dir": "/tmp/llm_tests",
    "max_test_time": 3600,  # 1 hour max per test
    "memory_threshold_gb": 8.0,  # Skip memory-intensive tests on low memory systems
}

# Create temp directory for test artifacts
os.makedirs(TEST_CONFIG["temp_dir"], exist_ok=True)

# Set deterministic behavior for testing
torch.manual_seed(TEST_CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(TEST_CONFIG["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False