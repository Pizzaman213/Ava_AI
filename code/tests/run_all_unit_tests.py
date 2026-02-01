#!/usr/bin/env python3
"""
Unified Test Runner for Ava MoE Framework
==========================================

Runs all unit tests across all modules with configurable options.

Usage:
    python code/tests/run_all_unit_tests.py                    # Run all tests
    python code/tests/run_all_unit_tests.py --module training  # Run training tests only
    python code/tests/run_all_unit_tests.py --coverage         # Run with coverage report
    python code/tests/run_all_unit_tests.py --parallel         # Run in parallel
    python code/tests/run_all_unit_tests.py --quick            # Run fast tests only
    python code/tests/run_all_unit_tests.py --cpu-only         # Force CPU mode
    python code/tests/run_all_unit_tests.py --list             # List available modules

Development Guidelines:
    - Testing must be performed on RTX 3060 (device 1)
    - RTX 3090 Ti (device 0) is reserved for production training
    - Use --cpu-only for CI environments
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Test modules and their paths
TEST_MODULES = {
    'training': 'code/tests/unit/training/',
    'phases': 'code/tests/unit/phases/',
    'models': 'code/tests/unit/models/',
    'cuda': 'code/tests/unit/cuda/',
    'data': 'code/tests/unit/data/',
    'optimizations': 'code/tests/unit/optimizations/',
    'config': 'code/tests/unit/config/',
}

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_test_paths(module: str = None) -> list:
    """Get test paths for specified module or all modules."""
    if module:
        if module not in TEST_MODULES:
            print(f"Unknown module: {module}")
            print(f"Available: {', '.join(TEST_MODULES.keys())}")
            sys.exit(1)
        return [str(PROJECT_ROOT / TEST_MODULES[module])]

    # All modules
    paths = []
    for path in TEST_MODULES.values():
        full_path = PROJECT_ROOT / path
        if full_path.exists():
            paths.append(str(full_path))
    return paths


def run_tests(args) -> int:
    """Run unit tests with specified options."""
    cmd = ['pytest']

    # Add test paths
    paths = get_test_paths(args.module)
    if not paths:
        print("No test paths found. Creating test directory structure...")
        (PROJECT_ROOT / 'code/tests/unit').mkdir(parents=True, exist_ok=True)
        paths = [str(PROJECT_ROOT / 'code/tests/unit/')]
    cmd.extend(paths)

    # Verbosity
    if args.verbose:
        cmd.append('-v')
    cmd.append('--tb=short')

    # Coverage
    if args.coverage:
        cmd.extend([
            '--cov=code/src/ava',
            '--cov-report=html',
            '--cov-report=term-missing',
            '--cov-fail-under=0',  # Don't fail on coverage threshold
        ])

    # Parallel execution
    if args.parallel:
        try:
            import pytest_xdist
            cmd.extend(['-n', 'auto'])
        except ImportError:
            print("Warning: pytest-xdist not installed. Running sequentially.")

    # Quick mode (skip slow tests)
    if args.quick:
        cmd.extend(['-m', 'not slow'])

    # CPU only mode
    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif not args.use_3090:
        # Default: use RTX 3060 (device 1) per development guidelines
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # Show durations
    cmd.extend(['--durations=10'])

    # Warnings
    cmd.extend([
        '-W', 'ignore::DeprecationWarning',
        '-W', 'ignore::UserWarning',
    ])

    # Color output
    cmd.append('--color=yes')

    # Print command
    print("=" * 60)
    print("Ava MoE Framework - Unit Test Runner")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print(f"Test paths: {paths}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 60)
    print()

    # Run tests
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def list_modules():
    """List available test modules."""
    print("Available test modules:")
    print("=" * 40)
    for name, path in TEST_MODULES.items():
        full_path = PROJECT_ROOT / path
        status = "[EXISTS]" if full_path.exists() else "[PENDING]"
        print(f"  {name:15} -> {path:30} {status}")
    print()
    print("Note: Modules marked [PENDING] will be created during implementation.")


def main():
    parser = argparse.ArgumentParser(
        description='Run Ava MoE Framework unit tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run all tests
  %(prog)s -m training              # Run training module tests only
  %(prog)s -m models                # Run model tests only
  %(prog)s --coverage               # Run with coverage report
  %(prog)s --parallel               # Run tests in parallel
  %(prog)s --quick                  # Skip slow tests
  %(prog)s --cpu-only               # Run without GPU
  %(prog)s -v                       # Verbose output

Development Guidelines:
  - Use RTX 3060 for testing (default, device 1)
  - RTX 3090 Ti is reserved for production training
  - Use --cpu-only for CI environments
        """
    )

    parser.add_argument(
        '--module', '-m',
        choices=list(TEST_MODULES.keys()),
        help='Run tests for specific module only'
    )
    parser.add_argument(
        '--coverage', '-c',
        action='store_true',
        help='Generate coverage report'
    )
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Run tests in parallel (requires pytest-xdist)'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run fast tests only (skip @pytest.mark.slow)'
    )
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Force CPU mode (no GPU)'
    )
    parser.add_argument(
        '--use-3090',
        action='store_true',
        help='Use RTX 3090 Ti instead of RTX 3060 (not recommended for testing)'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available test modules'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose test output'
    )

    args = parser.parse_args()

    if args.list:
        list_modules()
        sys.exit(0)

    sys.exit(run_tests(args))


if __name__ == '__main__':
    main()
