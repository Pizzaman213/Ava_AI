#!/usr/bin/env python3
"""
Simple Test Runner Script

Easy-to-use script for running the advanced LLM training test suite.
"""

import os
import sys
import subprocess
import argparse

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run Advanced LLM Training Tests")
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test suite (skip benchmarks and stress tests)'
    )
    parser.add_argument(
        '--suite',
        choices=['optimizers', 'progressive', 'schedulers', 'fp8', 'integration', 'benchmarks', 'stress'],
        help='Run specific test suite only'
    )
    parser.add_argument(
        '--output-dir',
        help='Output directory for test reports'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Build command
    cmd = [sys.executable, '-m', 'tests.test_runner']

    if args.quick:
        cmd.append('--quick')

    if args.suite:
        cmd.extend(['--suites', args.suite])

    if args.output_dir:
        cmd.extend(['--output-dir', args.output_dir])

    if args.verbose:
        cmd.extend(['--verbosity', '2'])

    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    # Run tests
    print("🚀 Starting Advanced LLM Training Test Suite...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Failed to run tests: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()