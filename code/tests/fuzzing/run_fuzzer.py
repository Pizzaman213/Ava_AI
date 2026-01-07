#!/usr/bin/env python3
"""
Config Fuzzer CLI

Command-line interface for running the config fuzzer.

Usage:
    python run_fuzzer.py [options]

Examples:
    # Run all tests on CPU
    python run_fuzzer.py --device cpu

    # Quick mode with minimal strategies
    python run_fuzzer.py --quick --verbose

    # Test specific parameters
    python run_fuzzer.py --params hidden_size,num_experts

    # Test specific sections
    python run_fuzzer.py --sections model,training
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Config Fuzzer for Ava LLM Training Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --device cpu --output fuzz_results
  %(prog)s --quick --verbose
  %(prog)s --params hidden_size,num_experts,dropout
  %(prog)s --sections model --max-mutations 5
        """,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run tests on (default: cpu)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for reports (default: code/tests/fuzz_results)",
    )

    parser.add_argument(
        "--max-mutations",
        type=int,
        default=10,
        help="Maximum mutations per parameter (default: 10)",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout per test in seconds (default: 30)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use minimal strategy set (boundary, zero, negative)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output during fuzzing",
    )

    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Comma-separated list of parameters to test (e.g., hidden_size,num_experts)",
    )

    parser.add_argument(
        "--sections",
        type=str,
        default=None,
        help="Comma-separated list of config sections to test (model,training,data)",
    )

    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Don't print summary to console",
    )

    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only output JSON (no console summary)",
    )

    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0.0,
        help="Exit with error if pass rate is below this threshold (0.0-1.0)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Import fuzzer components
    from config_fuzzer import ConfigFuzzer
    from report_generator import generate_report

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "fuzz_results"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse params and sections
    params = args.params.split(",") if args.params else None
    sections = args.sections.split(",") if args.sections else None

    # Create fuzzer
    fuzzer = ConfigFuzzer(
        seed=args.seed,
        device=args.device,
        max_mutations_per_param=args.max_mutations,
        timeout_per_test=args.timeout,
        quick_mode=args.quick,
        verbose=args.verbose,
    )

    # Print header
    if not args.json_only:
        print("=" * 70)
        print("AVA CONFIG FUZZER")
        print("=" * 70)
        print(f"Seed:       {args.seed}")
        print(f"Device:     {args.device}")
        print(f"Quick Mode: {args.quick}")
        print(f"Max Mutations/Param: {args.max_mutations}")
        if params:
            print(f"Parameters: {params}")
        if sections:
            print(f"Sections:   {sections}")
        print("=" * 70)

    # Run fuzzer
    start_time = time.time()

    try:
        results = fuzzer.run_all_tests(params=params, sections=sections)
    except KeyboardInterrupt:
        print("\nFuzzing interrupted by user")
        results = fuzzer.results

    total_time = time.time() - start_time

    # Generate report
    timestamp = int(time.time())
    report_path = output_dir / f"fuzz_report_{timestamp}.json"

    report = generate_report(
        results=results,
        seed=args.seed,
        device=args.device,
        output_path=str(report_path),
        print_summary=not args.no_summary and not args.json_only,
    )

    # Print timing
    if not args.json_only:
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Report saved: {report_path}")

    # Check threshold
    if args.fail_threshold > 0:
        if report.pass_rate < args.fail_threshold:
            print(
                f"\nFAILED: Pass rate {report.pass_rate:.1%} is below "
                f"threshold {args.fail_threshold:.1%}"
            )
            return 1

    # Return exit code based on whether there were failures
    return 0 if report.passed_count == report.total_tests else 0


if __name__ == "__main__":
    sys.exit(main())
