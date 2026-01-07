"""
Config Fuzzer for Ava LLM Training Framework

A comprehensive fuzzing toolkit for finding validation gaps and bugs
in configuration handling.

Usage:
    from tests.fuzzing import ConfigFuzzer, generate_report

    fuzzer = ConfigFuzzer(seed=42, device="cpu", quick_mode=True)
    results = fuzzer.run_all_tests()
    report = generate_report(results, seed=42, device="cpu")
    report.print_summary()

CLI Usage:
    python -m tests.fuzzing.run_fuzzer --quick --verbose
"""

from .config_fuzzer import ConfigFuzzer, FuzzTestResult, create_fuzzer
from .mutation_strategies import (
    ALL_STRATEGIES,
    BoundaryValueStrategy,
    CrossFieldConstraintStrategy,
    EmptyMissingStrategy,
    InvalidEnumStrategy,
    MutatedValue,
    MutationStrategy,
    NegativeValueStrategy,
    RandomValueStrategy,
    TypeMutationStrategy,
    ZeroValueStrategy,
    get_default_strategies,
    get_quick_strategies,
)
from .parameter_registry import (
    REGISTRY,
    ParameterRegistry,
    ParameterSpec,
    get_baseline_config,
)
from .report_generator import (
    BrokenFunction,
    FuzzReport,
    ValidationGap,
    generate_report,
)
from .test_stages import StageResult, TestStageRunner, run_single_test

__all__ = [
    # Main classes
    "ConfigFuzzer",
    "FuzzTestResult",
    "create_fuzzer",
    # Mutation strategies
    "MutationStrategy",
    "MutatedValue",
    "BoundaryValueStrategy",
    "TypeMutationStrategy",
    "InvalidEnumStrategy",
    "CrossFieldConstraintStrategy",
    "RandomValueStrategy",
    "EmptyMissingStrategy",
    "NegativeValueStrategy",
    "ZeroValueStrategy",
    "ALL_STRATEGIES",
    "get_default_strategies",
    "get_quick_strategies",
    # Parameter registry
    "ParameterRegistry",
    "ParameterSpec",
    "REGISTRY",
    "get_baseline_config",
    # Test stages
    "TestStageRunner",
    "StageResult",
    "run_single_test",
    # Report generation
    "FuzzReport",
    "BrokenFunction",
    "ValidationGap",
    "generate_report",
]

__version__ = "1.0.0"
