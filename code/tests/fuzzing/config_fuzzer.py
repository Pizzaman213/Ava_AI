"""
Config Fuzzer - Main Orchestrator

Coordinates mutation generation, test execution, and result collection.
"""

import copy
import gc
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import torch

try:
    from .mutation_strategies import (
        MutatedValue,
        MutationStrategy,
        get_default_strategies,
        get_quick_strategies,
    )
    from .parameter_registry import REGISTRY, ParameterSpec, get_baseline_config
    from .test_stages import StageResult, TestStageRunner
except ImportError:
    from mutation_strategies import (
        MutatedValue,
        MutationStrategy,
        get_default_strategies,
        get_quick_strategies,
    )
    from parameter_registry import REGISTRY, ParameterSpec, get_baseline_config
    from test_stages import StageResult, TestStageRunner


@dataclass
class FuzzTestResult:
    """Result of a single fuzz test case."""

    test_id: str
    description: str
    param_name: str
    mutation_value: Any
    mutation_strategy: str
    stage_results: List[StageResult] = field(default_factory=list)
    expected_failure_stage: Optional[str] = None
    total_time: float = 0.0

    @property
    def passed(self) -> bool:
        """Test passed if all stages passed."""
        return all(r.passed for r in self.stage_results)

    @property
    def failed_stage(self) -> Optional[str]:
        """Get the stage where failure occurred."""
        for r in self.stage_results:
            if not r.passed:
                return r.stage
        return None

    @property
    def error_type(self) -> Optional[str]:
        """Get the error type from the failed stage."""
        for r in self.stage_results:
            if not r.passed:
                return r.error_type
        return None

    @property
    def error_message(self) -> Optional[str]:
        """Get the error message from the failed stage."""
        for r in self.stage_results:
            if not r.passed:
                return r.error_message
        return None

    @property
    def function_location(self) -> Optional[str]:
        """Get the function location from the failed stage."""
        for r in self.stage_results:
            if not r.passed:
                return r.function_location
        return None

    @property
    def is_validation_gap(self) -> bool:
        """
        Check if this reveals a validation gap.

        A validation gap occurs when:
        - Expected failure at config stage but failed at a later stage
        - Expected to pass but failed
        """
        if self.expected_failure_stage == "config" and self.failed_stage not in [
            "config_validation",
            None,
        ]:
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "test_id": self.test_id,
            "description": self.description,
            "param_name": self.param_name,
            "mutation_value": str(self.mutation_value),
            "mutation_strategy": self.mutation_strategy,
            "passed": self.passed,
            "failed_stage": self.failed_stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "function_location": self.function_location,
            "is_validation_gap": self.is_validation_gap,
            "expected_failure_stage": self.expected_failure_stage,
            "total_time": self.total_time,
            "stage_results": [r.to_dict() for r in self.stage_results],
        }


class ConfigFuzzer:
    """Main fuzzer orchestrator."""

    def __init__(
        self,
        seed: int = 42,
        device: str = "cpu",
        strategies: Optional[List[MutationStrategy]] = None,
        max_mutations_per_param: int = 10,
        timeout_per_test: float = 30.0,
        quick_mode: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the config fuzzer.

        Args:
            seed: Random seed for reproducibility
            device: Device to run tests on ("cpu" or "cuda")
            strategies: List of mutation strategies (default: all)
            max_mutations_per_param: Maximum mutations to test per parameter
            timeout_per_test: Timeout per test in seconds
            quick_mode: Use minimal strategy set for quick testing
            verbose: Print progress during fuzzing
        """
        self.seed = seed
        self.device = device
        self.rng = random.Random(seed)
        self.timeout = timeout_per_test
        self.max_mutations_per_param = max_mutations_per_param
        self.verbose = verbose

        # Select strategies
        if strategies is not None:
            self.strategies = strategies
        elif quick_mode:
            self.strategies = get_quick_strategies()
        else:
            self.strategies = get_default_strategies()

        self.registry = REGISTRY
        self.stage_runner = TestStageRunner(
            device=device, timeout=timeout_per_test
        )

        self.results: List[FuzzTestResult] = []
        self._test_counter = 0

    def get_baseline_config(self) -> Dict[str, Any]:
        """Get baseline valid configuration."""
        return get_baseline_config()

    def apply_mutation(
        self,
        base_config: Dict[str, Any],
        param_path: str,
        value: Any,
    ) -> Dict[str, Any]:
        """
        Apply a mutation to a config at the specified path.

        Args:
            base_config: Base configuration dictionary
            param_path: Dot-separated path (e.g., "model.hidden_size")
            value: Value to set

        Returns:
            New config dict with mutation applied
        """
        config = copy.deepcopy(base_config)
        parts = param_path.split(".")

        # Navigate to parent
        obj = config
        for part in parts[:-1]:
            if part not in obj:
                obj[part] = {}
            obj = obj[part]

        # Set value
        obj[parts[-1]] = value
        return config

    def generate_test_cases(
        self, param_spec: ParameterSpec
    ) -> List[tuple[MutatedValue, Dict[str, Any]]]:
        """
        Generate test cases for a single parameter.

        Returns:
            List of (mutation, config) tuples
        """
        base_config = self.get_baseline_config()
        test_cases = []
        seen_values: Set[str] = set()

        for strategy in self.strategies:
            mutations = strategy.generate_mutations(
                param_spec, self.seed + self._test_counter
            )

            for mutation in mutations:
                # Deduplicate by value representation
                value_key = f"{mutation.value}_{mutation.description}"
                if value_key in seen_values:
                    continue
                seen_values.add(value_key)

                # Apply mutation to config
                try:
                    config = self.apply_mutation(
                        base_config, param_spec.name, mutation.value
                    )
                    test_cases.append((mutation, config))
                except Exception:
                    # Skip mutations that can't be applied
                    continue

                # Limit mutations per parameter
                if len(test_cases) >= self.max_mutations_per_param:
                    break

            if len(test_cases) >= self.max_mutations_per_param:
                break

        return test_cases

    def run_single_test(
        self,
        mutation: MutatedValue,
        config: Dict[str, Any],
        param_spec: ParameterSpec,
    ) -> FuzzTestResult:
        """Run a single fuzz test."""
        self._test_counter += 1
        test_id = f"FUZZ_{self._test_counter:04d}"

        description = f"{param_spec.name}={mutation.value} ({mutation.description})"

        if self.verbose:
            print(f"  Running {test_id}: {description[:60]}...")

        start_time = time.time()

        try:
            stage_results, _ = self.stage_runner.run_all_stages(config)
        except Exception as e:
            # Catch any unexpected errors
            stage_results = [
                StageResult(
                    passed=False,
                    stage="unknown",
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
            ]

        total_time = time.time() - start_time

        result = FuzzTestResult(
            test_id=test_id,
            description=description,
            param_name=param_spec.name,
            mutation_value=mutation.value,
            mutation_strategy=mutation.strategy,
            stage_results=stage_results,
            expected_failure_stage=param_spec.validation_stage
            if mutation.expected_failure
            else None,
            total_time=total_time,
        )

        self.results.append(result)

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def fuzz_parameter(self, param_spec: ParameterSpec) -> List[FuzzTestResult]:
        """
        Fuzz a single parameter with all strategies.

        Returns:
            List of test results for this parameter
        """
        if self.verbose:
            print(f"\nFuzzing parameter: {param_spec.name}")

        test_cases = self.generate_test_cases(param_spec)
        results = []

        for mutation, config in test_cases:
            result = self.run_single_test(mutation, config, param_spec)
            results.append(result)

        return results

    def run_all_tests(
        self,
        params: Optional[List[str]] = None,
        sections: Optional[List[str]] = None,
    ) -> List[FuzzTestResult]:
        """
        Run fuzz tests on all or selected parameters.

        Args:
            params: Specific parameter names to test (optional)
            sections: Config sections to test ("model", "training", "data")

        Returns:
            List of all test results
        """
        self.results = []
        self._test_counter = 0

        # Select parameters to test
        if params:
            param_specs = [
                self.registry.get_param(p)
                for p in params
                if self.registry.get_param(p) is not None
            ]
        elif sections:
            param_specs = []
            for section in sections:
                param_specs.extend(self.registry.get_params_by_section(section))
        else:
            param_specs = list(self.registry.all_params.values())

        if self.verbose:
            print(f"Starting fuzz tests with {len(self.strategies)} strategies")
            print(f"Testing {len(param_specs)} parameters")
            print(f"Device: {self.device}, Seed: {self.seed}")
            print("-" * 60)

        start_time = time.time()

        for param_spec in param_specs:
            self.fuzz_parameter(param_spec)

        total_time = time.time() - start_time

        if self.verbose:
            print("-" * 60)
            print(f"Completed {len(self.results)} tests in {total_time:.2f}s")
            passed = sum(1 for r in self.results if r.passed)
            print(f"Passed: {passed}/{len(self.results)} ({100*passed/len(self.results):.1f}%)")

        return self.results

    def get_validation_gaps(self) -> List[FuzzTestResult]:
        """Get all results that reveal validation gaps."""
        return [r for r in self.results if r.is_validation_gap]

    def get_failures_by_stage(self) -> Dict[str, List[FuzzTestResult]]:
        """Group failures by the stage where they occurred."""
        failures: Dict[str, List[FuzzTestResult]] = {}
        for result in self.results:
            if not result.passed:
                stage = result.failed_stage or "unknown"
                if stage not in failures:
                    failures[stage] = []
                failures[stage].append(result)
        return failures

    def get_failures_by_error_type(self) -> Dict[str, List[FuzzTestResult]]:
        """Group failures by error type."""
        failures: Dict[str, List[FuzzTestResult]] = {}
        for result in self.results:
            if not result.passed:
                error_type = result.error_type or "unknown"
                if error_type not in failures:
                    failures[error_type] = []
                failures[error_type].append(result)
        return failures


def create_fuzzer(
    seed: int = 42,
    device: str = "cpu",
    quick: bool = False,
    verbose: bool = False,
) -> ConfigFuzzer:
    """
    Factory function to create a ConfigFuzzer.

    Args:
        seed: Random seed
        device: Device ("cpu" or "cuda")
        quick: Use quick mode with minimal strategies
        verbose: Print progress

    Returns:
        ConfigFuzzer instance
    """
    return ConfigFuzzer(
        seed=seed,
        device=device,
        quick_mode=quick,
        verbose=verbose,
    )
