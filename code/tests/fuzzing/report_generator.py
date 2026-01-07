"""
Report Generator for Config Fuzzer

Generates JSON reports with detailed analysis of fuzz test results.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from .config_fuzzer import FuzzTestResult
except ImportError:
    from config_fuzzer import FuzzTestResult


@dataclass
class BrokenFunction:
    """Information about a function that failed during fuzzing."""

    function: str
    file: str
    line: int
    frequency: int
    error_types: List[str]
    triggers: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function": self.function,
            "file": self.file,
            "line": self.line,
            "frequency": self.frequency,
            "error_types": self.error_types,
            "triggers": self.triggers[:10],  # Limit triggers
        }


@dataclass
class ValidationGap:
    """Information about a validation gap."""

    param_name: str
    mutation_value: str
    expected_stage: str
    actual_stage: str
    error_type: str
    error_message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "param_name": self.param_name,
            "mutation_value": self.mutation_value,
            "expected_stage": self.expected_stage,
            "actual_stage": self.actual_stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass
class FuzzReport:
    """Complete fuzz testing report."""

    results: List[FuzzTestResult]
    seed: int
    device: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Compute derived statistics."""
        self._compute_statistics()

    def _compute_statistics(self):
        """Compute all statistics from results."""
        self._error_types = Counter()
        self._broken_functions: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "frequency": 0,
                "triggers": [],
                "error_types": set(),
                "file": "",
                "line": 0,
            }
        )
        self._validation_gaps: List[ValidationGap] = []
        self._failures_by_stage = defaultdict(list)
        self._failures_by_param = defaultdict(list)

        for result in self.results:
            if not result.passed:
                # Count error types
                if result.error_type:
                    self._error_types[result.error_type] += 1

                # Track failures by stage
                if result.failed_stage:
                    self._failures_by_stage[result.failed_stage].append(result)

                # Track failures by parameter
                self._failures_by_param[result.param_name].append(result)

                # Track broken functions
                if result.function_location:
                    loc = result.function_location
                    parts = loc.rsplit(":", 2)
                    if len(parts) >= 2:
                        func_name = parts[-2] if len(parts) >= 2 else "unknown"
                        file_path = parts[0] if len(parts) >= 1 else "unknown"
                        line_no = int(parts[-1]) if parts[-1].isdigit() else 0

                        key = f"{file_path}:{func_name}"
                        self._broken_functions[key]["frequency"] += 1
                        self._broken_functions[key]["triggers"].append(
                            result.description
                        )
                        if result.error_type:
                            self._broken_functions[key]["error_types"].add(
                                result.error_type
                            )
                        self._broken_functions[key]["file"] = file_path
                        self._broken_functions[key]["line"] = line_no
                        self._broken_functions[key]["function"] = func_name

                # Check for validation gaps
                if result.is_validation_gap:
                    self._validation_gaps.append(
                        ValidationGap(
                            param_name=result.param_name,
                            mutation_value=str(result.mutation_value),
                            expected_stage=result.expected_failure_stage or "config",
                            actual_stage=result.failed_stage or "unknown",
                            error_type=result.error_type or "unknown",
                            error_message=result.error_message or "",
                        )
                    )

    @property
    def total_tests(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return self.total_tests - self.passed_count

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total_tests if self.total_tests > 0 else 0.0

    @property
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            "total_tests": self.total_tests,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "pass_rate": self.pass_rate,
            "unique_errors": len(self._error_types),
            "broken_functions": len(self._broken_functions),
            "validation_gaps": len(self._validation_gaps),
            "error_types": dict(self._error_types),
            "failures_by_stage": {
                k: len(v) for k, v in self._failures_by_stage.items()
            },
        }

    @property
    def broken_functions(self) -> List[BrokenFunction]:
        """Get list of broken functions sorted by frequency."""
        functions = []
        for key, data in sorted(
            self._broken_functions.items(), key=lambda x: -x[1]["frequency"]
        ):
            functions.append(
                BrokenFunction(
                    function=data["function"],
                    file=data["file"],
                    line=data["line"],
                    frequency=data["frequency"],
                    error_types=list(data["error_types"]),
                    triggers=data["triggers"],
                )
            )
        return functions

    @property
    def validation_gaps(self) -> List[ValidationGap]:
        """Get list of validation gaps."""
        return self._validation_gaps

    def get_failures_by_param(self) -> Dict[str, List[FuzzTestResult]]:
        """Get failures grouped by parameter name."""
        return dict(self._failures_by_param)

    def get_most_problematic_params(self, top_n: int = 10) -> List[tuple[str, int]]:
        """Get parameters with the most failures."""
        return sorted(
            [(k, len(v)) for k, v in self._failures_by_param.items()],
            key=lambda x: -x[1],
        )[:top_n]

    def to_json(self, include_all_results: bool = True) -> str:
        """
        Export report to JSON string.

        Args:
            include_all_results: Include all individual test results
        """
        data = {
            "summary": self.summary,
            "broken_functions": [bf.to_dict() for bf in self.broken_functions],
            "validation_gaps": [vg.to_dict() for vg in self.validation_gaps],
            "most_problematic_params": self.get_most_problematic_params(),
            "seed": self.seed,
            "device": self.device,
            "timestamp": self.timestamp,
        }

        if include_all_results:
            data["results"] = [r.to_dict() for r in self.results]

        return json.dumps(data, indent=2, default=str)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return json.loads(self.to_json())

    def print_summary(self):
        """Print human-readable summary to console."""
        s = self.summary

        print(f"\n{'='*70}")
        print(f"CONFIG FUZZER REPORT - {self.timestamp}")
        print(f"{'='*70}")
        print(f"Seed: {self.seed} | Device: {self.device}")
        print(f"{'-'*70}")

        # Test results
        print(f"\nTEST RESULTS:")
        print(f"  Total Tests:  {s['total_tests']}")
        print(f"  Passed:       {s['passed']} ({s['pass_rate']*100:.1f}%)")
        print(f"  Failed:       {s['failed']}")

        # Error types
        if s["error_types"]:
            print(f"\nERROR TYPES:")
            for error_type, count in sorted(
                s["error_types"].items(), key=lambda x: -x[1]
            ):
                print(f"  {error_type}: {count}")

        # Failures by stage
        if s["failures_by_stage"]:
            print(f"\nFAILURES BY STAGE:")
            for stage, count in sorted(
                s["failures_by_stage"].items(), key=lambda x: -x[1]
            ):
                print(f"  {stage}: {count}")

        # Broken functions
        if self.broken_functions:
            print(f"\nBROKEN FUNCTIONS (top 10):")
            for bf in self.broken_functions[:10]:
                print(f"  [{bf.frequency}x] {bf.function} ({bf.file}:{bf.line})")
                if bf.triggers:
                    print(f"       Triggers: {bf.triggers[0][:50]}...")

        # Validation gaps
        if self.validation_gaps:
            print(f"\nVALIDATION GAPS ({len(self.validation_gaps)} found):")
            for vg in self.validation_gaps[:10]:
                print(f"  {vg.param_name}={vg.mutation_value[:30]}")
                print(f"    Expected: {vg.expected_stage}, Actual: {vg.actual_stage}")

        # Most problematic params
        problematic = self.get_most_problematic_params(5)
        if problematic:
            print(f"\nMOST PROBLEMATIC PARAMETERS:")
            for param, count in problematic:
                print(f"  {param}: {count} failures")

        print(f"\n{'='*70}")


def generate_report(
    results: List[FuzzTestResult],
    seed: int,
    device: str,
    output_path: Optional[str] = None,
    print_summary: bool = True,
) -> FuzzReport:
    """
    Generate a fuzz report from test results.

    Args:
        results: List of fuzz test results
        seed: Random seed used
        device: Device tests ran on
        output_path: Optional path to save JSON report
        print_summary: Print summary to console

    Returns:
        FuzzReport instance
    """
    report = FuzzReport(
        results=results,
        seed=seed,
        device=device,
    )

    if print_summary:
        report.print_summary()

    if output_path:
        with open(output_path, "w") as f:
            f.write(report.to_json())
        print(f"\nReport saved to: {output_path}")

    return report
