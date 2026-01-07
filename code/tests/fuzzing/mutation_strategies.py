"""
Mutation Strategies for Config Fuzzer

Implements various strategies for generating mutated config values to test validation.
"""

import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

try:
    from .parameter_registry import ParameterSpec
except ImportError:
    from parameter_registry import ParameterSpec


@dataclass
class MutatedValue:
    """A mutated value with description."""

    value: Any
    description: str
    strategy: str = ""
    expected_failure: bool = True  # Whether this should cause a failure


class MutationStrategy(ABC):
    """Base class for mutation strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for reporting."""
        pass

    @abstractmethod
    def generate_mutations(
        self, param_spec: ParameterSpec, seed: int
    ) -> List[MutatedValue]:
        """Generate mutations for a parameter."""
        pass

    def _add_strategy_name(self, mutations: List[MutatedValue]) -> List[MutatedValue]:
        """Add strategy name to all mutations."""
        for m in mutations:
            m.strategy = self.name
        return mutations


class BoundaryValueStrategy(MutationStrategy):
    """Generate boundary value mutations (0, -1, max values, edge cases)."""

    @property
    def name(self) -> str:
        return "boundary"

    def generate_mutations(
        self, param_spec: ParameterSpec, seed: int
    ) -> List[MutatedValue]:
        mutations = []

        if param_spec.param_type == int:
            mutations.extend([
                MutatedValue(0, "zero"),
                MutatedValue(-1, "negative_one"),
                MutatedValue(1, "one"),
                MutatedValue(2**31 - 1, "max_int32"),
                MutatedValue(-(2**31), "min_int32"),
            ])

            # Range-specific boundaries
            if param_spec.valid_range:
                min_val, max_val = param_spec.valid_range
                if min_val is not None:
                    mutations.append(
                        MutatedValue(int(min_val) - 1, f"below_min_{min_val}")
                    )
                    mutations.append(
                        MutatedValue(int(min_val), f"at_min_{min_val}", expected_failure=False)
                    )
                if max_val is not None:
                    mutations.append(
                        MutatedValue(int(max_val) + 1, f"above_max_{max_val}")
                    )
                    mutations.append(
                        MutatedValue(int(max_val), f"at_max_{max_val}", expected_failure=False)
                    )

        elif param_spec.param_type == float:
            mutations.extend([
                MutatedValue(0.0, "zero"),
                MutatedValue(-1.0, "negative_one"),
                MutatedValue(1.0, "one"),
                MutatedValue(float("inf"), "positive_infinity"),
                MutatedValue(float("-inf"), "negative_infinity"),
                MutatedValue(float("nan"), "nan"),
                MutatedValue(1e308, "very_large"),
                MutatedValue(-1e308, "very_large_negative"),
                MutatedValue(1e-308, "very_small_positive"),
            ])

            # Range-specific boundaries
            if param_spec.valid_range:
                min_val, max_val = param_spec.valid_range
                if min_val is not None:
                    mutations.append(
                        MutatedValue(min_val - 0.001, f"below_min_{min_val}")
                    )
                if max_val is not None:
                    mutations.append(
                        MutatedValue(max_val + 0.001, f"above_max_{max_val}")
                    )

        elif param_spec.param_type == bool:
            mutations.extend([
                MutatedValue(True, "true", expected_failure=False),
                MutatedValue(False, "false", expected_failure=False),
            ])

        return self._add_strategy_name(mutations)


class TypeMutationStrategy(MutationStrategy):
    """Generate type confusion mutations (wrong types for fields)."""

    @property
    def name(self) -> str:
        return "type_mutation"

    def generate_mutations(
        self, param_spec: ParameterSpec, seed: int
    ) -> List[MutatedValue]:
        mutations = [
            MutatedValue(None, "none_value"),
        ]

        if param_spec.param_type == int:
            mutations.extend([
                MutatedValue("8", "string_number"),
                MutatedValue("eight", "string_word"),
                MutatedValue(8.5, "float_with_decimal"),
                MutatedValue(8.0, "float_whole"),
                MutatedValue([8], "list_single"),
                MutatedValue({"value": 8}, "dict"),
                MutatedValue(True, "boolean_true"),
                MutatedValue(complex(8, 0), "complex_number"),
            ])

        elif param_spec.param_type == float:
            mutations.extend([
                MutatedValue("0.5", "string_decimal"),
                MutatedValue("half", "string_word"),
                MutatedValue([0.5], "list_single"),
                MutatedValue({"value": 0.5}, "dict"),
            ])

        elif param_spec.param_type == str:
            mutations.extend([
                MutatedValue(123, "integer"),
                MutatedValue(1.5, "float"),
                MutatedValue([], "empty_list"),
                MutatedValue({}, "empty_dict"),
                MutatedValue(True, "boolean"),
            ])

        elif param_spec.param_type == bool:
            mutations.extend([
                MutatedValue(0, "zero_int"),
                MutatedValue(1, "one_int"),
                MutatedValue("true", "string_true"),
                MutatedValue("false", "string_false"),
                MutatedValue("yes", "string_yes"),
                MutatedValue("", "empty_string"),
            ])

        return self._add_strategy_name(mutations)


class InvalidEnumStrategy(MutationStrategy):
    """Generate invalid enum values for string parameters with valid_values."""

    @property
    def name(self) -> str:
        return "invalid_enum"

    def generate_mutations(
        self, param_spec: ParameterSpec, seed: int
    ) -> List[MutatedValue]:
        mutations = []

        if param_spec.valid_values:
            # Invalid string values
            mutations.extend([
                MutatedValue("invalid_value", "invalid_string"),
                MutatedValue("", "empty_string"),
                MutatedValue(" ", "whitespace"),
                MutatedValue("MIXTRAL", "uppercase_valid"),  # Case sensitivity
                MutatedValue("mixtral ", "trailing_space"),
                MutatedValue(" mixtral", "leading_space"),
            ])

            # Wrong type for enum
            mutations.extend([
                MutatedValue(123, "integer_for_enum"),
                MutatedValue(1.5, "float_for_enum"),
                MutatedValue(["mixtral"], "list_for_enum"),
            ])

            # Similar but wrong values
            if "mixtral" in param_spec.valid_values:
                mutations.append(MutatedValue("mistral", "typo_mixtral"))
            if "deepseek" in param_spec.valid_values:
                mutations.append(MutatedValue("deep_seek", "typo_deepseek"))
            if "swiglu" in param_spec.valid_values:
                mutations.append(MutatedValue("swish", "similar_activation"))

        return self._add_strategy_name(mutations)


class CrossFieldConstraintStrategy(MutationStrategy):
    """Generate mutations that violate cross-field constraints."""

    @property
    def name(self) -> str:
        return "cross_field"

    def generate_mutations(
        self, param_spec: ParameterSpec, seed: int
    ) -> List[MutatedValue]:
        mutations = []

        # Handle specific cross-field constraints
        if "hidden_size" in param_spec.name:
            # Generate hidden_size values not divisible by common head counts
            mutations.extend([
                MutatedValue(65, "not_divisible_by_4"),
                MutatedValue(127, "prime_number"),
                MutatedValue(100, "not_divisible_by_common_heads"),
                MutatedValue(33, "small_prime"),
            ])

        if "num_experts_per_token" in param_spec.name:
            # Values that could exceed num_experts
            mutations.extend([
                MutatedValue(100, "likely_exceeds_experts"),
                MutatedValue(1000, "definitely_exceeds_experts"),
            ])

        if "max_position_embeddings" in param_spec.name:
            # Very small values that might cause issues
            mutations.extend([
                MutatedValue(1, "minimum_sequence"),
                MutatedValue(2, "very_short_sequence"),
            ])

        if "intermediate_size" in param_spec.name:
            # Should typically be larger than hidden_size
            mutations.extend([
                MutatedValue(1, "smaller_than_hidden"),
                MutatedValue(16, "very_small_ffn"),
            ])

        return self._add_strategy_name(mutations)


class RandomValueStrategy(MutationStrategy):
    """Generate seeded random values within plausible ranges."""

    @property
    def name(self) -> str:
        return "random"

    def generate_mutations(
        self, param_spec: ParameterSpec, seed: int
    ) -> List[MutatedValue]:
        rng = random.Random(seed)
        mutations = []

        if param_spec.param_type == int:
            # Random positive values
            mutations.extend([
                MutatedValue(
                    rng.randint(1, 10000),
                    f"random_positive_{seed}",
                    expected_failure=False
                ),
                MutatedValue(
                    rng.randint(-1000, -1),
                    f"random_negative_{seed}"
                ),
                MutatedValue(
                    rng.randint(1, 100) * rng.randint(1, 100),
                    f"random_product_{seed}",
                    expected_failure=False
                ),
            ])

        elif param_spec.param_type == float:
            # Random float values
            mutations.extend([
                MutatedValue(
                    rng.uniform(0, 10),
                    f"random_positive_{seed}",
                    expected_failure=False
                ),
                MutatedValue(
                    rng.uniform(-10, 0),
                    f"random_negative_{seed}"
                ),
                MutatedValue(
                    rng.uniform(0, 1),
                    f"random_probability_{seed}",
                    expected_failure=False
                ),
            ])

            # For dropout-like params, test around boundaries
            if param_spec.valid_range == (0.0, 1.0):
                mutations.extend([
                    MutatedValue(
                        rng.uniform(-1, 0),
                        f"random_below_zero_{seed}"
                    ),
                    MutatedValue(
                        rng.uniform(1, 2),
                        f"random_above_one_{seed}"
                    ),
                ])

        elif param_spec.param_type == str and param_spec.valid_values:
            # Random selection from valid values (should pass)
            if param_spec.valid_values:
                mutations.append(
                    MutatedValue(
                        rng.choice(param_spec.valid_values),
                        f"random_valid_{seed}",
                        expected_failure=False
                    )
                )
            # Random garbage string
            mutations.append(
                MutatedValue(
                    "".join(rng.choices("abcdefghijklmnop", k=8)),
                    f"random_string_{seed}"
                )
            )

        return self._add_strategy_name(mutations)


class EmptyMissingStrategy(MutationStrategy):
    """Generate empty, null, and missing value mutations."""

    @property
    def name(self) -> str:
        return "empty_missing"

    def generate_mutations(
        self, param_spec: ParameterSpec, seed: int
    ) -> List[MutatedValue]:
        mutations = [
            MutatedValue(None, "none"),
        ]

        if param_spec.param_type == str:
            mutations.extend([
                MutatedValue("", "empty_string"),
                MutatedValue("   ", "whitespace_only"),
                MutatedValue("\n", "newline_only"),
                MutatedValue("\t", "tab_only"),
            ])

        if param_spec.param_type == int:
            mutations.append(MutatedValue("", "empty_string_for_int"))

        if param_spec.param_type == float:
            mutations.append(MutatedValue("", "empty_string_for_float"))

        # List and dict empties
        mutations.extend([
            MutatedValue([], "empty_list"),
            MutatedValue({}, "empty_dict"),
        ])

        return self._add_strategy_name(mutations)


class NegativeValueStrategy(MutationStrategy):
    """Generate negative values for typically positive parameters."""

    @property
    def name(self) -> str:
        return "negative"

    def generate_mutations(
        self, param_spec: ParameterSpec, seed: int
    ) -> List[MutatedValue]:
        mutations = []

        if param_spec.param_type == int:
            mutations.extend([
                MutatedValue(-1, "negative_one"),
                MutatedValue(-100, "negative_hundred"),
                MutatedValue(-sys.maxsize, "min_system_int"),
            ])

        elif param_spec.param_type == float:
            mutations.extend([
                MutatedValue(-0.001, "small_negative"),
                MutatedValue(-1.0, "negative_one"),
                MutatedValue(-100.0, "negative_hundred"),
                MutatedValue(-1e10, "large_negative"),
            ])

        return self._add_strategy_name(mutations)


class ZeroValueStrategy(MutationStrategy):
    """Generate zero values for parameters that should be positive."""

    @property
    def name(self) -> str:
        return "zero"

    def generate_mutations(
        self, param_spec: ParameterSpec, seed: int
    ) -> List[MutatedValue]:
        mutations = []

        if param_spec.param_type == int:
            mutations.append(MutatedValue(0, "zero_int"))

        elif param_spec.param_type == float:
            mutations.extend([
                MutatedValue(0.0, "zero_float"),
                MutatedValue(-0.0, "negative_zero"),
            ])

        return self._add_strategy_name(mutations)


class SpecialCharacterStrategy(MutationStrategy):
    """Generate values with special characters and unicode."""

    @property
    def name(self) -> str:
        return "special_chars"

    def generate_mutations(
        self, param_spec: ParameterSpec, seed: int
    ) -> List[MutatedValue]:
        mutations = []

        if param_spec.param_type == str:
            mutations.extend([
                MutatedValue("router\x00type", "null_byte"),
                MutatedValue("router\ntype", "newline"),
                MutatedValue("router\ttype", "tab"),
                MutatedValue("router type", "space_in_middle"),
                MutatedValue("\u200b", "zero_width_space"),
                MutatedValue("mixtral\u0000", "null_terminated"),
            ])

        return self._add_strategy_name(mutations)


# List of all available strategies
ALL_STRATEGIES: List[MutationStrategy] = [
    BoundaryValueStrategy(),
    TypeMutationStrategy(),
    InvalidEnumStrategy(),
    CrossFieldConstraintStrategy(),
    RandomValueStrategy(),
    EmptyMissingStrategy(),
    NegativeValueStrategy(),
    ZeroValueStrategy(),
    SpecialCharacterStrategy(),
]


def get_default_strategies() -> List[MutationStrategy]:
    """Get the default set of mutation strategies."""
    return [
        BoundaryValueStrategy(),
        TypeMutationStrategy(),
        InvalidEnumStrategy(),
        CrossFieldConstraintStrategy(),
        RandomValueStrategy(),
        NegativeValueStrategy(),
        ZeroValueStrategy(),
    ]


def get_quick_strategies() -> List[MutationStrategy]:
    """Get a minimal set of strategies for quick testing."""
    return [
        BoundaryValueStrategy(),
        ZeroValueStrategy(),
        NegativeValueStrategy(),
    ]
