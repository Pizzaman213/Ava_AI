"""
Test Stage Runners for Config Fuzzer

Implements progressive testing through 4 stages:
1. Config validation (dataclass __post_init__, ConfigValidator)
2. Model instantiation
3. Forward pass
4. Backward pass
"""

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure src is in path
_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)


@dataclass
class StageResult:
    """Result of running a single test stage."""

    passed: bool
    stage: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    function_location: Optional[str] = None
    execution_time: float = 0.0
    traceback_str: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "stage": self.stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "function_location": self.function_location,
            "execution_time": self.execution_time,
        }


class TestStageRunner:
    """Runs fuzz tests through progressive stages."""

    STAGES = ["config_validation", "model_instantiation", "model_forward", "model_backward"]

    def __init__(
        self,
        device: str = "cpu",
        timeout: float = 30.0,
        batch_size: int = 2,
        seq_len: int = 16,
    ):
        self.device = torch.device(device)
        self.timeout = timeout
        self.batch_size = batch_size
        self.seq_len = seq_len
        self._model_cache: Optional[nn.Module] = None

    def run_all_stages(
        self, config_dict: Dict[str, Any]
    ) -> Tuple[List[StageResult], Optional[nn.Module]]:
        """
        Run all test stages sequentially.

        Returns:
            Tuple of (list of stage results, model if instantiated)
        """
        results = []
        model = None

        # Stage 1: Config validation
        result = self.run_config_validation(config_dict)
        results.append(result)
        if not result.passed:
            return results, None

        # Stage 2: Model instantiation
        result, model = self.run_model_instantiation(config_dict)
        results.append(result)
        if not result.passed or model is None:
            return results, None

        # Stage 3: Forward pass
        result = self.run_forward_pass(model, config_dict)
        results.append(result)
        if not result.passed:
            return results, model

        # Stage 4: Backward pass
        result = self.run_backward_pass(model, config_dict)
        results.append(result)

        return results, model

    def run_config_validation(self, config_dict: Dict[str, Any]) -> StageResult:
        """
        Stage 1: Test config validation.

        Tests:
        - ModelConfig dataclass __post_init__ validation
        - ConfigValidator validation (if available)
        """
        start = time.time()
        stage = "config_validation"

        try:
            # Add src to path if needed
            self._ensure_imports()

            # Test ModelConfig __post_init__
            from ava.config.training_config import ModelConfig

            model_config = config_dict.get("model", {})
            if model_config:
                # Create ModelConfig to trigger __post_init__ validation
                ModelConfig(**model_config)

            # Try ConfigValidator if available
            try:
                from ava.config.validator import ConfigValidator

                validator = ConfigValidator(config_dict)
                is_valid, errors = validator.validate()
                if not is_valid:
                    return StageResult(
                        passed=False,
                        stage=stage,
                        error_type="ConfigValidationError",
                        error_message="; ".join(errors) if errors else "Validation failed",
                        execution_time=time.time() - start,
                    )
            except ImportError:
                pass  # ConfigValidator not available, skip

            return StageResult(
                passed=True, stage=stage, execution_time=time.time() - start
            )

        except Exception as e:
            return StageResult(
                passed=False,
                stage=stage,
                error_type=type(e).__name__,
                error_message=str(e),
                function_location=self._extract_location(e),
                traceback_str=traceback.format_exc(),
                execution_time=time.time() - start,
            )

    def run_model_instantiation(
        self, config_dict: Dict[str, Any]
    ) -> Tuple[StageResult, Optional[nn.Module]]:
        """
        Stage 2: Test model instantiation.

        Attempts to create EnhancedMoEModel from config.
        """
        start = time.time()
        stage = "model_instantiation"

        try:
            self._ensure_imports()

            from ava.models.moe import EnhancedMoEConfig, EnhancedMoEModel

            model_config = config_dict.get("model", {})

            # Create config object
            moe_config = EnhancedMoEConfig(**model_config)

            # Create model
            model = EnhancedMoEModel(moe_config)
            model.to(self.device)
            model.eval()

            return (
                StageResult(
                    passed=True, stage=stage, execution_time=time.time() - start
                ),
                model,
            )

        except Exception as e:
            return (
                StageResult(
                    passed=False,
                    stage=stage,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    function_location=self._extract_location(e),
                    traceback_str=traceback.format_exc(),
                    execution_time=time.time() - start,
                ),
                None,
            )

    def run_forward_pass(
        self, model: nn.Module, config_dict: Dict[str, Any]
    ) -> StageResult:
        """
        Stage 3: Test forward pass.

        Runs a forward pass with dummy input data.
        """
        start = time.time()
        stage = "model_forward"

        try:
            model_config = config_dict.get("model", {})
            vocab_size = model_config.get("vocab_size", 1000)
            max_pos = model_config.get("max_position_embeddings", 512)

            # Use smaller sequence length if max_position_embeddings is small
            seq_len = min(self.seq_len, max_pos) if max_pos > 0 else self.seq_len

            # Create dummy input
            input_ids = torch.randint(
                0, max(1, vocab_size),
                (self.batch_size, seq_len),
                device=self.device
            )
            attention_mask = torch.ones_like(input_ids)

            # Run forward pass
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Verify outputs
            if outputs is None:
                return StageResult(
                    passed=False,
                    stage=stage,
                    error_type="OutputError",
                    error_message="Model returned None",
                    execution_time=time.time() - start,
                )

            return StageResult(
                passed=True, stage=stage, execution_time=time.time() - start
            )

        except Exception as e:
            return StageResult(
                passed=False,
                stage=stage,
                error_type=type(e).__name__,
                error_message=str(e),
                function_location=self._extract_location(e),
                traceback_str=traceback.format_exc(),
                execution_time=time.time() - start,
            )

    def run_backward_pass(
        self, model: nn.Module, config_dict: Dict[str, Any]
    ) -> StageResult:
        """
        Stage 4: Test backward pass.

        Runs forward pass, computes loss, and performs backward pass.
        """
        start = time.time()
        stage = "model_backward"

        try:
            model_config = config_dict.get("model", {})
            vocab_size = model_config.get("vocab_size", 1000)
            max_pos = model_config.get("max_position_embeddings", 512)

            # Use smaller sequence length if max_position_embeddings is small
            seq_len = min(self.seq_len, max_pos) if max_pos > 0 else self.seq_len

            # Create dummy input
            input_ids = torch.randint(
                0, max(1, vocab_size),
                (self.batch_size, seq_len),
                device=self.device
            )
            labels = torch.randint(
                0, max(1, vocab_size),
                (self.batch_size, seq_len),
                device=self.device
            )

            # Enable training mode and gradients
            model.train()
            model.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids)

            # Extract logits from outputs
            logits = self._extract_logits(outputs)

            if logits is None:
                return StageResult(
                    passed=False,
                    stage=stage,
                    error_type="OutputError",
                    error_message="Could not extract logits from model output",
                    execution_time=time.time() - start,
                )

            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

            # Backward pass
            loss.backward()

            # Verify gradients exist
            has_grads = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters()
                if p.requires_grad
            )

            if not has_grads:
                return StageResult(
                    passed=False,
                    stage=stage,
                    error_type="GradientError",
                    error_message="No gradients computed or all gradients are zero",
                    execution_time=time.time() - start,
                )

            return StageResult(
                passed=True, stage=stage, execution_time=time.time() - start
            )

        except Exception as e:
            return StageResult(
                passed=False,
                stage=stage,
                error_type=type(e).__name__,
                error_message=str(e),
                function_location=self._extract_location(e),
                traceback_str=traceback.format_exc(),
                execution_time=time.time() - start,
            )
        finally:
            # Clean up
            if model is not None:
                model.zero_grad()

    def _extract_logits(self, outputs: Any) -> Optional[torch.Tensor]:
        """Extract logits from model output (handles different output formats)."""
        if outputs is None:
            return None

        # Dict output
        if isinstance(outputs, dict):
            for key in ["logits", "lm_logits", "prediction_logits"]:
                if key in outputs:
                    return outputs[key]

        # Named tuple or object with attributes
        if hasattr(outputs, "logits"):
            return outputs.logits
        if hasattr(outputs, "lm_logits"):
            return outputs.lm_logits

        # Tuple output (assume first element is logits)
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]

        # Direct tensor output
        if isinstance(outputs, torch.Tensor):
            return outputs

        return None

    def _extract_location(self, exception: Exception) -> str:
        """Extract function/file location from exception traceback."""
        tb = traceback.extract_tb(exception.__traceback__)
        if tb:
            # Find the last frame that's not in this file
            for frame in reversed(tb):
                if "test_stages.py" not in frame.filename:
                    return f"{frame.filename}:{frame.name}:{frame.lineno}"
            # Fallback to last frame
            last_frame = tb[-1]
            return f"{last_frame.filename}:{last_frame.name}:{last_frame.lineno}"
        return "unknown"

    def _ensure_imports(self):
        """Ensure the ava package is importable."""
        import os

        # Add src directory to path if not already there
        src_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "src")
        )
        if src_path not in sys.path:
            sys.path.insert(0, src_path)


def run_single_test(
    config_dict: Dict[str, Any],
    device: str = "cpu",
    stages: Optional[List[str]] = None,
) -> List[StageResult]:
    """
    Convenience function to run a single fuzz test.

    Args:
        config_dict: Configuration dictionary to test
        device: Device to run on ("cpu" or "cuda")
        stages: List of stages to run (default: all)

    Returns:
        List of StageResult objects
    """
    runner = TestStageRunner(device=device)

    if stages is None:
        results, _ = runner.run_all_stages(config_dict)
        return results

    results = []
    model = None

    for stage in stages:
        if stage == "config_validation":
            result = runner.run_config_validation(config_dict)
            results.append(result)
            if not result.passed:
                break

        elif stage == "model_instantiation":
            result, model = runner.run_model_instantiation(config_dict)
            results.append(result)
            if not result.passed:
                break

        elif stage == "model_forward":
            if model is None:
                result, model = runner.run_model_instantiation(config_dict)
                if not result.passed:
                    results.append(result)
                    break
            result = runner.run_forward_pass(model, config_dict)
            results.append(result)
            if not result.passed:
                break

        elif stage == "model_backward":
            if model is None:
                result, model = runner.run_model_instantiation(config_dict)
                if not result.passed:
                    results.append(result)
                    break
            result = runner.run_backward_pass(model, config_dict)
            results.append(result)

    return results
