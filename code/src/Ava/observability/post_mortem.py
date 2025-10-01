"""
Post-Mortem Analysis (Phase 7.2)

Automatic failure analysis and recommendations for training issues.
"""

import time
import traceback
import json
import pickle
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import logging


class FailureType(Enum):
    """Types of training failures."""
    LOSS_EXPLOSION = "loss_explosion"
    LOSS_STAGNATION = "loss_stagnation"
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    OUT_OF_MEMORY = "out_of_memory"
    CONVERGENCE_FAILURE = "convergence_failure"
    NUMERICAL_INSTABILITY = "numerical_instability"
    DATA_PIPELINE_ERROR = "data_pipeline_error"
    MODEL_ERROR = "model_error"
    SYSTEM_ERROR = "system_error"
    UNKNOWN = "unknown"


class Severity(Enum):
    """Severity levels for failures."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FailureEvidence:
    """Evidence collected about a failure."""
    evidence_type: str
    description: str
    data: Any
    timestamp: float
    confidence: float = 1.0  # 0.0 to 1.0
    relevance: float = 1.0   # 0.0 to 1.0


@dataclass
class FailureReport:
    """Comprehensive failure analysis report."""
    timestamp: float
    failure_type: FailureType
    severity: Severity
    summary: str
    description: str

    # Evidence and analysis
    evidence: List[FailureEvidence] = field(default_factory=list)
    root_causes: List[str] = field(default_factory=list)
    contributing_factors: List[str] = field(default_factory=list)

    # Recommendations
    immediate_actions: List[str] = field(default_factory=list)
    prevention_measures: List[str] = field(default_factory=list)
    configuration_changes: Dict[str, Any] = field(default_factory=dict)

    # Context
    training_state: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    model_state: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    analysis_version: str = "1.0"
    confidence_score: float = 0.0


class FailureDetector:
    """Detect various types of training failures."""

    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.gradient_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=1000)

    def update_metrics(self, **kwargs):
        """Update metrics for failure detection."""
        timestamp = time.time()
        metrics = {'timestamp': timestamp, **kwargs}
        self.metrics_history.append(metrics)

        # Update specific histories
        if 'loss' in kwargs:
            self.loss_history.append((timestamp, kwargs['loss']))

        if 'gradient_norm' in kwargs:
            self.gradient_history.append((timestamp, kwargs['gradient_norm']))

    def detect_failures(self) -> List[Tuple[FailureType, float, Dict[str, Any]]]:
        """
        Detect potential failures from metrics history.

        Returns:
            List of (failure_type, confidence, evidence_data) tuples
        """
        failures = []

        # Loss-based failure detection
        failures.extend(self._detect_loss_failures())

        # Gradient-based failure detection
        failures.extend(self._detect_gradient_failures())

        # System-based failure detection
        failures.extend(self._detect_system_failures())

        return failures

    def _detect_loss_failures(self) -> List[Tuple[FailureType, float, Dict[str, Any]]]:
        """Detect loss-related failures."""
        failures = []

        if len(self.loss_history) < 10:
            return failures

        recent_losses = [loss for _, loss in list(self.loss_history)[-10:]]

        # Loss explosion detection
        if recent_losses:
            latest_loss = recent_losses[-1]
            if latest_loss > 100:  # Very high loss
                confidence = min(1.0, latest_loss / 1000)
                failures.append((
                    FailureType.LOSS_EXPLOSION,
                    confidence,
                    {
                        'latest_loss': latest_loss,
                        'recent_losses': recent_losses,
                        'loss_increase_factor': latest_loss / (sum(recent_losses[:-1]) / len(recent_losses[:-1]) + 1e-8)
                    }
                ))

            # Loss stagnation detection
            if len(recent_losses) >= 10:
                loss_std = np.std(recent_losses)
                loss_mean = np.mean(recent_losses)

                if loss_std < 1e-6 and loss_mean > 0.1:  # Very stable but high loss
                    failures.append((
                        FailureType.LOSS_STAGNATION,
                        0.8,
                        {
                            'loss_std': loss_std,
                            'loss_mean': loss_mean,
                            'recent_losses': recent_losses
                        }
                    ))

            # Check for loss oscillation (numerical instability)
            if len(recent_losses) >= 5:
                changes = [recent_losses[i] - recent_losses[i-1] for i in range(1, len(recent_losses))]
                sign_changes = sum(1 for i in range(1, len(changes)) if changes[i] * changes[i-1] < 0)

                if sign_changes >= len(changes) * 0.8:  # 80% sign changes
                    failures.append((
                        FailureType.NUMERICAL_INSTABILITY,
                        0.7,
                        {
                            'sign_changes': sign_changes,
                            'total_changes': len(changes),
                            'oscillation_ratio': sign_changes / len(changes),
                            'recent_losses': recent_losses
                        }
                    ))

        return failures

    def _detect_gradient_failures(self) -> List[Tuple[FailureType, float, Dict[str, Any]]]:
        """Detect gradient-related failures."""
        failures = []

        if len(self.gradient_history) < 5:
            return failures

        recent_grads = [grad for _, grad in list(self.gradient_history)[-10:]]

        if recent_grads:
            latest_grad = recent_grads[-1]

            # Gradient explosion
            if latest_grad > 100:
                confidence = min(1.0, latest_grad / 1000)
                failures.append((
                    FailureType.GRADIENT_EXPLOSION,
                    confidence,
                    {
                        'latest_gradient_norm': latest_grad,
                        'recent_gradients': recent_grads,
                        'explosion_factor': latest_grad / (np.mean(recent_grads[:-1]) + 1e-8)
                    }
                ))

            # Gradient vanishing (increased threshold from 1e-8 to 1e-6 for large models with gradient accumulation)
            if latest_grad < 1e-6:
                confidence = 0.9
                failures.append((
                    FailureType.GRADIENT_VANISHING,
                    confidence,
                    {
                        'latest_gradient_norm': latest_grad,
                        'recent_gradients': recent_grads,
                        'vanishing_threshold': 1e-6
                    }
                ))

        return failures

    def _detect_system_failures(self) -> List[Tuple[FailureType, float, Dict[str, Any]]]:
        """Detect system-related failures."""
        failures = []

        # Check recent metrics for memory issues
        recent_metrics = list(self.metrics_history)[-5:] if self.metrics_history else []

        for metrics in recent_metrics:
            # OOM detection
            if 'gpu_memory_used' in metrics and 'gpu_memory_total' in metrics:
                memory_ratio = metrics['gpu_memory_used'] / metrics['gpu_memory_total']
                if memory_ratio > 0.98:
                    failures.append((
                        FailureType.OUT_OF_MEMORY,
                        0.9,
                        {
                            'memory_ratio': memory_ratio,
                            'gpu_memory_used': metrics['gpu_memory_used'],
                            'gpu_memory_total': metrics['gpu_memory_total']
                        }
                    ))

        return failures


class FailureAnalyzer:
    """Analyze failures and generate detailed reports."""

    def __init__(self):
        self.analysis_templates = self._load_analysis_templates()

    def analyze_failure(
        self,
        failure_type: FailureType,
        evidence_data: Dict[str, Any],
        training_context: Dict[str, Any] = None
    ) -> FailureReport:
        """
        Analyze a specific failure and generate a comprehensive report.

        Args:
            failure_type: Type of failure detected
            evidence_data: Data supporting the failure detection
            training_context: Current training context (model, config, etc.)

        Returns:
            Detailed failure analysis report
        """
        training_context = training_context or {}

        # Create base report
        report = FailureReport(
            timestamp=time.time(),
            failure_type=failure_type,
            severity=self._determine_severity(failure_type, evidence_data),
            summary=self._generate_summary(failure_type, evidence_data),
            description=self._generate_description(failure_type, evidence_data)
        )

        # Collect evidence
        report.evidence = self._collect_evidence(failure_type, evidence_data, training_context)

        # Analyze root causes
        report.root_causes = self._analyze_root_causes(failure_type, evidence_data, training_context)
        report.contributing_factors = self._identify_contributing_factors(failure_type, evidence_data, training_context)

        # Generate recommendations
        report.immediate_actions = self._generate_immediate_actions(failure_type, evidence_data)
        report.prevention_measures = self._generate_prevention_measures(failure_type, evidence_data)
        report.configuration_changes = self._suggest_config_changes(failure_type, evidence_data, training_context)

        # Capture context
        report.training_state = self._capture_training_state(training_context)
        report.system_state = self._capture_system_state()
        report.model_state = self._capture_model_state(training_context)

        # Calculate confidence score
        report.confidence_score = self._calculate_confidence(report)

        return report

    def _determine_severity(self, failure_type: FailureType, evidence_data: Dict[str, Any]) -> Severity:
        """Determine failure severity."""
        if failure_type in [FailureType.OUT_OF_MEMORY, FailureType.SYSTEM_ERROR]:
            return Severity.CRITICAL
        elif failure_type in [FailureType.LOSS_EXPLOSION, FailureType.GRADIENT_EXPLOSION]:
            return Severity.HIGH
        elif failure_type in [FailureType.LOSS_STAGNATION, FailureType.CONVERGENCE_FAILURE]:
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _generate_summary(self, failure_type: FailureType, evidence_data: Dict[str, Any]) -> str:
        """Generate failure summary."""
        templates = {
            FailureType.LOSS_EXPLOSION: "Training loss exploded to {latest_loss:.4f}",
            FailureType.LOSS_STAGNATION: "Training loss stagnated at {loss_mean:.4f}",
            FailureType.GRADIENT_EXPLOSION: "Gradients exploded to {latest_gradient_norm:.4f}",
            FailureType.GRADIENT_VANISHING: "Gradients vanished to {latest_gradient_norm:.2e}",
            FailureType.OUT_OF_MEMORY: "GPU memory exhausted ({memory_ratio:.1%} used)",
            FailureType.NUMERICAL_INSTABILITY: "Numerical instability detected (loss oscillating)",
        }

        template = templates.get(failure_type, f"Failure detected: {failure_type.value}")

        try:
            return template.format(**evidence_data)
        except (KeyError, ValueError):
            return f"Failure detected: {failure_type.value}"

    def _generate_description(self, failure_type: FailureType, evidence_data: Dict[str, Any]) -> str:
        """Generate detailed failure description."""
        descriptions = {
            FailureType.LOSS_EXPLOSION: (
                "The training loss has increased dramatically, indicating potential "
                "numerical instability or learning rate issues. This typically occurs "
                "when gradients are too large or the learning rate is too high."
            ),
            FailureType.LOSS_STAGNATION: (
                "The training loss has stopped decreasing and appears to be stuck "
                "at a high value. This suggests the model may have reached a local "
                "minimum or the learning rate is too low."
            ),
            FailureType.GRADIENT_EXPLOSION: (
                "The gradient norm has become extremely large, which can cause "
                "numerical instability and prevent proper weight updates. This "
                "often requires gradient clipping or learning rate reduction."
            ),
            FailureType.GRADIENT_VANISHING: (
                "The gradient norm has become extremely small, which means the "
                "model weights are barely being updated. This can prevent learning "
                "and is common in deep networks without proper initialization."
            ),
            FailureType.OUT_OF_MEMORY: (
                "GPU memory has been exhausted, preventing further training. "
                "This requires reducing memory usage through smaller batch sizes, "
                "gradient accumulation, or model optimization techniques."
            ),
            FailureType.NUMERICAL_INSTABILITY: (
                "The training process shows signs of numerical instability with "
                "rapid oscillations in loss values. This can be caused by "
                "inappropriate learning rates or precision issues."
            )
        }

        return descriptions.get(failure_type, f"A {failure_type.value} failure has been detected.")

    def _collect_evidence(
        self,
        failure_type: FailureType,
        evidence_data: Dict[str, Any],
        training_context: Dict[str, Any]
    ) -> List[FailureEvidence]:
        """Collect evidence supporting the failure analysis."""
        evidence = []

        # Add primary evidence from detection
        evidence.append(FailureEvidence(
            evidence_type="detection_data",
            description=f"Primary evidence for {failure_type.value}",
            data=evidence_data,
            timestamp=time.time(),
            confidence=0.9,
            relevance=1.0
        ))

        # Add training configuration evidence
        if 'config' in training_context:
            config = training_context['config']
            evidence.append(FailureEvidence(
                evidence_type="training_config",
                description="Training configuration at time of failure",
                data={
                    'learning_rate': getattr(config, 'learning_rate', None),
                    'batch_size': getattr(config, 'batch_size', None),
                    'optimizer': getattr(config, 'optimizer_type', None),
                },
                timestamp=time.time(),
                confidence=0.8,
                relevance=0.9
            ))

        # Add model architecture evidence
        if 'model' in training_context:
            model = training_context['model']
            try:
                param_count = sum(p.numel() for p in model.parameters())
                evidence.append(FailureEvidence(
                    evidence_type="model_architecture",
                    description="Model architecture information",
                    data={
                        'total_parameters': param_count,
                        'model_type': type(model).__name__,
                    },
                    timestamp=time.time(),
                    confidence=0.8,
                    relevance=0.7
                ))
            except Exception:
                pass

        return evidence

    def _analyze_root_causes(
        self,
        failure_type: FailureType,
        evidence_data: Dict[str, Any],
        training_context: Dict[str, Any]
    ) -> List[str]:
        """Analyze potential root causes."""
        causes = []

        if failure_type == FailureType.LOSS_EXPLOSION:
            causes.extend([
                "Learning rate too high",
                "Gradient clipping not applied or insufficient",
                "Numerical precision issues (consider mixed precision)",
                "Batch size too small causing noisy gradients",
                "Poor model initialization"
            ])

        elif failure_type == FailureType.LOSS_STAGNATION:
            causes.extend([
                "Learning rate too low",
                "Model has reached capacity limit",
                "Data quality issues or insufficient diversity",
                "Local minimum reached without proper scheduling",
                "Optimizer state corruption"
            ])

        elif failure_type == FailureType.GRADIENT_EXPLOSION:
            causes.extend([
                "Missing or insufficient gradient clipping",
                "Learning rate too high",
                "RNN/LSTM unrolling too deep",
                "Numerical instability in model architecture",
                "Improper weight initialization"
            ])

        elif failure_type == FailureType.GRADIENT_VANISHING:
            causes.extend([
                "Poor weight initialization (especially for deep networks)",
                "Activation functions causing saturation",
                "Learning rate too low",
                "Model architecture not suitable for gradient flow",
                "Batch normalization issues"
            ])

        elif failure_type == FailureType.OUT_OF_MEMORY:
            causes.extend([
                "Batch size too large for available GPU memory",
                "Model too large for GPU memory",
                "Memory leaks in data loading or model forward pass",
                "Gradient accumulation not properly implemented",
                "Inefficient memory usage in model architecture"
            ])

        return causes

    def _identify_contributing_factors(
        self,
        failure_type: FailureType,
        evidence_data: Dict[str, Any],
        training_context: Dict[str, Any]
    ) -> List[str]:
        """Identify contributing factors."""
        factors = []

        # Check for common contributing factors
        if 'config' in training_context:
            config = training_context['config']

            # Learning rate factors
            lr = getattr(config, 'learning_rate', None)
            if lr and lr > 1e-2:
                factors.append(f"High learning rate ({lr})")
            elif lr and lr < 1e-6:
                factors.append(f"Very low learning rate ({lr})")

            # Batch size factors
            batch_size = getattr(config, 'batch_size', None)
            if batch_size and batch_size < 4:
                factors.append(f"Very small batch size ({batch_size})")
            elif batch_size and batch_size > 128:
                factors.append(f"Large batch size ({batch_size})")

        # System factors
        try:
            if torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if memory_usage > 0.9:
                    factors.append("High GPU memory usage")
        except Exception:
            pass

        return factors

    def _generate_immediate_actions(self, failure_type: FailureType, evidence_data: Dict[str, Any]) -> List[str]:
        """Generate immediate actions to take."""
        actions = []

        if failure_type == FailureType.LOSS_EXPLOSION:
            actions.extend([
                "Immediately reduce learning rate by 10x",
                "Apply gradient clipping (max_norm=1.0)",
                "Reduce batch size if memory allows",
                "Check for NaN values in model parameters",
                "Consider reverting to previous checkpoint"
            ])

        elif failure_type == FailureType.GRADIENT_EXPLOSION:
            actions.extend([
                "Apply gradient clipping with max_norm=1.0",
                "Reduce learning rate by 5x",
                "Check model for numerical issues",
                "Monitor gradient norms continuously"
            ])

        elif failure_type == FailureType.OUT_OF_MEMORY:
            actions.extend([
                "Reduce batch size immediately",
                "Clear GPU cache: torch.cuda.empty_cache()",
                "Enable gradient checkpointing if available",
                "Consider gradient accumulation",
                "Check for memory leaks"
            ])

        elif failure_type == FailureType.LOSS_STAGNATION:
            actions.extend([
                "Increase learning rate gradually",
                "Apply learning rate scheduling",
                "Check data pipeline for issues",
                "Verify model is in training mode",
                "Consider changing optimizer"
            ])

        return actions

    def _generate_prevention_measures(self, failure_type: FailureType, evidence_data: Dict[str, Any]) -> List[str]:
        """Generate prevention measures for future training."""
        measures = []

        if failure_type in [FailureType.LOSS_EXPLOSION, FailureType.GRADIENT_EXPLOSION]:
            measures.extend([
                "Implement gradient clipping from start",
                "Use learning rate warmup",
                "Monitor gradient norms continuously",
                "Implement automatic learning rate reduction on anomalies",
                "Use more conservative learning rates initially"
            ])

        elif failure_type == FailureType.OUT_OF_MEMORY:
            measures.extend([
                "Profile memory usage before training",
                "Implement dynamic batch sizing",
                "Use gradient accumulation for effective large batches",
                "Monitor GPU memory usage continuously",
                "Consider model parallelism for large models"
            ])

        elif failure_type == FailureType.LOSS_STAGNATION:
            measures.extend([
                "Implement learning rate scheduling",
                "Use early stopping with patience",
                "Implement periodic learning rate restarts",
                "Monitor validation metrics alongside training loss",
                "Use more sophisticated optimizers (AdamW, etc.)"
            ])

        return measures

    def _suggest_config_changes(
        self,
        failure_type: FailureType,
        evidence_data: Dict[str, Any],
        training_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Suggest specific configuration changes."""
        changes = {}

        if failure_type == FailureType.LOSS_EXPLOSION:
            if 'latest_loss' in evidence_data:
                current_lr = training_context.get('config', {}).get('learning_rate', 1e-3)
                changes['learning_rate'] = current_lr / 10
                changes['gradient_clip_norm'] = 1.0
                changes['batch_size'] = max(1, training_context.get('config', {}).get('batch_size', 32) // 2)

        elif failure_type == FailureType.OUT_OF_MEMORY:
            if 'memory_ratio' in evidence_data:
                current_batch = training_context.get('config', {}).get('batch_size', 32)
                changes['batch_size'] = max(1, current_batch // 2)
                changes['gradient_accumulation_steps'] = 2

        elif failure_type == FailureType.LOSS_STAGNATION:
            current_lr = training_context.get('config', {}).get('learning_rate', 1e-3)
            changes['learning_rate'] = current_lr * 2
            changes['use_lr_scheduler'] = True
            changes['scheduler_type'] = 'ReduceLROnPlateau'

        return changes

    def _capture_training_state(self, training_context: Dict[str, Any]) -> Dict[str, Any]:
        """Capture current training state."""
        state = {}

        if 'step' in training_context:
            state['current_step'] = training_context['step']
        if 'epoch' in training_context:
            state['current_epoch'] = training_context['epoch']
        if 'loss' in training_context:
            state['current_loss'] = training_context['loss']

        return state

    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture system state."""
        state = {}

        try:
            if torch.cuda.is_available():
                state['gpu_available'] = True
                state['gpu_count'] = torch.cuda.device_count()
                state['current_device'] = torch.cuda.current_device()
                state['gpu_memory_allocated'] = torch.cuda.memory_allocated()
                state['gpu_memory_reserved'] = torch.cuda.memory_reserved()
                state['gpu_memory_total'] = torch.cuda.get_device_properties(0).total_memory
            else:
                state['gpu_available'] = False
        except Exception as e:
            state['gpu_error'] = str(e)

        return state

    def _capture_model_state(self, training_context: Dict[str, Any]) -> Dict[str, Any]:
        """Capture model state."""
        state = {}

        if 'model' in training_context:
            model = training_context['model']
            try:
                state['model_type'] = type(model).__name__
                state['total_parameters'] = sum(p.numel() for p in model.parameters())
                state['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
                state['model_device'] = str(next(model.parameters()).device)
                state['model_dtype'] = str(next(model.parameters()).dtype)
            except Exception as e:
                state['model_error'] = str(e)

        return state

    def _calculate_confidence(self, report: FailureReport) -> float:
        """Calculate confidence score for the analysis."""
        # Base confidence from evidence
        evidence_confidence = sum(e.confidence * e.relevance for e in report.evidence)
        evidence_weight = sum(e.relevance for e in report.evidence) or 1.0

        base_confidence = evidence_confidence / evidence_weight

        # Adjust based on amount of evidence
        evidence_bonus = min(0.2, len(report.evidence) * 0.05)

        # Adjust based on failure severity (higher severity = higher confidence in detection)
        severity_bonus = {
            Severity.CRITICAL: 0.2,
            Severity.HIGH: 0.15,
            Severity.MEDIUM: 0.1,
            Severity.LOW: 0.05
        }.get(report.severity, 0)

        final_confidence = min(1.0, base_confidence + evidence_bonus + severity_bonus)
        return final_confidence

    def _load_analysis_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load analysis templates for different failure types."""
        # This would load from external files in a real implementation
        return {}


class PostMortemAnalyzer:
    """
    Main post-mortem analysis system for training failures.

    Features:
    - Automatic failure detection
    - Comprehensive failure analysis
    - Actionable recommendations
    - Failure history tracking
    - Pattern recognition across failures
    """

    def __init__(self, enable_auto_detection: bool = True):
        self.enable_auto_detection = enable_auto_detection
        self.detector = FailureDetector()
        self.analyzer = FailureAnalyzer()

        # Storage
        self.failure_reports = []
        self.analysis_history = []

        # Pattern recognition
        self.failure_patterns = defaultdict(list)

    def update_training_metrics(self, **kwargs):
        """Update training metrics for failure detection."""
        self.detector.update_metrics(**kwargs)

        if self.enable_auto_detection:
            # Check for failures
            detected_failures = self.detector.detect_failures()

            for failure_type, confidence, evidence_data in detected_failures:
                if confidence > 0.7:  # High confidence threshold
                    self._handle_detected_failure(failure_type, evidence_data, confidence, kwargs)

    def analyze_failure(
        self,
        failure_type: FailureType = None,
        evidence_data: Dict[str, Any] = None,
        training_context: Dict[str, Any] = None
    ) -> FailureReport:
        """
        Perform comprehensive failure analysis.

        Args:
            failure_type: Type of failure (auto-detected if None)
            evidence_data: Evidence data (auto-collected if None)
            training_context: Training context for analysis

        Returns:
            Detailed failure analysis report
        """
        if failure_type is None:
            # Auto-detect failure
            detected_failures = self.detector.detect_failures()
            if detected_failures:
                failure_type, confidence, evidence_data = max(detected_failures, key=lambda x: x[1])
            else:
                failure_type = FailureType.UNKNOWN
                evidence_data = {}

        if evidence_data is None:
            evidence_data = {}

        # Perform analysis
        report = self.analyzer.analyze_failure(failure_type, evidence_data, training_context)

        # Store report
        self.failure_reports.append(report)
        self.failure_patterns[failure_type].append(report)

        # Print summary
        self._print_failure_summary(report)

        return report

    def _handle_detected_failure(
        self,
        failure_type: FailureType,
        evidence_data: Dict[str, Any],
        confidence: float,
        training_context: Dict[str, Any]
    ):
        """Handle automatically detected failure."""
        print(f"\n🚨 FAILURE DETECTED: {failure_type.value} (confidence: {confidence:.2f})")

        # Perform analysis
        report = self.analyze_failure(failure_type, evidence_data, training_context)

        # Auto-apply critical fixes if confidence is very high
        if confidence > 0.9 and report.severity in [Severity.CRITICAL, Severity.HIGH]:
            print("🔧 Auto-applying critical fixes...")
            self._auto_apply_fixes(report)

    def _auto_apply_fixes(self, report: FailureReport):
        """Auto-apply critical fixes (placeholder)."""
        # This would integrate with the training system to apply fixes
        for action in report.immediate_actions[:2]:  # Apply first 2 actions
            print(f"  - {action}")

    def _print_failure_summary(self, report: FailureReport):
        """Print failure analysis summary."""
        print(f"\n📋 FAILURE ANALYSIS REPORT")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.timestamp))}")
        print(f"Type: {report.failure_type.value}")
        print(f"Severity: {report.severity.value.upper()}")
        print(f"Confidence: {report.confidence_score:.2f}")
        print(f"\nSummary: {report.summary}")
        print(f"Description: {report.description}")

        if report.root_causes:
            print(f"\n🔍 Likely Root Causes:")
            for cause in report.root_causes[:3]:  # Top 3
                print(f"  • {cause}")

        if report.immediate_actions:
            print(f"\n⚡ Immediate Actions:")
            for action in report.immediate_actions[:3]:  # Top 3
                print(f"  • {action}")

        if report.configuration_changes:
            print(f"\n⚙️ Suggested Config Changes:")
            for key, value in report.configuration_changes.items():
                print(f"  • {key}: {value}")

    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get statistics about failures."""
        if not self.failure_reports:
            return {"message": "No failures recorded"}

        failure_counts = defaultdict(int)
        severity_counts = defaultdict(int)

        for report in self.failure_reports:
            failure_counts[report.failure_type.value] += 1
            severity_counts[report.severity.value] += 1

        return {
            "total_failures": len(self.failure_reports),
            "failure_types": dict(failure_counts),
            "severity_distribution": dict(severity_counts),
            "most_common_failure": max(failure_counts.items(), key=lambda x: x[1])[0] if failure_counts else None,
            "recent_failures": [
                {
                    "type": r.failure_type.value,
                    "severity": r.severity.value,
                    "timestamp": r.timestamp,
                    "summary": r.summary
                }
                for r in self.failure_reports[-5:]
            ]
        }

    def export_analysis(self, filepath: str, format: str = 'json'):
        """Export failure analysis to file."""
        if not self.failure_reports:
            return

        export_data = {
            "analysis_metadata": {
                "export_timestamp": time.time(),
                "total_reports": len(self.failure_reports),
                "analyzer_version": "1.0"
            },
            "failure_reports": [
                {
                    "timestamp": report.timestamp,
                    "failure_type": report.failure_type.value,
                    "severity": report.severity.value,
                    "summary": report.summary,
                    "description": report.description,
                    "root_causes": report.root_causes,
                    "immediate_actions": report.immediate_actions,
                    "prevention_measures": report.prevention_measures,
                    "configuration_changes": report.configuration_changes,
                    "confidence_score": report.confidence_score,
                    "evidence_count": len(report.evidence)
                }
                for report in self.failure_reports
            ]
        }

        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Convenience functions
def create_post_mortem_analyzer(auto_detect: bool = True) -> PostMortemAnalyzer:
    """Create configured post-mortem analyzer."""
    return PostMortemAnalyzer(enable_auto_detection=auto_detect)


def analyze_training_failure(
    loss_history: List[float] = None,
    gradient_history: List[float] = None,
    **context
) -> FailureReport:
    """Quick failure analysis from metrics."""
    analyzer = PostMortemAnalyzer(enable_auto_detection=False)

    # Update with provided history
    if loss_history:
        for i, loss in enumerate(loss_history):
            analyzer.detector.loss_history.append((time.time() - len(loss_history) + i, loss))

    if gradient_history:
        for i, grad in enumerate(gradient_history):
            analyzer.detector.gradient_history.append((time.time() - len(gradient_history) + i, grad))

    return analyzer.analyze_failure(training_context=context)