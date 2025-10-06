"""
Feature Compatibility Validation Matrix (Phase 6.1)

Prevents incompatible feature combinations and validates startup configurations
to avoid training failures caused by conflicting features.

Implements the compatibility rules from Phase 6:
- Quantization XOR Mixed Precision (not both)
- Gradient Surgery XOR DeepSpeed (not both)
- Multi-task requires multiple loss components
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class CompatibilityLevel(Enum):
    """Compatibility issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CompatibilityIssue:
    """Represents a feature compatibility issue."""
    level: CompatibilityLevel
    message: str
    conflicting_features: List[str]
    fix_suggestions: List[str]
    category: str = "feature_conflict"


class FeatureCompatibilityValidator:
    """
    Validates feature combinations to prevent incompatible configurations.

    This validator implements the compatibility matrix from Phase 6 to ensure
    that conflicting features are not enabled simultaneously.
    """

    def __init__(self):
        """Initialize the compatibility validator."""
        self.compatibility_rules = self._build_compatibility_matrix()
        self.feature_dependencies = self._build_dependency_matrix()

    def _build_compatibility_matrix(self) -> Dict[str, Dict[str, Any]]:
        """
        Build the feature compatibility matrix.

        Returns:
            Dictionary mapping feature combinations to conflict types
        """
        return {
            # Quantization vs Mixed Precision conflicts
            "quantization_mixed_precision": {
                "features": ["quantization_aware", "mixed_precision"],
                "conflict_type": "exclusive",
                "reason": "Quantization and mixed precision cannot be used together due to numerical precision conflicts",
                "fix": "Choose either quantization OR mixed precision, not both"
            },

            # Gradient Surgery vs DeepSpeed conflicts
            "gradient_surgery_deepspeed": {
                "features": ["gradient_surgery", "deepspeed"],
                "conflict_type": "exclusive",
                "reason": "Gradient surgery interferes with DeepSpeed's gradient handling",
                "fix": "Use DeepSpeed's built-in gradient management instead of gradient surgery"
            },

            # Multi-task training requirements
            "multi_task_loss_components": {
                "features": ["multi_task", "single_loss_component"],
                "conflict_type": "dependency",
                "reason": "Multi-task training requires multiple loss components to be effective",
                "fix": "Enable multiple loss components (focal, contrastive, diversity) for multi-task training"
            },

            # Progressive training feature interactions
            "progressive_training_static_batch": {
                "features": ["progressive_training", "fixed_batch_size"],
                "conflict_type": "suboptimal",
                "reason": "Progressive training works best with dynamic batch sizing",
                "fix": "Enable dynamic batch sizing for optimal progressive training performance"
            },

            # RAG system memory requirements
            "rag_limited_memory": {
                "features": ["rag_system", "minimal_memory_mode"],
                "conflict_type": "suboptimal",
                "reason": "RAG system requires significant memory for knowledge base indexing",
                "fix": "Increase memory allocation or disable RAG for memory-constrained environments"
            },

            # Episodic memory conflicts
            "episodic_memory_streaming": {
                "features": ["episodic_memory", "pure_streaming"],
                "conflict_type": "suboptimal",
                "reason": "Episodic memory works better with some data persistence",
                "fix": "Enable data caching or reduce episodic memory capacity for streaming"
            },

            # Observability overhead in ultra-fast mode
            "ultra_fast_detailed_observability": {
                "features": ["ultra_fast_mode", "detailed_observability"],
                "conflict_type": "performance",
                "reason": "Detailed observability adds overhead that conflicts with ultra-fast mode",
                "fix": "Use lightweight observability or disable ultra-fast mode"
            }
        }

    def _build_dependency_matrix(self) -> Dict[str, List[str]]:
        """
        Build feature dependency requirements.

        Returns:
            Dictionary mapping features to their required dependencies
        """
        return {
            "multi_task": ["use_focal_loss", "use_contrastive_loss", "use_diversity_loss"],
            "rag_system": ["use_cross_attention"],
            "gradient_surgery": ["adaptive_gradient_surgery"],
            "progressive_training": ["enable_curriculum", "enable_dynamic_batch"],
            "quantization_aware": ["mixed_precision_disabled"],
            "episodic_memory": ["memory_capacity_set"],
            "distributed_training": ["process_group_backend"],
            "ultra_fast_mode": ["minimal_logging", "reduced_validation"]
        }

    def validate_configuration(self, config) -> Tuple[bool, List[CompatibilityIssue]]:
        """
        Validate the entire training configuration for compatibility issues.

        Args:
            config: Enhanced training configuration object

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Extract feature states from config
        feature_states = self._extract_feature_states(config)

        # Check compatibility rules
        compatibility_issues = self._check_compatibility_rules(feature_states)
        issues.extend(compatibility_issues)

        # Check feature dependencies
        dependency_issues = self._check_feature_dependencies(feature_states)
        issues.extend(dependency_issues)

        # Check resource requirements
        resource_issues = self._check_resource_requirements(feature_states, config)
        issues.extend(resource_issues)

        # Determine if configuration is valid
        critical_errors = [issue for issue in issues if issue.level == CompatibilityLevel.CRITICAL]
        errors = [issue for issue in issues if issue.level == CompatibilityLevel.ERROR]

        is_valid = len(critical_errors) == 0 and len(errors) == 0

        return is_valid, issues

    def _extract_feature_states(self, config) -> Dict[str, bool]:
        """Extract feature states from configuration object."""
        feature_states = {}

        try:
            # Architecture features
            if hasattr(config, 'architecture'):
                feature_states.update({
                    'use_moh': getattr(config.architecture, 'use_moh', False),
                    'use_moa': getattr(config.architecture, 'use_moa', False),
                    'use_cross_attention': getattr(config.architecture, 'use_cross_attention', False),
                    'use_alibi': getattr(config.architecture, 'use_alibi', False),
                })

            # RAG features
            if hasattr(config, 'rag'):
                feature_states['rag_system'] = getattr(config.rag, 'use_rag', False)

            # Quantization features
            if hasattr(config, 'quantization'):
                feature_states.update({
                    'quantization_aware': getattr(config.quantization, 'quantization_aware', False),
                    'use_nvfp4': getattr(config.quantization, 'use_nvfp4', False),
                })

            # Performance features
            if hasattr(config, 'performance'):
                feature_states.update({
                    'ultra_fast_mode': getattr(config.performance, 'ultra_fast_mode', False),
                    'mixed_precision': getattr(config.performance, 'mixed_precision', False),
                })

            # Training features
            if hasattr(config, 'training'):
                feature_states.update({
                    'progressive_training': getattr(config.training, 'progressive', False) and
                                          getattr(config.training.progressive, 'enable_progressive_training', False) if hasattr(config.training, 'progressive') else False,
                })

            # Gradient features
            if hasattr(config, 'gradient'):
                feature_states.update({
                    'gradient_surgery': getattr(config.gradient, 'gradient_surgery', False),
                    'adaptive_gradient_surgery': getattr(config.gradient, 'adaptive_gradient_surgery', False),
                })

            # Loss features
            if hasattr(config, 'loss'):
                feature_states.update({
                    'use_focal_loss': getattr(config.loss, 'use_focal_loss', False),
                    'use_contrastive_loss': getattr(config.loss, 'use_contrastive_loss', False),
                    'use_diversity_loss': getattr(config.loss, 'use_diversity_loss', False),
                    'multi_task': getattr(config.loss, 'use_focal_loss', False) and
                                 getattr(config.loss, 'use_contrastive_loss', False) and
                                 getattr(config.loss, 'use_diversity_loss', False),
                })

                # Single loss component detection
                loss_components = sum([
                    getattr(config.loss, 'use_focal_loss', False),
                    getattr(config.loss, 'use_contrastive_loss', False),
                    getattr(config.loss, 'use_diversity_loss', False)
                ])
                feature_states['single_loss_component'] = loss_components <= 1

            # Episodic memory features
            if hasattr(config, 'episodic_memory'):
                feature_states.update({
                    'episodic_memory': getattr(config.episodic_memory, 'use_episodic_memory', False),
                })

            # Data features
            if hasattr(config, 'data'):
                feature_states.update({
                    'pure_streaming': getattr(config.data, 'streaming', False),
                })

            # DeepSpeed detection
            feature_states['deepspeed'] = getattr(config, 'use_deepspeed', False)

            # Observability features
            if hasattr(config, 'observability'):
                feature_states.update({
                    'detailed_observability': getattr(config.observability, 'enable_detailed_monitoring', False),
                })

        except Exception as e:
            logger.warning(f"Error extracting feature states: {e}")

        return feature_states

    def _check_compatibility_rules(self, feature_states: Dict[str, bool]) -> List[CompatibilityIssue]:
        """Check compatibility rules against feature states."""
        issues = []

        for rule_name, rule_config in self.compatibility_rules.items():
            features = rule_config["features"]
            conflict_type = rule_config["conflict_type"]

            # Check if conflicting features are both enabled
            if conflict_type == "exclusive":
                enabled_features = [f for f in features if feature_states.get(f, False)]
                if len(enabled_features) > 1:
                    level = CompatibilityLevel.CRITICAL if conflict_type == "exclusive" else CompatibilityLevel.WARNING

                    issue = CompatibilityIssue(
                        level=level,
                        message=f"Incompatible features detected: {', '.join(enabled_features)}. {rule_config['reason']}",
                        conflicting_features=enabled_features,
                        fix_suggestions=[rule_config["fix"]],
                        category="exclusive_conflict"
                    )
                    issues.append(issue)

            elif conflict_type == "dependency":
                # Check dependency requirements
                primary_feature = features[0]
                required_state = features[1] if len(features) > 1 else "enabled_dependency"

                if feature_states.get(primary_feature, False) and feature_states.get(required_state, True):
                    issue = CompatibilityIssue(
                        level=CompatibilityLevel.ERROR,
                        message=f"Feature dependency violation: {primary_feature} requires {required_state}. {rule_config['reason']}",
                        conflicting_features=[primary_feature, required_state],
                        fix_suggestions=[rule_config["fix"]],
                        category="dependency_violation"
                    )
                    issues.append(issue)

            elif conflict_type in ["suboptimal", "performance"]:
                enabled_features = [f for f in features if feature_states.get(f, False)]
                if len(enabled_features) > 1:
                    level = CompatibilityLevel.WARNING

                    issue = CompatibilityIssue(
                        level=level,
                        message=f"Suboptimal feature combination: {', '.join(enabled_features)}. {rule_config['reason']}",
                        conflicting_features=enabled_features,
                        fix_suggestions=[rule_config["fix"]],
                        category=conflict_type
                    )
                    issues.append(issue)

        return issues

    def _check_feature_dependencies(self, feature_states: Dict[str, bool]) -> List[CompatibilityIssue]:
        """Check that enabled features have their required dependencies."""
        issues = []

        for feature, dependencies in self.feature_dependencies.items():
            if feature_states.get(feature, False):
                missing_deps = []
                for dep in dependencies:
                    # Handle special dependency checks
                    if dep == "mixed_precision_disabled":
                        if feature_states.get("mixed_precision", False):
                            missing_deps.append("mixed_precision must be disabled")
                    elif dep == "memory_capacity_set":
                        # This would need to be checked against actual config values
                        pass
                    elif dep == "minimal_logging":
                        # This would need to be checked against logging config
                        pass
                    elif not feature_states.get(dep, False):
                        missing_deps.append(dep)

                if missing_deps:
                    issue = CompatibilityIssue(
                        level=CompatibilityLevel.ERROR,
                        message=f"Feature '{feature}' is missing required dependencies: {', '.join(missing_deps)}",
                        conflicting_features=[feature] + missing_deps,
                        fix_suggestions=[f"Enable required dependencies: {', '.join(missing_deps)}"],
                        category="missing_dependency"
                    )
                    issues.append(issue)

        return issues

    def _check_resource_requirements(self, feature_states: Dict[str, bool], config) -> List[CompatibilityIssue]:
        """Check resource requirements for enabled features."""
        issues = []

        # Check memory-intensive feature combinations
        memory_intensive_features = [
            'rag_system', 'episodic_memory', 'detailed_observability',
            'progressive_training', 'use_cross_attention'
        ]

        enabled_memory_features = [f for f in memory_intensive_features if feature_states.get(f, False)]

        if len(enabled_memory_features) >= 3:
            issue = CompatibilityIssue(
                level=CompatibilityLevel.WARNING,
                message=f"Multiple memory-intensive features enabled: {', '.join(enabled_memory_features)}. This may cause OOM errors.",
                conflicting_features=enabled_memory_features,
                fix_suggestions=[
                    "Consider disabling some memory-intensive features",
                    "Increase GPU memory allocation",
                    "Enable gradient checkpointing to reduce memory usage"
                ],
                category="resource_constraint"
            )
            issues.append(issue)

        # Check performance feature conflicts
        if feature_states.get('ultra_fast_mode', False):
            performance_conflicting_features = [
                'detailed_observability', 'episodic_memory', 'rag_system'
            ]
            enabled_conflicts = [f for f in performance_conflicting_features if feature_states.get(f, False)]

            if enabled_conflicts:
                issue = CompatibilityIssue(
                    level=CompatibilityLevel.WARNING,
                    message=f"Ultra-fast mode conflicts with: {', '.join(enabled_conflicts)}. Performance may be degraded.",
                    conflicting_features=['ultra_fast_mode'] + enabled_conflicts,
                    fix_suggestions=[
                        "Disable conflicting features for maximum performance",
                        "Use lightweight alternatives",
                        "Disable ultra-fast mode if full features are needed"
                    ],
                    category="performance_conflict"
                )
                issues.append(issue)

        return issues

    def get_compatibility_report(self, config) -> str:
        """Generate a human-readable compatibility report."""
        is_valid, issues = self.validate_configuration(config)

        report = []
        report.append("=" * 60)
        report.append("FEATURE COMPATIBILITY VALIDATION REPORT")
        report.append("=" * 60)

        if is_valid:
            report.append("✅ Configuration is VALID - No critical compatibility issues found")
        else:
            report.append("❌ Configuration has COMPATIBILITY ISSUES")

        if issues:
            report.append(f"\nFound {len(issues)} compatibility issues:")

            # Group issues by level
            by_level = {}
            for issue in issues:
                level = issue.level.value
                if level not in by_level:
                    by_level[level] = []
                by_level[level].append(issue)

            # Report issues by severity
            for level in ['critical', 'error', 'warning', 'info']:
                if level in by_level:
                    level_issues = by_level[level]
                    report.append(f"\n{level.upper()} ({len(level_issues)} issues):")

                    for i, issue in enumerate(level_issues, 1):
                        report.append(f"  {i}. {issue.message}")
                        if issue.fix_suggestions:
                            report.append(f"     Fix: {issue.fix_suggestions[0]}")
        else:
            report.append("\n✅ No compatibility issues detected")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


def validate_training_config(config) -> Tuple[bool, List[CompatibilityIssue]]:
    """
    Convenience function to validate a training configuration.

    Args:
        config: Enhanced training configuration

    Returns:
        Tuple of (is_valid, issues_list)
    """
    validator = FeatureCompatibilityValidator()
    return validator.validate_configuration(config)


def print_compatibility_report(config):
    """
    Print a detailed compatibility report for the configuration.

    Args:
        config: Enhanced training configuration
    """
    validator = FeatureCompatibilityValidator()
    report = validator.get_compatibility_report(config)
    print(report)