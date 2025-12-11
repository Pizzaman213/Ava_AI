"""
Ava MoE++ Architecture Package

A comprehensive implementation of advanced LLM architectures with:
- Enhanced Mixture of Experts (MoE++) with hierarchical routing
- Advanced training optimizations (dynamic batching, gradient checkpointing)
- Multiple data loading strategies (streaming, pre-tokenized, conversation-aware)

IMPORTANT: This module uses lazy imports to avoid expensive module loading
during dataloader worker initialization. Imports only happen when attributes
are accessed, not when this package is imported.
"""

__version__ = "2.0.0"

# Lazy import placeholders - these are loaded on first access
_lazy_imports = {
    # Core models
    "EnhancedMoEModel": "Ava.models.moe_model",
    "EnhancedMoEConfig": "Ava.models.moe_model",
    # Configuration
    "TrainingConfig": "Ava.config.training_config",
    "DataConfig": "Ava.config.training_config",
    "OutputConfig": "Ava.config.training_config",
    "PerformanceConfig": "Ava.config.training_config",
    "WandBConfig": "Ava.config.training_config",
    # Training context
    "TrainingContext": "Ava.training.train.base",
    "TrainingComponent": "Ava.training.train.base",
}


def __getattr__(name: str):
    """Lazy import mechanism for expensive modules."""
    if name in _lazy_imports:
        module_path = _lazy_imports[name]
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module 'Ava' has no attribute '{name}'")


__all__ = [
    # Core models
    "EnhancedMoEModel",
    "EnhancedMoEConfig",
    # Configuration
    "TrainingConfig",
    "DataConfig",
    "OutputConfig",
    "PerformanceConfig",
    "WandBConfig",
    # Training
    "TrainingContext",
    "TrainingComponent",
]
