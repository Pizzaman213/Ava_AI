"""Core training loop implementation."""

from .trainer import EnhancedModularTrainer

# Alias for backward compatibility
EnhancedTrainer = EnhancedModularTrainer

__all__ = ["EnhancedTrainer", "EnhancedModularTrainer"]
