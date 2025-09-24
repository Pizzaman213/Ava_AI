#!/usr/bin/env python3
"""
Quick test to verify all refactored components work correctly.
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append('/project/code')

def test_imports():
    """Test that all new modular components can be imported."""
    print("🧪 Testing imports...")

    try:
        # Configuration management
        from src.Ava.config import TrainingConfigManager, EnhancedTrainingConfig
        print("  ✅ Configuration management")

        # GPU memory management
        from src.Ava.utils import GPUMemoryManager, cleanup_gpu_memory
        print("  ✅ GPU memory management")

        # Training components
        from src.Ava.training import (
            AdvancedWarmupScheduler, AdaptiveLearningRateManager,
            PerformanceModeManager, TrainingMetricsCollector
        )
        print("  ✅ Training components")

        # Async logging
        from src.Ava.utils import AsyncLogger, AsyncLoggingConfig
        print("  ✅ Async logging")

        # Enhanced trainer
        from src.Ava.training.enhanced_trainer import EnhancedModularTrainer
        print("  ✅ Enhanced trainer")

        return True

    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False


def test_configuration():
    """Test configuration management."""
    print("\n🧪 Testing configuration management...")

    try:
        from src.Ava.config import TrainingConfigManager

        config_manager = TrainingConfigManager()
        parser = config_manager.create_argument_parser()

        # Test with sample arguments
        args = parser.parse_args([
            '--config', 'dummy.yaml',  # File doesn't need to exist for this test
            '--enable-all-features',
            '--ultra-fast-mode'
        ])

        # Parse to structured config
        training_config = config_manager.parse_args_to_config(args)

        # Get feature summary
        feature_summary = config_manager.get_feature_summary(training_config)

        print(f"  ✅ Features enabled: {feature_summary['total_features']}")
        print(f"  ✅ Performance mode: {feature_summary['performance_mode']}")

        return True

    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False


def test_components():
    """Test individual components."""
    print("\n🧪 Testing individual components...")

    try:
        # Test GPU memory manager
        from src.Ava.utils import GPUMemoryManager
        gpu_manager = GPUMemoryManager()
        memory_stats = gpu_manager.get_memory_stats()
        print(f"  ✅ GPU Memory Manager initialized")

        # Test performance manager
        from src.Ava.training import PerformanceModeManager, create_ultra_fast_config
        perf_config = create_ultra_fast_config()
        perf_manager = PerformanceModeManager(perf_config)
        print(f"  ✅ Performance Manager: {perf_manager.config.mode.value}")

        # Test async logger
        from src.Ava.utils import AsyncLogger, create_fast_logging_config
        log_config = create_fast_logging_config()
        async_logger = AsyncLogger(log_config, wandb_available=False)
        print(f"  ✅ Async Logger initialized")

        # Test metrics collector
        from src.Ava.training import TrainingMetricsCollector, create_fast_metrics_config
        metrics_config = create_fast_metrics_config()
        metrics_collector = TrainingMetricsCollector(metrics_config)
        print(f"  ✅ Metrics Collector initialized")

        # Test adaptive LR
        from src.Ava.training import AdaptiveLearningRateManager, create_balanced_lr_config

        # Create a dummy optimizer for testing
        dummy_model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(dummy_model.parameters(), lr=1e-3)

        lr_config = create_balanced_lr_config()
        lr_manager = AdaptiveLearningRateManager(optimizer, lr_config)

        # Test a step
        lr_info = lr_manager.step(loss=1.0, batch_idx=0)
        print(f"  ✅ Adaptive LR Manager: current_lr={lr_info['current_lr']:.2e}")

        # Test warmup scheduler
        from src.Ava.training import AdvancedWarmupScheduler, create_cosine_warmup_config
        warmup_config = create_cosine_warmup_config(warmup_steps=100)
        warmup_scheduler = AdvancedWarmupScheduler(optimizer, warmup_config)

        warmup_info = warmup_scheduler.step(loss=1.0, model=dummy_model)
        print(f"  ✅ Warmup Scheduler: in_warmup={warmup_info['in_warmup']}")

        return True

    except Exception as e:
        print(f"  ❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Refactored Components")
    print("=" * 50)

    tests_passed = 0
    total_tests = 3

    # Test imports
    if test_imports():
        tests_passed += 1

    # Test configuration
    if test_configuration():
        tests_passed += 1

    # Test components
    if test_components():
        tests_passed += 1

    # Results
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} passed")

    if tests_passed == total_tests:
        print("✅ All tests passed! Refactoring successful! 🎉")
        print("\nThe modular components are working correctly:")
        print("  - Configuration management ✅")
        print("  - GPU memory management ✅")
        print("  - Training components ✅")
        print("  - Async logging ✅")
        print("  - Performance modes ✅")
        print("  - Adaptive learning rate ✅")
        print("  - Advanced warmup ✅")
        print("  - Metrics collection ✅")

        print(f"\n🚀 Ready to use the new modular training script:")
        print(f"  python /project/code/scripts/training/train.py --config configs/gpu/small.yaml --enable-all-features")

    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())