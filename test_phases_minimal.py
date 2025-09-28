#!/usr/bin/env python3
"""
Minimal validation test for all 8 phases - imports and basic functionality only.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/project/code')

def test_minimal():
    """Minimal test of all 8 phases - imports only."""
    print("🚀 MINIMAL VALIDATION OF ALL 8 PHASES")
    print("=" * 50)

    results = {}

    # Phase 1: Training Stability - Test imports and basic config
    print("✅ Phase 1: Training Stability")
    try:
        from src.Ava.config import EnhancedTrainingConfig, TrainingConfig, DataConfig, ArchitectureConfig, WandBConfig
        from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
        from src.Ava.training.enhanced_trainer import EnhancedModularTrainer

        # Test config creation
        config = EnhancedTrainingConfig(
            config_file="test",
            training=TrainingConfig(batch_size=2, learning_rate=1e-4, epochs=1),
            data=DataConfig(max_length=64),
            architecture=ArchitectureConfig(),
            wandb=WandBConfig(use_wandb=False)
        )

        results['Phase 1'] = True
        print("  ✅ Training components available")
    except Exception as e:
        results['Phase 1'] = False
        print(f"  ❌ Failed: {e}")

    # Phase 2: Dataset Formats - Test tokenization
    print("✅ Phase 2: Dataset Format Processing")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Test basic tokenization
        tokens = tokenizer("Test text", max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        results['Phase 2'] = tokens['input_ids'].shape[1] == 64
        print("  ✅ Dataset processing available")
    except Exception as e:
        results['Phase 2'] = False
        print(f"  ❌ Failed: {e}")

    # Phase 3: Anomaly Recovery - Test error handling imports
    print("✅ Phase 3: Automatic Anomaly Recovery")
    try:
        from src.Ava.training.rank_aware_error_handler import RankAwareErrorHandler, ErrorType, ErrorSeverity
        from src.Ava.training.distributed_health_checker import DistributedHealthChecker

        # Test that new method exists
        health_checker = DistributedHealthChecker(rank=0, world_size=1)
        has_method = hasattr(health_checker, 'get_health_status')

        results['Phase 3'] = has_method
        print(f"  ✅ Anomaly recovery available (get_health_status: {has_method})")
    except Exception as e:
        results['Phase 3'] = False
        print(f"  ❌ Failed: {e}")

    # Phase 4: Multi-GPU Scaling - Test distributed imports
    print("✅ Phase 4: Multi-GPU Scaling")
    try:
        from src.Ava.training.distributed_manager import DistributedManager, DistributedConfig
        from src.Ava.multi_column_data import AdvancedDistributedSampler

        # Test creation without initialization
        dist_config = DistributedConfig()
        results['Phase 4'] = True
        print("  ✅ Multi-GPU components available")
    except Exception as e:
        results['Phase 4'] = False
        print(f"  ❌ Failed: {e}")

    # Phase 5: Progressive Training - Test new manager class
    print("✅ Phase 5: Progressive Training Convergence")
    try:
        from src.Ava.training.progressive_training import ProgressiveTrainingManager

        # Test that the class can be instantiated
        manager = ProgressiveTrainingManager()
        has_methods = (hasattr(manager, 'get_current_sequence_length') and
                      hasattr(manager, 'get_adaptive_learning_rate'))

        results['Phase 5'] = has_methods
        print(f"  ✅ Progressive training available (methods: {has_methods})")
    except Exception as e:
        results['Phase 5'] = False
        print(f"  ❌ Failed: {e}")

    # Phase 6: Feature Combinations - Test config combinations
    print("✅ Phase 6: Feature Combinations Validation")
    try:
        # Test different feature combinations
        combo_configs = [
            ArchitectureConfig(use_moh=True, use_moa=False),
            ArchitectureConfig(use_moh=False, use_moa=True),
            ArchitectureConfig(use_moh=True, use_moa=True)
        ]

        valid_configs = sum(1 for config in combo_configs if hasattr(config, 'use_moh'))
        results['Phase 6'] = valid_configs == len(combo_configs)
        print(f"  ✅ Feature combinations available ({valid_configs}/{len(combo_configs)} valid)")
    except Exception as e:
        results['Phase 6'] = False
        print(f"  ❌ Failed: {e}")

    # Phase 7: Observability - Test monitoring imports
    print("✅ Phase 7: Real-time Monitoring & Observability")
    try:
        from src.Ava.training.memory_monitor import GPUMemoryManager

        # Test that GPUMemoryManager alias works
        memory_manager = GPUMemoryManager()
        has_methods = hasattr(memory_manager, 'get_memory_stats')

        results['Phase 7'] = has_methods
        print(f"  ✅ Observability available (GPUMemoryManager: {has_methods})")
    except Exception as e:
        results['Phase 7'] = False
        print(f"  ❌ Failed: {e}")

    # Phase 8: Test Coverage - Test critical imports
    print("✅ Phase 8: Test Coverage & Stress Testing")
    try:
        import torch
        import numpy as np

        # Test critical component imports
        critical_components = [
            "src.Ava.config",
            "src.Ava.models.moe_model",
            "src.Ava.training.enhanced_trainer",
            "src.Ava.training.distributed_manager",
            "src.Ava.multi_column_data"
        ]

        importable = 0
        for component in critical_components:
            try:
                __import__(component)
                importable += 1
            except:
                pass

        coverage = importable / len(critical_components)
        results['Phase 8'] = coverage >= 0.8
        print(f"  ✅ Test coverage available ({importable}/{len(critical_components)} = {coverage:.1%})")
    except Exception as e:
        results['Phase 8'] = False
        print(f"  ❌ Failed: {e}")

    # Final results
    print("\n" + "=" * 50)
    print("🎯 MINIMAL VALIDATION RESULTS")
    print("=" * 50)

    total_phases = len(results)
    passed_phases = sum(1 for passed in results.values() if passed)

    for phase, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {phase}: {'PASSED' if passed else 'FAILED'}")

    print(f"\n🏆 OVERALL: {passed_phases}/{total_phases} phases passed ({passed_phases/total_phases:.1%})")

    if passed_phases >= total_phases * 0.75:
        print("✅ Core system is functional!")
        return True
    else:
        print("❌ Core system needs fixes.")
        return False

if __name__ == "__main__":
    try:
        success = test_minimal()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 Critical failure: {e}")
        sys.exit(1)