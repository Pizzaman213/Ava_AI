#!/usr/bin/env python3
"""
Quick validation script to check if all new Colossal-AI modules import correctly.
This doesn't require torch to be installed.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""

    results = {}

    # Test shardformer_integration
    try:
        from src.Ava.training.shardformer_integration import (
            ShardformerConfig,
            ShardformerIntegration,
            auto_shard_huggingface_model,
        )
        results["shardformer_integration"] = "✅ OK"
    except Exception as e:
        results["shardformer_integration"] = f"❌ FAILED: {e}"

    # Test distributed_optimizers
    try:
        from src.Ava.training.distributed_optimizers import (
            OptimizerConfig,
            OptimizerFactory,
            create_distributed_optimizer,
            get_optimizer_recommendations,
        )
        results["distributed_optimizers"] = "✅ OK"
    except Exception as e:
        results["distributed_optimizers"] = f"❌ FAILED: {e}"

    # Test colossalai_enhanced_features
    try:
        from src.Ava.training.colossalai_enhanced_features import (
            GradientCompressor,
            GradientCompressionConfig,
            AutoParallelismHelper,
            PerformanceMonitor,
            create_optimal_config,
        )
        results["colossalai_enhanced_features"] = "✅ OK"
    except Exception as e:
        results["colossalai_enhanced_features"] = f"❌ FAILED: {e}"

    # Test existing colossalai_integration
    try:
        from src.Ava.training.colossalai_integration import (
            ColossalAIConfig,
            ColossalAIIntegration,
            ParallelismStrategy,
        )
        results["colossalai_integration"] = "✅ OK"
    except Exception as e:
        results["colossalai_integration"] = f"❌ FAILED: {e}"

    # Test unified_distributed_manager
    try:
        from src.Ava.training.unified_distributed_manager import (
            UnifiedDistributedManager,
            DistributedBackend,
        )
        results["unified_distributed_manager"] = "✅ OK"
    except Exception as e:
        results["unified_distributed_manager"] = f"❌ FAILED: {e}"

    return results

def main():
    print("=" * 80)
    print("Validating Colossal-AI Module Imports")
    print("=" * 80)

    results = test_imports()

    all_ok = True
    for module, status in results.items():
        print(f"{module:40s} {status}")
        if "FAILED" in status:
            all_ok = False

    print("=" * 80)
    if all_ok:
        print("✅ All modules imported successfully!")
        return 0
    else:
        print("❌ Some modules failed to import")
        return 1

if __name__ == "__main__":
    sys.exit(main())
