#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Colossal-AI Features

This script tests all the new Colossal-AI integrations including:
- Shardformer integration
- Distributed optimizers
- Gradient compression
- Auto-parallelism helpers
- Performance monitoring
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.Ava.training.shardformer_integration import (
    ShardformerConfig,
    ShardformerIntegration,
    auto_shard_huggingface_model,
)
from src.Ava.training.distributed_optimizers import (
    OptimizerConfig,
    OptimizerFactory,
    create_distributed_optimizer,
    get_optimizer_recommendations,
)
from src.Ava.training.colossalai_enhanced_features import (
    GradientCompressor,
    GradientCompressionConfig,
    AutoParallelismHelper,
    PerformanceMonitor,
    create_optimal_config,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleTestModel(nn.Module):
    """Simple model for testing"""

    def __init__(self, hidden_size=768, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.norm(x)


def test_shardformer_integration():
    """Test Shardformer integration"""
    logger.info("=" * 80)
    logger.info("Testing Shardformer Integration")
    logger.info("=" * 80)

    try:
        # Create test model
        model = SimpleTestModel(hidden_size=512, num_layers=4)
        logger.info(f"Created test model with {sum(p.numel() for p in model.parameters())} parameters")

        # Test ShardformerConfig
        config = ShardformerConfig(
            tensor_parallel_size=1,
            enable_flash_attention=True,
            enable_jit_fused=True,
        )
        logger.info("✓ ShardformerConfig created successfully")

        # Test ShardformerIntegration
        integration = ShardformerIntegration(config)
        logger.info(f"✓ ShardformerIntegration initialized (available: {integration._initialized})")

        # Test model info
        info = integration.get_model_info(model)
        logger.info(f"✓ Model info: {info}")

        # Test sharding (will fallback if Shardformer not available)
        sharded_model = integration.shard_model(model)
        logger.info("✓ Model sharding completed (or fallback to original)")

        logger.info("✅ Shardformer integration tests PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Shardformer integration test FAILED: {e}")
        return False


def test_distributed_optimizers():
    """Test distributed optimizers"""
    logger.info("=" * 80)
    logger.info("Testing Distributed Optimizers")
    logger.info("=" * 80)

    try:
        model = SimpleTestModel()

        # Test OptimizerConfig
        config = OptimizerConfig(
            optimizer_type="hybrid_adam",
            lr=1e-4,
            weight_decay=0.01,
        )
        logger.info("✓ OptimizerConfig created successfully")

        # Test optimizer creation
        optimizer = create_distributed_optimizer(model, config)
        logger.info(f"✓ Optimizer created: {type(optimizer).__name__}")

        # Test GaLore config
        galore_config = OptimizerConfig(
            optimizer_type="galore_adamw",
            lr=1e-4,
            use_galore=True,
            galore_rank=256,
        )
        galore_optimizer = create_distributed_optimizer(model, galore_config)
        logger.info(f"✓ GaLore optimizer created: {type(galore_optimizer).__name__}")

        # Test optimizer recommendations
        model_size = sum(p.numel() for p in model.parameters())
        recommendations = get_optimizer_recommendations(
            model_size=model_size,
            gpu_memory_gb=16.0,
            world_size=1,
        )
        logger.info(f"✓ Optimizer recommendations: {recommendations}")

        # Test OptimizerFactory
        factory_optimizer = OptimizerFactory.create_auto(model, lr=1e-4)
        logger.info(f"✓ Factory optimizer created: {type(factory_optimizer).__name__}")

        logger.info("✅ Distributed optimizer tests PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Distributed optimizer test FAILED: {e}")
        return False


def test_gradient_compression():
    """Test gradient compression"""
    logger.info("=" * 80)
    logger.info("Testing Gradient Compression")
    logger.info("=" * 80)

    try:
        # Create compression config
        config = GradientCompressionConfig(
            enabled=True,
            compression_ratio=0.1,
            compression_type="topk",
            warmup_steps=0,  # Disable warmup for testing
        )
        logger.info("✓ GradientCompressionConfig created")

        # Create compressor
        compressor = GradientCompressor(config)
        logger.info("✓ GradientCompressor created")

        # Test compression/decompression
        test_tensor = torch.randn(1000, 1000)
        compressed, metadata = compressor.compress(test_tensor)
        logger.info(f"✓ Compressed tensor from {test_tensor.numel()} to {compressed.numel()} elements")
        logger.info(f"  Metadata: {metadata}")

        decompressed = compressor.decompress(compressed, metadata)
        logger.info(f"✓ Decompressed tensor back to {decompressed.shape}")

        # Test statistics
        compressor.step()
        stats = compressor.get_stats()
        logger.info(f"✓ Compression stats: {stats}")

        # Test different compression types
        for comp_type in ["topk", "randomk", "threshold"]:
            config.compression_type = comp_type
            compressor = GradientCompressor(config)
            compressed, _ = compressor.compress(test_tensor)
            logger.info(f"✓ {comp_type} compression: {compressed.shape}")

        logger.info("✅ Gradient compression tests PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Gradient compression test FAILED: {e}")
        return False


def test_auto_parallelism():
    """Test auto-parallelism helpers"""
    logger.info("=" * 80)
    logger.info("Testing Auto-Parallelism Helpers")
    logger.info("=" * 80)

    try:
        helper = AutoParallelismHelper()

        # Test configuration suggestions for different scenarios
        test_cases = [
            {"model_size": 1e8, "num_gpus": 1, "gpu_memory_gb": 16, "name": "Small model, 1 GPU"},
            {"model_size": 1e9, "num_gpus": 4, "gpu_memory_gb": 24, "name": "1B model, 4 GPUs"},
            {"model_size": 7e9, "num_gpus": 8, "gpu_memory_gb": 40, "name": "7B model, 8 GPUs"},
            {"model_size": 7e9, "num_gpus": 2, "gpu_memory_gb": 16, "name": "7B model, 2 GPUs (constrained)"},
        ]

        for case in test_cases:
            config = helper.suggest_parallelism_config(
                model_size=int(case["model_size"]),
                num_gpus=case["num_gpus"],
                gpu_memory_gb=case["gpu_memory_gb"],
            )
            logger.info(f"✓ {case['name']}: {config}")

        # Test optimization presets
        throughput_config = helper.optimize_for_throughput(num_gpus=4)
        logger.info(f"✓ Throughput optimization: {throughput_config}")

        memory_config = helper.optimize_for_memory(num_gpus=4)
        logger.info(f"✓ Memory optimization: {memory_config}")

        logger.info("✅ Auto-parallelism tests PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Auto-parallelism test FAILED: {e}")
        return False


def test_performance_monitoring():
    """Test performance monitoring"""
    logger.info("=" * 80)
    logger.info("Testing Performance Monitoring")
    logger.info("=" * 80)

    try:
        monitor = PerformanceMonitor()
        logger.info("✓ PerformanceMonitor created")

        # Simulate training steps
        for step in range(10):
            monitor.start_step()

            # Forward phase
            monitor.start_phase()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            monitor.end_phase("forward")

            # Backward phase
            monitor.start_phase()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            monitor.end_phase("backward")

            # Optimizer phase
            monitor.start_phase()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            monitor.end_phase("optimizer")

            monitor.end_step()

        logger.info("✓ Simulated 10 training steps")

        # Get summary
        summary = monitor.get_summary(last_n_steps=10)
        logger.info(f"✓ Performance summary: {summary}")

        # Log summary
        monitor.log_summary(step=10, last_n_steps=10)
        logger.info("✓ Logged performance summary")

        logger.info("✅ Performance monitoring tests PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Performance monitoring test FAILED: {e}")
        return False


def test_optimal_config_creation():
    """Test optimal configuration creation"""
    logger.info("=" * 80)
    logger.info("Testing Optimal Config Creation")
    logger.info("=" * 80)

    try:
        model = SimpleTestModel(hidden_size=1024, num_layers=12)

        # Test different optimization targets
        for optimize_for in ["balanced", "throughput", "memory"]:
            config = create_optimal_config(
                model=model,
                num_gpus=4,
                gpu_memory_gb=24.0,
                optimize_for=optimize_for,
            )
            logger.info(f"✓ {optimize_for.capitalize()} config: {config}")

        logger.info("✅ Optimal config creation tests PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Optimal config creation test FAILED: {e}")
        return False


def test_integration():
    """Test complete integration workflow"""
    logger.info("=" * 80)
    logger.info("Testing Complete Integration Workflow")
    logger.info("=" * 80)

    try:
        # Create model
        model = SimpleTestModel(hidden_size=768, num_layers=8)
        logger.info("✓ Created test model")

        # Get optimal config
        optimal_config = create_optimal_config(model, optimize_for="balanced")
        logger.info(f"✓ Generated optimal config: {optimal_config}")

        # Create optimizer
        optimizer = OptimizerFactory.create_auto(model, lr=1e-4)
        logger.info(f"✓ Created optimizer: {type(optimizer).__name__}")

        # Setup compression
        compression_config = GradientCompressionConfig(enabled=True, compression_ratio=0.1)
        compressor = GradientCompressor(compression_config)
        logger.info("✓ Setup gradient compression")

        # Setup monitoring
        monitor = PerformanceMonitor()
        logger.info("✓ Setup performance monitoring")

        # Simulate training step
        monitor.start_step()

        # Forward
        monitor.start_phase()
        x = torch.randn(2, 10, 768)
        if torch.cuda.is_available():
            x = x.cuda()
            model = model.cuda()
        output = model(x)
        loss = output.mean()
        monitor.end_phase("forward")

        # Backward
        monitor.start_phase()
        loss.backward()
        monitor.end_phase("backward")

        # Compress gradients
        for param in model.parameters():
            if param.grad is not None:
                compressed, metadata = compressor.compress(param.grad)
                # In real training, compressed gradients would be communicated
                # and then decompressed on other ranks

        # Optimizer step
        monitor.start_phase()
        optimizer.step()
        optimizer.zero_grad()
        monitor.end_phase("optimizer")

        monitor.end_step()
        compressor.step()

        logger.info("✓ Completed simulated training step")

        # Get statistics
        perf_summary = monitor.get_summary(last_n_steps=1)
        comp_stats = compressor.get_stats()

        logger.info(f"✓ Performance: {perf_summary}")
        logger.info(f"✓ Compression: {comp_stats}")

        logger.info("✅ Integration workflow test PASSED")
        return True

    except Exception as e:
        logger.error(f"❌ Integration workflow test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Enhanced Colossal-AI Features")
    parser.add_argument(
        "--test",
        choices=[
            "all",
            "shardformer",
            "optimizers",
            "compression",
            "parallelism",
            "monitoring",
            "config",
            "integration",
        ],
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()

    results = {}

    if args.test in ["all", "shardformer"]:
        results["shardformer"] = test_shardformer_integration()

    if args.test in ["all", "optimizers"]:
        results["optimizers"] = test_distributed_optimizers()

    if args.test in ["all", "compression"]:
        results["compression"] = test_gradient_compression()

    if args.test in ["all", "parallelism"]:
        results["parallelism"] = test_auto_parallelism()

    if args.test in ["all", "monitoring"]:
        results["monitoring"] = test_performance_monitoring()

    if args.test in ["all", "config"]:
        results["config"] = test_optimal_config_creation()

    if args.test in ["all", "integration"]:
        results["integration"] = test_integration()

    # Print summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name.upper()}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED! 🎉")
        return 0
    else:
        logger.error("\n⚠️  SOME TESTS FAILED ⚠️")
        return 1


if __name__ == "__main__":
    sys.exit(main())
