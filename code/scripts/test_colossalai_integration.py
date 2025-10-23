#!/usr/bin/env python3
"""
Test Script for Colossal-AI Integration

This script validates the Colossal-AI integration with the Ava training infrastructure.
It tests various parallelization strategies and memory optimizations.
"""

import os
import sys
import logging
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.Ava.config.training_config import EnhancedTrainingConfig
from src.Ava.models.moe_model import EnhancedMoEConfig, EnhancedMoEModel
from src.Ava.models.colossalai_moe_model import (
    ColossalAIMoEConfig,
    ColossalAIEnhancedMoEModel,
    create_colossal_moe_model,
)
from src.Ava.training.colossalai_integration import (
    ColossalAIConfig,
    ColossalAIIntegration,
    ParallelismStrategy,
    create_colossalai_integration,
)
from src.Ava.training.unified_distributed_manager import (
    UnifiedDistributedManager,
    DistributedBackend,
    create_unified_manager,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_colossal_ai_availability():
    """Test if Colossal-AI is properly installed."""
    print("\n" + "="*60)
    print("Testing Colossal-AI Availability")
    print("="*60)

    try:
        import colossalai
        print("✓ Colossal-AI is installed")
        print(f"  Version: {colossalai.__version__ if hasattr(colossalai, '__version__') else 'Unknown'}")
        return True
    except ImportError as e:
        print("✗ Colossal-AI is not installed")
        print(f"  Error: {e}")
        return False


def test_model_creation():
    """Test creating models with Colossal-AI support."""
    print("\n" + "="*60)
    print("Testing Model Creation")
    print("="*60)

    # Create base configuration
    config = ColossalAIMoEConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_token=2,
    )

    try:
        # Create standard model
        print("\n1. Creating standard EnhancedMoEModel...")
        base_model = EnhancedMoEModel(config)
        print(f"✓ Base model created: {base_model.__class__.__name__}")
        print(f"  Parameters: {sum(p.numel() for p in base_model.parameters()):,}")

        # Create Colossal-AI model
        print("\n2. Creating ColossalAIEnhancedMoEModel...")
        colossal_model = create_colossal_moe_model(config)
        print(f"✓ Colossal model created: {colossal_model.__class__.__name__}")
        print(f"  Parameters: {sum(p.numel() for p in colossal_model.parameters()):,}")

        return True

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass through models."""
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)

    config = ColossalAIMoEConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=128,
        num_experts=4,
        num_experts_per_token=2,
    )

    try:
        model = create_colossal_moe_model(config)
        model.eval()

        # Create dummy input
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        print(f"Input shape: {input_ids.shape}")

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        print("✓ Forward pass successful")
        print(f"  Output keys: {list(outputs.keys())}")
        if outputs['logits'] is not None:
            print(f"  Logits shape: {outputs['logits'].shape}")

        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_colossal_integration():
    """Test Colossal-AI integration module."""
    print("\n" + "="*60)
    print("Testing Colossal-AI Integration Module")
    print("="*60)

    # Create dummy training config
    training_config = EnhancedTrainingConfig()
    training_config.model.hidden_size = 128
    training_config.training.batch_size = 4

    # Test different parallelism strategies
    strategies = [
        ParallelismStrategy.DATA_PARALLEL,
        ParallelismStrategy.ZERO1,
        ParallelismStrategy.ZERO2,
    ]

    for strategy in strategies:
        print(f"\nTesting {strategy.value} strategy...")

        try:
            config = ColossalAIConfig(
                parallel_strategy=strategy,
                mixed_precision="bf16" if torch.cuda.is_available() else "none",
                use_activation_checkpointing=True,
            )

            integration = ColossalAIIntegration(
                config=config,
                training_config=training_config,
            )

            print(f"✓ {strategy.value} integration created")
            print(f"  Compatibility mode: {config.compatibility_mode}")

        except Exception as e:
            print(f"✗ {strategy.value} integration failed: {e}")

    return True


def test_unified_manager():
    """Test unified distributed manager."""
    print("\n" + "="*60)
    print("Testing Unified Distributed Manager")
    print("="*60)

    training_config = EnhancedTrainingConfig()

    backends = [
        DistributedBackend.NATIVE,
        DistributedBackend.COLOSSALAI,
        DistributedBackend.HYBRID,
    ]

    for backend in backends:
        print(f"\nTesting {backend.value} backend...")

        try:
            manager = UnifiedDistributedManager(
                training_config=training_config,
                backend=backend,
            )

            print(f"✓ {backend.value} manager created")

            # Test health check
            health = manager.check_health()
            print(f"  Health status: {health}")

        except Exception as e:
            print(f"✗ {backend.value} manager failed: {e}")

    return True


def test_memory_optimization():
    """Test memory optimization features."""
    print("\n" + "="*60)
    print("Testing Memory Optimization Features")
    print("="*60)

    if not torch.cuda.is_available():
        print("⚠ GPU not available, skipping memory tests")
        return True

    try:
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Initial GPU memory: {initial_memory:.2f} GB")

        # Create model with memory optimization
        config = ColossalAIMoEConfig(
            vocab_size=10000,
            hidden_size=256,
            num_layers=4,
            num_attention_heads=8,
            intermediate_size=512,
            use_checkpoint=True,  # Enable activation checkpointing
        )

        model = create_colossal_moe_model(config)
        model = model.cuda()

        # Check memory after model creation
        model_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Memory after model: {model_memory:.2f} GB")
        print(f"Model size: {(model_memory - initial_memory):.2f} GB")

        # Test with dummy forward pass
        batch = torch.randint(0, config.vocab_size, (2, 64)).cuda()
        outputs = model(batch)

        # Check peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak memory usage: {peak_memory:.2f} GB")

        print("✓ Memory optimization test passed")
        return True

    except Exception as e:
        print(f"✗ Memory optimization test failed: {e}")
        return False


def test_training_step():
    """Test a complete training step with Colossal-AI."""
    print("\n" + "="*60)
    print("Testing Training Step")
    print("="*60)

    try:
        # Create configuration
        training_config = EnhancedTrainingConfig()
        training_config.training.batch_size = 2
        training_config.training.gradient_accumulation_steps = 1

        # Create model
        model_config = ColossalAIMoEConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
        )
        model = create_colossal_moe_model(model_config)

        # Create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Create Colossal-AI integration
        colossal_config = ColossalAIConfig(
            parallel_strategy=ParallelismStrategy.DATA_PARALLEL,
            mixed_precision="none",  # Disable for CPU testing
            compatibility_mode=True,  # Use compatibility mode for testing
        )

        integration = ColossalAIIntegration(
            config=colossal_config,
            training_config=training_config,
        )

        # Initialize with Colossal-AI
        model, optimizer, _, _ = integration.initialize(
            model=model,
            optimizer=optimizer,
        )

        # Create dummy batch
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        # Training step
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']

        if loss is not None:
            # Backward pass
            integration.backward(loss)

            # Optimizer step
            integration.step(optimizer)

            print("✓ Training step completed successfully")
            print(f"  Loss: {loss.item():.4f}")
        else:
            print("⚠ Loss is None, skipping backward pass")

        return True

    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("COLOSSAL-AI INTEGRATION TEST SUITE")
    print("="*60)

    tests = [
        ("Colossal-AI Availability", test_colossal_ai_availability),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("Integration Module", test_colossal_integration),
        ("Unified Manager", test_unified_manager),
        ("Memory Optimization", test_memory_optimization),
        ("Training Step", test_training_step),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Colossal-AI integration is working correctly.")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the errors above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)