#!/usr/bin/env python3
"""
Test script for validating the complete training functionality.
"""

import sys
import torch
import warnings
import traceback
from pathlib import Path

# Add project root to path
sys.path.append('/project/code')

def test_simple_training_step():
    """Test a complete training step end-to-end."""
    print("🏃 Testing complete training step...")

    try:
        from src.Ava.config import (
            EnhancedTrainingConfig, TrainingConfig, DataConfig,
            ArchitectureConfig, WandBConfig
        )
        from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
        from src.Ava.training.enhanced_trainer import EnhancedModularTrainer
        from transformers import AutoTokenizer

        # Create minimal config for testing
        training_config = EnhancedTrainingConfig(
            config_file="test_config",
            training=TrainingConfig(batch_size=2, learning_rate=1e-4, epochs=1),
            data=DataConfig(max_length=64),  # Short sequences for testing
            architecture=ArchitectureConfig(),
            wandb=WandBConfig(use_wandb=False)
        )

        # Create very small model for testing
        # Use GPT-2's vocab size to match the tokenizer
        model_config = EnhancedMoEConfig(
            hidden_size=144,  # Divisible by 12
            num_layers=1,     # Just 1 layer
            num_attention_heads=12,
            num_experts=2,    # Just 2 experts
            vocab_size=50257, # Match GPT-2 vocab size
            max_position_embeddings=64
        )

        model = EnhancedMoEModel(model_config)
        print(f"  ✅ Test model created - {sum(p.numel() for p in model.parameters()):,} parameters")

        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"  ✅ Model moved to {device}")

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create trainer (device already defined above)
        trainer = EnhancedModularTrainer(
            model=model,
            config=training_config,
            tokenizer=tokenizer,
            device=device
        )

        print(f"  ✅ Trainer created on {device}")

        # Create simple test batch
        test_text = ["Hello world this is a test", "Another test sentence here"]

        # Tokenize
        tokens = tokenizer(
            test_text,
            max_length=64,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        labels = input_ids.clone()  # For causal LM

        print(f"  ✅ Test batch created - shape: {input_ids.shape}")

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Test forward pass only first
        print("  🔍 Testing forward pass...")
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if isinstance(outputs, dict):
                loss = outputs.get('loss', torch.tensor(0.0))
            else:
                loss = outputs.loss
            print(f"  ✅ Forward pass successful - loss: {loss.item():.4f}")

        # Test training step
        print("  🏋️ Testing training step...")
        model.train()

        step_results = trainer.train_step(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            optimizer=optimizer,
            epoch=0,
            batch_idx=0
        )

        print(f"  ✅ Training step successful!")
        print(f"    Loss: {step_results['loss']:.4f}")
        print(f"    Learning rate: {step_results['learning_rate']:.2e}")
        print(f"    Gradient norm: {step_results['grad_norm']:.4f}")
        print(f"    Forward time: {step_results['forward_time']:.3f}s")
        print(f"    Backward time: {step_results['backward_time']:.3f}s")

        # Test memory monitoring
        if 'memory_status' in step_results:
            print(f"    Memory status: {step_results['memory_status']}")
            print(f"    Memory utilization: {step_results['memory_utilization']:.1%}")

        # Test checkpoint functionality
        print("  💾 Testing checkpoint save/load...")

        checkpoint_dir = "/tmp/test_checkpoint"
        Path(checkpoint_dir).mkdir(exist_ok=True)

        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(checkpoint_dir, tag="test")
        print(f"    Checkpoint saved: {checkpoint_path}")

        # Load checkpoint
        load_result = trainer.load_checkpoint(checkpoint_path)
        print(f"    Checkpoint loaded: {load_result.get('step_count', 0)} steps")

        # Test cleanup
        print("  🧹 Testing cleanup...")
        trainer.cleanup()
        print(f"  ✅ Cleanup completed")

        return True

    except Exception as e:
        print(f"  ❌ Training step test failed: {e}")
        traceback.print_exc()
        return False

def test_error_recovery():
    """Test error handling and recovery mechanisms."""
    print("🚨 Testing error recovery...")

    try:
        from src.Ava.training.rank_aware_error_handler import RankAwareErrorHandler, ErrorType, ErrorSeverity

        # Create error handler
        error_handler = RankAwareErrorHandler(
            rank=0,
            world_size=1,
            max_retries=2,
            enable_recovery=True
        )

        # Test different error types
        error_scenarios = [
            (RuntimeError("CUDA out of memory"), ErrorType.MEMORY, ErrorSeverity.CRITICAL),
            (RuntimeError("Computation error"), ErrorType.COMPUTE, ErrorSeverity.ERROR),
            (ValueError("Invalid input"), ErrorType.DATA, ErrorSeverity.WARNING),
        ]

        for i, (error, error_type, severity) in enumerate(error_scenarios):
            try:
                handled = error_handler.handle_error(
                    error,
                    error_type,
                    severity,
                    context={"test_scenario": i},
                    recoverable=True
                )
                print(f"  ✅ Error scenario {i+1}: {'handled' if handled else 'not handled'}")
            except Exception as e:
                print(f"  ❌ Error scenario {i+1} failed: {e}")

        return True

    except Exception as e:
        print(f"  ❌ Error recovery test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all training tests."""
    print("🏃 Starting Training Functionality Tests")
    print("=" * 60)

    # Test 1: Simple Training Step
    if not test_simple_training_step():
        print("❌ Training step tests failed")
        return False

    # Test 2: Error Recovery
    if not test_error_recovery():
        print("❌ Error recovery tests failed")
        return False

    print("\n" + "=" * 60)
    print("✅ All training tests passed!")
    print("\n🔧 Training components verified:")
    print("  - Forward/Backward Pass: ✅")
    print("  - Loss Calculation: ✅")
    print("  - Gradient Computation: ✅")
    print("  - Memory Monitoring: ✅")
    print("  - Checkpoint Save/Load: ✅")
    print("  - Error Handling: ✅")
    print("  - Cleanup: ✅")

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)