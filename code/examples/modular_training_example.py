#!/usr/bin/env python3
"""
Example: Using the New Modular Training Components

This example demonstrates how to use all the newly extracted modular components
from the train.py refactoring for enhanced LLM training.
"""

import torch
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append('/project/code')

# Import the new modular components
from src.Ava.config import TrainingConfigManager, EnhancedTrainingConfig
from src.Ava.training.enhanced_trainer import EnhancedModularTrainer
from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
from transformers import AutoTokenizer


def create_sample_config():
    """Create a sample configuration for demonstration."""
    # This would normally be loaded from a YAML file
    return {
        'model': {
            'hidden_size': 512,
            'num_layers': 6,
            'num_attention_heads': 8,
            'num_experts': 4,
            'vocab_size': 50257
        },
        'training': {
            'batch_size': 8,
            'learning_rate': 5e-5,
            'num_epochs': 1,
            'gradient_accumulation_steps': 2
        }
    }


def main():
    """Demonstrate modular training components."""
    print("🚀 Modular Training Components Example")
    print("=" * 50)

    # 1. Create Configuration Manager
    print("\n📋 1. Setting up Training Configuration")
    config_manager = TrainingConfigManager()

    # Create argument parser and simulate command line args
    parser = config_manager.create_argument_parser()

    # Simulate command line arguments
    args = parser.parse_args([
        '--config', 'configs/gpu/small.yaml',  # This file may not exist, it's just for demo
        '--enable-all-features',
        '--use-rag',
        '--gradient-surgery',
        '--fast-progress'
    ])

    # Parse to structured config
    training_config = config_manager.parse_args_to_config(args)

    # Validate configuration
    validation_messages = config_manager.validate_config(training_config)
    for message in validation_messages:
        print(f"⚠️ {message}")

    # Get feature summary
    feature_summary = config_manager.get_feature_summary(training_config)
    print(f"✅ Enabled features ({feature_summary['total_features']}): {', '.join(feature_summary['enabled_features'])}")
    print(f"⚡ Performance mode: {feature_summary['performance_mode']}")

    # 2. Initialize Model and Tokenizer
    print("\n🤖 2. Initializing Model and Tokenizer")

    # Create model config
    model_config = EnhancedMoEConfig(
        hidden_size=512,
        num_layers=6,
        num_attention_heads=8,
        num_experts=4,
        vocab_size=50257,
        use_moh=training_config.architecture.use_moh,
        use_moa=training_config.architecture.use_moa
    )

    # Initialize model and tokenizer
    model = EnhancedMoEModel(model_config)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"✅ Model initialized: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print(f"✅ Device: {device}")

    # 3. Initialize Enhanced Modular Trainer
    print("\n🎯 3. Initializing Enhanced Modular Trainer")

    trainer = EnhancedModularTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        config=training_config
    )

    # 4. Setup Training
    print("\n⚙️ 4. Setting up Training Components")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    setup_info = trainer.setup_training(optimizer)

    print("✅ Training setup:")
    for key, value in setup_info.items():
        print(f"  - {key}: {value}")

    # 5. Demonstrate Training Step
    print("\n🏃 5. Demonstrating Training Step")

    # Create dummy batch
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # Perform training step
    step_results = trainer.train_step(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        optimizer=optimizer,
        epoch=0,
        batch_idx=0
    )

    print("✅ Training step completed:")
    print(f"  - Loss: {step_results['loss']:.4f}")
    print(f"  - Learning Rate: {step_results['learning_rate']:.2e}")
    print(f"  - Grad Norm: {step_results['grad_norm']:.4f}")
    print(f"  - Forward Time: {step_results['forward_time']*1000:.1f}ms")
    print(f"  - Backward Time: {step_results['backward_time']*1000:.1f}ms")

    # 6. Get Training Statistics
    print("\n📊 6. Training Statistics")

    stats = trainer.get_training_statistics()
    print("✅ Current training statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  - {key}:")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    print(f"    - {sub_key}: {sub_value:.4f}")
                else:
                    print(f"    - {sub_key}: {sub_value}")
        else:
            print(f"  - {key}: {value}")

    # 7. Performance Monitoring
    print("\n⚡ 7. Performance Monitoring")

    perf_summary = trainer.performance_manager.get_performance_summary()
    print("✅ Performance configuration:")
    print(f"  - Mode: {perf_summary['mode']}")
    print(f"  - Active optimizations: {', '.join(perf_summary['active_optimizations'])}")
    print(f"  - Expected benefits: {', '.join(perf_summary['expected_benefits'])}")

    # 8. GPU Memory Statistics
    print("\n💾 8. GPU Memory Management")

    memory_stats = trainer.gpu_manager.get_memory_stats()
    if 'total_memory_gb' in memory_stats:
        print(f"✅ GPU Memory:")
        print(f"  - Total: {memory_stats['total_memory_gb']:.1f}GB")
        print(f"  - Allocated: {memory_stats['allocated_gb']:.2f}GB")
        print(f"  - Reserved: {memory_stats['reserved_gb']:.2f}GB")
        print(f"  - Utilization: {memory_stats['utilization_percent']:.1f}%")

    # 9. Cleanup
    print("\n🧹 9. Cleanup")
    trainer.cleanup()

    print("\n" + "=" * 50)
    print("✅ Modular Training Components Demo Completed!")
    print("\nKey Benefits of Modular Architecture:")
    print("  ✨ Better code organization and maintainability")
    print("  ✨ Reusable components across different training scripts")
    print("  ✨ Easier testing and debugging of individual features")
    print("  ✨ Flexible configuration and feature toggling")
    print("  ✨ Improved separation of concerns")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()