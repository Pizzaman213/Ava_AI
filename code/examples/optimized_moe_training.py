#!/usr/bin/env python3
"""
Example script showing how to use the optimized MoE training features
"""

import torch
from src.model.moe_transformer import MoEConfig, MoEForCausalLM
from src.training.simple_trainer import SimpleTrainingConfig, SimpleMoETrainer

def main():
    # Create optimized MoE configuration
    config = MoEConfig(
        vocab_size=50257,
        hidden_size=1024,
        num_layers=12,
        num_attention_heads=16,
        num_key_value_heads=4,  # GQA
        intermediate_size=4096,
        num_experts=8,
        num_experts_per_tok=2,
        
        # Enable all optimizations
        use_parallel_experts=True,  # Parallel expert processing
        use_memory_efficient_moe=True,  # Memory-efficient layer with checkpointing
        expert_dropout=0.1,  # 10% expert dropout for regularization
        use_adaptive_capacity=True,  # Dynamic expert capacity
        use_torch_compile=True,  # PyTorch 2.0 compile
        torch_compile_mode="reduce-overhead",
        torch_compile_backend="inductor",
        checkpoint_experts=True,  # Checkpoint each expert
        activation_pool_size=8,  # Larger pool for activation reuse
        
        # MoE loss coefficients
        aux_loss_coef=0.001,  # Reduced for stability
        router_z_loss_coef=0.001,
        router_aux_loss_coef=0.001,
        
        # Other features
        use_mod=True,  # Mixture of Depths
        mod_mode="adaptive",
        mod_skip_fraction=0.2,
        use_flash_attn=True,
        gradient_checkpointing=True,
        mixed_precision="bf16",
    )
    
    # Create model
    model = MoEForCausalLM(config)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params/1e6:.1f}M")
    print(f"Trainable parameters: {trainable_params/1e6:.1f}M")
    
    # Example: Create dummy data
    batch_size = 4
    seq_length = 512
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    dummy_labels = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Forward pass
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(dummy_input, labels=dummy_labels)
        print(f"Loss: {outputs['loss'].item():.4f}")
        print(f"Auxiliary loss: {outputs['aux_loss'].item():.6f}")
    
    # Training configuration with optimizations
    train_config = SimpleTrainingConfig(
        model_config=config,
        batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=3e-4,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_epochs=3,
        mixed_precision="bf16",
        optimizer="fused_adamw",  # Use fused optimizer
        output_dir="./outputs/optimized_moe",
        experiment_name="optimized_moe_experiment",
    )
    
    print("\nOptimized MoE Features Enabled:")
    print("✓ Parallel expert processing")
    print("✓ Memory-efficient MoE layer with gradient checkpointing")
    print("✓ Expert dropout for regularization")
    print("✓ Adaptive expert capacity")
    print("✓ PyTorch 2.0 compile with inductor backend")
    print("✓ Advanced load balancing with smooth L1 loss")
    print("✓ Activation pooling for memory reuse")
    print("✓ Mixed precision training (BF16)")
    print("✓ Fused AdamW optimizer")
    
    print("\nQuick optimization tips:")
    print("1. Increase num_experts_per_tok to 3-4 for better model capacity")
    print("2. Use expert_dropout=0.1-0.2 to prevent overfitting")
    print("3. Enable checkpoint_experts=True for large models to save memory")
    print("4. Set torch_compile_mode='max-autotune' for best performance (slower compilation)")
    print("5. Use adaptive capacity for dynamic workloads")

if __name__ == "__main__":
    main()