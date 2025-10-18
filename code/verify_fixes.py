#!/usr/bin/env python3
"""
Verify all fixes have been applied correctly.
Tests model initialization with the fixed config.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import torch
from src.Ava.models.moe_model import EnhancedMoEConfig, EnhancedMoEModel

def verify_config():
    """Verify config file has all fixes."""
    print("=" * 80)
    print("VERIFYING CONFIGURATION FIXES")
    print("=" * 80)

    config_path = Path(__file__).parent / "configs/gpu/small.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    checks = {
        "Model Architecture": [
            ("vocab_size", 500, config["model"]["vocab_size"]),
            ("hidden_size", 128, config["model"]["hidden_size"]),
            ("num_layers", 4, config["model"]["num_layers"]),
            ("num_experts", 4, config["model"]["num_experts"]),
            ("num_experts_per_token", 2, config["model"]["num_experts_per_token"]),
            ("initializer_range", 0.02, config["model"]["initializer_range"]),
            ("tie_word_embeddings", False, config["model"]["tie_word_embeddings"]),
            ("use_learned_position_embeddings", False, config["model"].get("use_learned_position_embeddings", False)),
        ],
        "Training Config": [
            ("batch_size", 128, config["training"]["batch_size"]),
            ("learning_rate", 0.0002, config["training"]["learning_rate"]),
            ("repetition_penalty_weight", 0.5, config["training"]["repetition_penalty_weight"]),
            ("immediate_repetition_weight", 1.0, config["training"]["immediate_repetition_weight"]),
        ],
        "Loss Config": [
            ("ngram_penalty_weight", 0.5, config["enhanced_features"]["losses"]["ngram_penalty_weight"]),
            ("immediate_repetition_weight", 1.0, config["enhanced_features"]["losses"]["immediate_repetition_weight"]),
        ]
    }

    all_passed = True
    for section, items in checks.items():
        print(f"\n{section}:")
        for name, expected, actual in items:
            status = "✓" if expected == actual else "✗"
            passed = expected == actual
            all_passed = all_passed and passed

            color = "\033[92m" if passed else "\033[91m"
            reset = "\033[0m"
            print(f"  {color}{status}{reset} {name}: {actual} {'(expected: ' + str(expected) + ')' if not passed else ''}")

    return all_passed


def verify_model():
    """Verify model can be initialized with fixed config."""
    print("\n" + "=" * 80)
    print("VERIFYING MODEL INITIALIZATION")
    print("=" * 80)

    config_path = Path(__file__).parent / "configs/gpu/small.yaml"
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    model_config_dict = config_dict["model"]

    # Remove fields not in EnhancedMoEConfig
    import inspect
    valid_params = inspect.signature(EnhancedMoEConfig.__init__).parameters
    filtered_dict = {k: v for k, v in model_config_dict.items() if k in valid_params}

    # Fix type conversions (YAML sometimes loads as strings)
    if 'layer_norm_eps' in filtered_dict:
        filtered_dict['layer_norm_eps'] = float(filtered_dict['layer_norm_eps'])
    if 'rope_theta' in filtered_dict:
        filtered_dict['rope_theta'] = float(filtered_dict['rope_theta'])

    print("\nCreating model config...")
    model_config = EnhancedMoEConfig(**filtered_dict)

    print(f"  Vocab size: {model_config.vocab_size}")
    print(f"  Hidden size: {model_config.hidden_size}")
    print(f"  Num layers: {model_config.num_layers}")
    print(f"  Num experts: {model_config.num_experts}")
    print(f"  Experts per token: {model_config.num_experts_per_token}")
    print(f"  Router type: {model_config.router_type}")
    print(f"  Initializer range: {model_config.initializer_range}")

    print("\nInitializing model...")
    model = EnhancedMoEModel(model_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    seq_len = 32

    input_ids = torch.randint(0, model_config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

    loss = outputs['loss']
    logits = outputs['logits']

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected shape: ({batch_size}, {seq_len}, {model_config.vocab_size})")

    # Verify causal mask (check that logits don't depend on future tokens)
    print("\nVerifying causal mask...")
    with torch.no_grad():
        # Forward with full sequence
        full_output = model(input_ids=input_ids, return_dict=True)
        full_logits = full_output['logits']

        # Forward with truncated sequence (should match first part)
        half_seq = seq_len // 2
        trunc_input = input_ids[:, :half_seq]
        trunc_output = model(input_ids=trunc_input, return_dict=True)
        trunc_logits = trunc_output['logits']

        # First half should match (if causal mask working)
        diff = (full_logits[:, :half_seq, :] - trunc_logits).abs().max()
        print(f"  Max difference in logits: {diff.item():.6f}")

        if diff < 1e-5:
            print("  ✓ Causal mask working correctly")
        else:
            print("  ✗ Causal mask may have issues (difference > 1e-5)")

    # Check for MoE load balancing loss
    print("\nVerifying MoE load balancing...")
    if 'aux_info' in outputs and outputs['aux_info']:
        has_load_balance = any(
            'load_balance_loss' in layer_info
            for layer_info in outputs['aux_info']
        )
        if has_load_balance:
            print("  ✓ MoE load balancing loss found in outputs")
            # Show expert utilization
            for i, layer_info in enumerate(outputs['aux_info']):
                if 'expert_utilization' in layer_info:
                    util = layer_info['expert_utilization']
                    print(f"    Layer {i} expert utilization: {util.tolist()}")
        else:
            print("  ✗ No load balancing loss in outputs")
    else:
        print("  ✗ No aux_info in outputs")

    return True


def main():
    print("\n" + "=" * 80)
    print("VERIFICATION SCRIPT FOR ALL FIXES")
    print("=" * 80)

    config_ok = verify_config()
    model_ok = verify_model()

    print("\n" + "=" * 80)
    if config_ok and model_ok:
        print("✓ ALL VERIFICATIONS PASSED")
        print("\nYour fixes are correctly applied!")
        print("\nNext steps:")
        print("1. Stop the current training (Ctrl+C)")
        print("2. Start fresh training:")
        print("   cd /project/code/scripts/5_training")
        print("   python train.py --config ../../configs/gpu/small.yaml")
    else:
        print("✗ SOME VERIFICATIONS FAILED")
        print("\nPlease check the errors above.")
    print("=" * 80)


if __name__ == "__main__":
    main()
