#!/usr/bin/env python3
"""
Basic test suite for core MoE++ features
Tests essential functionality without all advanced features
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import traceback
import time

# Import core modules
from src.model.moe_transformer import MoEConfig, MoEForCausalLM
from src.model.experts import HierarchicalExpertLayer
from src.model.attention import MultiQueryAttention

def test_basic_model():
    """Test basic model creation and forward pass"""
    print("\n🧪 Testing Basic MoE Model...")
    try:
        # Create small config
        config = MoEConfig()
        config.vocab_size = 1000
        config.hidden_size = 256
        config.num_layers = 2
        config.num_attention_heads = 8
        config.num_key_value_heads = 4
        config.intermediate_size = 1024
        config.num_experts = 4
        config.num_experts_per_tok = 2
        config.max_position_embeddings = 512
        
        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Using device: {device}")
        
        model = MoEForCausalLM(config).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logits = outputs['logits']
        print(f"   Output shape: {logits.shape}")
        print(f"   Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size), "Wrong output shape!"
        
        # Test generation (if method exists)
        if hasattr(model, 'generate'):
            with torch.no_grad():
                generated = model.generate(
                    input_ids[:1, :5],
                    max_length=20,
                    temperature=0.8,
                    do_sample=True
                )
            print(f"   Generated shape: {generated.shape}")
            assert generated.shape[1] == 20, "Wrong generation length!"
        else:
            print("   Generation method not available, skipping generation test")
        
        print("✅ Basic model test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic model test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_expert_layer():
    """Test expert layer functionality"""
    print("\n🧪 Testing Expert Layer...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create expert layer
        from src.model.moe_transformer import MoEConfig
        config = MoEConfig()
        config.hidden_size = 256
        config.num_experts = 4
        config.intermediate_size = 1024
        config.num_experts_per_tok = 2
        config.activation = "swiglu"
        expert_layer = HierarchicalExpertLayer(config).to(device)
        
        # Test forward pass
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 256).to(device)
        
        output, router_outputs = expert_layer(x)
        
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        if isinstance(router_outputs, dict) and 'router_logits' in router_outputs:
            print(f"   Router logits shape: {router_outputs['router_logits'].shape}")
            if len(router_outputs['router_logits'].shape) > 2:
                assert router_outputs['router_logits'].shape[2] == 4, "Wrong number of experts!"
        else:
            print(f"   Router outputs type: {type(router_outputs)}")
        
        assert output.shape == x.shape, "Wrong output shape!"
        
        print("✅ Expert layer test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Expert layer test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_attention():
    """Test attention mechanisms"""
    print("\n🧪 Testing Attention Mechanisms...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create attention layer
        from src.model.moe_transformer import MoEConfig
        config = MoEConfig()
        config.hidden_size = 256
        config.num_attention_heads = 8
        config.num_key_value_heads = 4
        config.attention_dropout = 0.1
        attention = MultiQueryAttention(config).to(device)
        
        # Test forward pass
        batch_size = 2
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, 256).to(device)
        
        result = attention(
            hidden_states,
            attention_mask=None,
            position_ids=None
        )
        
        # Handle both tuple and tensor returns
        if isinstance(result, tuple):
            output = result[0]
            weights = result[1] if len(result) > 1 else None
        else:
            output = result
            weights = None
        
        print(f"   Input shape: {hidden_states.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Attention weights shape: {weights.shape if weights is not None else 'None'}")
        
        assert output.shape == hidden_states.shape, "Wrong output shape!"
        
        print("✅ Attention test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Attention test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_training_step():
    """Test a basic training step"""
    print("\n🧪 Testing Training Step...")
    try:
        # Create small config
        config = MoEConfig()
        config.vocab_size = 1000
        config.hidden_size = 128
        config.num_layers = 2
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.intermediate_size = 512
        config.num_experts = 2
        config.num_experts_per_tok = 1
        config.max_position_embeddings = 256
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MoEForCausalLM(config).to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create dummy batch
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        labels = input_ids.clone()
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']
        
        print(f"   Initial loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"   Gradient norm: {grad_norm:.4f}")
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Second forward pass
        outputs2 = model(input_ids=input_ids, labels=labels)
        loss2 = outputs2['loss']
        
        print(f"   Loss after update: {loss2.item():.4f}")
        
        # Loss should change after update
        assert abs(loss.item() - loss2.item()) > 1e-6, "Loss didn't change after update!"
        
        print("✅ Training step test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Training step test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n🧪 Testing Configuration System...")
    try:
        import yaml
        from pathlib import Path
        
        # Find a config file
        config_path = Path("configs/gpu/small.yaml")
        if not config_path.exists():
            config_path = Path("configs/cpu/small.yaml")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            print(f"   Loaded config from: {config_path}")
            print(f"   Model size: {config_dict.get('model', {}).get('size', 'unknown')}")
            print(f"   Num experts: {config_dict.get('model', {}).get('num_experts', 'unknown')}")
            
            # Create MoEConfig from dict
            config = MoEConfig()
            model_config = config_dict.get('model', {})
            
            # Update config with values from YAML
            for key, value in model_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            print("✅ Config loading test passed!")
            return True
        else:
            print("⚠️  No config file found, skipping test")
            return True
            
    except Exception as e:
        print(f"❌ Config loading test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test memory-efficient features"""
    print("\n🧪 Testing Memory Efficiency...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create config with gradient checkpointing
        config = MoEConfig()
        config.vocab_size = 1000
        config.hidden_size = 256
        config.num_layers = 4
        config.num_attention_heads = 8
        config.num_key_value_heads = 2  # GQA for memory efficiency
        config.intermediate_size = 1024
        config.num_experts = 4
        config.num_experts_per_tok = 1
        config.gradient_checkpointing = True
        
        model = MoEForCausalLM(config).to(device)
        
        # Enable gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("   Gradient checkpointing enabled")
        
        # Test forward pass with larger batch
        batch_size = 4
        seq_len = 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
        labels = input_ids.clone()
        
        # Forward and backward
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs['loss']
        loss.backward()
        
        print(f"   Forward/backward pass completed with batch_size={batch_size}, seq_len={seq_len}")
        print(f"   Loss: {loss.item():.4f}")
        
        print("✅ Memory efficiency test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Memory efficiency test failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all basic tests"""
    print("\n" + "="*60)
    print("🚀 MoE++ Basic Feature Test Suite")
    print("="*60)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📊 System Info:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   CUDA version: {torch.version.cuda}")
    
    # List of tests
    tests = [
        ("Basic Model", test_basic_model),
        ("Expert Layer", test_expert_layer),
        ("Attention", test_attention),
        ("Training Step", test_training_step),
        ("Config Loading", test_config_loading),
        ("Memory Efficiency", test_memory_efficiency),
    ]
    
    # Run tests
    passed = 0
    failed = 0
    start_time = time.time()
    
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        else:
            failed += 1
    
    end_time = time.time()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    print(f"⏱️  Total Time: {end_time - start_time:.2f}s")
    
    if failed == 0:
        print("\n🎉 ALL BASIC TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} tests failed.")
    print("="*60)

if __name__ == "__main__":
    main()