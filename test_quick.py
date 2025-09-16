import sys
sys.path.insert(0, '/project/code/scripts/training')
from qwen_test_2 import MoEConfig, EnhancedMoEModel
import torch

# Create a simple config
config = MoEConfig(
    hidden_size=128,
    num_layers=2,
    num_attention_heads=4,
    num_experts=2,
    num_experts_per_tok=1
)

# Create model
model = EnhancedMoEModel(config)

# Test forward pass
input_ids = torch.randint(0, config.vocab_size, (1, 16))
outputs = model(input_ids)

print("✅ Model forward pass successful!")
print(f"Output keys: {outputs.keys()}")
print(f"Logits shape: {outputs['logits'].shape}")

# Test with labels for loss
outputs_with_loss = model(input_ids, labels=input_ids)
if outputs_with_loss['loss'] is not None:
    print(f"Loss value: {outputs_with_loss['loss'].item():.4f}")
    print("✅ Loss computation successful!")
