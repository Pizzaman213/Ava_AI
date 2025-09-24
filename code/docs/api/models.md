# Models API Reference

## EnhancedMoEModel

Main model class implementing the MoE++ architecture with DeepSpeed integration and enhanced features.

```python
from src.Ava.models import EnhancedMoEModel, EnhancedMoEConfig
```

### Class: `EnhancedMoEModel`

```python
class EnhancedMoEModel(nn.Module):
    def __init__(self, config: EnhancedMoEConfig)
```

#### Parameters
- `config` (EnhancedMoEConfig): Model configuration object

#### Methods

##### `forward`
```python
def forward(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    return_expert_stats: bool = False
) -> Dict[str, torch.Tensor]
```

**Parameters:**
- `input_ids` (torch.Tensor): Input token IDs, shape `[batch_size, seq_len]`
- `attention_mask` (Optional[torch.Tensor]): Attention mask, shape `[batch_size, seq_len]`
- `labels` (Optional[torch.Tensor]): Target labels for language modeling
- `return_expert_stats` (bool): Whether to return expert utilization statistics

**Returns:**
- Dictionary containing:
  - `logits`: Output logits, shape `[batch_size, seq_len, vocab_size]`
  - `loss`: Language modeling loss (if labels provided)
  - `aux_losses`: Dictionary of auxiliary losses
  - `expert_stats`: Expert statistics (if requested)

**Example:**
```python
config = EnhancedMoEConfig(hidden_size=768, num_layers=12)
model = EnhancedMoEModel(config)

# Forward pass
outputs = model(input_ids, attention_mask=mask, labels=labels)
loss = outputs['loss']
logits = outputs['logits']
```

##### `generate`
```python
def generate(
    input_ids: torch.Tensor,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    do_sample: bool = True
) -> torch.Tensor
```

**Parameters:**
- `input_ids` (torch.Tensor): Starting token IDs
- `max_length` (int): Maximum generation length
- `temperature` (float): Sampling temperature
- `top_k` (int): Top-k filtering parameter
- `top_p` (float): Top-p (nucleus) filtering parameter
- `do_sample` (bool): Whether to sample or use greedy decoding

**Returns:**
- Generated token IDs, shape `[batch_size, generated_length]`

**Example:**
```python
# Generate text
prompt_ids = tokenizer.encode("Once upon a time", return_tensors='pt')
generated_ids = model.generate(prompt_ids, max_length=100, temperature=0.8)
generated_text = tokenizer.decode(generated_ids[0])
```

### Class: `EnhancedMoEConfig`

Configuration class for EnhancedMoEModel.

```python
@dataclass
class EnhancedMoEConfig:
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    num_experts: int = 8
    expert_capacity: float = 1.25
    top_k_experts: int = 2
    ffn_hidden_size: int = 3072
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    expert_dropout: float = 0.1
    use_expert_balancing: bool = True
    use_adaptive_routing: bool = True
    temperature: float = 1.0
    confidence_threshold: float = 0.5
    sinkhorn_iterations: int = 3
    aux_loss_weight: float = 0.01
    layer_norm_epsilon: float = 1e-5
```

#### Attributes

**Model Architecture:**
- `hidden_size` (int): Hidden dimension size
- `num_layers` (int): Number of transformer layers
- `num_attention_heads` (int): Number of attention heads
- `ffn_hidden_size` (int): Feedforward network hidden size
- `vocab_size` (int): Vocabulary size
- `max_position_embeddings` (int): Maximum sequence length

**MoE Configuration:**
- `num_experts` (int): Number of experts per MoE layer
- `expert_capacity` (float): Capacity factor for expert routing
- `top_k_experts` (int): Number of experts to route to per token
- `use_expert_balancing` (bool): Whether to use load balancing
- `use_adaptive_routing` (bool): Whether to use adaptive routing

**Regularization:**
- `dropout_rate` (float): General dropout probability
- `attention_dropout` (float): Attention dropout probability
- `expert_dropout` (float): Expert dropout probability

**Training:**
- `temperature` (float): Temperature for routing softmax
- `confidence_threshold` (float): Confidence threshold for expert selection
- `sinkhorn_iterations` (int): Number of Sinkhorn normalization iterations
- `aux_loss_weight` (float): Weight for auxiliary losses
- `layer_norm_epsilon` (float): Layer normalization epsilon

**Example:**
```python
# Create custom configuration
config = EnhancedMoEConfig(
    hidden_size=1024,
    num_layers=16,
    num_experts=16,
    top_k_experts=4,
    dropout_rate=0.2
)

# Load from dictionary
config_dict = {
    'hidden_size': 768,
    'num_layers': 12,
    'num_experts': 8
}
config = EnhancedMoEConfig(**config_dict)
```

## TransformerBlock

Individual transformer block with MoE layer.

```python
class TransformerBlock(nn.Module):
    def __init__(self, config: EnhancedMoEConfig)
```

### Methods

##### `forward`
```python
def forward(
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_aux_loss: bool = True
) -> Dict[str, torch.Tensor]
```

**Parameters:**
- `hidden_states` (torch.Tensor): Input tensor, shape `[batch_size, seq_len, hidden_size]`
- `attention_mask` (Optional[torch.Tensor]): Attention mask
- `return_aux_loss` (bool): Whether to return auxiliary losses

**Returns:**
- Dictionary containing:
  - `hidden_states`: Output tensor
  - `aux_loss`: Auxiliary loss (if requested)

## Usage Examples

### Basic Training Setup
```python
from src.Ava.models import EnhancedMoEModel, EnhancedMoEConfig
import torch

# Configure model
config = EnhancedMoEConfig(
    hidden_size=768,
    num_layers=12,
    num_experts=8,
    top_k_experts=2
)

# Initialize model
model = EnhancedMoEModel(config)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training step
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

for batch in dataloader:
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs['loss']

    # Add auxiliary losses
    if 'aux_losses' in outputs:
        for aux_loss in outputs['aux_losses'].values():
            loss = loss + 0.01 * aux_loss

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Inference Example
```python
# Load trained model
checkpoint = torch.load('outputs/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate text
with torch.no_grad():
    prompt = "The future of AI is"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    generated = model.generate(
        input_ids,
        max_length=100,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )

    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(output_text)
```

### Expert Statistics
```python
# Get expert utilization statistics
outputs = model(input_ids, return_expert_stats=True)
expert_stats = outputs['expert_stats']

print(f"Expert activations: {expert_stats['expert_activations']}")
print(f"Expert loads: {expert_stats['expert_loads']}")
```

### Custom Configuration
```python
# Load configuration from YAML
import yaml

with open('configs/custom.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

# Extract model configuration
model_config_dict = config_dict.get('model', {})

# Filter to valid fields
from dataclasses import fields
valid_fields = {f.name for f in fields(EnhancedMoEConfig)}
filtered_config = {k: v for k, v in model_config_dict.items() if k in valid_fields}

# Create configuration
config = EnhancedMoEConfig(**filtered_config)

# Initialize model
model = EnhancedMoEModel(config)
```

## Model Saving and Loading

### Saving
```python
# Save model and configuration
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.__dict__,
    'epoch': epoch,
    'loss': loss.item()
}, 'checkpoint.pt')
```

### Loading
```python
# Load checkpoint
checkpoint = torch.load('checkpoint.pt')

# Recreate configuration
config = EnhancedMoEConfig(**checkpoint['config'])

# Initialize and load model
model = EnhancedMoEModel(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Performance Tips

1. **Memory Optimization:**
   ```python
   config = EnhancedMoEConfig(
       gradient_checkpointing=True,  # Enable gradient checkpointing
       expert_capacity=1.0,          # Reduce expert capacity
       top_k_experts=1               # Use fewer experts
   )
   ```

2. **Speed Optimization:**
   ```python
   config = EnhancedMoEConfig(
       use_cache=False,              # Disable KV cache during training
       num_experts=4,                # Use fewer experts
       ffn_hidden_size=2048         # Smaller FFN
   )
   ```

3. **Quality Optimization:**
   ```python
   config = EnhancedMoEConfig(
       num_experts=16,               # More experts
       top_k_experts=4,              # Use more experts per token
       sinkhorn_iterations=5,        # Better load balancing
       aux_loss_weight=0.1          # Stronger auxiliary losses
   )
   ```