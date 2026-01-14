# API Reference

Module-by-module documentation of the Ava framework.

## Package Structure

```
code/src/ava/
├── config/           # Configuration management
├── core/             # Core utilities
├── cuda/             # CUDA utilities
├── data/             # Data loading
├── nn/               # Neural network layers
├── kernels/          # Triton kernels
├── models/           # Model architectures
├── optim/            # Optimizers
├── training/         # Training pipeline
├── optimizations/    # Training optimizations
├── strategies/       # Training strategies
└── eval/             # Evaluation
```

---

## Configuration Module

### `ava.config.training_config`

#### DynamicConfig

Flexible configuration with dot-notation access.

```python
from ava.config.training_config import DynamicConfig

config = DynamicConfig({
    'model': {'hidden_size': 1024},
    'training': {'batch_size': 32}
})

# Access via dot notation
hidden = config.model.hidden_size  # 1024

# Dictionary-style access
batch = config['training']['batch_size']  # 32

# Safe access with default
lr = config.get('training.learning_rate', 0.001)
```

**Methods**:
- `to_dict()` - Convert to dictionary
- `validate()` - Check for circular references
- `validate_required_fields(fields)` - Verify required fields exist
- `validate_schema(schema)` - Validate against schema

#### TrainingConfigManager

Manages configuration loading and validation.

```python
from ava.config.training_config import TrainingConfigManager

manager = TrainingConfigManager()
config = manager.load_yaml_config("config.yaml")

# Validate
errors = manager.validate_dynamic_config(config)

# Get feature summary
summary = manager.get_feature_summary(config)
```

---

## Models Module

### `ava.models.moe`

#### EnhancedMoEConfig

Model configuration dataclass.

```python
from ava.models.moe import EnhancedMoEConfig

config = EnhancedMoEConfig(
    vocab_size=50680,
    hidden_size=1024,
    num_layers=16,
    num_attention_heads=16,
    intermediate_size=4096,
    num_experts=8,
    num_experts_per_token=2,
    router_type='mixtral',
)
```

#### EnhancedMoEModel

Main model class.

```python
from ava.models.moe import EnhancedMoEModel

model = EnhancedMoEModel(config)

# Forward pass
outputs = model(
    input_ids,           # [batch, seq_len]
    attention_mask=mask, # [batch, seq_len]
    labels=labels,       # [batch, seq_len] optional
)

# Outputs
logits = outputs.logits          # [batch, seq_len, vocab]
loss = outputs.loss              # scalar if labels provided
aux_loss = outputs.aux_loss      # MoE auxiliary loss
```

**Key Methods**:
- `forward(input_ids, attention_mask, labels)` - Forward pass
- `generate(input_ids, max_length, ...)` - Text generation
- `get_num_parameters()` - Total parameter count

### `ava.models.moe_layer`

#### SparseMoELayer

Mixture of Experts layer.

```python
from ava.models.moe_layer import SparseMoELayer

layer = SparseMoELayer(
    hidden_size=1024,
    intermediate_size=4096,
    num_experts=8,
    num_experts_per_token=2,
    router_type='mixtral',
)

output, aux_loss = layer(hidden_states)
```

---

## Neural Network Layers

### `ava.models.routing`

#### MixtralRouter

Standard learned gating router.

```python
from ava.models.routing import MixtralRouter

router = MixtralRouter(
    hidden_size=1024,
    num_experts=8,
    num_experts_per_token=2,
)

routing_weights, selected_experts = router(hidden_states)
# routing_weights: [batch, seq, k]
# selected_experts: [batch, seq, k]
```

#### DeepSeekRouter

Hybrid router with shared experts.

```python
from ava.models.routing import DeepSeekRouter

router = DeepSeekRouter(
    hidden_size=1024,
    num_experts=8,
    num_shared_experts=2,
    num_experts_per_token=2,
)
```

### `ava.models.experts`

#### HighPerformanceExpert

Optimized FFN expert.

```python
from ava.models.experts import HighPerformanceExpert

expert = HighPerformanceExpert(
    hidden_size=1024,
    intermediate_size=4096,
    activation='swiglu',
)

output = expert(hidden_states)
```

---

## Training Module

### `ava.training.pipeline`

#### TrainingPipeline

Central orchestrator for training.

```python
from ava.training.pipeline import TrainingPipeline
from ava.training.context import TrainingContext

context = TrainingContext(model=model, device=device)
pipeline = TrainingPipeline(context)

# Register components
pipeline.register('model', model_builder)
pipeline.register('optimizer', optimizer_manager)
pipeline.register('data', data_manager)
pipeline.register('training', training_loop)

# Lifecycle
pipeline.initialize_all()

for epoch in range(num_epochs):
    pipeline.on_epoch_start(epoch)
    # ... training loop ...
    pipeline.on_epoch_end(epoch)

pipeline.cleanup_all()
```

**Lifecycle Methods**:
- `initialize_all()` - Initialize components (FATAL on error)
- `cleanup_all()` - Cleanup in reverse order
- `on_epoch_start(epoch)` - Epoch start hook
- `on_epoch_end(epoch)` - Epoch end hook
- `on_step_start(step)` - Step start hook
- `on_step_end(step, loss)` - Step end hook
- `on_error(error)` - Error handling hook

### `ava.training.context`

#### TrainingContext

Shared state container.

```python
from ava.training.context import TrainingContext

context = TrainingContext(
    model=model,
    device=device,
    config=config,
)

# Access shared state
context.epoch       # Current epoch
context.step        # Current step
context.current_loss  # Latest loss
```

#### TrainingComponent

Base class for pipeline components.

```python
from ava.training.context import TrainingComponent

class MyComponent(TrainingComponent):
    def initialize(self):
        """Called before training."""
        pass

    def cleanup(self):
        """Called after training."""
        pass

    def is_initialized(self) -> bool:
        """Check initialization status."""
        return self._initialized
```

### `ava.training.loop`

#### TrainingLoopManager

Core training loop implementation.

```python
from ava.training.loop import TrainingLoopManager

loop = TrainingLoopManager(context)

# Train one epoch
avg_loss = loop.train_epoch(
    train_loader,
    optimizer,
    scheduler,
)
```

---

## Data Module

### `ava.data.streaming`

#### StreamingDataset

Memory-efficient streaming dataset.

```python
from ava.data.streaming import StreamingDataset

dataset = StreamingDataset(
    data_dir="data/processed",
    buffer_size=50000,
    shuffle=True,
    seed=42,
)

for batch in dataset:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
```

### `ava.data.factory`

#### create_streaming_dataloaders

Factory function for dataloaders.

```python
from ava.data.factory import create_streaming_dataloaders

train_loader, val_loader = create_streaming_dataloaders(
    config=config,
    tokenizer=tokenizer,
    rank=0,
    world_size=1,
)
```

### `ava.data.pretokenized`

#### UltraFastPretokenizedDataset

High-performance pre-tokenized loading with zero-copy Arrow access.

```python
from ava.data.pretokenized import UltraFastPretokenizedDataset

dataset = UltraFastPretokenizedDataset(
    data_dir="data/tokenized",
    split="train",
    max_length=512,
)
```

---

## Optimizations Module

### `ava.optimizations.checkpointing`

#### GradientCheckpointer

Activation checkpointing utilities.

```python
from ava.optimizations.checkpointing import checkpoint

# Wrap expensive forward pass
output = checkpoint(expensive_forward, input, use_reentrant=False)
```

### `ava.optimizations.fp8`

#### FP8Training

FP8 mixed precision support.

```python
from ava.optimizations.fp8 import FP8Config, enable_fp8

fp8_config = FP8Config(enabled=True, format='e4m3')
model = enable_fp8(model, fp8_config)
```

---

## CUDA Utilities

### `ava.cuda.streams`

#### StreamPool

CUDA stream management.

```python
from ava.cuda.streams import StreamPool

pool = StreamPool(num_streams=4)

with pool.get_stream() as stream:
    # Operations on this stream
    tensor.copy_(other)
```

### `ava.cuda.metrics`

#### AsyncMetricsTracker

Non-blocking metrics tracking.

```python
from ava.cuda.metrics import AsyncMetricsTracker

tracker = AsyncMetricsTracker()

tracker.log({'loss': 0.5, 'lr': 1e-4})
metrics = tracker.get_metrics()
```

---

## Optimizer Module

### `ava.optim.lr_managers`

#### AdaptiveLearningRateManager

Dynamic learning rate scheduling.

```python
from ava.optimizations.lr_managers import AdaptiveLearningRateManager

lr_manager = AdaptiveLearningRateManager(
    optimizer,
    warmup_steps=1000,
    total_steps=100000,
    min_lr_ratio=0.1,
)

for step in range(total_steps):
    lr_manager.step()
    current_lr = lr_manager.get_lr()
```

---

## Evaluation Module

### `ava.eval.coherence`

#### CoherenceEvaluator

Text coherence metrics.

```python
from ava.training.coherence import CoherenceEvaluator

evaluator = CoherenceEvaluator(config)

scores = evaluator.evaluate(
    model,
    tokenizer,
    prompts=["Once upon a time"],
    num_samples=10,
)

print(f"Coherence: {scores['coherence_score']:.3f}")
print(f"Perplexity: {scores['perplexity']:.2f}")
print(f"Repetition: {scores['repetition_score']:.3f}")
```

---

## Kernels Module

### `ava.kernels.moe`

Triton kernels for MoE operations.

```python
from ava.cuda.moe_kernels import fused_softmax_topk

# Fused softmax and top-k selection
weights, indices = fused_softmax_topk(logits, k=2)
```

### `ava.kernels.activations`

Fused activation kernels.

```python
from ava.cuda.kernel_activations import fused_swiglu

output = fused_swiglu(gate_proj, up_proj)
```

---

## Utilities

### `ava.core.paths`

Path management utilities.

```python
from ava.core.paths import get_project_root, get_data_dir, get_outputs_dir

root = get_project_root()      # /root/Ava_AI
data = get_data_dir()          # /root/Ava_AI/code/data
outputs = get_outputs_dir()    # /root/Ava_AI/code/outputs
```

### `ava.core.checkpoint`

Checkpoint utilities.

```python
from ava.core.checkpoint import save_checkpoint, load_checkpoint

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    step=step,
    path="checkpoint.pt",
)

checkpoint = load_checkpoint("checkpoint.pt", device="cuda")
```

---

## Type Hints

Common type aliases used throughout:

```python
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor

# Common types
Config = Union[DynamicConfig, Dict[str, Any]]
Device = Union[str, torch.device]
LossDict = Dict[str, float]
MetricsDict = Dict[str, Union[float, int, str]]
```

---

## Error Handling

### ComponentError

Pipeline component errors.

```python
from ava.training.pipeline import ComponentError, ErrorSeverity

try:
    component.initialize()
except Exception as e:
    error = ComponentError(
        component_name='model',
        error=e,
        severity=ErrorSeverity.FATAL,
    )
    raise RuntimeError(f"Pipeline error: {error}")
```

---

## Extending Ava

### Custom Component

```python
from ava.training.context import TrainingComponent, ManagerInterface

class CustomMetrics(TrainingComponent, ManagerInterface):
    def initialize(self):
        self.metrics = {}
        self._initialized = True

    def cleanup(self):
        self.save_metrics()
        self._initialized = False

    def on_step_end(self, step: int, loss: float):
        self.metrics[step] = loss

    def get_status(self) -> Dict[str, Any]:
        return {'num_metrics': len(self.metrics)}
```

### Custom Router

```python
from ava.models.routing import BaseRouter

class CustomRouter(BaseRouter):
    def forward(self, hidden_states: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.gate(hidden_states)
        # Custom routing logic
        weights, indices = self.custom_selection(logits)
        return weights, indices
```
