# Troubleshooting Guide

Common issues and solutions when using Ava.

## Quick Diagnosis

```bash
# Check GPU status
nvidia-smi

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"

# Verify installation
python code/scripts/check_dependencies.py
```

---

## Memory Issues

### Out of Memory (OOM)

**Symptoms**:
```
CUDA out of memory. Tried to allocate X GiB
RuntimeError: CUDA error: out of memory
```

**Solutions**:

1. **Reduce batch size**
```yaml
training:
  batch_size: 16  # Reduce from 32
  gradient_accumulation_steps: 4  # Compensate
```

2. **Enable gradient checkpointing**
```yaml
model:
  gradient_checkpointing: true
```

3. **Use mixed precision**
```yaml
hardware:
  mixed_precision: 'bf16'
```

5. **Reduce sequence length**
```yaml
data:
  max_length: 256  # Reduce from 512
```

### Memory Leak

**Symptoms**:
- GPU memory grows over time
- Eventually OOM after many steps

**Solutions**:

1. **Clear cache periodically**
```python
import torch
torch.cuda.empty_cache()
```

2. **Disable persistent workers**
```yaml
data:
  persistent_workers: false
```

3. **Check for tensor accumulation**
```python
# Don't accumulate tensors
total_loss += loss.item()  # Use .item()
# Not: total_loss += loss
```

---

## Training Issues

### Loss Not Decreasing

**Symptoms**:
- Loss stuck at high value
- No improvement after many steps

**Solutions**:

1. **Check learning rate**
```bash
python code/scripts/5_training/train_pipeline.py \
    --config config.yaml \
    --run-lr-finder
```

2. **Increase warmup**
```yaml
training:
  warmup_steps: 2000  # Increase from 1000
```

3. **Verify data loading**
```python
# Debug: print first batch
for batch in train_loader:
    print(batch['input_ids'].shape)
    print(batch['input_ids'][:2])
    break
```

4. **Check for all-padding sequences**
```python
# Ensure sequences have content
mask_sum = batch['attention_mask'].sum(dim=1)
assert (mask_sum > 0).all(), "Found all-padding sequences"
```

### Loss Exploding (NaN/Inf)

**Symptoms**:
```
Loss: nan or inf
RuntimeError: Loss is nan
```

**Solutions**:

1. **Lower learning rate**
```yaml
training:
  learning_rate: 0.0001  # Reduce from 0.001
```

2. **Enable gradient clipping**
```yaml
training:
  max_gradient_norm: 0.5  # Stricter clipping
```

3. **Check for numerical issues**
```python
# Add to training loop
if torch.isnan(loss) or torch.isinf(loss):
    print(f"Bad loss at step {step}")
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN grad in {name}")
```

4. **Use stable softmax**
```yaml
model:
  router_z_loss_coef: 0.001  # Add z-loss
```

### Training Too Slow

**Symptoms**:
- Low GPU utilization
- Slow iterations

**Solutions**:

1. **Enable Flash Attention**
```yaml
model:
  use_flash_attention: true
```

2. **Increase data workers**
```yaml
data:
  num_workers: 8
  prefetch_factor: 4
  persistent_workers: true
```

3. **Enable TF32**
```yaml
performance:
  enable_tf32: true
  enable_cudnn_benchmark: true
```

4. **Use Triton kernels**
```yaml
model:
  use_triton_kernels: true
```

5. **Check data loading bottleneck**
```yaml
dev_log:
  enabled: true
  show_step_breakdown: true
```

---

## Data Issues

### Data Not Found

**Symptoms**:
```
FileNotFoundError: Data directory not found
No Arrow files found in directory
```

**Solutions**:

1. **Check path**
```bash
ls -la code/data/processed/*.arrow
```

2. **Verify config path**
```yaml
data:
  data_dir: "code/data/processed"  # Absolute or relative to project root
```

3. **Download data**
```bash
python code/scripts/1_data_download/unified_download.py
```

### Tokenizer Mismatch

**Symptoms**:
- Garbled output
- `IndexError: index out of range`

**Solutions**:

1. **Verify tokenizer path**
```yaml
data:
  tokenizer_name: "code/data/Ava_Ai/tokenizer"
```

2. **Check vocab size match**
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")
print(f"Vocab size: {len(tokenizer)}")
# Should match model.vocab_size
```

3. **Re-tokenize data**
```bash
python code/scripts/1_data_download/build_pretokenized_data.py \
    --tokenizer code/data/Ava_Ai/tokenizer
```

### Empty Batches

**Symptoms**:
```
RuntimeError: Expected non-empty tensor
Zero-size tensor
```

**Solutions**:

1. **Check data file**
```python
import pyarrow as pa
table = pa.ipc.open_file(pa.memory_map("data.arrow", 'r')).read_all()
print(f"Rows: {len(table)}")
print(f"Columns: {table.column_names}")
```

2. **Add drop_last**
```yaml
data:
  dataloader_drop_last: true
```

---

## Distributed Training Issues

### NCCL Timeout

**Symptoms**:
```
NCCL timeout
RuntimeError: NCCL watchdog timeout
```

**Solutions**:

1. **Increase timeout**
```bash
export NCCL_TIMEOUT=1800
```

2. **Check network**
```bash
# Test connectivity between nodes
ping other-node
```

3. **Verify NCCL settings**
```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
```

### Rank Mismatch

**Symptoms**:
```
All processes must use same world size
Rank out of bounds
```

**Solutions**:

1. **Consistent launch**
```bash
# All nodes must use same nnodes
torchrun --nproc_per_node=4 --nnodes=2 ...
```

2. **Check environment**
```python
import os
print(f"RANK: {os.environ.get('RANK')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
```

### Hanging at Barrier

**Symptoms**:
- Training hangs
- No progress on one or more GPUs

**Solutions**:

1. **Check all processes started**
```bash
ps aux | grep python
```

2. **Verify master address**
```bash
# All nodes must reach master
ping $MASTER_ADDR
```

3. **Debug with NCCL**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

---

## Generation Issues

### Repetitive Output

**Symptoms**:
- Model repeats same phrase
- Degenerate output

**Solutions**:

1. **Add repetition penalty**
```yaml
generation:
  repetition_penalty: 1.2
  no_repeat_ngram_size: 3
```

2. **Adjust temperature**
```yaml
generation:
  temperature: 0.8
  top_p: 0.9
```

3. **Check training quality**
```bash
python code/scripts/7_generation/generate.py \
    --prompt "Test prompt" \
    --checkpoint path/to/best_model.pt
```

### Incoherent Output

**Symptoms**:
- Nonsense text
- Grammatically incorrect

**Solutions**:

1. **Verify checkpoint loaded**
```python
checkpoint = torch.load("model.pt")
print(f"Step: {checkpoint.get('step')}")
print(f"Loss: {checkpoint.get('best_loss')}")
```

2. **Check tokenizer**
```python
text = "Hello world"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
assert text == decoded, "Tokenizer roundtrip failed"
```

3. **Train longer**
```yaml
training:
  epochs: 10  # Increase epochs
```

---

## Configuration Issues

### Config Not Loading

**Symptoms**:
```
FileNotFoundError: Config file not found
yaml.scanner.ScannerError
```

**Solutions**:

1. **Check path**
```bash
ls -la code/configs/moe/large.yaml
```

2. **Validate YAML**
```bash
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

3. **Check indentation**
```yaml
# Correct (2 spaces)
model:
  hidden_size: 1024

# Wrong (mixed tabs/spaces)
model:
	hidden_size: 1024
```

### Missing Config Key

**Symptoms**:
```
AttributeError: 'NoneType' object has no attribute
KeyError: 'learning_rate'
```

**Solutions**:

1. **Use defaults**
```python
lr = config.training.learning_rate or 0.001
```

2. **Enable strict mode**
```python
DynamicConfig.enable_strict_mode()
# Will raise error for missing keys
```

---

## Checkpoint Issues

### Cannot Load Checkpoint

**Symptoms**:
```
RuntimeError: Error(s) in loading state_dict
Missing keys / Unexpected keys
```

**Solutions**:

1. **Check model architecture**
```python
# Compare keys
model_keys = set(model.state_dict().keys())
ckpt_keys = set(checkpoint['model_state_dict'].keys())
print(f"Missing: {model_keys - ckpt_keys}")
print(f"Unexpected: {ckpt_keys - model_keys}")
```

2. **Load partial checkpoint**
```python
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

3. **Convert checkpoint**
```python
# Remove 'module.' prefix from DDP
new_state = {k.replace('module.', ''): v
             for k, v in checkpoint['model_state_dict'].items()}
model.load_state_dict(new_state)
```

### Checkpoint Corruption

**Symptoms**:
```
pickle.UnpicklingError
RuntimeError: storage offset
```

**Solutions**:

1. **Load with weights_only**
```python
checkpoint = torch.load("model.pt", weights_only=True)
```

2. **Use backup checkpoint**
```bash
ls -la outputs/checkpoints/
# Use previous checkpoint
```

3. **Enable checkpoint validation**
```yaml
logging:
  log_checkpoint_validation: true
```

---

## Environment Issues

### CUDA Not Available

**Symptoms**:
```
RuntimeError: CUDA is not available
torch.cuda.is_available() returns False
```

**Solutions**:

1. **Check NVIDIA driver**
```bash
nvidia-smi
```

2. **Check PyTorch CUDA**
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

3. **Reinstall PyTorch**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Package Version Conflicts

**Symptoms**:
```
ImportError: cannot import name
ModuleNotFoundError
```

**Solutions**:

1. **Check versions**
```bash
pip list | grep -E "torch|transformers|flash"
```

2. **Reinstall requirements**
```bash
pip install -r requirements.txt --force-reinstall
```

---

## Getting Help

### Debug Mode

```yaml
logging:
  verbosity: 'debug'

dev_log:
  enabled: true
  show_step_breakdown: true
```

### Collect Diagnostics

```bash
# System info
nvidia-smi
python --version
pip list

# Training log
python code/scripts/5_training/train_pipeline.py \
    --config config.yaml 2>&1 | tee training.log
```

### Report Issue

Include:
1. Error message and full traceback
2. Configuration file (sanitized)
3. System info (GPU, CUDA version)
4. Steps to reproduce

## Next Steps

- [Training Guide](./03_TRAINING_GUIDE.md) - Training best practices
- [Configuration Reference](./04_CONFIGURATION.md) - All parameters
- [Performance Tuning](./10_PERFORMANCE.md) - Optimization guide
