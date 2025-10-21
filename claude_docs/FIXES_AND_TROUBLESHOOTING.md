# Fixes & Troubleshooting Guide

Comprehensive guide to common issues and their solutions.

## 🔧 Common Issues & Solutions

### 1. Out of Memory (OOM) Errors

**Symptom**:
```
RuntimeError: CUDA out of memory. Tried to allocate X.00 GiB
```

**Root Causes**:
- Batch size too large for GPU
- Sequence length too long
- Not using gradient checkpointing
- Model too large for GPU

**Solutions**:

Option A - Reduce batch size:
```yaml
training:
  batch_size: 4  # Reduce from 8
```

Option B - Increase gradient accumulation:
```yaml
training:
  gradient_accumulation_steps: 8  # Increase from 4
  batch_size: 4                    # Also reduce batch size
```

Option C - Enable memory optimizations:
```yaml
performance:
  gradient_checkpointing: true
  use_flash_attention: true
  mixed_precision: "bf16"
```

Option D - Reduce sequence length:
```yaml
training:
  max_length: 512  # Reduce from 1024
```

**If still OOM**: Use DeepSpeed ZeRO-2:
```yaml
deepspeed:
  use_deepspeed: true
  zero_stage: 2
```

### 2. Loss Not Decreasing / Training Stalled

**Symptom**:
```
Step 100: Loss = 10.0
Step 500: Loss = 10.0
Step 1000: Loss = 10.0  # Not improving
```

**Root Causes**:
- Learning rate too low (< 0.0001 for pre-training)
- Wrong loss function
- Data not being loaded properly
- Model not learning (gradient flow problem)

**Solutions**:

Check learning rate (CRITICAL):
```yaml
training:
  learning_rate: 0.006  # Pre-training LR should be 0.001-0.01
```

**Common mistake**: Using fine-tuning LR (0.0001) for pre-training
- Pre-training: LR = 0.001-0.01
- Fine-tuning: LR = 0.0001-0.001
- RLHF: LR = 1e-7-1e-6

Check data is loading:
```bash
# Verify data file exists and has content
wc -l data/processed/data.jsonl
# Should show number > 0

# Check data format
head -1 data/processed/data.jsonl
# Should be valid JSON with "text" field
```

Check gradient flow:
```bash
# Test with small batch
python scripts/5_training/train.py \
  --config configs/gpu/small.yaml \
  --batch-size 1 \
  --dry-run
```

### 3. NaN Loss (Training Collapse)

**Symptom**:
```
Step 500: Loss = NaN
Step 501: Loss = NaN  # All NaN from here
```

**Root Causes**:
- Learning rate too high
- Numerical instability
- Extreme gradients (exploding or vanishing)

**Solutions**:

Reduce learning rate immediately:
```yaml
training:
  learning_rate: 0.001  # Reduce from current value
  max_gradient_norm: 0.5  # Tighter clipping
```

Use full precision (no mixed precision):
```yaml
performance:
  mixed_precision: "fp32"  # Avoid bf16/fp16
```

Enable gradient clipping:
```yaml
training:
  max_gradient_norm: 1.0
```

Add small epsilon for stability:
```python
# In training code
optimizer = torch.optim.Adam(
    params, 
    eps=1e-8,
    betas=(0.9, 0.95)
)
```

### 4. Loss Spikes During Training

**Symptom**:
```
Step 1000: Loss = 2.0
Step 2000: Loss = 1.8
Step 3000: Loss = 1.9
Step 4000: Loss = 5.0  # Sudden spike!
Step 5000: Loss = 1.8  # Back to normal
```

**Root Causes**:
- Learning rate schedule issues
- Bad batch in data
- Model instability

**Solutions**:

Smooth learning rate schedule:
```yaml
training:
  warmup_steps: 2000  # Longer warmup
  lr_schedule: "cosine"
```

Reduce learning rate:
```yaml
training:
  learning_rate: 0.003  # Reduce slightly
```

### 5. Model Generating Gibberish

**Symptom**:
```
Prompt: "What is machine learning?"
Output: "たintuitive exponentStudio Aristotle Kah..."  # Nonsense
```

**Root Causes**:
- Model undertrained (low loss value but tokens wrong)
- Tokenizer mismatch
- Sampling temperature too high
- Wrong token IDs in tokenizer

**Solutions**:

Check training progress:
```bash
# Monitor loss at specific steps
tail -100 outputs/runs/latest/logs/training.log | grep "Step 10000"
# Should show: Loss < 5.0 at 10k steps
```

Verify tokenizer:
```bash
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('/project/code/models/tokenizer/enhanced-65536')
print(f'Vocab size: {tokenizer.vocab_size}')
print(f'Sample encode: {tokenizer.encode(\"hello\")}')
"
```

Reduce generation temperature:
```python
# In generation script
output = model.generate(
    input_ids,
    temperature=0.7,  # Reduce from 1.0
    top_p=0.9
)
```

### 6. Repetition in Generated Text

**Symptom**:
```
Generated: "The the the the the cat cat cat cat..."
```

**Root Causes**:
- Model collapse
- Over-training
- Sampling strategy (temperature too low)
- Penalty parameters not set

**Solutions**:

Enable repetition penalty:
```python
output = model.generate(
    input_ids,
    repetition_penalty=1.2,  # Add penalty
    no_repeat_ngram_size=3   # Prevent trigram repeat
)
```

Reduce training epochs:
```yaml
training:
  num_epochs: 1  # Was 3
```

Increase sampling temperature:
```python
output = model.generate(
    input_ids,
    temperature=1.0,  # Increase from 0.7
    top_p=0.95
)
```

### 7. Data Pipeline Errors

**Symptom**:
```
FileNotFoundError: data/processed/data.jsonl not found
```

**Solutions**:

Prepare data:
```bash
python scripts/data_prep/prepare_data.py \
  --input raw_data.txt \
  --output data/processed/data.jsonl

# Verify
python -c "import jsonlines; print(sum(1 for _ in jsonlines.open('data/processed/data.jsonl')))"
```

Check data format:
```bash
# Should have "text" field
head -1 data/processed/data.jsonl | python -m json.tool
```

### 8. GPU Memory Not Being Used

**Symptom**:
```
nvidia-smi shows GPU at 1-2% utilization
Training very slow
```

**Root Causes**:
- Using CPU instead of GPU
- Batch size too small
- Data loading bottleneck

**Solutions**:

Verify GPU is being used:
```bash
# Check if CUDA available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor during training
watch -n 1 nvidia-smi
```

Increase batch size:
```yaml
training:
  batch_size: 32  # Increase if memory allows
```

Enable dataloader workers:
```python
# In training code
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Use multiple workers
    pin_memory=True
)
```

### 9. Slow Training Speed

**Symptom**:
```
Throughput: ~100 tokens/sec (should be 2000+)
```

**Solutions**:

Enable Flash Attention:
```yaml
performance:
  use_flash_attention: true
```

Enable torch.compile:
```yaml
performance:
  use_torch_compile: true
  compile_mode: "reduce-overhead"
```

Increase batch size:
```yaml
training:
  batch_size: 32  # Increase for better GPU utilization
```

Use mixed precision:
```yaml
performance:
  mixed_precision: "bf16"
```

### 10. Configuration Loading Errors

**Symptom**:
```
KeyError: 'learning_rate' not found in config
```

**Solutions**:

Verify config file exists:
```bash
ls -la configs/gpu/small.yaml
# Should exist
```

Validate YAML syntax:
```bash
python -c "import yaml; yaml.safe_load(open('configs/gpu/small.yaml'))"
# Should print nothing (success)
```

Check required fields:
```yaml
# Must have these
model:
  vocab_size: 65536
  
training:
  learning_rate: 0.006
  batch_size: 8
```

## 🆘 Emergency Procedures

### If Training Crashes

**Step 1**: Check error message
```bash
tail -100 outputs/runs/latest/logs/training.log
```

**Step 2**: Match to solution above

**Step 3**: Resume training
```bash
python scripts/5_training/train.py \
  --config configs/gpu/small.yaml \
  --resume outputs/runs/latest/checkpoint
```

### If Model Quality Is Bad

**Step 1**: Check loss is reasonable
```bash
# Loss should decrease over time
grep "Loss" outputs/runs/latest/logs/training.log | tail -20
```

**Step 2**: Check generation at different steps
```bash
for ckpt in outputs/runs/latest/checkpoints/checkpoint_step_{10000,50000,100000}; do
  python scripts/evaluation/test_generation.py --checkpoint $ckpt
done
```

**Step 3**: If all losses high: check data quality
```bash
# Verify data file
python -c "
import jsonlines
with jsonlines.open('data/processed/data.jsonl') as f:
    for i, obj in enumerate(f):
        print(f'Example {i}: {obj[\"text\"][:50]}...')
        if i >= 5: break
"
```

### If Stuck / Not Sure

**Checklist**:
- [ ] GPU is working (`nvidia-smi`)
- [ ] Data file exists and has content
- [ ] Config file is valid YAML
- [ ] Learning rate is reasonable (0.001-0.01 for pre-training)
- [ ] Batch size fits in GPU memory
- [ ] Loss is not NaN
- [ ] Loss is decreasing over time

## 📊 Expected Training Curves

### Healthy Training
```
Step 1,000:    Loss = 6.5  ← Learning started ✓
Step 5,000:    Loss = 4.5  ← Steady progress ✓
Step 10,000:   Loss = 3.5  ← Good convergence ✓
Step 50,000:   Loss = 2.0  ← Excellent quality ✓
Step 100,000:  Loss = 1.8  ← Final result ✓
```

### Problem: LR Too Low
```
Step 1,000:    Loss = 10.0  ← No learning ✗
Step 5,000:    Loss = 9.9   ← Barely changed ✗
Step 10,000:   Loss = 9.8   ← Still no learning ✗
→ Solution: Increase learning rate to 0.006-0.01
```

### Problem: LR Too High
```
Step 100:      Loss = 10.0
Step 500:      Loss = 5.0
Step 1,000:    Loss = NaN   ← Exploded ✗
→ Solution: Reduce learning rate to 0.001
```

## 🔍 Diagnostic Scripts

### Check System Status
```bash
# All at once
python -c "
import torch
import torch.cuda as cuda
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {cuda.is_available()}')
print(f'CUDA device: {cuda.get_device_name(0) if cuda.is_available() else \"N/A\"}')
print(f'CUDA capability: {cuda.get_device_capability(0) if cuda.is_available() else \"N/A\"}')
print(f'CUDA memory: {cuda.get_device_properties(0).total_memory / 1e9:.1f}GB' if cuda.is_available() else 'N/A')
"
```

### Check Data Quality
```bash
python scripts/data_prep/validate_data.py --input data/processed/data.jsonl
```

### Test Model Loading
```bash
python -c "
from src.Ava.models import EnhancedMoE
from src.Ava.config import load_config
config = load_config('configs/gpu/small.yaml')
model = EnhancedMoE(config)
print(f'Model loaded successfully')
print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
"
```

### Check GPU Memory
```bash
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader
```

---

**Status**: ✅ Complete
**Last Updated**: 2025-10-21

For more information, see TRAINING_AND_CONFIGURATION.md for setup or ARCHITECTURE_AND_FEATURES.md for deep dives.
