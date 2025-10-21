# Validation & Testing Guide

Guide to validating training, monitoring progress, and testing model quality.

## 📊 Key Metrics to Monitor

### Training Metrics

**Primary Metric: Loss**
- Most important indicator of training progress
- Should decrease smoothly over time
- Plateau indicates convergence

```
Healthy progression:
Step 1,000:    Loss ≈ 6.0-7.0
Step 10,000:   Loss ≈ 3.5-4.0
Step 50,000:   Loss ≈ 2.0-2.5
Step 100,000:  Loss ≈ 1.8-2.0
```

**Gradient Statistics**:
- Mean absolute gradient (should be reasonable)
- Gradient norm (watch for spikes)
- Gradient variance (should be stable)

**Learning Rate**:
- Follows schedule (warmup then decay)
- Should match configuration
- Use warmup for stability

**Batch Statistics**:
- Batch size matches config
- Sequence lengths reasonable
- No empty batches

### Validation Metrics

**Perplexity**:
```
Perplexity = exp(Loss)

Step 10,000:  PPL ≈ 50-100   ← Still learning
Step 50,000:  PPL ≈ 10-20    ← Good convergence
Step 100,000: PPL ≈ 5-10     ← Excellent
```

**Generation Quality**:
- Coherence (0-100%): Does text make sense?
- Diversity (0-1): How varied is the output?
- Fluency (0-100%): How natural is the language?
- Repetition (0-1): How much text repeats?

## 🎯 Validation During Training

### Checkpointing Strategy

**Every N steps**:
```yaml
training:
  eval_steps: 1000  # Evaluate every 1000 steps
  save_steps: 1000  # Save checkpoint every 1000 steps
```

**Keeps last N checkpoints**:
```yaml
training:
  keep_last_checkpoints: 3
```

### Checkpoint Structure
```
outputs/runs/run_20251021_120000_abcdef/
├── checkpoints/
│   ├── checkpoint_step_1000.pt
│   ├── checkpoint_step_2000.pt
│   └── latest_model.pt (symlink to best)
├── logs/
│   ├── training.log
│   ├── metrics.json
│   └── config.yaml
└── wandb/
    └── wandb run files
```

## 📈 Monitoring with W&B

### Setup W&B
```bash
# Login
wandb login

# Verify (optional)
python -c "import wandb; print(f'Logged in as: {wandb.api.default_entity}')"
```

### During Training
- Loss decreases over time
- Learning rate follows schedule
- Throughput (tokens/sec) is consistent
- GPU memory stable

### Important Dashboards
1. **Metrics Tab**: Loss, learning rate, throughput
2. **System Tab**: GPU memory, GPU utilization, CPU
3. **Console Tab**: Training logs and errors
4. **Files Tab**: Saved checkpoints and configs

### Interpreting W&B Plots

**Good Loss Curve**:
- Monotonic decrease (always down or flat)
- Smooth without erratic spikes
- Plateaus at low value eventually

**Bad Loss Curve**:
- Increasing loss (training not working)
- Wild oscillations (LR too high)
- NaN from some point (numerical error)

## 🧪 Testing Model Quality

### Quick Generation Test
```bash
python scripts/evaluation/test_generation.py \
  --checkpoint outputs/runs/latest \
  --prompt "What is machine learning?" \
  --num-samples 5
```

### Test at Multiple Checkpoints
```bash
# Test every 10k steps
for step in 10000 20000 50000 100000; do
  echo "Testing checkpoint step $step..."
  python scripts/evaluation/test_generation.py \
    --checkpoint outputs/runs/latest/checkpoints/checkpoint_step_$step \
    --prompt "Explain AI" \
    --num-samples 3
done
```

### Batch Generation Test
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("outputs/runs/latest")
tokenizer = AutoTokenizer.from_pretrained("/project/code/models/tokenizer/enhanced-65536")

prompts = [
    "What is AI?",
    "Tell me about machine learning",
    "Explain neural networks"
]

for prompt in prompts:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100)
    text = tokenizer.decode(output[0])
    print(f"Prompt: {prompt}")
    print(f"Output: {text}\n")
```

### Coherence Evaluation

Measure how coherent generated text is:
```python
def measure_coherence(text):
    """Simple coherence heuristic"""
    # Check for complete sentences
    sentences = text.split('.')
    complete_sentences = sum(1 for s in sentences if len(s.split()) > 3)
    
    # Check for repetition
    words = text.split()
    unique_words = len(set(words))
    repetition = 1 - (unique_words / len(words)) if words else 0
    
    # Score: 0-100
    score = 50 + (complete_sentences * 10) - (repetition * 30)
    return max(0, min(100, score))

# Test
texts = [generated_text for _ in range(10)]
scores = [measure_coherence(text) for text in texts]
print(f"Average coherence: {sum(scores) / len(scores):.1f}/100")
```

## 📊 Test Results Recording

### Create Test Report
```bash
# After training reaches milestone (e.g., step 100k)
python -c "
import json
from datetime import datetime

report = {
    'date': datetime.now().isoformat(),
    'checkpoint': 'outputs/runs/latest',
    'training_steps': 100000,
    'loss': 1.85,  # From metrics.json
    'generation_samples': 10,
    'coherence_score': 82.5,
    'notes': 'Model producing coherent, well-structured text'
}

with open('reports/test_report_100k.json', 'w') as f:
    json.dump(report, f, indent=2)

print('Report saved')
"
```

## ✅ Validation Checklist

Before considering training complete:

- [ ] Loss converged (no longer decreasing significantly)
- [ ] Generation produces coherent text
- [ ] No obvious repetition problems
- [ ] Grammatical quality is good
- [ ] Diversity in generated outputs
- [ ] Checkpoint loads without errors
- [ ] Tokenizer matches model vocabulary
- [ ] At least 100k training steps completed

## 📊 Expected Progression

### Step 1,000
```
Loss: 6.0-7.0
Generation: "the cat the dog the house..."  (Repetitive, word choices)
Quality: Poor - just learning tokens
```

### Step 10,000
```
Loss: 3.5-4.0
Generation: "the quick brown fox jumps over..."
Quality: Fair - word order improving, some coherence
```

### Step 50,000
```
Loss: 2.0-2.5
Generation: "The quick brown fox jumps over the lazy dog..."
Quality: Good - natural language, proper grammar
```

### Step 100,000
```
Loss: 1.8-2.0
Generation: "Artificial intelligence is rapidly transforming..."
Quality: Excellent - coherent, complex ideas
```

## 🔍 Comparative Testing

### Before vs After Optimization

Compare same model with different configs:

```bash
# Test unoptimized model
python scripts/evaluation/test_generation.py \
  --checkpoint outputs/runs/baseline \
  --num-samples 10 \
  > results_baseline.txt

# Test optimized model
python scripts/evaluation/test_generation.py \
  --checkpoint outputs/runs/optimized \
  --num-samples 10 \
  > results_optimized.txt

# Compare
diff -u results_baseline.txt results_optimized.txt
```

### Benchmark Different Model Sizes

```bash
# Test each model size
for config in tiny small base large; do
  echo "Testing $config model..."
  python scripts/5_training/train.py \
    --config configs/gpu/$config.yaml \
    --steps 1000 \
    --benchmark
done
```

## 📈 Performance Metrics

### Throughput
```
tokens/sec = (batch_size × seq_length) / seconds_per_step

RTX 3090 Ti, small model:
- Baseline: ~2,000 tokens/sec
- With Flash Attention: ~3,500 tokens/sec
- With torch.compile: ~4,500 tokens/sec
- With both: ~5,500 tokens/sec
```

### Model FLOPS Utilization (MFU)
```
MFU = actual_throughput / theoretical_peak

RTX 3090 Ti (40 TFLOPS for bfloat16):
- 2,000 tokens/sec = 5% MFU
- 4,000 tokens/sec = 10% MFU
- 8,000 tokens/sec = 20% MFU
Goal: 30-40% MFU is good
```

### Memory Usage
```
Expected for small model (233M):
- Model weights: ~900 MB
- Optimizer state: ~1.8 GB (2x for Adam)
- Activations: ~2-4 GB (depends on batch size)
- Total: ~5-7 GB

Should fit in 8GB+ GPU
```

## 🧪 Regression Testing

Create test suite to catch regressions:

```python
import torch
from src.Ava.models import EnhancedMoE

# Test 1: Model loads
def test_model_loads():
    model = EnhancedMoE(config)
    assert model is not None

# Test 2: Forward pass works
def test_forward_pass():
    model = EnhancedMoE(config)
    input_ids = torch.randint(0, 65536, (2, 128))
    output = model(input_ids)
    assert output.shape == (2, 128, 65536)

# Test 3: Loss is reasonable
def test_loss_computation():
    model = EnhancedMoE(config)
    input_ids = torch.randint(0, 65536, (2, 128))
    labels = torch.randint(0, 65536, (2, 128))
    output = model(input_ids, labels=labels)
    assert 0 < output.loss < 20  # Reasonable range

# Test 4: Gradient flow
def test_gradient_flow():
    model = EnhancedMoE(config)
    input_ids = torch.randint(0, 65536, (2, 128))
    labels = torch.randint(0, 65536, (2, 128))
    output = model(input_ids, labels=labels)
    output.loss.backward()
    for param in model.parameters():
        assert param.grad is not None

if __name__ == "__main__":
    test_model_loads()
    test_forward_pass()
    test_loss_computation()
    test_gradient_flow()
    print("All tests passed!")
```

## 🎯 Validation Examples

### Example 1: Daily Test
```bash
#!/bin/bash
# daily_test.sh

# Run daily at training checkpoints
for checkpoint in outputs/runs/*/checkpoints/checkpoint_step_{10000,50000,100000}; do
    echo "Testing $(basename $checkpoint)..."
    python scripts/evaluation/test_generation.py \
        --checkpoint $checkpoint \
        --prompt "Explain artificial intelligence" \
        --num-samples 5 | tee "logs/test_$(basename $checkpoint).log"
done
```

### Example 2: Quality Dashboard
```python
import json
import matplotlib.pyplot as plt

# Load metrics
with open('outputs/runs/latest/logs/metrics.json') as f:
    metrics = [json.loads(line) for line in f]

# Extract data
steps = [m['step'] for m in metrics]
losses = [m['loss'] for m in metrics]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(steps, losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.grid(True)
plt.savefig('training_progress.png')
print("Saved to training_progress.png")
```

### Example 3: Continuous Testing
```bash
#!/bin/bash
# Monitor training and test every hour

while true; do
    # Get latest checkpoint
    latest=$(ls -td outputs/runs/*/checkpoints/* | head -1)
    
    # Test it
    python scripts/evaluation/test_generation.py \
        --checkpoint $latest \
        --prompt "test" \
        >> "logs/continuous_test.log"
    
    # Wait 1 hour
    sleep 3600
done
```

## 📝 Development Log

Record important milestones and findings:

```
## 2025-10-21

### Training Run: small_model_v1
- **Start**: Step 0
- **Duration**: 48 hours on RTX 3090 Ti
- **Config**: configs/gpu/small.yaml
- **Data**: 5M examples from wikitext + books

#### Results
- Final loss: 1.85
- Final perplexity: 6.3
- Generation quality: Excellent
- Best checkpoint: step_95000

#### Observations
- Loss decreased smoothly
- No training instability
- Generated text coherent and diverse
- Model generalizes well

#### Next Steps
- Test RLHF fine-tuning
- Compare with base model
- Deploy for evaluation
```

---

**Status**: ✅ Complete
**Last Updated**: 2025-10-21

For more details on specific features, see ARCHITECTURE_AND_FEATURES.md or troubleshooting in FIXES_AND_TROUBLESHOOTING.md.
