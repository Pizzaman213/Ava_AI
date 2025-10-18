# NeuML "Train a Language Model from Scratch" Configuration

**Reference:** https://neuml.hashnode.dev/train-a-language-model-from-scratch

**Status:** ✅ Configuration matched to NeuML specifications

---

## NeuML Original Specifications

### Model Architecture (BERT Micromodel)
```python
BertConfig(
    vocab_size=500,
    hidden_size=50,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=100,
)
```

**Result:**
- Parameters: 94,450 (0.09% of BERT-base's 110M)
- Model size: 386KB PyTorch binary
- Training time: ~5 minutes

### Training Configuration
```python
HFTrainer(
    fp16=True,
    per_device_train_batch_size=128,
    num_train_epochs=10,
    dataloader_num_workers=2
)
```

### Dataset
- Source: `ag_news` (Hugging Face datasets)
- Split: train
- Task: Language modeling (next token prediction)

### Tokenizer
- Base: bert-base-uncased
- Retrained with 500 vocab size
- Max length: 512 tokens

---

## Our Implementation

### Model Architecture (MoE Micromodel)

**Adjusted for RoPE compatibility:**
```yaml
model:
  hidden_size: 64          # Adjusted from 50 (must be divisible by num_heads)
  num_layers: 2            # NeuML: 2 layers
  num_attention_heads: 2   # NeuML: 2 heads (head_dim=32, even for RoPE)
  intermediate_size: 128   # Adjusted from 100 (2x hidden, like NeuML)
  vocab_size: 500          # NeuML: 500 vocab
  max_position_embeddings: 512  # NeuML: 512 sequences

  # MoE extensions (adds capacity beyond plain BERT)
  num_experts: 4
  num_experts_per_token: 2
  router_type: deepseek
```

**Result:**
- Parameters: **289,936** (~290K, 3x NeuML's 94K due to MoE)
- Still ultra-compact micromodel
- MoE adds expert capacity without bloating too much

### Why Different from NeuML's 50/100?

**RoPE Constraint:** Our model uses Rotary Position Embeddings (RoPE) which requires:
- `hidden_size` divisible by `num_attention_heads`
- `head_dim = hidden_size / num_heads` must be even for rotation matrices

With NeuML's 50/2:
- head_dim = 25 (odd) ❌ Causes tensor dimension mismatch in RoPE

With our 64/2:
- head_dim = 32 (even) ✅ Works perfectly with RoPE

**Trade-off:** Slightly larger (290K vs 94K) but still a micromodel with better position encoding.

### Training Configuration
```yaml
training:
  batch_size: 128          # NeuML: 128
  num_epochs: 10           # NeuML: 10
  learning_rate: 0.0002    # Standard for small models
  gradient_accumulation_steps: 1
  max_gradient_norm: 1.0
  warmup_steps: 500
  lr_scheduler_type: cosine

# Mixed precision
fp16: true                 # NeuML: fp16=True
bf16: false
```

### Dataset Configuration
```yaml
data:
  data_dir: /project/code/data/ag_news/processed
  max_length: 512          # NeuML: 512 token sequences
  tokenizer_name: /project/code/models/tokenizer/enhanced-500
  padding_side: right
  truncation: true
```

**Dataset verified:**
- ✅ AG News downloaded and processed
- ✅ Train: 180MB (train.jsonl)
- ✅ Val: 29MB (combined_val.jsonl)
- ✅ Format: JSONL with text, input_ids, attention_mask, labels

---

## Comparison: NeuML vs Our Implementation

| Feature | NeuML Original | Our Implementation | Difference |
|---------|---------------|-------------------|------------|
| **Architecture** | BERT | Transformer + MoE | Added experts |
| **Hidden size** | 50 | 64 | +28% (RoPE requirement) |
| **Layers** | 2 | 2 | Same |
| **Heads** | 2 | 2 | Same |
| **Intermediate** | 100 | 128 | +28% (2x hidden) |
| **Vocab** | 500 | 500 | Same |
| **Max length** | 512 | 512 | Same |
| **Parameters** | 94K | 290K | +207% (MoE overhead) |
| **Model size** | 386KB | ~1.2MB | Larger but still tiny |
| **Batch size** | 128 | 128 | Same |
| **Epochs** | 10 | 10 | Same |
| **FP16** | Yes | Yes | Same |
| **Position encoding** | Learned | RoPE | Different (better extrapolation) |
| **Dataset** | AG News | AG News | Same |

---

## Key Differences Explained

### 1. MoE vs Plain BERT
**NeuML:** Single FFN per layer
**Ours:** 4 experts per layer, route to top-2

**Why:** MoE provides better capacity and specialization without massive parameter growth. 290K params is still a micromodel.

### 2. RoPE vs Learned Positions
**NeuML:** Standard learned position embeddings
**Ours:** Rotary Position Embeddings (RoPE)

**Why:**
- Better extrapolation to longer sequences
- No position embedding parameters to learn
- State-of-the-art position encoding (used in LLaMA, GPT-NeoX)

**Trade-off:** Requires even head_dim (hence 64 instead of 50)

### 3. Size Adjustment (50→64, 100→128)
**Constraint:** `head_dim = hidden_size / num_heads` must be even

**Math:**
- NeuML: 50 / 2 = 25 (odd) → RoPE fails
- Ours: 64 / 2 = 32 (even) → RoPE works

**Impact:**
- 28% more parameters in transformer layers
- Still ultra-compact at 290K total
- Better than switching to 4 heads (would be 200 hidden!)

---

## Expected Training Performance

### NeuML Results (5 minutes training)
- Model: 94K params, 386KB file
- Task: AG News language modeling
- Quality: Functional micromodel for search/embeddings

### Our Expected Results (~10-15 minutes)
- Model: 290K params, ~1.2MB file
- Loss progression:
  - Step 0: ~6.2 (random)
  - Step 1000: ~3.5-4.0
  - Step 5000: ~2.5-3.0
  - Final: ~2.0-2.5
- Quality: Better than NeuML due to MoE capacity

### Training Time Estimate
**NeuML:** 5 minutes for 94K params
**Ours:**
- 3x more parameters (290K vs 94K)
- MoE routing overhead
- **Estimated:** 10-15 minutes total

Still fast enough for rapid iteration!

---

## Verification Results

```
Model Architecture:
  ✓ vocab_size: 500
  ✓ hidden_size: 64 (adjusted for RoPE)
  ✓ num_layers: 2
  ✓ num_experts: 4
  ✓ num_experts_per_token: 2
  ✓ initializer_range: 0.02
  ✓ tie_word_embeddings: False
  ✓ use_learned_position_embeddings: False

Training Config:
  ✓ batch_size: 128
  ✓ learning_rate: 0.0002
  ✓ num_epochs: 10
  ✓ fp16: true

Model Initialized Successfully:
  Total parameters: 289,936
  Trainable parameters: 289,936
  Forward pass: ✓ (loss 6.23)
  MoE load balancing: ✓

Dataset:
  ✓ AG News processed (180MB train, 29MB val)
  ✓ Format: JSONL with tokenized sequences
  ✓ Max length: 512 tokens
```

---

## Start Training (NeuML Style)

```bash
cd /project/code/scripts/5_training
python train.py --config ../../configs/gpu/small.yaml
```

**Expected behavior:**
- Fast training (~10-15 min for 10 epochs)
- Stable loss curve (no explosions)
- All 4 experts utilized
- Final model: ~1.2MB PyTorch binary

**Monitor for:**
- Loss dropping from 6.2 → 2.0-2.5
- Gradient norms stable (~1-5 range)
- No repetition issues (fixed all 23 bugs!)
- MoE expert utilization balanced

---

## Philosophy: Micromodels

Following NeuML's philosophy:
> "Micromodels can be fully rebuilt in hours using the most up-to-date knowledge available. If properly constructed, prepared and trained, micromodels have the potential to be a viable choice for limited resource environments."

**Our approach:**
- ✅ Ultra-compact (290K params, ~1.2MB)
- ✅ Fast training (10-15 minutes)
- ✅ Modern architecture (RoPE + MoE)
- ✅ Easy to retrain with new data
- ✅ Low resource requirements

**Use cases:**
- Edge deployment
- Rapid prototyping
- Domain-specific micromodels
- Educational purposes
- Resource-constrained environments

---

## Files Modified for NeuML Compatibility

1. [configs/gpu/small.yaml](configs/gpu/small.yaml)
   - Model: 64 hidden, 2 layers, 2 heads, 128 intermediate
   - Training: batch 128, 10 epochs, fp16
   - Data: 512 max_length

2. Dataset: Already set up
   - [data/ag_news/processed/](data/ag_news/processed/)

3. Tokenizer: Already set up
   - [models/tokenizer/enhanced-500/](models/tokenizer/enhanced-500/)
   - 500 vocab (matches NeuML)

---

## Next Steps

1. **Start training:**
   ```bash
   cd /project/code/scripts/5_training
   python train.py --config ../../configs/gpu/small.yaml
   ```

2. **Monitor progress:**
   - Watch loss drop from ~6.2
   - Check expert utilization in logs
   - Verify no gradient explosions

3. **Test at checkpoints:**
   ```bash
   python /project/code/test_step_1000.py
   ```

4. **Export to ONNX (optional):**
   - Follow NeuML's quantization approach
   - Target: ~600KB quantized model

---

**Status:** ✅ Ready to train
**Configuration:** Matches NeuML philosophy with MoE enhancements
**Expected time:** 10-15 minutes for full training

**Generated:** 2025-10-17
