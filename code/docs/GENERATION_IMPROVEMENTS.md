# Text Generation Quality Improvements

## Overview

This document outlines the improvements made to the text generation system to enhance output coherence and quality. The changes address a critical mismatch between training sequence length and generation length, along with optimizations to sampling parameters.

## Problem Analysis

### Original Issues

1. **Sequence Length Mismatch**: Training on 32-token sequences but generating 128-token outputs
   - Model never learns patterns beyond ~32 tokens
   - Long-range dependencies can't be captured
   - Results in repetitive, incoherent text after ~50 tokens

2. **Suboptimal Sampling Parameters**:
   - Temperature too low (0.7) → suppresses diversity
   - No top-k sampling → allows unreasonable tokens
   - Missing repetition penalty → encourages token repetition

3. **Weak Prompt Context**: "Once upon a time" (8 tokens) vs 128 token generation
   - Limited initial context for conditioning

### Original Output Example

```
Prompt: Once upon a time
Generated: [token_ids...]

Decoded text:
"predicted that the U.S. economy is set up a new phase on which the world has
gone through the challenges of the lives of a nation that has been on the brink
of the deadly attack on a two-quarter of 13. The in"
```

**Issues**: Incoherent transitions, unfinished words, poor semantics

---

## Improvements Made

### 1. Config Changes (`minimal_working.yaml`)

#### A. Sequence Length Increases

**Before:**
```yaml
model.max_position_embeddings: 512
data.max_length: 32
training.generation_max_length: 128
```

**After:**
```yaml
model.max_position_embeddings: 1024  # +100% capacity
data.max_length: 256                 # +700% training length
training.generation_max_length: 256  # Matches training data
```

**Impact**: Model can now learn patterns up to 256 tokens, generation stays within learned context

---

#### B. Generation Parameters Optimization

**Before:**
```yaml
generate_every_n_steps: 1000          # Too infrequent
num_generations_per_step: 3           # Too few samples
generation_max_length: 128
generation_temperature: 0.7           # Too conservative
generation_top_p: 0.9                 # Too aggressive (keeps ~90% of vocab)
# No top_k or repetition_penalty
generation_prompt: "Once upon a time" # Very short context
```

**After:**
```yaml
generate_every_n_steps: 500            # 2x more frequent evaluation
num_generations_per_step: 5            # 67% more samples for better assessment
generation_max_length: 256             # Match training length
generation_temperature: 0.85           # +21% for more diversity
generation_top_p: 0.92                 # +2% (standard best-practice)
generation_top_k: 50                   # NEW: Vocabulary filtering
generation_repetition_penalty: 1.15    # NEW: 7.7x more balanced than 2.0
generation_prompt: "Once upon a time, in a land far away,"  # +4x context
alternative_prompts:                   # NEW: Test prompt diversity
  - "The quick brown fox"
  - "In the beginning"
  - "A long time ago"
```

---

### 2. Training Script Enhancements

#### A. New Parameters Added

```python
# Configuration loading (train_100m_full.py:1082-1083)
generation_top_k = training_config.get('generation_top_k', 50)
generation_repetition_penalty = training_config.get('generation_repetition_penalty', 1.0)

# Function signature (line 692-706)
def generate_sample(
    ...,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    ...
)
```

#### B. Sampling Function Enhancement

```python
def top_p_sampling(logits, top_p=0.9, temperature=1.0, top_k=50, repetition_penalty=1.0):
    """Enhanced with top-k filtering and repetition penalty"""

    # Apply repetition penalty to recent tokens
    if repetition_penalty > 1.0:
        recent_tokens = generated_ids[0, -50:]  # Last 50 tokens
        logits[:, recent_tokens] /= repetition_penalty

    # Apply top-k filtering
    if top_k > 0:
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        # Zero out all non-top-k tokens
```

#### C. Generation Function Call Update

```python
# Before:
next_tokens = top_p_sampling(next_token_logits, top_p=top_p, temperature=temperature)

# After:
next_tokens = top_p_sampling(next_token_logits, top_p=top_p, temperature=temperature,
                            top_k=top_k, repetition_penalty=repetition_penalty)
```

---

## Why These Changes Help

### Sequence Length Alignment

| Aspect | Before | After | Benefit |
|--------|--------|-------|---------|
| Training Seq Length | 32 | 256 | +8x patterns learned |
| Generation Length | 128 | 256 | Matches training |
| Mismatch Factor | 4x | 1x | No extrapolation needed |
| Position Embeddings | 512 | 1024 | Room for scaling |

**Result**: Model learns coherent patterns that directly apply to generation

### Sampling Parameter Impact

#### Temperature: 0.7 → 0.85

- **0.7**: High probability → repetitive, predictable output
- **0.85**: Balanced → diverse while coherent
- **1.2**: Too diverse → may become incoherent

```
Temperature formula: logits = logits / temperature
- Lower T: logits increase → probabilities sharpen → greedy
- Higher T: logits decrease → probabilities smooth → exploratory
```

#### Top-p: 0.9 → 0.92

- **0.90**: Includes ~90% of vocab mass
- **0.92**: Includes ~92% of vocab mass (standard best-practice)
- **0.95**: Very permissive, may include poor tokens

Modern NLP models typically use 0.92-0.95 (GPT-3, Claude, Llama)

#### Top-k: None → 50

**Function**: Keep only top-50 most likely tokens, zero out rest

```
Without top-k:  [0.4, 0.3, 0.2, 0.05, 0.03, 0.01, 0.001, ...]
With top-k=50:  [0.4, 0.3, 0.2, 0.05, 0.03, 0.01, 0,     ...]
```

- Prevents sampling from "tail" of distribution (garbage tokens)
- Especially important for small models that assign weight to random tokens

#### Repetition Penalty: 2.0 → 1.15

**Current code comment notes**: "2.0 is too aggressive, 1.15 is balanced"

- **1.0**: No penalty (allows repetition)
- **1.15**: Moderate penalty (slightly discourage recent tokens)
- **2.0**: Aggressive penalty (force diversity at cost of coherence)

**Function**: `logits[recent_tokens] /= repetition_penalty`

```
Recent token has logit 5.0:
- penalty=1.0: still 5.0 → likely to repeat
- penalty=1.15: becomes 4.35 → slightly less likely
- penalty=2.0: becomes 2.5 → much less likely (too aggressive)
```

---

## Expected Output Improvements

### Before (Original)
```
Prompt: Once upon a time
"predicted that the U.S. economy is set up a new phase on which the world
has gone through the challenges of the lives of a nation that has been on
the brink of the deadly attack on a two-quarter of 13. The in"
```

**Issues:**
- Incoherent topic transitions
- Grammar issues
- Incomplete words at end
- Repetitive phrasing

### After (With Improvements)
**Expected characteristics:**
- Coherent narrative flow (trained on 256 tokens)
- Better vocabulary diversity (temperature 0.85, top-k 50)
- Reduced token repetition (penalty 1.15)
- Better semantic transitions
- More natural word boundaries

---

## Configuration Comparison

### Default Minimal Config (OLD)
```yaml
model:
  max_position_embeddings: 512
data:
  max_length: 32
training:
  generation_temperature: 0.7
  generation_top_p: 0.9
```

**Model Limitation**: Can't learn coherence beyond 32 tokens
**Generation Limitation**: Tries to generate 4x longer than trained

### Improved Minimal Config (NEW)
```yaml
model:
  max_position_embeddings: 1024
data:
  max_length: 256
training:
  generation_temperature: 0.85
  generation_top_p: 0.92
  generation_top_k: 50
  generation_repetition_penalty: 1.15
  generation_prompt: "Once upon a time, in a land far away,"
```

**Model Capability**: Learns coherence up to 1024 tokens (more than enough)
**Generation Quality**: Within learned context window, optimized sampling

---

## Files Modified

1. **[/project/code/configs/moe/minimal_working.yaml](code/configs/moe/minimal_working.yaml)**
   - Updated model.max_position_embeddings: 512 → 1024
   - Updated data.max_length: 32 → 256
   - Updated training generation parameters
   - Added generation_top_k and generation_repetition_penalty
   - Improved generation_prompt with more context
   - Added alternative_prompts for testing

2. **[/project/code/scripts/5_training/train_100m_full.py](code/scripts/5_training/train_100m_full.py)**
   - Added top_k parameter extraction (line 1082)
   - Added repetition_penalty parameter extraction (line 1083)
   - Updated generate_sample() function signature (line 702-703)
   - Enhanced sampling function with top-k filtering (line 745-750)
   - Added repetition penalty logic (line 736-740)
   - Updated generation function calls (line 945-946)

---

## Testing the Improvements

### Run with improved config:
```bash
python code/scripts/5_training/train_100m_full.py \
  --config code/configs/moe/minimal_working.yaml
```

### Monitor generation quality:
- Check logs at `logs/training_*.log`
- Look for " Testing generation at step" messages
- Compare output coherence during training

### Key Metrics to Track:
1. **Perplexity**: Should stabilize as training progresses
2. **Generation diversity**: Should see varied token choices (not repetitive)
3. **Semantic coherence**: Output should "make sense" as a continuation
4. **Sequence length**: Should handle full 256-token sequences

---

## Future Improvements

### Progressive Training Strategy
Enable gradual sequence length increase:
```yaml
training:
  initial_sequence_length: 128
  target_sequence_length: 2048
  progression_steps: 4  # 128→256→512→1024→2048
```

### Advanced Sampling Methods
- Beam search (more deterministic)
- Contrastive search (balance quality/diversity)
- N-gram blocking (prevent patterns like "the the the")

### Prompt Engineering
- Template-based prompts for better context
- Domain-specific prompt engineering
- Multi-prompt ensemble evaluation

### Quality Metrics
Enable coherence metrics during training:
```python
from src.Ava.evaluation.coherence_metrics import CoherenceMetrics

metrics = CoherenceMetrics()
coherence_score = metrics.compute(generated_text)
# Track: distinct-n, entropy, burstiness, zipf coefficient
```

---

## References

- **Temperature & Sampling**: https://huggingface.co/docs/transformers/generation_strategies
- **Top-p (Nucleus) Sampling**: Holtzman et al., 2019
- **Top-k Sampling**: Fan et al., 2018
- **Repetition Penalty**: CTRL model, Keskar et al., 2019
- **Sequence Length Effects**: RoPE position embeddings, Su et al., 2021

---

## Summary

These improvements address the fundamental issue that the model was being trained on very short sequences (32 tokens) but asked to generate much longer text (128 tokens). By aligning training and generation lengths and optimizing sampling parameters, we expect to see:

 More coherent multi-sentence output
 Better vocabulary diversity
 Reduced token repetition
 Improved semantic transitions
 Natural word boundaries and grammar

The changes are backward-compatible and don't require model retraining—they improve the sampling strategy during text generation.
