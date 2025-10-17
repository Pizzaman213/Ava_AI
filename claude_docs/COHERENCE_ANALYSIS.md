# Coherence Configuration Analysis

## Overview

Analyzing your configuration for factors affecting output coherence beyond the LR finder.

---

## ✅ 1. Model Architecture - EXCELLENT for Coherence

```yaml
model:
  hidden_size: 512              # Good for 83M model
  num_layers: 6                 # ✅ Good depth
  num_attention_heads: 8        # ✅ Good attention
  intermediate_size: 2048       # ✅ 4x hidden (standard)
  vocab_size: 65536             # ✅ Large vocab (custom tokenizer)
  max_position_embeddings: 2048 # ✅ Long context

  # MoE Architecture (DeepSeek-style)
  num_experts: 2                # ✅ Small but functional
  num_experts_per_token: 1      # ✅ Sparse routing
  router_type: deepseek         # ✅ Good router

  # Regularization
  attention_dropout: 0.1        # ✅ Good
  hidden_dropout: 0.05          # ✅ Low dropout
  dropout: 0.1                  # ✅ Balanced

  # Advanced Features
  use_flash_attention: true     # ✅ Efficient attention
  use_cache: true               # ✅ Good for inference
```

### Assessment: ✅ EXCELLENT

**Strengths:**
- ✅ **6 layers** - Good depth for understanding context
- ✅ **8 attention heads** - Can learn multiple relationships
- ✅ **2048 context length** - Can maintain long conversations
- ✅ **65k vocab** - Rich token representation (custom tokenizer)
- ✅ **MoE architecture** - Can specialize (2 experts)
- ✅ **Low dropout (0.05)** - Won't forget context easily

**For Coherence:**
- **6 layers** is sufficient for short-to-medium coherence
- **8 heads** can track multiple dependencies
- **2048 tokens** allows maintaining context across paragraphs
- **MoE** can specialize one expert for coherence patterns

**Potential Concerns:**
- ⚠️ Only **2 experts** (minimal specialization)
- ⚠️ **83M params** is small (may limit complex reasoning)

**Recommendation:** ✅ Keep as-is. Good architecture for this size.

---

## ✅ 2. Training Duration - GOOD

```yaml
training:
  max_steps: 30000              # ✅ Good for initial training
  warmup_steps: 1000            # ✅ 3.3% warmup (good)
  num_epochs: 1                 # ✅ One pass through data

  eval_steps: 1000              # ✅ Frequent evaluation
  save_steps: 1000              # ✅ Frequent checkpoints
  early_stopping_patience: 5    # ✅ Will stop if not improving
```

### Assessment: ✅ GOOD

**For Coherence:**
- ✅ **30,000 steps** is reasonable for:
  - Batch size: 16
  - Gradient accumulation: 16
  - Effective batch: 256
  - Total tokens: 30k × 256 × 512 ≈ **3.9B tokens**

**Is This Enough?**

| Model Size | Recommended Tokens | Your Training | Status |
|------------|-------------------|---------------|---------|
| 83M params | 1-10B tokens | ~3.9B tokens | ✅ Good |
| Optimal | 10-20B tokens | - | ⚠️ Could be more |
| Minimum | 500M-1B tokens | - | ✅ Well above |

**Coherence Timeline:**
```
Steps 0-5k:    Basic syntax, grammar
Steps 5k-10k:  Sentence structure, basic coherence
Steps 10k-20k: Paragraph coherence, context tracking
Steps 20k-30k: Refinement, style consistency
```

**Recommendation:**
- ✅ 30k steps is **good** for initial training
- 💡 For better coherence, consider extending to **50-100k steps**
- 💡 Monitor loss curve - if still decreasing at 30k, continue training

---

## ⚠️ 3. Generation Parameters - NEEDS FIXES!

```yaml
generation:
  repetition_penalty: 1.8       # ✅ Good (1.5-2.0 range)
  eos_penalty: 3.0              # 🔴 TOO HIGH! (should be 0.5-2.0)
  min_length: 50                # ⚠️ TOO HIGH! (should be 10-30)
  temperature: 0.7              # ✅ Good (balanced)
  top_k: 40                     # ✅ Good (standard)
  top_p: 0.85                   # ✅ Good (nucleus sampling)
  use_ngram_blocking: true      # ✅ Excellent (prevents loops)
  ngram_size: 2                 # ✅ Good (blocks bigram repetition)
  do_sample: true               # ✅ Good (diverse outputs)
  num_beams: 1                  # ✅ Good (greedy or sampling)
```

### Assessment: ⚠️ NEEDS FIXES

### 🔴 **Critical Issue: EOS Penalty**

```yaml
# CURRENT (WRONG for generation):
eos_penalty: 3.0     # This PREVENTS ending!

# SHOULD BE:
eos_penalty: 0.5     # Gentle encouragement to end
# OR
eos_penalty: 1.0     # Neutral (no penalty)
# OR
eos_penalty: 2.0     # Strong encouragement to end
```

**Why This Matters:**
- During **training**: `eos_penalty_weight: 0.3, eos_logit_bias: -0.5` ✅ Correct
- During **generation**: `eos_penalty: 3.0` 🔴 **Contradicts training!**

**Impact:**
- Training teaches model to end naturally
- Generation penalizes ending (forces continuation)
- **Result: Incoherent rambling, repetition, loops**

### ⚠️ **Issue: Min Length**

```yaml
# CURRENT:
min_length: 50       # Forces at least 50 tokens

# SHOULD BE:
min_length: 10       # Allow short answers
# OR
min_length: 20       # Medium minimum
# OR
min_length: 30       # Conservative (current training setting)
```

**Why This Matters:**
- User: "What is 2+2?"
- Model wants to say: "4" (1 token)
- Forced to generate: "4... [48 more tokens of padding/repetition]"

**Impact:**
- Short answers become padded
- Padding = repetition/incoherence
- Model learns bad patterns

### ✅ **Good Settings:**

```yaml
repetition_penalty: 1.8    # ✅ Prevents word repetition
temperature: 0.7           # ✅ Balanced randomness
top_k: 40                  # ✅ Limits to top 40 tokens
top_p: 0.85                # ✅ Nucleus sampling (good)
use_ngram_blocking: true   # ✅ Prevents phrase repetition
ngram_size: 2              # ✅ Blocks bigrams
```

---

## 🔧 Recommended Fixes for Generation

### Fix 1: EOS Penalty (Critical!)

```yaml
generation:
  eos_penalty: 1.0  # Changed from 3.0 to 1.0 (neutral)
```

**Or for more natural endings:**
```yaml
generation:
  eos_penalty: 0.5  # Gentle encouragement to end
```

### Fix 2: Min Length

```yaml
generation:
  min_length: 20  # Changed from 50 to 20 (reasonable)
```

**Or match training setting:**
```yaml
generation:
  min_length: 30  # Match training min_sequence_length
```

### Fix 3: Consider Adjusting Temperature

**For more coherent (but less creative) outputs:**
```yaml
generation:
  temperature: 0.6  # Reduced from 0.7 (more focused)
```

**For more creative (but less coherent) outputs:**
```yaml
generation:
  temperature: 0.8  # Increased from 0.7 (more random)
```

---

## 📊 Coherence Score Summary

| Factor | Current Status | Coherence Impact | Recommendation |
|--------|---------------|------------------|----------------|
| **Model Architecture** | ✅ Excellent | ✅ High | Keep as-is |
| **Training Duration** | ✅ Good | ✅ Medium-High | 30k is good, 50-100k better |
| **Generation: EOS Penalty** | 🔴 Wrong (3.0) | 🔴 **CRITICAL** | Fix to 0.5-1.0 |
| **Generation: Min Length** | ⚠️ High (50) | ⚠️ Medium | Fix to 20-30 |
| **Generation: Repetition** | ✅ Good (1.8) | ✅ High | Keep as-is |
| **Generation: Temperature** | ✅ Good (0.7) | ✅ Medium | Keep or adjust |
| **Generation: Top-p/k** | ✅ Good | ✅ Medium | Keep as-is |
| **Generation: N-gram Block** | ✅ Excellent | ✅ High | Keep as-is |

---

## 🎯 Action Items for Coherence

### Immediate (Before Training):
1. ✅ **Training config is correct** - Already fixed!
   - eos_penalty_weight: 0.3 ✅
   - eos_logit_bias: -0.5 ✅
   - min_sequence_length: 30 ✅

2. 🔴 **Fix generation config** - Do this now:
   ```yaml
   generation:
     eos_penalty: 1.0      # Change from 3.0
     min_length: 30        # Change from 50
   ```

### After Training:
3. ✅ Train for 30k steps (good baseline)
4. 💡 Test generation quality at checkpoints
5. 💡 If coherence is good but loss still decreasing, extend to 50k

### During Generation:
6. ✅ Use repetition_penalty: 1.8
7. ✅ Use temperature: 0.6-0.8 (adjust based on use case)
8. ✅ Use top_p: 0.85
9. ✅ Use ngram_blocking: true

---

## 🎓 Expected Coherence Results

### With Current Training Config (After Fixes):
```
Steps 0-5k:    ⭐⭐☆☆☆ Basic sentences
Steps 5k-10k:  ⭐⭐⭐☆☆ Simple coherence
Steps 10k-20k: ⭐⭐⭐⭐☆ Good coherence
Steps 20k-30k: ⭐⭐⭐⭐⭐ Excellent coherence
```

### With Fixed Generation Config:
- ✅ Natural sentence endings
- ✅ No forced padding
- ✅ Minimal repetition
- ✅ Context maintained across turns
- ✅ Appropriate response lengths

### Model Limitations (83M params):
- ⚠️ May struggle with very long context (>1000 tokens)
- ⚠️ May lack deep reasoning (small model)
- ⚠️ May need multiple experts for complex tasks

---

## 🚀 Final Recommendations

### 1. Fix Generation Config NOW (Before Training):

```yaml
generation:
  repetition_penalty: 1.8   # ✅ Keep
  eos_penalty: 1.0          # 🔧 FIX: Changed from 3.0
  min_length: 30            # 🔧 FIX: Changed from 50
  temperature: 0.7          # ✅ Keep (or adjust to 0.6 for more focus)
  top_k: 40                 # ✅ Keep
  top_p: 0.85               # ✅ Keep
  use_ngram_blocking: true  # ✅ Keep
  ngram_size: 2             # ✅ Keep
  do_sample: true           # ✅ Keep
  num_beams: 1              # ✅ Keep
```

### 2. Training Duration:
- ✅ Start with 30k steps
- 💡 Monitor loss curve
- 💡 Extend to 50-100k if still improving

### 3. Model Architecture:
- ✅ Current architecture is good
- ✅ No changes needed

---

## ✅ Summary

**Training Config:** ✅ **EXCELLENT** (all fixes applied)
- EOS penalties: Perfect
- Repetition penalties: Perfect
- Min sequence length: Perfect
- Learning rate: Optimal (from LR finder)

**Generation Config:** 🔴 **NEEDS FIXES**
- eos_penalty: 3.0 → 1.0 (critical!)
- min_length: 50 → 30 (important!)

**Model & Duration:** ✅ **GOOD**
- Architecture: Excellent for 83M
- 30k steps: Good baseline
- Can extend if needed

**Fix these 2 generation settings and you're ready for coherent training!** 🎯

