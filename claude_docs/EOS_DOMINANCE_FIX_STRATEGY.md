# EOS Dominance Fix Strategy

## Problem Summary

**Current Status:** Model at step 42,000 has learned to predict EOS (end-of-text) as the top token with **76.94% probability**

**Impact:**
- Generation stops immediately after prompt
- With forced continuation (high EOS penalty), outputs gibberish/repetition
- Model learned "when to end" before learning "what to say"

**Root Cause:** Training data likely contains many short sequences, causing model to learn EOS is the most common continuation

---

## Multi-Pronged Fix Strategy

### Fix 1: Logit Bias During Training (IMMEDIATE - Most Important)

**What:** Add negative bias to EOS token logits during forward pass of training

**Why:** Current EOS penalty only works on target labels. The model's **predictions** need bias too.

**Implementation:**
```python
# In model forward pass, before computing loss
if training and self.eos_token_id is not None:
    # Subtract bias from EOS logits to discourage prediction
    logits[:, :, self.eos_token_id] -= 5.0  # Configurable
```

**Location:** `/project/code/src/Ava/models/moe_model.py` in the forward method

**Impact:** 🔥 HIGH - Directly reduces EOS probability during training

---

### Fix 2: Minimum Length Filtering in Data Pipeline (HIGH PRIORITY)

**What:** Filter out sequences shorter than minimum length during data loading

**Why:** Short sequences ending in EOS teach model to end early

**Implementation:**
```python
# In data loader
def filter_sequences(example):
    tokens = tokenizer.encode(example['text'])
    return len(tokens) >= min_sequence_length  # e.g., 30 tokens
```

**Location:** `/project/code/src/Ava/data_streaming.py`

**Impact:** 🔥 HIGH - Prevents learning from bad examples

---

### Fix 3: Increase eos_penalty_weight (QUICK WIN)

**What:** Increase from 5.0 to 10.0 or higher

**Why:** Stronger penalty for early EOS in targets

**Implementation:**
```yaml
# In small.yaml
training:
  eos_penalty_weight: 10.0  # Was 5.0
```

**Impact:** 🟡 MEDIUM - Helps but not sufficient alone

---

### Fix 4: Validation Sampling with Strong EOS Penalty (MONITORING)

**What:** Add EOS penalty to validation generation

**Why:** Current validation uses greedy decoding, hitting EOS immediately

**Implementation:**
```python
# In enhanced_trainer.py validation
outputs = model.generate(
    inputs,
    max_length=100,
    min_length=30,  # Force continuation
    eos_penalty=3.0,  # Discourage EOS
    do_sample=True,
    temperature=0.8
)
```

**Impact:** 🟢 LOW - Doesn't fix training but gives better monitoring

---

### Fix 5: EOS Token Masking for First N Tokens (NUCLEAR OPTION)

**What:** Completely mask out EOS token for first N positions

**Why:** Forces model to continue for minimum length

**Implementation:**
```python
# During training, mask EOS in loss computation
if position < min_sequence_length:
    logits[:, :, eos_token_id] = float('-inf')
```

**Impact:** 🔥 HIGH - Very effective but aggressive

---

## Recommended Implementation Order

### Phase 1: Quick Wins (Can do now without restarting)
1. ✅ Update config: `eos_penalty_weight: 10.0`
2. ✅ Update config: `min_sequence_length: 40` (was 20)
3. ✅ Git commit config changes

### Phase 2: Code Changes (Requires restart)
1. 🔧 Implement Fix #1: Logit bias in model forward
2. 🔧 Implement Fix #2: Data filtering
3. 🔧 Implement Fix #4: Better validation sampling
4. 🔧 Test and validate

### Phase 3: Monitor & Iterate
1. 📊 Watch EOS probability every 5k steps
2. 📊 Check generation quality at 50k, 60k, 70k
3. 📊 Adjust penalties if needed

---

## Detailed Implementation: Fix #1 (Logit Bias)

### File: `/project/code/src/Ava/models/moe_model.py`

**Current forward method (around line 360-380):**
```python
def forward(self, input_ids, attention_mask=None, labels=None):
    # ... embedding and transformer layers ...

    # Get logits
    logits = self.lm_head(hidden_states)

    # Compute loss if labels provided
    if labels is not None:
        loss = compute_loss(logits, labels)
```

**Add EOS bias BEFORE loss computation:**
```python
def forward(self, input_ids, attention_mask=None, labels=None):
    # ... embedding and transformer layers ...

    # Get logits
    logits = self.lm_head(hidden_states)

    # CRITICAL FIX: Apply negative bias to EOS token during training
    if self.training and labels is not None:
        eos_bias = getattr(self.config, 'eos_logit_bias', 5.0)
        if eos_bias > 0:
            # Subtract bias from EOS logits (makes EOS less likely)
            eos_token_id = 151643  # Qwen EOS token
            logits[:, :, eos_token_id] = logits[:, :, eos_token_id] - eos_bias

    # Compute loss if labels provided
    if labels is not None:
        loss = compute_loss(logits, labels)
```

**Add to config:**
```yaml
# In small.yaml under training:
training:
  eos_logit_bias: 5.0  # NEW: Negative bias for EOS during training
```

---

## Detailed Implementation: Fix #2 (Data Filtering)

### File: `/project/code/src/Ava/data_streaming.py`

**Find the data loading function** (around line 100-200):
```python
def load_dataset_streaming(...):
    # Current code loads all examples
    dataset = load_dataset(...)
    return dataset
```

**Add filtering:**
```python
def load_dataset_streaming(...):
    dataset = load_dataset(...)

    # Filter out short sequences
    min_length = config.get('min_sequence_length', 30)

    def filter_fn(example):
        # Get text length in tokens
        if 'text' in example:
            text = example['text']
        elif 'content' in example:
            text = example['content']
        else:
            return True  # Keep if no text field

        # Quick length check (chars * 0.3 ≈ tokens for English)
        if len(text) < min_length * 3:
            return False

        return True

    dataset = dataset.filter(filter_fn)
    return dataset
```

---

## Detailed Implementation: Fix #4 (Validation Sampling)

### File: `/project/code/src/Ava/training/enhanced_trainer.py`

**Find validation generation** (around line 780):
```python
# Current validation (greedy)
outputs = self.model.generate(
    input_ids=inputs['input_ids'],
    max_length=50,
    do_sample=False  # Greedy
)
```

**Update to use sampling with EOS penalty:**
```python
# Fixed validation with EOS penalty
outputs = self.model.generate(
    input_ids=inputs['input_ids'],
    max_length=100,
    min_length=30,  # Force minimum continuation
    do_sample=True,  # Use sampling
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.5,
    eos_penalty=3.0,  # CRITICAL: Discourage EOS
    pad_token_id=self.tokenizer.pad_token_id,
    eos_token_id=self.tokenizer.eos_token_id
)
```

---

## Expected Results Timeline

### After Implementing Fixes (Restart Training)

**Step 0-10k:** Model relearning with new biases
- EOS probability should drop from 77% to 50%

**Step 10k-20k:** Learning alternative patterns
- EOS probability should drop to 30-40%
- Some gibberish but attempting continuation

**Step 20k-30k:** Coherent phrases emerging
- EOS probability should drop to 15-25%
- Short phrases appear in generation

**Step 30k-50k:** Sentence-level coherence
- EOS probability should drop to 5-10%
- Multi-word coherent outputs

**Step 50k+:** Natural generation
- EOS probability < 5%
- Fluent text generation

---

## Testing the Fix

### Before Retraining (Test Current Checkpoint)

Run diagnostic to confirm EOS dominance:
```bash
python code/scripts/validation/test_checkpoint_step_41000.py
```

Expected output:
```
EOS token rank: 1 (probability: 76.94%) ← PROBLEM
```

### After Implementing Fixes (Every 5k Steps)

Monitor EOS probability:
```bash
# At step 50k, 55k, 60k, etc.
python code/scripts/validation/test_checkpoint_step_41000.py
```

Expected improvement:
```
Step 50k: EOS token rank: 5 (probability: 30.00%) ← Better!
Step 60k: EOS token rank: 12 (probability: 12.00%) ← Much better!
Step 70k: EOS token rank: 25 (probability: 3.00%) ← Good!
```

### Generation Quality Test

```bash
python scripts/6_generation/generate.py \
    --prompt "Once upon a time" \
    --max-length 100 \
    --temperature 0.8 \
    --cpu
```

**Before fix:**
```
Output: (empty or gibberish)
```

**After fix (step 60k+):**
```
Output: "Once upon a time there was a young girl who lived in a small village..."
```

---

## Alternative: Quick Restart Strategy

If you want to fix this NOW without waiting for 100k steps:

### Option A: Restart from Scratch with Fixes
1. Implement all 4 fixes
2. Delete current run
3. Start fresh training
4. Should see coherent output by step 30k

**Pros:** Clean slate, optimal learning
**Cons:** Lose 42k steps of work

### Option B: Continue Current Training with Fixes
1. Implement all 4 fixes
2. Resume from step 42k
3. Model will adapt but slowly
4. May need 80-100k total steps

**Pros:** Don't lose progress
**Cons:** Takes longer to overcome bad pattern

### Option C: Fine-tune from Good Checkpoint
1. Find earlier checkpoint where EOS wasn't dominant
2. Resume from there with fixes
3. Retrain forward with new biases

**Pros:** Best of both worlds
**Cons:** Need to find "good" checkpoint

---

## Recommendation

**Best Path Forward:**

1. **Implement Fixes 1-4 NOW** (30 min work)
2. **Restart training from scratch** with fixed config
3. **Monitor EOS probability** at steps 5k, 10k, 15k, 20k
4. **By step 30k** should see coherent generation
5. **By step 50k** should have good quality

**Why restart?**
- Current checkpoint has deeply learned "EOS first" pattern
- Will take 40-60k MORE steps to unlearn
- Fresh start with fixes = faster to good generation
- Only "lost" 42k steps, but those steps learned wrong pattern

**Total time investment:**
- Fix implementation: 30-60 minutes
- Retraining to step 50k: ~5-6 hours
- Total: Same time as waiting for step 100k with current model

---

## Code Change Summary

### Files to Modify:
1. ✅ `/project/code/configs/gpu/small.yaml` - Update penalties
2. 🔧 `/project/code/src/Ava/models/moe_model.py` - Add logit bias
3. 🔧 `/project/code/src/Ava/data_streaming.py` - Add filtering
4. 🔧 `/project/code/src/Ava/training/enhanced_trainer.py` - Fix validation

### Config Changes (Immediate):
```yaml
training:
  min_sequence_length: 40  # Was 20
  eos_penalty_weight: 10.0  # Was 5.0
  eos_logit_bias: 5.0  # NEW
```

### Estimated Lines of Code: ~50 lines total

---

## Conclusion

The EOS dominance problem is **fixable** with the right strategy. The model itself is training well (good loss, no collapse), it just learned the wrong pattern from the data.

**Key Insight:** The model learned "end early" because the data showed "end early". Fix the incentives, fix the behavior.

**Next Step:** Choose between restart (faster) or continue (safer), then implement the fixes.
