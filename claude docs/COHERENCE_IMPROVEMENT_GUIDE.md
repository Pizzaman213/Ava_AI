# LLM Coherence Improvement Guide

Complete guide to understanding and improving LLM coherence in this codebase.

**Last Updated:** 2025-11-09
**Status:** ✅ Complete Implementation

---

## Table of Contents

1. [What is LLM Coherence?](#what-is-llm-coherence)
2. [Quick Start](#quick-start)
3. [Coherence Metrics](#coherence-metrics)
4. [Improvement Strategies](#improvement-strategies)
5. [Configuration Guide](#configuration-guide)
6. [Tools & Scripts](#tools--scripts)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## What is LLM Coherence?

**Coherence** refers to how well an LLM generates logical, diverse, and natural-sounding text without excessive repetition or randomness.

### Key Aspects:

1. **Diversity** - Vocabulary variety (measured by distinct-n metrics)
2. **Consistency** - Logical flow and topic coherence
3. **Naturalness** - Human-like token distributions (Zipf's law)
4. **Non-repetition** - Avoiding phrase/word repetition
5. **Fluency** - Proper grammar and sentence structure

### Why It Matters:

- ❌ **Poor coherence**: Repetitive, incoherent, or overly random outputs
- ✅ **Good coherence**: Natural, diverse, logically consistent text

---

## Quick Start

### 1. Enable Coherence Features (Already Done! ✅)

The configuration at [code/configs/moe/tiny_moe_ultra_low_mem.yaml](code/configs/moe/tiny_moe_ultra_low_mem.yaml) has been updated with:

```yaml
training:
  compute_coherence_metrics: true
  generation_during_eval: true
  test_generation_quality: true
  num_generations_per_step: 3

losses:
  primary_loss_type: deepseek
  use_deepseek_loss: true
  adaptive_temperature: true
  label_smoothing: 0.1

  use_ngram_penalty: true
  ngram_penalty_weight: 0.1

  use_immediate_repetition_detector: true
  immediate_repetition_weight: 0.05

  use_diversity_loss: true
  diversity_weight: 0.01

generation:
  temperature: 1.2
  top_p: 0.95
  repetition_penalty: 1.1
  no_repeat_ngram_size: 3
```

### 2. Train with Coherence Monitoring

```bash
python code/scripts/5_training/train.py --config code/configs/moe/tiny_moe_ultra_low_mem.yaml
```

Coherence metrics will be:
- Logged to console during training
- Tracked in WandB (if enabled)
- Saved in training logs

### 3. Evaluate Coherence

```bash
# Evaluate a trained model
python code/scripts/evaluation/evaluate_coherence.py \
    --model-path code/outputs/runs/YOUR_RUN/checkpoints/latest_model.pt
```

### 4. Filter Training Data (Optional)

```bash
# Filter data by coherence before training
python code/scripts/2_data_prep/filter_by_coherence.py \
    --input code/data/processed/raw_data.jsonl \
    --output code/data/processed/filtered_data.jsonl \
    --max-repetition 0.5 \
    --min-distinct-2 0.3
```

---

## Coherence Metrics

This codebase implements 7 research-backed coherence metrics:

### 1. **Distinct-n** (Li et al., 2016)

Measures vocabulary diversity by counting unique n-grams.

```python
distinct_n = unique_ngrams / total_ngrams
```

**Targets:**
- Distinct-1 (unigrams): >0.5
- Distinct-2 (bigrams): >0.7
- Distinct-4 (4-grams): >0.8

**What it means:**
- High = Diverse vocabulary
- Low = Repetitive vocabulary

### 2. **Repetition Ratio**

Percentage of repeated n-grams in the output.

```python
repetition = 1.0 - (unique_4grams / total_4grams)
```

**Target:** <0.3 (30%)

**What it means:**
- High = Repetitive text
- Low = Varied text

### 3. **Shannon Entropy**

Measures unpredictability in token distribution.

```python
entropy = -Σ(p(token) * log2(p(token)))
```

**Target:** >4.0

**What it means:**
- High = Diverse, unpredictable tokens
- Low = Monotonous, predictable tokens

### 4. **Burstiness** (Goh & Barabási, 2008)

Measures uniformity of token usage over time.

```python
burstiness = (σ - μ) / (σ + μ)
```

**Target:** <0.5

**What it means:**
- High = Bursty (tokens appear in clusters)
- Low = Uniform distribution

### 5. **Zipf Coefficient**

Measures adherence to Zipf's law (natural language distribution).

```python
zipf = slope of log(rank) vs log(frequency)
```

**Target:** 0.8-1.2

**What it means:**
- 1.0 = Perfect natural language distribution
- Far from 1.0 = Unnatural distribution

### 6. **Overall Coherence Score**

Composite score (0-100) combining all metrics with optimal weighting.

**Thresholds:**
- 75-100: ✅ Excellent
- 50-75: ⚠️ Moderate
- 0-50: ❌ Poor

### 7. **Perplexity** (Optional)

Language model confidence on held-out data.

```python
perplexity = exp(cross_entropy_loss)
```

**Target:** Lower is better (depends on model size)

---

## Improvement Strategies

### Strategy 1: Generation Parameters (Easiest)

**What:** Tune sampling parameters at inference time
**Effort:** Low
**Impact:** Medium-High

**Key Parameters:**

```yaml
generation:
  temperature: 1.2      # Higher = more diverse (0.8-1.5)
  top_p: 0.95          # Nucleus sampling (0.9-0.98)
  top_k: 50            # Top-k sampling (0 or 40-100)
  repetition_penalty: 1.1  # Penalize repetition (1.0-1.3)
  no_repeat_ngram_size: 3  # Block n-gram repetition (2-4)
```

**When to use:**
- Quick experiments
- Post-training tuning
- Different use cases need different settings

---

### Strategy 2: Training Loss Functions (Recommended)

**What:** Use coherence-aware losses during training
**Effort:** Low (config change)
**Impact:** High

**Available Losses:**

#### A. **DeepSeek Loss** (Primary - Recommended)
```yaml
losses:
  use_deepseek_loss: true
  temperature: 1.0
  adaptive_temperature: true  # Adjust during training
  label_smoothing: 0.1       # Prevent overconfidence
  eos_penalty_weight: 0.1    # Prevent premature stopping
```

**Benefits:**
- Adaptive temperature prevents mode collapse
- Label smoothing improves generalization
- EOS penalty prevents truncation

#### B. **N-gram Repetition Penalty**
```yaml
losses:
  use_ngram_penalty: true
  ngram_size: 4              # 4-gram recommended
  ngram_penalty_weight: 0.1  # Start with 0.05-0.15
```

**Benefits:**
- Directly penalizes repeated phrases
- Configurable n-gram size
- Low overhead

#### C. **Immediate Repetition Detector**
```yaml
losses:
  use_immediate_repetition_detector: true
  immediate_repetition_weight: 0.05  # Start with 0.01-0.1
```

**Benefits:**
- Prevents consecutive identical tokens
- Catches token-level repetition
- Very fast

#### D. **Diversity Loss**
```yaml
losses:
  use_diversity_loss: true
  diversity_weight: 0.01     # Start with 0.005-0.02
```

**Benefits:**
- Encourages diverse expert representations (MoE)
- Promotes vocabulary diversity
- Synergizes with other losses

#### E. **Adaptive Loss Scaling**
```yaml
losses:
  adaptive_loss_scaling: true
```

**Benefits:**
- Automatically balances multiple losses
- Prevents any one loss from dominating
- No manual tuning needed

**Recommended Combination:**
```yaml
losses:
  # Primary
  primary_loss_type: deepseek
  use_deepseek_loss: true
  adaptive_temperature: true
  label_smoothing: 0.1

  # Anti-repetition
  use_ngram_penalty: true
  ngram_penalty_weight: 0.1
  use_immediate_repetition_detector: true
  immediate_repetition_weight: 0.05

  # Diversity
  use_diversity_loss: true
  diversity_weight: 0.01

  # Auto-balance
  adaptive_loss_scaling: true
```

---

### Strategy 3: Data Quality Filtering (High Impact)

**What:** Filter training data to remove low-quality samples
**Effort:** Medium
**Impact:** Very High

**Use the filter script:**

```bash
python code/scripts/2_data_prep/filter_by_coherence.py \
    --input raw_data.jsonl \
    --output filtered_data.jsonl \
    --min-length 5 \
    --max-length 2048 \
    --max-repetition 0.5 \
    --min-distinct-2 0.3 \
    --min-entropy 2.0
```

**What gets filtered:**
- ❌ Duplicates (exact and near-duplicates)
- ❌ Too short (<5 tokens)
- ❌ Too long (>2048 tokens)
- ❌ High repetition (>50%)
- ❌ Low diversity (distinct-2 <0.3)
- ❌ Low entropy (<2.0)

**Expected results:**
- 10-30% of data filtered (typical)
- Higher quality training data
- Better model coherence
- Faster convergence

**When to use:**
- Before training new models
- When using scraped/noisy data
- When coherence is poor despite good losses

---

### Strategy 4: Training Configuration

**What:** Optimize training hyperparameters
**Effort:** Low (config change)
**Impact:** Medium-High

**Key Settings:**

```yaml
training:
  max_steps: 50000        # Train long enough! (50K-100K+)
  warmup_steps: 5000      # 10% of max_steps
  learning_rate: 0.0002   # Standard for MoE
  gradient_accumulation_steps: 8

  # Monitoring
  compute_coherence_metrics: true
  generation_during_eval: true
  test_generation_quality: true
  eval_steps: 1000        # Check coherence often
```

**Common mistakes:**
- ❌ Training too few steps (model undertrained)
- ❌ Too high learning rate (unstable training)
- ❌ No warmup (early training instability)
- ❌ Not monitoring coherence during training

---

### Strategy 5: Model Architecture (Advanced)

**What:** Architectural improvements for coherence
**Effort:** High (code changes)
**Impact:** High

**Already Implemented:**

1. **Rotary Position Embeddings (RoPE)**
   - Better long-range dependencies
   - Improves multi-sentence coherence

2. **Flash Attention**
   - More efficient attention
   - Better gradient flow

3. **Expert Load Balancing**
   - Prevents expert collapse
   - Maintains model capacity

**Potential Improvements:**

1. **Sequence-Level Coherence Modeling**
   - Cross-sentence attention
   - Discourse-aware position encodings

2. **Expert Specialization**
   - Dedicate experts to coherence aspects
   - Local vs global coherence experts

3. **Contrastive Learning**
   - Train on coherent vs incoherent pairs
   - Use ContrastiveLoss (already implemented)

---

### Strategy 6: Dataset Diversity (Recommended)

**What:** Train on diverse, high-quality data
**Effort:** Medium
**Impact:** High

**Current Status:**
- Primary dataset: OpenAssistant OASST2 (conversational)
- ⚠️ Limited diversity

**Recommendations:**

1. **Add diverse domains:**
   - Technical documentation
   - Narrative/stories
   - Reasoning tasks
   - Code
   - Scientific text

2. **Balance dataset composition:**
   - 40% conversational
   - 20% narrative
   - 20% informative/factual
   - 20% reasoning/problem-solving

3. **Quality over quantity:**
   - Use filtered, high-quality data
   - Remove duplicates
   - Filter by coherence metrics

---

## Configuration Guide

### Minimal Coherence Config

```yaml
# Minimal config for basic coherence
losses:
  use_deepseek_loss: true
  label_smoothing: 0.1
  use_ngram_penalty: true

generation:
  temperature: 1.2
  repetition_penalty: 1.1
```

### Recommended Coherence Config

```yaml
# Recommended for best results
training:
  compute_coherence_metrics: true
  generation_during_eval: true
  test_generation_quality: true
  num_generations_per_step: 3

losses:
  primary_loss_type: deepseek
  use_deepseek_loss: true
  adaptive_temperature: true
  label_smoothing: 0.1

  use_ngram_penalty: true
  ngram_penalty_weight: 0.1

  use_immediate_repetition_detector: true
  immediate_repetition_weight: 0.05

  use_diversity_loss: true
  diversity_weight: 0.01

  adaptive_loss_scaling: true

generation:
  temperature: 1.2
  top_p: 0.95
  repetition_penalty: 1.1
  no_repeat_ngram_size: 3
  min_length: 10
```

### Aggressive Anti-Repetition Config

```yaml
# For models with severe repetition issues
losses:
  use_ngram_penalty: true
  ngram_penalty_weight: 0.2     # Increased

  use_immediate_repetition_detector: true
  immediate_repetition_weight: 0.1  # Increased

  use_diversity_loss: true
  diversity_weight: 0.02        # Increased

generation:
  repetition_penalty: 1.3       # Increased
  no_repeat_ngram_size: 4       # Increased
```

---

## Tools & Scripts

### 1. Training with Coherence Monitoring

**Script:** `code/scripts/5_training/train.py`

**Features:**
- ✅ Automatic coherence metrics during validation
- ✅ WandB logging of all coherence metrics
- ✅ Console output with pass/fail indicators
- ✅ Sample generation for visual inspection

**Usage:**
```bash
python code/scripts/5_training/train.py --config your_config.yaml
```

**Output:**
```
🎯 Testing generation quality...
  Repetition: 25.3% (lower=better)
  Avg Length: 48 tokens
  Coherence: 72/100 (⚠️ Moderate)
    • Distinct-2: 0.685 ❌
    • Repetition: 0.253 ✅
    • Entropy: 4.52 ✅
  Sample: "Once upon a time, there was a..."
```

---

### 2. Coherence Evaluation

**Script:** `code/scripts/evaluation/evaluate_coherence.py`

**Features:**
- ✅ Comprehensive multi-prompt evaluation
- ✅ Detailed metrics report
- ✅ Actionable recommendations
- ✅ JSON export for tracking

**Usage:**
```bash
# Basic evaluation
python code/scripts/evaluation/evaluate_coherence.py \
    --model-path checkpoints/latest_model.pt

# With custom prompts
python code/scripts/evaluation/evaluate_coherence.py \
    --model-path checkpoints/latest_model.pt \
    --prompts my_prompts.txt \
    --output results.json

# Custom generation params
python code/scripts/evaluation/evaluate_coherence.py \
    --model-path checkpoints/latest_model.pt \
    --temperature 1.5 \
    --top-p 0.98 \
    --max-length 300
```

**Output:**
```
================================================================================
COHERENCE EVALUATION REPORT
================================================================================
Timestamp: 2025-11-09T15:30:00
Number of prompts: 15
Average length: 127.3 tokens

Generation Parameters:
  • max_length: 200
  • temperature: 1.2
  • top_p: 0.95
  • repetition_penalty: 1.1

Coherence Metrics:
  Overall Score: 78.5/100 ✅ EXCELLENT

Detailed Metrics:
  • Distinct-1: 0.623 (target: >0.5)
  • Distinct-2: 0.752 (target: >0.7)
  • Distinct-4: 0.891 (target: >0.8)
  • Repetition: 0.245 (target: <0.3)
  • Entropy: 4.82 (target: >4.0)
  • Burstiness: 0.412 (target: <0.5)
  • Zipf Coefficient: 0.987 (target: 0.8-1.2)

Recommendations:
  ✅ Coherence metrics are within acceptable ranges
================================================================================
```

---

### 3. Data Quality Filtering

**Script:** `code/scripts/2_data_prep/filter_by_coherence.py`

**Features:**
- ✅ Duplicate removal (exact + near-duplicates)
- ✅ Length filtering
- ✅ Repetition filtering
- ✅ Diversity filtering
- ✅ Entropy filtering
- ✅ Detailed statistics

**Usage:**
```bash
# Basic filtering
python code/scripts/2_data_prep/filter_by_coherence.py \
    --input raw_data.jsonl \
    --output filtered_data.jsonl

# Custom thresholds
python code/scripts/2_data_prep/filter_by_coherence.py \
    --input raw_data.jsonl \
    --output filtered_data.jsonl \
    --min-length 10 \
    --max-length 1024 \
    --max-repetition 0.4 \
    --min-distinct-2 0.4 \
    --min-entropy 2.5

# Disable duplicate removal
python code/scripts/2_data_prep/filter_by_coherence.py \
    --input raw_data.jsonl \
    --output filtered_data.jsonl \
    --no-dedup
```

**Output:**
```
============================================================
FILTERING STATISTICS
============================================================
Total samples: 100000
Passed filters: 73421 (73.4%)

Filtered out:
  • Too short: 1234 (1.2%)
  • Too long: 892 (0.9%)
  • Duplicates: 15234 (15.2%)
  • High repetition: 6789 (6.8%)
  • Low diversity: 2103 (2.1%)
  • Low entropy: 327 (0.3%)
============================================================
```

---

### 4. Coherence Metrics Library

**Module:** `code/src/Ava/evaluation/coherence_metrics.py`

**Quick usage in code:**

```python
from Ava.evaluation import quick_coherence_test

# Generate some text
generated_token_lists = [
    [1, 2, 3, 4, 5, ...],  # Token IDs
    [6, 7, 8, 9, 10, ...],
]

# Compute all coherence metrics
metrics = quick_coherence_test(generated_token_lists)

print(f"Coherence score: {metrics['coherence_score']}/100")
print(f"Distinct-2: {metrics['distinct_2']:.3f}")
print(f"Repetition: {metrics['repetition']:.3f}")
print(f"Entropy: {metrics['entropy']:.2f}")
```

---

## Best Practices

### ✅ DO

1. **Monitor coherence during training**
   - Enable `compute_coherence_metrics: true`
   - Check metrics every 1000 steps
   - Track trends over time

2. **Use multiple coherence losses**
   - DeepSeek + n-gram penalty + diversity loss
   - Enable adaptive loss scaling
   - Start with recommended weights

3. **Filter training data**
   - Remove duplicates
   - Filter by quality metrics
   - Use diverse data sources

4. **Tune generation parameters**
   - Start with recommended defaults
   - Adjust based on use case
   - Test multiple settings

5. **Evaluate regularly**
   - Use evaluation script
   - Test on diverse prompts
   - Compare against baselines

6. **Train long enough**
   - At least 50K steps for small models
   - 100K+ steps for larger models
   - Don't stop at first validation plateau

7. **Use proper warmup**
   - 10% of total training steps
   - Prevents early instability
   - Improves final coherence

### ❌ DON'T

1. **Don't ignore coherence metrics**
   - Perplexity alone is insufficient
   - Low perplexity ≠ good coherence

2. **Don't overtune repetition penalty**
   - >1.5 can make outputs unnatural
   - Start conservative (1.1-1.2)

3. **Don't train on unfiltered data**
   - Garbage in = garbage out
   - Always filter/deduplicate

4. **Don't use only one loss**
   - Multiple losses work synergistically
   - Single loss can cause overfitting

5. **Don't skip evaluation**
   - Numbers don't tell the whole story
   - Read generated samples
   - Get human feedback

6. **Don't change everything at once**
   - One change at a time
   - A/B test configurations
   - Track what works

---

## Troubleshooting

### Problem: High Repetition

**Symptoms:**
- Repetition ratio >0.4
- Distinct-2 <0.5
- Model repeats phrases

**Solutions:**
1. Enable n-gram penalty loss:
   ```yaml
   losses:
     use_ngram_penalty: true
     ngram_penalty_weight: 0.15  # Increase if needed
   ```

2. Increase repetition penalty:
   ```yaml
   generation:
     repetition_penalty: 1.3
     no_repeat_ngram_size: 4
   ```

3. Check if model is undertrained:
   - Train longer (increase max_steps)
   - Verify loss is decreasing

4. Filter training data for repetition:
   ```bash
   python filter_by_coherence.py --max-repetition 0.3
   ```

---

### Problem: Low Diversity

**Symptoms:**
- Distinct-2 <0.7
- Low entropy <4.0
- Monotonous outputs

**Solutions:**
1. Increase temperature:
   ```yaml
   generation:
     temperature: 1.3  # Increase from 1.2
   ```

2. Enable diversity loss:
   ```yaml
   losses:
     use_diversity_loss: true
     diversity_weight: 0.02  # Increase
   ```

3. Use nucleus sampling:
   ```yaml
   generation:
     top_p: 0.95
     top_k: 0  # Disable top-k
   ```

4. Check data diversity:
   - Add more diverse datasets
   - Balance domain distribution

---

### Problem: Incoherent/Nonsensical

**Symptoms:**
- Output doesn't make sense
- Topic drift
- Grammatical errors

**Solutions:**
1. Reduce temperature:
   ```yaml
   generation:
     temperature: 1.0  # Decrease from 1.2
   ```

2. Reduce repetition penalty:
   ```yaml
   generation:
     repetition_penalty: 1.0  # Disable
   ```

3. Enable label smoothing:
   ```yaml
   losses:
     label_smoothing: 0.15  # Increase
   ```

4. Train longer:
   - Model may be undertrained
   - Check if validation loss is still decreasing

---

### Problem: Premature EOS

**Symptoms:**
- Generations are too short
- Model stops mid-sentence
- Average length <20 tokens

**Solutions:**
1. Enable EOS penalty:
   ```yaml
   losses:
     eos_penalty_weight: 0.15  # Increase
   ```

2. Set minimum length:
   ```yaml
   generation:
     min_length: 20
   ```

3. Reduce EOS token probability:
   - Check training data for short samples
   - Filter data by minimum length

---

### Problem: Expert Collapse (MoE)

**Symptoms:**
- Only 1-2 experts used
- Poor model capacity
- Low coherence

**Solutions:**
1. Increase load balancing:
   ```yaml
   model:
     load_balance_loss_coef: 0.1  # Increase from 0.05
   ```

2. Add router jitter:
   ```yaml
   model:
     router_jitter_noise: 0.02  # Increase from 0.01
   ```

3. Enable diversity loss:
   ```yaml
   losses:
     use_diversity_loss: true
     diversity_weight: 0.015
   ```

---

### Problem: Mode Collapse

**Symptoms:**
- Model always generates similar text
- Ignores prompt
- Very low distinct-1

**Solutions:**
1. Use adaptive temperature:
   ```yaml
   losses:
     adaptive_temperature: true
   ```

2. Increase training diversity:
   - Add more diverse data
   - Filter duplicates more aggressively

3. Reduce model capacity:
   - May be overfitting
   - Increase dropout
   - Add regularization

---

## References

### Papers

1. **Distinct-n Metrics**
   - Li et al. (2016): "A Diversity-Promoting Objective Function for Neural Conversation Models"

2. **Burstiness**
   - Goh & Barabási (2008): "Burstiness and Memory in Complex Systems"

3. **Zipf's Law**
   - Zipf (1949): "Human Behavior and the Principle of Least Effort"

4. **Mixture of Experts**
   - Shazeer et al. (2017): "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"

5. **DeepSeek MoE**
   - DeepSeek-AI (2024): "DeepSeekMoE: Towards Ultimate Expert Specialization"

### Documentation

- [Training Guide](code/docs/02_TRAINING_GUIDE.md)
- [Loss Functions](code/docs/04_LOSS_FUNCTIONS.md)
- [Evaluation & Generation](code/docs/06_EVALUATION_GENERATION.md)
- [Coherence Metrics](code/src/Ava/evaluation/coherence_metrics.py)

---

## Summary

### ✅ Implemented Features

- [x] 7 coherence metrics (distinct-n, entropy, burstiness, Zipf, etc.)
- [x] 5+ coherence-aware loss functions
- [x] Comprehensive evaluation suite
- [x] Data quality filtering
- [x] WandB integration for coherence tracking
- [x] Generation parameter optimization
- [x] Advanced architecture (RoPE, Flash Attention)

### 📊 Expected Results

With proper configuration, you should achieve:

- **Coherence Score:** 70-85/100
- **Distinct-2:** 0.7-0.85
- **Repetition:** 0.2-0.3
- **Entropy:** 4.0-5.5
- **Burstiness:** 0.3-0.5
- **Zipf:** 0.9-1.1

### 🚀 Next Steps

1. ✅ **Phase 1 (Complete):** Enable coherence features in config
2. ✅ **Phase 2 (Complete):** Add data quality filtering
3. ⏭️ **Phase 3:** Train with coherence monitoring
4. ⏭️ **Phase 4:** Evaluate and iterate on results
5. ⏭️ **Phase 5:** Expand dataset diversity
6. ⏭️ **Phase 6:** Experiment with advanced architectures

---

**For questions or issues:**
- Check [Troubleshooting](#troubleshooting) section
- Review training logs for coherence metrics
- Run evaluation script for detailed analysis

**Good luck improving your LLM coherence! 🎯**
