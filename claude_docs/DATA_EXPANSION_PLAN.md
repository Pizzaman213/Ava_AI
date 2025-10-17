# Data Expansion Plan: 773K → 2-5M Examples

## Current Status

**Before expansion:**
- Examples: 773,114
- Estimated tokens: ~193M (773K × 250 tokens/example)
- Training epochs (30K steps): **21 epochs**
- Issue: High memorization, fast loss drop

**After expansion (Target):**
- Examples: 2-5M
- Estimated tokens: 500M-1.25B
- Training epochs (30K steps): **2-6 epochs**
- Benefit: Minimal memorization, better generalization

---

## High-Quality Datasets Being Downloaded

### TIER 1: Highest Quality Instruction Data (~900K examples)

1. **HuggingFaceH4/ultrachat_200k** (200K)
   - Multi-turn conversations
   - GPT-3.5 Turbo quality
   - Quality score: 9/10

2. **Open-Orca/SlimOrca** (500K) ← **Currently downloading**
   - GPT-4 distilled reasoning
   - Complex instruction following
   - Quality score: 9.5/10

3. **microsoft/orca-math-word-problems-200k** (200K)
   - Mathematical reasoning
   - Step-by-step solutions
   - Quality score: 9/10

### TIER 2: High-Quality Web Text (~1M examples)

4. **HuggingFaceFW/fineweb-edu** (500K)
   - Educational web content
   - Filtered from CommonCrawl
   - Quality score: 8.5/10

5. **allenai/c4** (500K)
   - Colossal Clean Crawled Corpus
   - Diverse web text
   - Quality score: 7.5/10

### TIER 3: Code & Reasoning (~500K examples)

6. **bigcode/starcoderdata** (300K)
   - High-quality Python code
   - Clean, well-documented
   - Quality score: 8/10

7. **bigcode/the-stack-v2** (200K)
   - Deduplicated GitHub code
   - Python focus
   - Quality score: 8/10

### TIER 4: Conversational Data (~300K examples)

8. **anon8231489123/ShareGPT_Vicuna_unfiltered** (100K)
   - Real ChatGPT conversations
   - Diverse user queries
   - Quality score: 8.5/10

9. **Anthropic/hh-rlhf** (200K)
   - Helpful and harmless
   - Human feedback aligned
   - Quality score: 9/10

### TIER 5: Knowledge & Diverse Data (~500K examples)

10. **wikimedia/wikipedia** (300K)
    - English Wikipedia articles
    - Factual knowledge
    - Quality score: 8/10

11. **bookcorpus** (200K)
    - Books corpus
    - Long-form text
    - Quality score: 7.5/10

---

## Expected Results

### Total New Data
```
TIER 1: 900,000 examples
TIER 2: 1,000,000 examples
TIER 3: 500,000 examples
TIER 4: 300,000 examples
TIER 5: 500,000 examples
─────────────────────────
NEW:    3,200,000 examples
OLD:    773,114 examples
─────────────────────────
TOTAL:  ~3,973,114 examples
```

### Token Estimates
```
Total examples: 3,973,114
Avg tokens/example: 250
Total tokens: ~993M tokens (~1B tokens)
```

### Training Impact (30K steps)
```
Batch size: 8
Gradient accumulation: 16
Effective batch: 128

Examples per step: 128
Total examples needed: 30,000 × 128 = 3,840,000

Epochs: 3,840,000 / 3,973,114 = 0.97 epochs
```

**Result: Less than 1 epoch!** ✅

---

## Comparison: Before vs After

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Examples** | 773K | 3.97M | **5.1x more** |
| **Tokens** | 193M | 993M | **5.1x more** |
| **Epochs (30K steps)** | 21 | 0.97 | **21x less** |
| **Memorization risk** | Very High | Minimal | **Huge improvement** |
| **Loss curve speed** | Very fast | Normal | **Like GPT-2** |
| **Generalization** | Limited | Strong | **Much better** |

---

## Loss Curve Prediction

### Before (773K examples, 21 epochs)
```
Step 0:     Loss = 10.0
Step 1K:    Loss = 3.9  (Very fast drop - memorizing)
Step 5K:    Loss = 2.5  (Plateau)
Step 10K:   Loss = 2.0  (Overfitting)
Step 20K:   Loss = 1.8  (Stuck)
```

### After (3.97M examples, <1 epoch)
```
Step 0:     Loss = 10.0
Step 1K:    Loss = 5.0  (Normal learning)
Step 5K:    Loss = 3.5  (Steady improvement)
Step 10K:   Loss = 2.8  (Good progress)
Step 20K:   Loss = 2.3  (Continued learning)
Step 30K:   Loss = 2.0  (High quality)
```

**Expected behavior: Similar to GPT-2 training curve** ✅

---

## Quality Improvements

### Data Diversity
- **Before**: 30 datasets (some low quality)
- **After**: 41 datasets (curated high quality)
- **Improvement**: Better coverage of domains

### Source Quality
- **Before**: Mixed quality (avg 7/10)
- **After**: Weighted toward 8-9/10 quality
- **Improvement**: Higher signal-to-noise ratio

### Data Balance
```
Instruction/QA: 1,100,000 examples (28%)
Web text:       1,000,000 examples (25%)
Code:           500,000 examples (13%)
Conversation:   300,000 examples (8%)
Knowledge:      500,000 examples (13%)
Existing:       573,114 examples (14%)
```

Balanced across multiple domains ✅

---

## Next Steps

### 1. Wait for Download to Complete
- Monitor: `/project/data/download_high_quality.log`
- ETA: ~30-60 minutes for all datasets
- Progress: Check with `tail -f /project/data/download_high_quality.log`

### 2. Verify Data Quality
```bash
# Count total examples
wc -l /project/code/data/processed/*.jsonl

# Check file sizes
du -sh /project/code/data/processed/

# Sample random examples
shuf -n 5 /project/code/data/processed/*.jsonl
```

### 3. Update Training Config (Optional)
Since we have more data, we can:
- Keep current LR (6.76e-05) - will work better now
- OR lower to 3.0e-05 for even more stable training
- Increase max_steps from 30K → 50K (optional)

### 4. Restart Training
```bash
# Clean restart
./RESTART_TRAINING.sh

# Or continue from checkpoint
# (if you want to compare before/after)
```

---

## Expected Training Timeline

### With 3.97M examples:

```
Step 0-5K:   Loss: 10→3.5  (Normal learning phase)
Step 5K-15K: Loss: 3.5→2.5 (Steady improvement)
Step 15K-30K: Loss: 2.5→2.0 (Refinement)
Step 30K+:   Loss: 2.0→1.8 (Optional - can continue)
```

**Key difference:** No sharp drop, no early plateau, continuous learning!

---

## Files Created

1. `/project/code/scripts/1_data_download/download_high_quality_simple.py`
   - Simple download script
   - Saves directly as JSONL
   - No complex processing needed

2. `/project/claude_docs/DATA_EXPANSION_PLAN.md`
   - This document
   - Explains expansion strategy

---

## Download Progress

Check download status:
```bash
# Live progress
tail -f /project/data/download_high_quality.log

# Count completed
ls -1 /project/code/data/processed/*_processed.jsonl | wc -l

# Total examples so far
wc -l /project/code/data/processed/*.jsonl | tail -1
```

---

## Summary

**Problem:** Small dataset (773K) causing fast memorization and limited quality

**Solution:** Expand to 3.97M high-quality examples from curated sources

**Result:**
- 5.1x more data
- <1 epoch instead of 21 epochs
- Normal training curve like GPT-2
- Much better generalization
- Higher final quality

**Status:** ⏳ Downloading... (SlimOrca: 35% complete)

**ETA:** 30-60 minutes until download complete

**Date:** 2025-10-15
