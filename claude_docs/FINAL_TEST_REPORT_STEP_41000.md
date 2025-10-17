# Training Collapse Analysis - Step 26000

## SEVERE REPETITION COLLAPSE DETECTED

**Repetition: 97.9% at Step 26,000**

The model has collapsed into repeating only the input prompt with no new generation.

## Collapse Timeline

Step 17k: 0.0% → ✅ Healthy
Step 18k: 1.4% → ⚠️ Starting  
Step 19k: 5.6% → ⚠️ Growing
Step 20k: 36.1% → COLLAPSING
Step 21k: 44.4% → Severe
Step 22k: 77.1% → Critical
Step 23k+: 97.9% → COLLAPSED

## Root Cause

**The EOS/repetition penalties we added were TOO STRONG.**

This created a vicious cycle:
1. Strong penalties discourage EOS + repetition
2. Model learns "safest" output = copy the prompt
3. Loss still decreases (model "succeeding")
4. Repetition becomes dominant strategy
5. Full collapse

## The Fix

**Reduce all penalties to find balanced middle ground**

Current (TOO STRONG):
- min_sequence_length: 40
- eos_penalty_weight: 10.0
- eos_logit_bias: 5.0
- repetition_penalty_weight: 1.5
- immediate_repetition_weight: 3.0

Recommended (BALANCED):
- min_sequence_length: 30
- eos_penalty_weight: 3.0
- eos_logit_bias: 2.0
- repetition_penalty_weight: 1.2
- immediate_repetition_weight: 2.0
- learning_rate: 1.5e-06 (increased from 7.94e-07)

## Next Steps

1. Stop current training (collapsed, not recoverable)
2. Update config with balanced penalties
3. Restart training
4. Monitor carefully at steps 15k-25k
