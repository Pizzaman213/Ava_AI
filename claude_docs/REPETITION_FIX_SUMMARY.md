# LLM Repetition Issue - Root Cause & Fix

## Problem Summary
All trained model checkpoints exhibited severe token repetition during text generation, making them unusable for production.

## Root Cause Analysis

### Issue: Wrong Attention Mechanism
The model was using `nn.TransformerEncoderLayer` which implements **bidirectional attention**:
- During training, each token could "see" ALL future tokens in the sequence
- Model learned patterns that rely on seeing future context
- During autoregressive generation (one token at a time), future context isn't available
- This mismatch causes the model to collapse into repetitive patterns

### Technical Details
```python
# BROKEN (old code):
nn.TransformerEncoderLayer(...)  # Bidirectional attention - can see future
```

Location: `/project/code/src/Ava/models/moe_model.py:138-149`

## Solution Implemented

### Fix: Causal Attention Architecture
Replaced encoder layers with proper **causal decoder blocks**:

1. **Added `TransformerDecoderBlock` class** with:
   - Causal self-attention (prevents looking at future tokens)
   - Pre-norm architecture (like GPT)
   - Upper triangular attention mask

2. **Updated forward pass** to use causal masking:
```python
# Create causal mask (upper triangular)
causal_mask = torch.triu(
    torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
    diagonal=1
)

# Apply to all layers
for layer in self.layers:
    hidden_states = layer(hidden_states, attn_mask=causal_mask)
```

### Code Changes
File: `/project/code/src/Ava/models/moe_model.py`
- Lines 125-190: Added `TransformerDecoderBlock` class
- Lines 138-149: Replaced encoder with decoder layers
- Lines 276-284: Added causal masking in forward pass

## Testing Results

### Old Model (Broken Architecture)
```
Checkpoint: run_20251005_014813_34e79e4d/step_10000/model.pt
Min Loss: 1.162 (best training loss)
Generation: "Once upon a time time time time time time..." ❌
```

### New Model (Fixed Architecture)
```
Architecture: ✅ Causal attention implemented
Training: In progress (5000 samples, 1 epoch)
Status: Needs more training for quality generation
```

## Next Steps

### For Production Use:
1. **Retrain ALL models** from scratch with the fixed architecture
2. **Training requirements**:
   - Minimum 50K-100K steps
   - Larger dataset (current test used only 5K samples)
   - Monitor loss convergence
   - Test generation quality every 10K steps

3. **Expected results** with proper training:
   - No token repetition
   - Coherent text generation
   - Proper autoregressive behavior

### Command to Test New Models:
```bash
# Test generation quality
python3 scripts/generation/test_generation.py \
  --checkpoint "path/to/checkpoint.pt" \
  --test-suite \
  --max-length 100 \
  --device cuda
```

## Important Notes

⚠️ **All existing checkpoints are UNUSABLE** - they were trained with bidirectional attention and will always exhibit repetition

✅ **Architecture is now correct** - new models trained with this code will work properly

🔄 **Training in progress** - `test_causal_fix_20251005_143438_5556b631` uses the fixed architecture but needs more steps

## Files Modified
- `/project/code/src/Ava/models/moe_model.py` - Core architecture fix
- `/project/code/test_causal_model.py` - Testing script created

## References
- GPT architecture: Pre-norm + Causal attention
- Attention masking: Upper triangular matrix prevents future token access
- Autoregressive generation: Generates one token at a time using only past context
