# Perplexity Measurement Fix

## Problem

Perplexity was not being measured correctly because **padding tokens were included in the loss calculation**. This resulted in inaccurate perplexity scores.

### Root Cause

In `code/src/ava/eval/coherence.py`, both the `CoherenceMeasurer` and `FastCoherenceMeasurer` classes were computing perplexity without using attention masks:

**Before (INCORRECT):**
```python
# CoherenceMeasurer._compute_perplexity
def _compute_perplexity(self, input_ids: torch.Tensor) -> float:
    labels = input_ids[:, 1:].contiguous()
    inputs = input_ids[:, :-1].contiguous()

    # No attention mask!
    outputs = self.model(inputs)

    # Loss computed on ALL tokens, including padding
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction='mean'
    )
    ...
```

This meant:
- Padding tokens (usually token ID 0) were treated as real tokens
- Loss was computed on padding positions, corrupting the perplexity value
- Sequences with different amounts of padding would give different perplexity scores even for the same content

## Solution

Updated both perplexity calculation methods to:
1. **Create attention masks** to identify real vs padding tokens
2. **Pass attention masks to the model** during forward pass
3. **Mask out padding tokens** in the loss calculation

**After (CORRECT):**
```python
# CoherenceMeasurer._compute_perplexity
def _compute_perplexity(self, input_ids: torch.Tensor) -> float:
    # Create attention mask (1 for real tokens, 0 for padding)
    pad_token_id = 0
    if self.tokenizer is not None and hasattr(self.tokenizer, 'pad_token_id'):
        if self.tokenizer.pad_token_id is not None:
            pad_token_id = self.tokenizer.pad_token_id

    attention_mask = (input_ids != pad_token_id).long()

    labels = input_ids[:, 1:].contiguous()
    inputs = input_ids[:, :-1].contiguous()
    attention_mask_shifted = attention_mask[:, :-1].contiguous()

    # Forward pass WITH attention mask
    outputs = self.model(inputs, attention_mask=attention_mask_shifted)

    # Create loss mask to exclude padding
    loss_mask = attention_mask[:, 1:].contiguous()
    labels_masked = labels_flat.clone()
    labels_masked[loss_mask_flat == 0] = -100  # Ignore padding

    # Loss computed only on real tokens
    loss = F.cross_entropy(
        logits_flat,
        labels_masked,
        ignore_index=-100,
        reduction='mean'
    )
    ...
```

## Files Changed

1. **[code/src/ava/eval/coherence.py](code/src/ava/eval/coherence.py):**
   - `CoherenceMeasurer._compute_perplexity()` - Lines 416-481
   - `FastCoherenceMeasurer._measure_single_batch()` - Lines 841-890
   - `FastCoherenceMeasurer._compute_perplexity_from_logits()` - Lines 1053-1103

## Impact

### Before Fix:
- Perplexity values were **artificially biased** by padding tokens
- Different padding amounts gave different perplexity scores for identical content
- Model selection based on perplexity was unreliable

### After Fix:
- Perplexity now **correctly excludes padding tokens**
- Perplexity is computed only on actual content
- Consistent perplexity regardless of sequence padding
- More accurate model quality assessment

## Testing

Verified the fix with tests showing:
- ✓ Padding tokens are correctly excluded from loss calculation
- ✓ Perplexity is consistent regardless of padding amount
- ✓ Both `CoherenceMeasurer` and `FastCoherenceMeasurer` work correctly

```python
# Test results
Test 1: Sequence WITHOUT padding
  Perplexity: 100.0000

Test 2: Same sequence WITH padding
  Perplexity: 100.0000

✓ Perplexity matches regardless of padding!
```

## Related Metrics

This fix ensures that:
- **Validation loss** measurement (which already used attention masks) and **perplexity** are now consistent
- **Model quality scores** (which combine val_loss, coherence, and perplexity) are more accurate
- **Best model selection** is based on correct metrics

## Migration Notes

No action required - this is a pure bug fix. The next time you run training:
- Perplexity values will be more accurate
- They may differ from previous runs (previous values were incorrect)
- Model selection will be more reliable
