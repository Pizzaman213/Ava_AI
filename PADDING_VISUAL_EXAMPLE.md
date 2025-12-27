# Visual Example: How Your Text Becomes a Padded Batch

## Example: Processing 3 TinyStories

### Step 1: Original Stories
```
Story A: "Once upon a time there was a cat."
Story B: "The dog."
Story C: "A bird flew in the sky today."
```

### Step 2: Tokenization (using your tokenizer)
```
Story A → [542, 891, 12, 234, 567, 89, 12, 423]      (8 tokens)
Story B → [234, 567]                                  (2 tokens)
Story C → [12, 987, 654, 32, 234, 567, 321]          (7 tokens)
```

### Step 3: Add BOS (2) and EOS (1)
```
Story A: [2, 542, 891, 12, 234, 567, 89, 12, 423, 1]     (10 tokens)
Story B: [2, 234, 567, 1]                                (4 tokens)
Story C: [2, 12, 987, 654, 32, 234, 567, 321, 1]         (9 tokens)
```

### Step 4: Create Attention Masks (1 = real, 0 = padding)
```
Story A mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]             (10 ones)
Story B mask: [1, 1, 1, 1]                               (4 ones)
Story C mask: [1, 1, 1, 1, 1, 1, 1, 1, 1]                (9 ones)
```

### Step 5: Pad to max_length (let's say 16 for this example)
```
Story A: [2, 542, 891, 12, 234, 567, 89, 12, 423, 1, 0, 0, 0, 0, 0, 0]
          ^                                          ^  ^              ^
          BOS         real tokens                   EOS    PAD tokens

Mask A:  [1,   1,   1,  1,   1,   1,  1,  1,   1, 1, 0, 0, 0, 0, 0, 0]
          ^                                          ^  ^              ^
          real tokens (attend to these)              real PAD (ignore)


Story B: [2, 234, 567, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          ^            ^  ^                                  ^
          BOS   real  EOS         PAD tokens

Mask B:  [1,   1,   1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
          ^            ^  ^                                  ^
          real tokens  real PAD (ignore all these)


Story C: [2, 12, 987, 654, 32, 234, 567, 321, 1, 0, 0, 0, 0, 0, 0, 0]
          ^                                   ^  ^                   ^
          BOS         real tokens            EOS     PAD tokens

Mask C:  [1,  1,   1,   1,  1,   1,   1,   1, 1, 0, 0, 0, 0, 0, 0, 0]
          ^                                   ^  ^                   ^
          real tokens                        real PAD (ignore)
```

### Step 6: Stack into Batch
```python
batch = {
    'input_ids': [
        [2, 542, 891, 12, 234, 567, 89, 12, 423, 1, 0, 0, 0, 0, 0, 0],  # Story A
        [2, 234, 567, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],           # Story B
        [2, 12, 987, 654, 32, 234, 567, 321, 1, 0, 0, 0, 0, 0, 0, 0],   # Story C
    ],  # Shape: [3, 16]
    
    'attention_mask': [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Story A mask
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Story B mask
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # Story C mask
    ],  # Shape: [3, 16]
}
```

## What Happens in the Model

### Attention Masking (for Story B, token position 2)

Token 2 can attend to:
```
Position:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
Input:    [2, 234, 567, 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
          BOS word word EOS PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD PAD

Causal:   [✓   ✓   ✓   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗]
          Can see past    Can't see future (autoregressive)

Padding:  [✓   ✓   ✓   ✓   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗]
          Real tokens     Padding (ignore)

Final:    [✓   ✓   ✓   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗   ✗]
          Attend to      Don't attend (future or padding)
```

So token 2 ("567") can only see:
- Position 0 (BOS: 2)
- Position 1 (234)
- Position 2 (567, itself)

It CANNOT see:
- Position 3 (EOS) - future token
- Positions 4-15 (PAD) - padding tokens

### Loss Calculation

```
Input IDs:  [2, 234, 567, 1,   0,   0,   0, ...]
Labels:     [2, 234, 567, 1, -100, -100, -100, ...]
                           ^   ^^^^  ^^^^  ^^^^
                          real  ignored in loss

Loss is computed as:
- Cross-entropy between predicted and actual for positions 0-3
- Positions 4-15 are ignored (label = -100)
```

## Efficiency Comparison

### WITHOUT Sequence Packing (your old setup)
```
Batch of 3 stories:
Story A: 10 real + 6 PAD  = 16 tokens  (38% waste)
Story B:  4 real + 12 PAD = 16 tokens  (75% waste)
Story C:  9 real + 7 PAD  = 16 tokens  (44% waste)
────────────────────────────────────────────────
Total:   23 real + 25 PAD = 48 tokens  (52% waste) ❌
```

### WITH Sequence Packing (your current setup)
```
Packed sequence:
[Story A (10) | Story B (4) | Story C (9)] + 9 PAD = 32 tokens
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         23 real tokens

Total:   23 real + 9 PAD = 32 tokens  (28% waste) ✓
Savings: 33% less memory, 33% faster training!
```

## Your Actual Data (from analysis)

```
Average story: 215.5 tokens
Max length: 512 tokens
Padding overhead: 296.5 tokens (58% waste)

WITH sequence packing enabled:
Effective waste: ~15-20% (much better!)
```

## Key Takeaways

1. **PAD token (0)** fills empty space to make all sequences the same length
2. **Attention mask** tells model which positions are real (1) vs padding (0)
3. **Causal mask** prevents seeing future tokens (autoregressive language modeling)
4. **Loss masking** excludes padding from gradient calculation (label = -100)
5. **Your setup uses sequence packing** → 70% less waste → faster training!

## See Also

- Full explanation: [code/docs/PADDING_EXPLAINED.md](code/docs/PADDING_EXPLAINED.md)
- Config reference: [code/configs/moe/minimal_working.yaml](code/configs/moe/minimal_working.yaml)
- Implementation: [code/src/ava/core/data_utils.py](code/src/ava/core/data_utils.py)
