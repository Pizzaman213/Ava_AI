# How Padding Works in Ava Training

## Overview

**Padding** makes all sequences the same length so they can be efficiently batched together on GPUs. Without padding, you'd have to process each sequence one at a time (very slow).

## Special Tokens in Ava

Your model uses these special token IDs (defined in config):

```python
PAD_TOKEN_ID = 0  # Padding (ignored during training)
EOS_TOKEN_ID = 1  # End of sequence
BOS_TOKEN_ID = 2  # Beginning of sequence
```

## Step-by-Step: From Text to Padded Batch

### 1. Original Stories (Variable Length)

```
Story 1: "Once upon a time, there was a cat."        (9 words)
Story 2: "The dog ran fast."                         (4 words)
Story 3: "A long story about many things today."     (7 words)
```

### 2. Tokenization

```python
# Tokenize each story
story1_tokens = tokenizer.encode("Once upon a time, there was a cat.")
# Result: [1234, 567, 89, 234, 567, ...]  (let's say 42 tokens)

story2_tokens = tokenizer.encode("The dog ran fast.")
# Result: [890, 234, 567, 123]  (15 tokens)

story3_tokens = tokenizer.encode("A long story about many things today.")
# Result: [456, 789, 123, ...]  (28 tokens)
```

### 3. Add BOS/EOS Tokens

```python
# Add BOS (2) at start, EOS (1) at end
story1 = [2] + story1_tokens + [1]  # Length: 44 tokens
story2 = [2] + story2_tokens + [1]  # Length: 17 tokens
story3 = [2] + story3_tokens + [1]  # Length: 30 tokens
```

**Visual:**
```
Story 1: [BOS] Once upon a time... [EOS]           (44 tokens)
Story 2: [BOS] The dog ran fast [EOS]              (17 tokens)
Story 3: [BOS] A long story... [EOS]               (30 tokens)
```

### 4. Create Attention Mask (Before Padding)

```python
# Attention mask: 1 = real token, 0 = padding
attention_mask1 = [1, 1, 1, ..., 1]  # 44 ones
attention_mask2 = [1, 1, 1, ..., 1]  # 17 ones
attention_mask3 = [1, 1, 1, ..., 1]  # 30 ones
```

### 5. Pad to Max Length (512 in your case)

```python
max_length = 512

# Story 1: pad with 468 PAD tokens (512 - 44 = 468)
story1_padded = [2, 1234, 567, ..., 1] + [0, 0, 0, ..., 0]
#                ^^^^ 44 real tokens ^^^^   ^^^^ 468 PAD ^^^^

# Story 2: pad with 495 PAD tokens (512 - 17 = 495)
story2_padded = [2, 890, 234, 567, 123, 1] + [0, 0, 0, ..., 0]
#                ^^^^^ 17 real tokens ^^^^^   ^^^^ 495 PAD ^^^^

# Story 3: pad with 482 PAD tokens (512 - 30 = 482)
story3_padded = [2, 456, 789, ..., 1] + [0, 0, 0, ..., 0]
#                ^^^^ 30 real tokens ^^^^   ^^^^ 482 PAD ^^^^

# Update attention masks
attention_mask1 = [1, 1, ..., 1] + [0, 0, ..., 0]  # 44 ones + 468 zeros
attention_mask2 = [1, 1, ..., 1] + [0, 0, ..., 0]  # 17 ones + 495 zeros
attention_mask3 = [1, 1, ..., 0] + [0, 0, ..., 0]  # 30 ones + 482 zeros
```

**Visual:**
```
Story 1: [BOS][tokens...][EOS][PAD][PAD][PAD]...[PAD]
         |<---- 44 real ---->|<----- 468 padding --->|  = 512 total
Mask 1:  [ 1    1  ...  1  1 ][ 0    0  ...    0   ]

Story 2: [BOS][tokens][EOS][PAD][PAD][PAD]...[PAD]
         |<- 17 real ->|<-------- 495 padding ------->|  = 512 total
Mask 2:  [ 1  1  ...  1][ 0    0    0  ...      0   ]

Story 3: [BOS][tokens...][EOS][PAD][PAD]...[PAD]
         |<--- 30 real --->|<----- 482 padding ---->|  = 512 total
Mask 3:  [ 1   1  ...  1  ][ 0    0  ...       0  ]
```

### 6. Batch Together

Now all sequences are length 512, so they can be batched:

```python
batch = {
    'input_ids': torch.tensor([
        story1_padded,  # [512]
        story2_padded,  # [512]
        story3_padded,  # [512]
    ]),  # Shape: [3, 512]

    'attention_mask': torch.tensor([
        attention_mask1,  # [512]
        attention_mask2,  # [512]
        attention_mask3,  # [512]
    ]),  # Shape: [3, 512]
}
```

## How the Model Uses Attention Mask

### In Self-Attention

The model combines **two types of masks**:

1. **Causal Mask** - prevents looking at future tokens (autoregressive)
2. **Padding Mask** - prevents attending to PAD tokens

```python
# From moe.py line 962-987
# 1. Create causal mask (upper triangular)
causal_mask = [
    [  0,  -inf, -inf, -inf, ...],  # Token 0 can only see itself
    [  0,    0,  -inf, -inf, ...],  # Token 1 can see 0,1
    [  0,    0,    0,  -inf, ...],  # Token 2 can see 0,1,2
    [  0,    0,    0,    0,  ...],  # etc.
]

# 2. Create padding mask from attention_mask
# Where attention_mask=0 (padding), set to -inf
# Where attention_mask=1 (real token), set to 0
padding_mask = torch.where(
    attention_mask == 0,
    -inf,  # Don't attend to padding
    0      # OK to attend
)

# 3. Combine both masks
final_mask = causal_mask + padding_mask
```

**Example for Story 2** (17 real tokens + 495 padding):
```
Attention mask input: [1,1,1,...,1,0,0,0,...,0]
                       ^^ 17 ones ^^  ^^ 495 zeros ^^

Final attention weights after masking:
Token 0: can attend to [0]                    (can't see future, no padding yet)
Token 1: can attend to [0, 1]                 (can't see future, no padding yet)
...
Token 16: can attend to [0,1,2,...,16]        (can see all real tokens)
Token 17-511: IGNORED (these are padding)     (-inf in mask)
```

### In Loss Calculation

Padding tokens are excluded from loss:

```python
# From data_utils.py line 155
labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
#                                           ^^^^
# -100 is PyTorch's special "ignore this token" value

# Real tokens get copied
labels[i, :seq_len] = input_ids[i, :seq_len]

# Padding positions stay -100 (ignored in loss)
# So loss is only calculated on real tokens
```

**Example:**
```
Input IDs:  [2, 1234, 567, 123, 1, 0, 0, 0, ...]
Labels:     [2, 1234, 567, 123, 1, -100, -100, -100, ...]
            ^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^
            Used in loss               Ignored in loss
```

## Code Flow

### 1. Data Download/Preparation
```python
# download_tinystories_improved.py lines 263-284
tokens = [BOS_TOKEN_ID] + encoded.ids + [EOS_TOKEN_ID]
attention_mask = [1] * len(tokens)

# Pad to max_length (512)
padding_length = max_length - len(tokens)
if padding_length > 0:
    tokens = tokens + [PAD_TOKEN_ID] * padding_length
    attention_mask = attention_mask + [0] * padding_length
```

### 2. DataLoader Collation
```python
# data_utils.py lines 116-192
def collate_batch(batch, max_length, pad_token_id=0):
    # Create tensors filled with PAD_TOKEN_ID
    input_ids = torch.full((batch_size, max_length), pad_token_id)
    attention_mask = torch.zeros((batch_size, max_length))
    labels = torch.full((batch_size, max_length), -100)

    # Fill in real data
    for i, item in enumerate(batch):
        seq_len = len(item['input_ids'])
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = item['attention_mask']
        labels[i, :seq_len] = item['input_ids']  # Autoregressive: predict next token
```

### 3. Model Forward Pass
```python
# moe.py lines 962-987
# Combine causal and padding masks
causal_mask = self._get_causal_mask(seq_len, device, dtype)
if attention_mask is not None:
    # Convert padding mask to -inf for masked positions
    padding_mask = torch.where(
        attention_mask[:, None, None, :] == 0,
        torch.finfo(dtype).min,  # -inf (effectively)
        0.0
    )
    attention_mask = causal_mask + padding_mask
```

### 4. Attention Computation
```python
# moe.py lines 455-459
attn_weights = torch.matmul(q, k.transpose(-2, -1)) / sqrt(head_dim)

# Apply mask (masked positions become -inf)
if attention_mask is not None:
    attn_weights = attn_weights + attention_mask

# Softmax turns -inf into 0 probability
attn_weights = F.softmax(attn_weights, dim=-1)
# Padded positions now have weight=0 (don't contribute to output)
```

## Why Padding is Necessary

### Without Padding (Sequential Processing)
```python
# Process one at a time - SLOW on GPU
for story in stories:
    output = model(story)  # GPU is mostly idle

# Throughput: ~10 stories/second
```

### With Padding (Batch Processing)
```python
# Process all together - FAST on GPU
batch = pad_and_batch(stories)
outputs = model(batch)  # GPU is fully utilized

# Throughput: ~500 stories/second (50x faster!)
```

## Padding Efficiency Analysis

From your data analysis:
```
Average sequence length: 215.5 tokens
Max length (with padding): 512 tokens
Padding overhead: 512 - 215.5 = 296.5 tokens (58% waste)
```

### Solutions to Reduce Waste

#### 1. Dynamic Padding (Already Used)
```yaml
# In data_utils.py, set use_fixed_padding=False
use_fixed_padding: false
# Pads to batch maximum instead of global maximum
# Reduces waste from 58% to ~20-30%
```

#### 2. Sequence Packing (Already Enabled)
```yaml
# In minimal_working.yaml line 242
use_sequence_packing: true
# Combines multiple short sequences into one sample
# Reduces waste from 58% to ~5-10%
```

**Example of sequence packing:**
```
WITHOUT packing:
Story 1 (100 tokens) + 412 padding = 512
Story 2 (80 tokens)  + 432 padding = 512
Story 3 (90 tokens)  + 422 padding = 512
Total: 270 real + 1266 padding = 82% waste!

WITH packing:
[Story 1 (100) | Story 2 (80) | Story 3 (90)] + 242 padding = 512
Total: 270 real + 242 padding = 47% waste (much better!)
```

## Your Current Setup

Looking at your config (minimal_working.yaml):

```yaml
model:
  pad_token_id: 0    # Padding ignored
  eos_token_id: 1    # End of sequence
  bos_token_id: 2    # Beginning of sequence
  max_position_embeddings: 512  # Maximum sequence length

data:
  max_length: 512              # Data pre-padded to 512
  add_special_tokens: false    # BOS/EOS already in data
  use_sequence_packing: true   # ✓ Reduces padding waste by 70%

  # Collation settings
  dataloader_drop_last: true   # Drop incomplete batches
  dataloader_pin_memory: true  # Faster GPU transfer
```

## Common Issues

### 1. "Attention mask shape mismatch"
```python
# Problem: attention_mask doesn't match input_ids
input_ids.shape = [batch, 512]
attention_mask.shape = [batch, 256]  # WRONG!

# Solution: Always pad both to same length
```

### 2. "Loss is NaN"
```python
# Problem: PAD tokens included in loss
labels = input_ids  # WRONG - includes padding!

# Solution: Set padding to -100
labels = torch.where(attention_mask == 1, input_ids, -100)
```

### 3. "Out of memory"
```python
# Problem: max_length too large
max_length = 2048  # Wastes memory on short sequences

# Solution: Use sequence packing or dynamic padding
use_sequence_packing: true
# OR
use_fixed_padding: false
```

## Summary

1. **Padding** adds PAD tokens (ID=0) to make all sequences the same length
2. **Attention mask** tells the model which positions are real (1) vs padding (0)
3. **The model ignores padding** via attention masking and loss masking
4. **Sequence packing** (enabled in your config) reduces padding waste by 70%
5. **Your setup is already optimized** with packing and proper masking

The padding is handled automatically by your data pipeline - you don't need to change anything!
