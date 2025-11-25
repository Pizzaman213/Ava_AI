# Turn-Aware Data Loading - Quick Start Guide

## 30-Second Overview

Turn-aware loading keeps conversations together during training, preserves speaker identity, and tracks conversation quality. This improves dialogue coherence and model understanding.

## Installation (Already Included)

The implementation is in: `code/src/Ava/data/conversation_turn_loader.py`

No additional dependencies needed (uses standard PyTorch).

## Basic Usage

### Step 1: Import
```python
from src.Ava.data.conversation_turn_loader import (
    TurnAwareConversationDataLoader
)
```

### Step 2: Create Dataloader
```python
train_loader = TurnAwareConversationDataLoader.create_dataloader(
    data_path="code/data/processed/conversations.jsonl",
    tokenizer=tokenizer,
    batch_size=32,
    max_length=2048,
)
```

### Step 3: Train
```python
for batch in train_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Your model forward pass
    outputs = model(input_ids, attention_mask)
    loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()
```

That's it! Your conversations are now:
-  Kept together (not split across batches)
-  Speaker-aware (with `<user>` and `<assistant>` markers)
-  Turn-structured (with `<turn_start>` and `<turn_end>`)
-  Quality-tracked (quality scores in metadata)

## Common Tasks

### Filter High-Quality Conversations
```python
loader = TurnAwareConversationDataLoader.create_dataloader(
    data_path="conversations.jsonl",
    tokenizer=tokenizer,
    quality_threshold=0.8,  # Only conversations with quality > 0.8
)
```

### Use Only Multi-Turn Conversations
```python
loader = TurnAwareConversationDataLoader.create_dataloader(
    data_path="conversations.jsonl",
    tokenizer=tokenizer,
    min_turns=2,  # Only 2+ turn conversations
)
```

### Weight Loss by Quality
```python
for batch in train_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    quality_scores = torch.tensor(
        batch["conversation_metadata"]["quality_scores"],
        device=input_ids.device
    )

    outputs = model(input_ids, attention_mask)
    loss = criterion(outputs, targets)  # Per-sample loss

    # Weight by quality
    weighted_loss = (loss * quality_scores).mean()
    weighted_loss.backward()
```

### Access Conversation Metadata
```python
for batch in train_loader:
    metadata = batch["conversation_metadata"]

    # Available fields:
    conversation_ids = metadata["conversation_ids"]  # Unique IDs
    sources = metadata["sources"]                    # Dataset names
    domains = metadata["domains"]                    # Conversation types
    quality_scores = metadata["quality_scores"]      # Quality (0-1)
    num_turns = metadata["num_turns"]               # Turn counts

    print(f"Average quality: {np.mean(quality_scores):.2f}")
    print(f"Average turns: {np.mean(num_turns):.1f}")
```

### Process Different Conversation Formats
```python
from src.Ava.data.conversation_turn_loader import ConversationParser

# Text format
conv = ConversationParser.parse_text_format(
    "User: Hello!\nAssistant: Hi!"
)

# HuggingFace messages
conv = ConversationParser.parse_messages_format([
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi!"}
])

# Auto-detect format
conv = ConversationParser.auto_parse(any_format)
```

## Data Format

Your JSONL file should have one conversation per line:

```json
{"text": "User: Hello!\nAssistant: Hi there!", "source": "my_dataset", "type": "conversation"}
{"text": "User: How are you?\nAssistant: I'm great!", "source": "my_dataset", "type": "conversation"}
```

**Optional fields** (automatically detected):
- `quality_score` (0-1): Quality metric
- `source`: Dataset name
- `type`: Conversation type

## Supported Conversation Formats

### Format 1: Text with Speaker Labels
```
User: Hello!
Assistant: Hi there!
```

### Format 2: HuggingFace Messages
```json
{"messages": [
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi!"}
]}
```

### Format 3: Dialogue List
```python
["Hello!", "Hi!", "How are you?", "Great!"]
```

The parser auto-detects and handles all formats!

## Expected Batch Structure

```python
batch = {
    "input_ids": torch.Tensor([batch_size, seq_len]),
    "attention_mask": torch.Tensor([batch_size, seq_len]),
    "conversation_metadata": {
        "conversation_ids": ["conv_001", "conv_002", ...],
        "sources": ["dataset_a", "dataset_b", ...],
        "domains": ["general", "technical", ...],
        "quality_scores": [0.95, 0.87, ...],
        "num_turns": [2, 4, ...]
    }
}
```

## Special Tokens

Conversations are marked with special tokens:

```
Original:  "User: Hi!\nAssistant: Hello!"
Tokenized: <turn_start> <user> Hi! </user> <turn_end>
           <turn_start> <assistant> Hello! </assistant> <turn_end>
```

Available tokens:
- `<turn_start>` / `<turn_end>` - Turn boundaries
- `<user>` / `</user>` - User messages
- `<assistant>` / `</assistant>` - Assistant messages
- `<context_start>` / `<context_end>` - Context sections

Disable speaker tokens if needed:
```python
from src.Ava.data.conversation_turn_loader import TurnAwareTokenizer

tokenizer = TurnAwareTokenizer(
    your_tokenizer,
    include_speaker_tokens=False,  # No <user>, <assistant>
)
```

## Performance Tips

### Faster Loading
```python
loader = TurnAwareConversationDataLoader.create_dataloader(
    ...,
    num_workers=4,  # Use multiple workers
)
```

### Lower Memory
```python
loader = TurnAwareConversationDataLoader.create_dataloader(
    ...,
    batch_size=16,  # Reduce batch size
    max_length=1024,  # Reduce max length
)
```

### Better Quality
```python
loader = TurnAwareConversationDataLoader.create_dataloader(
    ...,
    min_turns=2,  # Only multi-turn
    quality_threshold=0.8,  # Only high quality
)
```

## Testing Your Setup

Run the validation tests:
```bash
python code/scripts/validation/test_turn_aware_loader.py
```

Expected: All 6 tests pass 

Try the example training:
```bash
python code/scripts/examples/example_turn_aware_training.py
```

Expected: Training completes successfully

## Troubleshooting

### "No conversations loaded"
Make sure your JSONL file has a "text" field:
```json
{"text": "User: ...\nAssistant: ..."}  #  Correct
{"conversation": "..."}               #  Wrong field name
```

### "Quality scores are all 1.0"
Add quality_score to your JSONL:
```json
{"text": "...", "quality_score": 0.85}
```

### Memory running out
Try:
```python
loader = TurnAwareConversationDataLoader.create_dataloader(
    ...,
    batch_size=8,  # Smaller batches
    max_length=512,  # Shorter sequences
)
```

### Conversations being split
This shouldn't happen with the turn-aware loader - conversations stay together. If you see split conversations, check your data format.

## Integration Examples

### With Existing Training Loop
```python
# Before
train_loader, val_loader = create_streaming_dataloaders(...)

# After
train_loader = TurnAwareConversationDataLoader.create_dataloader(
    data_path="conversations.jsonl",
    tokenizer=tokenizer,
    batch_size=batch_size,
)
```

### With Configuration File
```yaml
# config.yaml
data:
  use_turn_aware_loader: true  # Enable turn-aware loading
  data_dir: "code/data/processed"
```

```python
# training.py
if config.data.use_turn_aware_loader:
    train_loader = TurnAwareConversationDataLoader.create_dataloader(...)
else:
    train_loader, val_loader = create_streaming_dataloaders(...)
```

### With Quality Weighting
```python
def train_step(batch, model, optimizer):
    input_ids = batch["input_ids"]
    quality = torch.tensor(
        batch["conversation_metadata"]["quality_scores"]
    )

    outputs = model(input_ids)
    loss = criterion(outputs, targets)
    weighted_loss = (loss * quality).mean()

    weighted_loss.backward()
    optimizer.step()
```

## Next Steps

1.  Install (already included)
2. → Prepare your JSONL data
3. → Create a dataloader (copy code from "Step 2")
4. → Update your training loop (copy code from "Step 3")
5. → Test with example_turn_aware_training.py

## Documentation

- **Full Guide**: See [TURN_AWARE_DATA_LOADING.md](TURN_AWARE_DATA_LOADING.md)
- **Implementation**: See [conversation_turn_loader.py](../src/Ava/data/conversation_turn_loader.py)
- **Tests**: See [test_turn_aware_loader.py](../scripts/validation/test_turn_aware_loader.py)
- **Example**: See [example_turn_aware_training.py](../scripts/examples/example_turn_aware_training.py)

## Summary

Turn-aware loading makes your conversational training:
- **Better**: Preserves dialogue coherence
- **Easier**: Auto-detects conversation formats
- **Transparent**: Tracks conversation quality
- **Flexible**: Works with any tokenizer/model

Just 3 lines to get started! 
