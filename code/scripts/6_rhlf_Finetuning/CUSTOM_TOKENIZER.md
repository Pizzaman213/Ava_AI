# Using Custom Tokenizer with RLHF

Your RLHF pipeline fully supports your custom enhanced 65k tokenizer!

## Automatic Detection

The training script automatically uses your custom tokenizer from the config:

```yaml
data:
  tokenizer_name: /project/code/models/tokenizer/enhanced-65536
```

## How It Works

1. **Config Loading**: Reads `tokenizer_name` from your existing config
2. **Custom Tokenizer First**: Tries to load as `PreTrainedTokenizerFast`
3. **Fallback**: Falls back to `AutoTokenizer` if needed
4. **Special Tokens**: Automatically handles pad tokens

## Example Usage

### With Config (Recommended)

```bash
# Uses tokenizer from config automatically
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config configs/gpu/small.yaml
```

The script will automatically use:
- Policy model vocab: 65536 (from your custom tokenizer)
- Judge model vocab: 65536 (same tokenizer)
- All embeddings properly sized

### Verify Tokenizer Loading

Check the logs for:
```
Loading tokenizer from /project/code/models/tokenizer/enhanced-65536
Loaded custom tokenizer with vocab size: 65536
```

## Custom Tokenizer Features Supported

✅ Enhanced 65k vocabulary
✅ Math/code optimized tokens
✅ Special tokens (PAD, EOS, etc.)
✅ Fast tokenization
✅ Custom token embeddings

## Model Compatibility

Your models trained with the custom tokenizer will work perfectly:

```python
# The RLHF trainer automatically:
# 1. Loads your custom tokenizer
# 2. Loads your model trained with that tokenizer
# 3. Ensures vocab sizes match
# 4. Preserves all learned embeddings
```

## Example: Full Pipeline

```bash
# 1. Your model was trained with custom tokenizer
#    (vocab_size: 65536)

# 2. Prepare RLHF prompts
python scripts/6_rhlf_Finetuning/prepare_prompts.py \
  --create-samples --num-samples 100

# 3. Run RLHF (automatically uses custom tokenizer)
python scripts/6_rhlf_Finetuning/train_rlhf.py \
  --config configs/gpu/small.yaml \
  --policy-model /project/code/outputs/runs/latest/model

# The script will:
# ✓ Load custom tokenizer (65k vocab)
# ✓ Load your trained model (65k embeddings)
# ✓ Judge model uses same tokenizer
# ✓ All vocab sizes match perfectly
```

## Testing with Custom Tokenizer

```bash
# Test the pipeline with your tokenizer
python scripts/6_rhlf_Finetuning/test_rlhf_cpu.py

# You should see:
# "Loaded custom tokenizer with vocab size: 65536"
```

## Troubleshooting

### Issue: Vocab Size Mismatch

**Symptom**: `IndexError: index out of range`

**Solution**: Ensure all models use the same tokenizer:
```yaml
rlhf:
  policy_model_path: /path/to/model  # Trained with custom tokenizer
  judge_model_path: /path/to/model   # Same model = same vocab
```

### Issue: Tokenizer Not Found

**Symptom**: `FileNotFoundError: Tokenizer not found`

**Solution**: Check path in config:
```yaml
data:
  tokenizer_name: /project/code/models/tokenizer/enhanced-65536
```

### Issue: Different Vocab Sizes

**Symptom**: Policy has 65536, judge has 50257

**Solution**: Use models trained with same tokenizer:
```bash
# Both should use your custom tokenizer
--policy-model /project/code/outputs/your_model
--judge-model /project/code/outputs/your_model  # Same model
```

## Advanced: Multiple Tokenizers

If you need different tokenizers for policy vs judge:

```python
# Modify train_rlhf.py to load different tokenizers
policy_tokenizer = load_tokenizer(policy_tokenizer_path)
judge_tokenizer = load_tokenizer(judge_tokenizer_path)
```

But **recommended**: Use same tokenizer for both (simpler and more stable).

## Benefits of Custom Tokenizer for RLHF

Your enhanced 65k tokenizer provides:

1. **Better Math/Code**: Optimized tokens for technical prompts
2. **Efficient Encoding**: Fewer tokens per prompt = faster training
3. **Consistent Vocab**: Same vocab across all training stages
4. **Better Rewards**: Judge model understands same tokens as policy

## Summary

✅ Custom tokenizer automatically detected from config
✅ Works with 65k vocab size
✅ No code changes needed
✅ Full compatibility with your trained models
✅ Same tokenizer used for policy and judge

Your RLHF pipeline is ready to use with your custom tokenizer! 🚀
