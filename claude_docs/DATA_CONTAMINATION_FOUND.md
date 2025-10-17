# CRITICAL: Data Contamination Found

## Issue Identified

Your training data contains **conversational AI artifacts** that are causing the model to learn incorrect patterns.

### Evidence

Sample generation at step 1000:
```
"Once upon a time time time time time time time time timeAssistantAssistantAssistantAssist..."
```

### Root Cause

Training data files contain dialogue from conversational AI datasets (OpenOrca/SlimOrca):

```json
{"text": "Assistant: You are an AI assistant. User will you give you a task...", "source": "Open-Orca/SlimOrca"}
```

The model is learning that "Assistant:" is a valid token to generate, which causes it to repeat this token during generation.

## Impact on Training

1. **Repetition Penalties ARE Working**: The improvement from 86% to 59% repetition proves the fix worked
2. **But Fighting Contaminated Data**: The penalties can't fully overcome bad training examples
3. **Model Learning Wrong Patterns**: Every time it sees "Assistant:" in training data, it reinforces this as valid output

## Why This Matters

- Repetition penalties can only do so much against systematic data issues
- The model will continue struggling with coherence until the data is cleaned
- Current performance (59% repetition, 15/100 coherence) is limited by data quality

## Recommendations

### Immediate (Continue Current Training)

**Keep training to step 5000-10000** to see if the repetition penalties can partially overcome the data issues through extended training. The penalties ARE working (59% vs 86%), they just need more time to reshape learned patterns.

### Short Term (Next Training Run)

1. **Clean Training Data**:
   ```bash
   # Remove conversational AI artifacts
   cd /project/code/data/processed
   for file in *.jsonl; do
       grep -v "Assistant:" "$file" | grep -v "User:" | grep -v "you are an AI assistant" > "${file}.cleaned"
       mv "${file}.cleaned" "$file"
   done
   ```

2. **Filter Sources**:
   - Remove or filter OpenOrca/SlimOrca entries
   - Keep only high-quality narrative/story datasets
   - Verify data quality before training

3. **Restart Training** with cleaned data

### Long Term

1. **Data Quality Pipeline**:
   - Add preprocessing to detect and remove conversational artifacts
   - Validate all training data before use
   - Monitor for contamination patterns

2. **Improved Data Sources**:
   - Focus on high-quality story datasets
   - Use datasets specifically designed for creative text generation
   - Avoid instruction-following or conversational datasets

## Expected Improvement After Cleaning

With clean data AND repetition penalties:

- **Repetition**: Should drop to 25-30% (vs current 59%)
- **Coherence**: Should reach 50-70/100 (vs current 15/100)
- **Samples**: Natural story text without "Assistant" artifacts

## Current Training Recommendation

**✅ CONTINUE** the current training run to step 5000-10000:

- Monitor if repetition continues to decrease
- The penalties ARE working, give them time
- This validates that our fix was correct
- Provides baseline to compare against clean data

Then prepare a new training run with cleaned data.

## Data Cleaning Script

Create `/project/code/scripts/data_prep/clean_conversational_artifacts.py`:

```python
import json
import re
from pathlib import Path

def clean_conversational_artifacts(input_file, output_file):
    """Remove conversational AI artifacts from training data."""

    contamination_patterns = [
        r'\\bAssistant:\\b',
        r'\\bUser:\\b',
        r'you are an AI assistant',
        r'I am an AI',
        r'as an AI language model',
    ]

    cleaned_count = 0
    total_count = 0

    with open(input_file) as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            total_count += 1
            data = json.loads(line)
            text = data.get('text', '')

            # Check for contamination
            is_contaminated = any(
                re.search(pattern, text, re.IGNORECASE)
                for pattern in contamination_patterns
            )

            if not is_contaminated:
                f_out.write(line)
                cleaned_count += 1

    removed = total_count - cleaned_count
    print(f"Processed {total_count} examples")
    print(f"Kept {cleaned_count} examples")
    print(f"Removed {removed} contaminated examples ({removed/total_count*100:.1f}%)")

if __name__ == "__main__":
    data_dir = Path("/project/code/data/processed")
    for input_file in data_dir.glob("*.jsonl"):
        if ".cleaned" not in input_file.name:
            output_file = input_file.with_suffix(".cleaned.jsonl")
            print(f"\\nCleaning {input_file.name}...")
            clean_conversational_artifacts(input_file, output_file)
```

## Files Affected

- Training data in `/project/code/data/processed/*.jsonl`
- Multiple sources appear contaminated based on sample inspection

## Next Steps

1. ✅ Continue current training (validate penalty fix)
2. ⏸️ Prepare data cleaning script
3. ⏸️ Clean all training data
4. ⏸️ Start new training run with clean data
5. ⏸️ Compare results (should see dramatic improvement)
