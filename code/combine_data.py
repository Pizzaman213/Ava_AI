#!/usr/bin/env python3
"""
Combine downloaded datasets into train.jsonl and val.jsonl for training
"""

import json
import random
from pathlib import Path
from tqdm import tqdm

# Datasets to combine (good quality instruction datasets)
DATASETS_TO_COMBINE = [
    "teknium_OpenHermes-2.5/train",
    "Open-Orca_OpenOrca/train",
    "meta-math_MetaMathQA/train",
    "m-a-p_Code-Feedback/train",
    "tatsu-lab_alpaca/train",
    "vicgalle_alpaca-gpt4/train",
    "yahma_alpaca-cleaned/train",
    "databricks_databricks-dolly-15k/train",
]

data_dir = Path("/project/code/data")
combined_dir = data_dir / "combined"
combined_dir.mkdir(exist_ok=True)

print(" Combining datasets...")

all_texts = []

for dataset_path in DATASETS_TO_COMBINE:
    full_path = data_dir / dataset_path / "data.json"

    if not full_path.exists():
        print(f" Skipping {dataset_path} - not found")
        continue

    print(f" Reading {dataset_path}...")

    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract texts from different formats
        for item in tqdm(data, desc=f"Processing {dataset_path.split('/')[0]}"):
            text = None

            # Try different text extraction methods
            if isinstance(item, dict):
                if 'text' in item:
                    text = item['text']
                elif 'conversations' in item:
                    # OpenHermes format
                    conv_text = ""
                    for conv in item['conversations']:
                        role = conv.get('from', conv.get('role', ''))
                        value = conv.get('value', conv.get('content', ''))
                        conv_text += f"{role}: {value}\n"
                    text = conv_text.strip()
                elif 'instruction' in item:
                    # Alpaca format
                    instruction = item['instruction']
                    input_text = item.get('input', '')
                    output = item.get('output', item.get('response', ''))

                    if input_text:
                        text = f"Instruction: {instruction}\nInput: {input_text}\nResponse: {output}"
                    else:
                        text = f"Instruction: {instruction}\nResponse: {output}"
                elif 'question' in item:
                    # Math format
                    question = item['question']
                    answer = item.get('answer', item.get('solution', ''))
                    text = f"Question: {question}\nAnswer: {answer}"
                elif 'prompt' in item:
                    # Code feedback format
                    prompt = item['prompt']
                    response = item.get('response', item.get('completion', ''))
                    text = f"Prompt: {prompt}\nResponse: {response}"

            if text and len(text.strip()) > 20:
                all_texts.append(text)

        print(f"   Added {len([t for t in all_texts if t])} texts")

    except Exception as e:
        print(f"   Error reading {dataset_path}: {e}")
        continue

print(f"\n Total texts collected: {len(all_texts)}")

# Shuffle and split
random.seed(42)
random.shuffle(all_texts)

# Create train/val split (90/10)
split_idx = int(len(all_texts) * 0.9)
train_texts = all_texts[:split_idx]
val_texts = all_texts[split_idx:]

print(f" Train: {len(train_texts)} examples")
print(f" Val: {len(val_texts)} examples")

# Write train file
train_file = combined_dir / "train.jsonl"
with open(train_file, 'w', encoding='utf-8') as f:
    for text in tqdm(train_texts, desc="Writing train.jsonl"):
        f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

# Write val file
val_file = combined_dir / "val.jsonl"
with open(val_file, 'w', encoding='utf-8') as f:
    for text in tqdm(val_texts, desc="Writing val.jsonl"):
        f.write(json.dumps({"text": text}, ensure_ascii=False) + '\n')

print(f"\n Combined dataset ready!")
print(f"  Train: {train_file} ({train_file.stat().st_size / 1024 / 1024:.1f} MB)")
print(f"  Val: {val_file} ({val_file.stat().st_size / 1024 / 1024:.1f} MB)")

# Sample some examples
print(f"\n Sample training examples:")
for i, text in enumerate(train_texts[:3], 1):
    print(f"\n--- Example {i} ---")
    print(text[:300] + "..." if len(text) > 300 else text)