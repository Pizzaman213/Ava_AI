#!/usr/bin/env python3
"""
Download ALL popular conversational datasets with greetings.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    os.system("pip install -q datasets tqdm")
    from datasets import load_dataset
    from tqdm import tqdm

OUTPUT_DIR = Path("/project/code/data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def has_greeting(text: str) -> bool:
    """Check if text contains greeting words."""
    text_lower = text.lower()
    greetings = [
        'hello', 'hi ', 'hi,', 'hi!', 'hi.', 'hey', 'greetings',
        'good morning', 'good afternoon', 'good evening', 'howdy',
        "what's up", 'whats up', 'how are you', 'how do you do',
        'nice to meet', 'pleased to meet', 'welcome', 'hiya',
        'hola', 'bonjour', 'ciao', 'aloha', 'namaste'
    ]
    return any(greeting in text_lower for greeting in greetings)


def extract_conversation(item: Dict, dataset_name: str) -> str:
    """Extract conversational text from various dataset formats."""

    # OpenAI messages format
    if 'messages' in item:
        messages = item['messages']
        if isinstance(messages, list):
            parts = []
            for msg in messages:
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                parts.append(f"{role.capitalize()}: {content}")
            return "\n".join(parts)

    # Conversations list format
    if 'conversations' in item or 'conversation' in item:
        convs = item.get('conversations', item.get('conversation', []))
        if isinstance(convs, list):
            parts = []
            for conv in convs:
                if isinstance(conv, dict):
                    role = conv.get('from', conv.get('role', conv.get('speaker', 'unknown')))
                    content = conv.get('value', conv.get('content', conv.get('text', '')))
                    parts.append(f"{role.capitalize()}: {content}")
                elif isinstance(conv, str):
                    parts.append(conv)
            return "\n".join(parts)

    # Dialog/dialogue format
    if 'dialog' in item or 'dialogue' in item:
        dialog = item.get('dialog', item.get('dialogue', []))
        if isinstance(dialog, list):
            parts = []
            for i, turn in enumerate(dialog):
                speaker = "User" if i % 2 == 0 else "Assistant"
                if isinstance(turn, dict):
                    text = turn.get('text', turn.get('utterance', ''))
                    speaker = turn.get('speaker', speaker)
                else:
                    text = str(turn)
                parts.append(f"{speaker}: {text}")
            return "\n".join(parts)

    # Utterances format
    if 'utterances' in item:
        utts = item['utterances']
        if isinstance(utts, list):
            parts = []
            for utt in utts:
                if isinstance(utt, dict):
                    speaker = utt.get('speaker', utt.get('actor_type', 'Speaker'))
                    text = utt.get('text', utt.get('utterance', ''))
                    parts.append(f"{speaker}: {text}")
                else:
                    parts.append(str(utt))
            return "\n".join(parts)

    # Prompt-response format
    if 'prompt' in item and 'response' in item:
        return f"User: {item['prompt']}\nAssistant: {item['response']}"

    # Instruction-response format
    if 'instruction' in item:
        instruction = item['instruction']
        response = item.get('response', item.get('output', ''))
        context = item.get('context', item.get('input', ''))
        if context:
            return f"User: {instruction}\nContext: {context}\nAssistant: {response}"
        return f"User: {instruction}\nAssistant: {response}"

    # Question-answer format
    if 'question' in item and 'answer' in item:
        return f"User: {item['question']}\nAssistant: {item['answer']}"

    # Plain text
    if 'text' in item:
        return item['text']

    # Fallback: combine text fields
    text_parts = []
    for key, value in item.items():
        if isinstance(value, str) and len(value) > 10 and key not in ['id', 'source', 'dataset']:
            text_parts.append(f"{key.capitalize()}: {value}")

    return "\n".join(text_parts) if text_parts else ""


def download_dataset(name: str, config: Optional[str] = None, split: str = "train", max_samples: int = 50000):
    """Download and process a conversational dataset."""
    print(f"\n{'='*70}")
    print(f"Processing: {name}")
    print(f"{'='*70}")

    try:
        # Try to load dataset (without trust_remote_code - it's deprecated)
        try:
            if config:
                dataset = load_dataset(name, config, split=split, streaming=True)
            else:
                dataset = load_dataset(name, split=split, streaming=True)
        except ValueError as e:
            # Try alternative splits
            if "train_sft" in str(e):
                dataset = load_dataset(name, split="train_sft", streaming=True)
            elif "validation" in str(e):
                dataset = load_dataset(name, split="validation", streaming=True)
            else:
                print(f"  ⚠ Skipping - split error: {e}")
                return 0
        except RuntimeError as e:
            if "Dataset scripts are no longer supported" in str(e):
                print(f"  ⚠ Skipping - old dataset format not supported: {e}")
                return 0
            else:
                raise

        # Output file
        safe_name = name.replace("/", "_").replace("-", "_")
        if config:
            safe_name += f"_{config.replace('/', '_')}"
        output_file = OUTPUT_DIR / f"{safe_name}_conversational.jsonl"

        # Check if already exists
        if output_file.exists():
            existing_lines = sum(1 for _ in open(output_file))
            if existing_lines > 100:
                print(f"  ✓ Already exists with {existing_lines} examples - skipping")
                return existing_lines

        count = 0
        saved = 0

        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(dataset, desc=f"  Downloading", total=max_samples):
                if count >= max_samples:
                    break

                text = extract_conversation(item, name)

                if text and len(text.strip()) > 20:
                    has_greet = has_greeting(text)

                    # Save all conversations, mark which have greetings
                    record = {
                        "text": text.strip(),
                        "source": name,
                        "has_greeting": has_greet
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    saved += 1

                count += 1

        # Show stats
        size_mb = output_file.stat().st_size / (1024 * 1024)
        greeting_count = sum(1 for line in open(output_file) if '"has_greeting": true' in line)

        print(f"  ✓ Saved {saved:,} examples ({greeting_count:,} with greetings)")
        print(f"  ✓ File size: {size_mb:.1f} MB")

        return saved

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    """Download all conversational datasets."""

    datasets = [
        # Already downloaded - will skip
        {"name": "Anthropic/hh-rlhf", "max_samples": 160000},
        {"name": "OpenAssistant/oasst1", "max_samples": 100000},
        {"name": "HuggingFaceH4/no_robots", "max_samples": 10000},

        # New datasets to download
        {"name": "daily_dialog", "max_samples": 15000},
        {"name": "empathetic_dialogues", "max_samples": 30000},
        {"name": "conv_ai_2", "max_samples": 20000},
        {"name": "blended_skill_talk", "max_samples": 10000},
        {"name": "google/Synthetic-Persona-Chat", "max_samples": 50000},
        {"name": "microsoft/wizard_of_wikipedia", "max_samples": 20000},
        {"name": "Salesforce/dialogstudio", "config": "TradeDial", "max_samples": 5000},
        {"name": "AllenAI/prosocial-dialog", "max_samples": 20000},
        {"name": "PygmalionAI/PIPPA", "max_samples": 30000},
        {"name": "meta-math/MetaMathQA", "max_samples": 20000},
        {"name": "HuggingFaceH4/self-instruct", "max_samples": 10000},
        {"name": "teknium/OpenHermes-2.5", "max_samples": 50000},
        {"name": "garage-bAInd/Open-Platypus", "max_samples": 25000},
        {"name": "WizardLM/WizardLM_evol_instruct_V2_196k", "max_samples": 50000},
        {"name": "fnlp/moss-002-sft-data", "max_samples": 20000},
        {"name": "timdettmers/openassistant-guanaco", "max_samples": 10000},
        {"name": "QingyiSi/Alpaca-CoT", "max_samples": 30000},
    ]

    print("\n" + "="*70)
    print("  DOWNLOADING ALL CONVERSATIONAL DATASETS WITH GREETINGS")
    print("="*70)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Total datasets: {len(datasets)}")
    print("="*70)

    total_examples = 0
    successful = 0

    for ds in datasets:
        name = ds["name"]
        config = ds.get("config")
        max_samples = ds.get("max_samples", 10000)

        count = download_dataset(name, config, max_samples=max_samples)

        if count > 0:
            total_examples += count
            successful += 1

    print("\n" + "="*70)
    print("  DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"  ✓ Successful: {successful}/{len(datasets)} datasets")
    print(f"  ✓ Total examples: {total_examples:,}")
    print("="*70)

    # List all files
    print("\n  Created files:")
    total_size = 0
    all_files = sorted(OUTPUT_DIR.glob("*conversational.jsonl"))
    all_files.extend(sorted(OUTPUT_DIR.glob("*greeting*.jsonl")))

    for file in all_files:
        if file.stat().st_size > 0:
            size_mb = file.stat().st_size / (1024 * 1024)
            lines = sum(1 for _ in open(file))
            total_size += size_mb
            print(f"    • {file.name}: {lines:,} examples ({size_mb:.1f} MB)")

    print(f"\n  Total size: {total_size:.1f} MB")
    print("="*70)
    print("\n  These will be automatically loaded during training!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
