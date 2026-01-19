#!/usr/bin/env python3
"""
Filter OpenOrca dataset to keep only narrative/creative content.

Removes:
- Task IDs (niv., flan., t0., cot., etc.)
- System prompts ("You are an AI assistant...")
- Task definitions ("You will be given a definition...")
- QA patterns ("the answer is", "question:", "answer:")
- RDF/JSON structured data
- Multiple choice questions
- Math word problems
- Short sequences (< 50 tokens)

Keeps:
- Natural narrative text
- Stories
- Creative writing
- Conversational text without instruction patterns
"""

import os
import sys
import re
import argparse
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
from transformers import AutoTokenizer


# Patterns to EXCLUDE (QA/instruction patterns)
EXCLUDE_PATTERNS = [
    # Task IDs
    r'^(niv|flan|t0|cot|niv_task)\s*\.\s*\d+',

    # System prompts
    r'you are an ai assistant',
    r'you are a helpful assistant',
    r'you will be given a (definition|task)',
    r'you must generate',
    r'think like you are answering',
    r'let\'s think step by step',
    r'let\'s be accurate',

    # Task instructions
    r'generate an? .*(sentence|paragraph|response)',
    r'please (answer|provide|generate|write)',
    r'this task is about',
    r'the input is .* the output',
    r'convert .* to .*(rdf|json|triplet)',
    r'resource description framework',

    # QA patterns
    r'(question|answer)\s*:',
    r'the answer is',
    r'choose your answer from',
    r'select the (best|correct|most)',
    r'\[?\s*\[\s*"[^"]+"\s*,\s*"[^"]+"\s*,\s*"[^"]+"',  # RDF triplets
    r'options?\s*:\s*[a-d]\)',
    r'^[a-d]\.\s+',  # Multiple choice options
    r'what is the (answer|solution|result)',
    r'true or false',
    r'which of the following',
    r'based on the (passage|context|text)',
    r'according to the (passage|context|text)',

    # Math problems
    r'how (much|many) (does|do|did|will)',
    r'calculate',
    r'\$\s*\d+',  # Dollar amounts
    r'\d+\s*[\+\-\*\/]\s*\d+',  # Math operations

    # Structured data
    r'^\s*\{',  # JSON objects
    r'^\s*\[',  # JSON arrays
    r'<[a-z]+>.*</[a-z]+>',  # XML tags

    # Meta/formatting
    r'^output\s*:',
    r'^input\s*:',
    r'^context\s*:',
    r'^passage\s*:',
]

# Patterns that INDICATE narrative content (positive signals)
NARRATIVE_INDICATORS = [
    r'^once upon a time',
    r'^there (was|were|lived)',
    r'^(one|a) (day|morning|evening|night)',
    r'^in a (land|place|town|city|village)',
    r'^long ago',
    r'^(she|he|they|it) (was|were|had|went|saw)',
    r'^\w+ was a (boy|girl|man|woman|child|person)',
]


def compile_patterns():
    """Compile regex patterns for efficiency."""
    exclude = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in EXCLUDE_PATTERNS]
    narrative = [re.compile(p, re.IGNORECASE) for p in NARRATIVE_INDICATORS]
    return exclude, narrative


def is_narrative_content(text: str, exclude_patterns: list, narrative_patterns: list) -> bool:
    """
    Check if text is narrative/creative content.

    STRICT MODE: Only keeps text with clear narrative indicators.
    This is intentionally aggressive to filter out QA/instruction content.
    """
    text_lower = text.lower().strip()

    # Too short = likely not useful
    if len(text_lower) < 150:
        return False

    # Check exclusion patterns first
    for pattern in exclude_patterns:
        if pattern.search(text_lower):
            return False

    # Too many special characters = likely structured data
    special_ratio = sum(1 for c in text if c in '[]{}|<>:') / max(len(text), 1)
    if special_ratio > 0.02:
        return False

    # STRICT: Must have narrative indicators
    has_narrative_indicator = any(p.search(text_lower) for p in narrative_patterns)

    if not has_narrative_indicator:
        # Check for story-like patterns if no explicit indicator
        story_patterns = [
            r'\b(said|asked|replied|answered|thought|felt|saw|heard)\b',  # Dialogue/action verbs
            r'\b(he|she|they|we|i)\s+(was|were|had|went|came|saw|did)\b',  # Narrative pronouns + past tense
            r'\b(morning|evening|night|day|year|month|week)\b.*\b(later|ago|passed)\b',  # Time references
        ]
        story_match = sum(1 for p in story_patterns if re.search(p, text_lower))
        if story_match < 2:
            return False

    # Reject if it looks like instruction/QA even with narrative elements
    qa_signals = [
        'step 1', 'step 2', 'step-by-step',
        'option a', 'option b', 'options:',
        'the correct answer', 'the answer is',
        'based on the passage', 'according to',
        'true or false', 'yes or no',
        'you are given', 'you will be given',
        'your task is', 'your job is',
        'translate', 'translation',
        'sentiment', 'classify',
        'premise:', 'hypothesis:',
        'solution:', 'output:',
        'definition:', 'input:',
    ]
    if any(sig in text_lower for sig in qa_signals):
        return False

    # Must have good sentence structure
    sentences = text.split('.')
    if len(sentences) < 3:
        return False

    # Average sentence length should be reasonable (not too short = headers, not too long = no punctuation)
    avg_sentence_len = sum(len(s) for s in sentences) / len(sentences)
    if avg_sentence_len < 20 or avg_sentence_len > 300:
        return False

    return True


def filter_arrow_file(
    input_path: str,
    output_path: str,
    tokenizer,
    exclude_patterns: list,
    narrative_patterns: list,
    min_tokens: int = 50,
    max_tokens: int = 2048,
) -> tuple:
    """
    Filter a single arrow file, keeping only narrative content.

    Returns (kept_count, total_count)
    """
    # Read input file
    with pa.memory_map(input_path, 'r') as source:
        reader = pa.ipc.open_stream(source)
        table = reader.read_all()

    total = len(table)
    kept_indices = []

    for i in range(total):
        tokens = table['input_ids'][i].as_py()

        # Length filter
        if len(tokens) < min_tokens or len(tokens) > max_tokens:
            continue

        # Decode and check content
        text = tokenizer.decode(tokens, skip_special_tokens=True)

        if is_narrative_content(text, exclude_patterns, narrative_patterns):
            kept_indices.append(i)

    if not kept_indices:
        return 0, total

    # Create filtered table
    filtered_data = {
        'input_ids': [table['input_ids'][i].as_py() for i in kept_indices],
        'attention_mask': [table['attention_mask'][i].as_py() for i in kept_indices],
    }

    # Build new table
    schema = pa.schema([
        ('input_ids', pa.list_(pa.int32())),
        ('attention_mask', pa.list_(pa.int32())),
    ])

    arrays = [
        pa.array(filtered_data['input_ids'], type=pa.list_(pa.int32())),
        pa.array(filtered_data['attention_mask'], type=pa.list_(pa.int32())),
    ]

    new_table = pa.table(dict(zip(schema.names, arrays)), schema=schema)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pa.OSFile(output_path, 'wb') as sink:
        writer = pa.ipc.new_stream(sink, schema)
        writer.write_table(new_table)
        writer.close()

    return len(kept_indices), total


def main():
    parser = argparse.ArgumentParser(description='Filter OpenOrca to narrative content only')
    parser.add_argument('--input-dir', type=str,
                        default='/root/Ava_AI/pretokenized_data/data/OpenOrca',
                        help='Input OpenOrca directory')
    parser.add_argument('--output-dir', type=str,
                        default='/root/Ava_AI/pretokenized_data/data/OpenOrca_narrative',
                        help='Output directory for filtered data')
    parser.add_argument('--tokenizer', type=str,
                        default='/root/Ava_AI/pretokenized_data/tokenizers/vocab',
                        help='Tokenizer path')
    parser.add_argument('--min-tokens', type=int, default=50,
                        help='Minimum sequence length')
    parser.add_argument('--max-tokens', type=int, default=2048,
                        help='Maximum sequence length')
    parser.add_argument('--sample', type=int, default=None,
                        help='Process only first N files (for testing)')
    parser.add_argument('--show-examples', action='store_true',
                        help='Show example filtered texts')

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    # Compile patterns
    exclude_patterns, narrative_patterns = compile_patterns()

    # Find all arrow files
    input_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.endswith('.arrow')
    ])

    if args.sample:
        input_files = input_files[:args.sample]

    print(f"Found {len(input_files)} files to process")
    print(f"Output directory: {args.output_dir}")

    # Process files
    total_kept = 0
    total_processed = 0

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in tqdm(input_files, desc="Filtering files"):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        kept, total = filter_arrow_file(
            input_path, output_path, tokenizer,
            exclude_patterns, narrative_patterns,
            args.min_tokens, args.max_tokens
        )

        total_kept += kept
        total_processed += total

        if args.show_examples and kept > 0:
            # Show a sample from this file
            with pa.memory_map(output_path, 'r') as source:
                reader = pa.ipc.open_stream(source)
                table = reader.read_all()
            if len(table) > 0:
                text = tokenizer.decode(table['input_ids'][0].as_py(), skip_special_tokens=True)
                print(f"\n[Sample from {filename}]:\n{text[:500]}...\n")

    # Summary
    retention_rate = (total_kept / total_processed * 100) if total_processed > 0 else 0
    print(f"\n{'='*60}")
    print(f"Filtering complete!")
    print(f"  Input directory:  {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Files processed:  {len(input_files)}")
    print(f"  Examples kept:    {total_kept:,} / {total_processed:,} ({retention_rate:.1f}%)")
    print(f"{'='*60}")

    if total_kept == 0:
        print("\nWARNING: No examples passed the filter!")
        print("This might mean the filtering is too strict.")
        print("Consider adjusting the patterns or using --show-examples to debug.")


if __name__ == '__main__':
    main()
