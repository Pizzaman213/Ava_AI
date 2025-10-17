#!/usr/bin/env python3
"""Convert simple_fairy_tales.txt to processed.jsonl format."""

import json
from pathlib import Path

def main():
    """Convert text file to JSONL format."""
    input_path = Path("/project/code/data/Testing/simple_fairy_tales.txt")
    output_path = Path("/project/code/data/Testing/processed.jsonl")

    print(f"Reading from: {input_path}")

    # Read the text file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines (each story is separated by a blank line)
    stories = [story.strip() for story in content.split('\n\n') if story.strip()]

    print(f"Found {len(stories)} stories")

    # Write to JSONL format
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, story in enumerate(stories):
            # Create a JSON object with the text
            json_obj = {
                "text": story,
                "id": f"fairy_tale_{i:05d}",
                "source": "simple_fairy_tales"
            }
            f.write(json.dumps(json_obj) + '\n')

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} stories...")

    print(f"\nSuccessfully created {output_path}")
    print(f"Total stories: {len(stories)}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
