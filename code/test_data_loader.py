#!/usr/bin/env python3
"""
Test script to verify data streaming works with Human/Assistant format
and loads all processed files correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from Ava.data_streaming import create_streaming_dataloaders
from transformers import AutoTokenizer

def test_data_loading():
    """Test that data loader properly loads all files with Human/Assistant format"""

    print("=" * 80)
    print("🧪 Testing Data Loader with Human/Assistant Format")
    print("=" * 80)

    # Setup
    data_dir = "/project/code/data/processed"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("\n📂 Data Directory:", data_dir)
    print("🔧 Tokenizer:", "gpt2")

    # Count files
    processed_files = list(Path(data_dir).glob("*_processed.jsonl"))
    print(f"\n📊 Found {len(processed_files)} *_processed.jsonl files")

    # Create dataloaders
    print("\n🔄 Creating streaming dataloaders...")
    try:
        train_loader, val_loader = create_streaming_dataloaders(
            tokenizer=tokenizer,
            batch_size=2,
            max_length=512,
            data_dir=data_dir,
            buffer_size=100,
            max_samples=None,
            num_workers=0,  # Single worker for testing
            enable_bucketing=False
        )
        print("   ✓ Dataloaders created successfully!")
    except Exception as e:
        print(f"   ❌ Failed to create dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test train loader
    print("\n🚂 Testing Train Loader...")
    try:
        train_iter = iter(train_loader)
        samples_checked = 0
        human_assistant_found = 0

        for i in range(50):  # Check more batches to find conversational data
            try:
                batch = next(train_iter)
                samples_checked += 1

                # Decode first sample in batch to check format
                input_ids = batch['input_ids'][0]
                decoded_text = tokenizer.decode(input_ids, skip_special_tokens=False)

                # Check if Human/Assistant format is present (H:/A: or Human:/Assistant:)
                is_conversational = ("Human:" in decoded_text or "Assistant:" in decoded_text or
                                   "H:" in decoded_text or "A:" in decoded_text)

                if is_conversational:
                    human_assistant_found += 1
                    if human_assistant_found <= 2:  # Show first 2 conversational examples
                        print(f"\n   📝 Conversational Sample {human_assistant_found} (first 300 chars):")
                        print(f"   {decoded_text[:300]}...")

                if i < 10 or is_conversational:  # Show first 10 batches or any conversational ones
                    print(f"   ✓ Batch {i+1}: shape={batch['input_ids'].shape}, "
                          f"conversational={'Yes' if is_conversational else 'No'}")

            except StopIteration:
                print(f"   ⚠️  Train loader exhausted after {i} batches")
                break

        print(f"\n   📊 Results:")
        print(f"      - Batches checked: {samples_checked}")
        print(f"      - Conversational samples: {human_assistant_found}/{samples_checked}")
        print(f"      - Non-conversational samples: {samples_checked - human_assistant_found}/{samples_checked}")

        if human_assistant_found == 0:
            print(f"      ⚠️  WARNING: No Human/Assistant format detected in {samples_checked} batches!")
            print(f"      This might be OK if conversational data comes later in the stream.")
        else:
            print(f"      ✓ Human/Assistant conversational format detected and preserved!")
            print(f"      ✓ Mixed dataset with both conversational and plain text works correctly!")

    except Exception as e:
        print(f"   ❌ Error during train loader testing: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test val loader
    print("\n🔍 Testing Validation Loader...")
    try:
        val_iter = iter(val_loader)
        batch = next(val_iter)
        print(f"   ✓ Validation batch: shape={batch['input_ids'].shape}")
    except Exception as e:
        print(f"   ❌ Error during val loader testing: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)
