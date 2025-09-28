#!/usr/bin/env python3
"""
Test script for the enhanced data preparation with quality fixing.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.append('/project/code')

def test_enhanced_data_prep():
    """Test the enhanced data preparation script."""
    print("🧪 Testing Enhanced Data Preparation")
    print("=" * 50)

    try:
        # Import the enhanced processor
        from scripts.data_prep.prepare_data_rapids import LocalDataProcessor, DatasetQualityAnalyzer

        # Test 1: Quality Analyzer
        print("\n🔍 Test 1: Quality Analyzer")
        analyzer = DatasetQualityAnalyzer()

        # Test sample with various quality issues
        test_samples = [
            {"text": "This is a good quality text sample with proper length and content."},  # Good
            {"text": "a"},  # Too short
            {"text": "word " * 1000},  # Too repetitive
            {"text": "This has â€™ encoding issues that need fixing."},  # Encoding issues
            {"invalid": "data"},  # No text field
            {"text": ""},  # Empty
            {"text": "This is another good quality sample for testing purposes."},  # Good
        ]

        processed_samples = []
        for sample in test_samples:
            analyzer.quality_metrics['total_samples'] += 1
            cleaned_sample, fixing_log = analyzer.filter_and_fix_sample(sample)
            if cleaned_sample:
                processed_samples.append(cleaned_sample)
                print(f"    ✅ Accepted: {cleaned_sample['text'][:50]}... (score: {fixing_log['quality_score']:.2f})")
            else:
                print(f"    ❌ Rejected: {fixing_log['rejected_reason']}")

        print(f"    📊 Quality filtering: {len(processed_samples)}/{len(test_samples)} samples accepted")

        # Test 2: Enhanced Data Processor
        print("\n🏗️  Test 2: Enhanced Data Processor")

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"
            test_data_dir = Path(temp_dir) / "test_data"
            test_data_dir.mkdir()

            # Create test dataset
            test_dataset = [
                {"text": "High quality sample with good content for testing purposes."},
                {"text": "Another quality sample."},
                {"text": "bad"},  # Low quality
                {"instruction": "What is AI?", "response": "AI is artificial intelligence."},  # Multi-column
                {"text": "This sample â€™ has encoding issues."},  # Encoding issue
            ]

            # Save test data
            test_file = test_data_dir / "test_dataset.jsonl"
            with open(test_file, 'w') as f:
                for sample in test_dataset:
                    f.write(json.dumps(sample) + '\n')

            # Test processor with quality filtering
            processor = LocalDataProcessor(
                output_dir=str(output_dir),
                use_gpu=False,
                enable_quality_filtering=True,
                quality_threshold=0.3
            )

            # Process the test dataset
            texts = processor.extract_text_efficiently(test_dataset)
            print(f"    📝 Extracted {len(texts)} texts from {len(test_dataset)} samples")

            # Check processing stats
            stats = processor.processing_stats
            print(f"    ✅ Accepted: {stats['samples_accepted']}")
            print(f"    ❌ Rejected: {stats['samples_rejected']}")
            print(f"    🔧 Encoding fixes: {stats['encoding_fixes']}")

        # Test 3: Multi-column Format Detection
        print("\n📋 Test 3: Multi-column Format Detection")

        instruction_samples = [
            {"instruction": "Explain photosynthesis", "response": "Photosynthesis is the process..."},
            {"instruction": "What is gravity?", "response": "Gravity is a fundamental force..."}
        ]

        qa_samples = [
            {"question": "What is Python?", "answer": "Python is a programming language.", "context": "Programming"},
            {"question": "How do computers work?", "answer": "Computers process data using..."}
        ]

        # Test format detection
        instruction_format = processor.detect_dataset_format(instruction_samples)
        qa_format = processor.detect_dataset_format(qa_samples)

        print(f"    📊 Instruction samples format: {instruction_format}")
        print(f"    📊 QA samples format: {qa_format}")

        # Test 4: JSON Fixing
        print("\n🔧 Test 4: JSON Fixing")

        # Create malformed JSON
        malformed_json_samples = [
            '{"text": "Good sample"}',  # Valid
            '{"text": "Sample with "quotes""}',  # Unescaped quotes
            '{"text": "Sample",}',  # Trailing comma
            '{text: "No quotes on key"}',  # Missing quotes on key
        ]

        fixed_count = 0
        for i, json_str in enumerate(malformed_json_samples):
            try:
                # Test if it's already valid
                json.loads(json_str)
                print(f"    ✅ Sample {i+1}: Already valid JSON")
                fixed_count += 1
            except json.JSONDecodeError:
                # Try to fix it
                fixed_json = processor._fix_malformed_json(json_str)
                if fixed_json:
                    try:
                        json.loads(fixed_json)
                        print(f"    🔧 Sample {i+1}: Fixed malformed JSON")
                        fixed_count += 1
                    except json.JSONDecodeError:
                        print(f"    ❌ Sample {i+1}: Could not fix")
                else:
                    print(f"    ❌ Sample {i+1}: Could not fix")

        print(f"    📊 JSON fixing: {fixed_count}/{len(malformed_json_samples)} samples valid/fixed")

        print("\n🎉 All tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_data_prep()
    sys.exit(0 if success else 1)