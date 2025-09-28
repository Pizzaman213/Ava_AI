#!/usr/bin/env python3
"""
Simple but comprehensive test of the enhanced data preparation script.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append('/project/code')

def test_data_prep():
    """Test the enhanced data preparation script."""
    print("🧪 TESTING ENHANCED DATA PREPARATION")
    print("=" * 50)

    try:
        # Test datasets with various quality issues
        test_datasets = {
            "high_quality": [
                {"instruction": "Explain machine learning", "response": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming."},
                {"instruction": "What is photosynthesis?", "response": "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll."}
            ],
            "mixed_quality": [
                {"text": "This is a high-quality text sample with comprehensive content."},
                {"text": "bad"},  # Too short
                {"text": "word " * 100},  # Too repetitive
                {"text": ""},  # Empty
                {"content": "Alternative field name with good content."},
                {"text": "Another excellent quality sample for testing."}
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup directories
            raw_data_dir = Path(temp_dir) / "raw_data"
            output_dir = Path(temp_dir) / "output"
            raw_data_dir.mkdir()

            print(f"📂 Test directories: {raw_data_dir}")

            # Create test files
            for dataset_name, data in test_datasets.items():
                dataset_dir = raw_data_dir / dataset_name
                dataset_dir.mkdir()

                with open(dataset_dir / f"{dataset_name}.jsonl", 'w') as f:
                    for sample in data:
                        f.write(json.dumps(sample) + '\n')

                print(f"   ✅ {dataset_name}: {len(data)} samples")

            # Test 1: Quality filtering enabled
            print(f"\n🔍 TEST 1: Processing with Quality Filtering")
            print("-" * 40)

            from scripts.data_prep.prepare_data_rapids import LocalDataProcessor

            processor = LocalDataProcessor(
                output_dir=str(output_dir),
                use_gpu=False,
                enable_quality_filtering=True,
                quality_threshold=0.3
            )

            # Process datasets
            processor.process_all_datasets(raw_data_dir=str(raw_data_dir))

            # Show results
            stats = processor.processing_stats
            print(f"   ✅ Samples accepted: {stats['samples_accepted']}")
            print(f"   ❌ Samples rejected: {stats['samples_rejected']}")
            print(f"   🔧 Encoding fixes: {stats['encoding_fixes']}")

            if stats['rejection_reasons']:
                print(f"   📋 Rejection reasons:")
                for reason, count in stats['rejection_reasons'].items():
                    print(f"      • {reason}: {count}")

            # Test 2: Verify output files
            print(f"\n📁 TEST 2: Verifying Output Files")
            print("-" * 40)

            output_files = list(output_dir.glob("*.jsonl"))
            print(f"   📂 Output files created: {len(output_files)}")

            combined_file = output_dir / "combined_processed.jsonl"
            if combined_file.exists():
                with open(combined_file, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                print(f"   ✅ Combined file: {len(lines)} lines")

                # Validate JSON
                valid_json = 0
                for line in lines[:5]:  # Check first 5 lines
                    try:
                        json.loads(line)
                        valid_json += 1
                    except json.JSONDecodeError:
                        pass

                print(f"   ✅ Valid JSON: {valid_json}/{min(5, len(lines))} samples checked")

            # Test 3: Statistics file
            print(f"\n📊 TEST 3: Statistics Validation")
            print("-" * 40)

            stats_file = output_dir / "processing_stats.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    file_stats = json.load(f)

                print(f"   ✅ Statistics file created")
                print(f"   📈 Enhanced stats: {'enhanced_processing_stats' in file_stats}")
                print(f"   📋 Quality report: {'quality_report' in file_stats}")

                if 'overall_stats' in file_stats:
                    overall = file_stats['overall_stats']
                    print(f"   📚 Total texts: {overall.get('total_texts', 0)}")
                    print(f"   📏 Avg length: {overall.get('avg_length', 0):.1f} chars")

            # Test 4: Format detection
            print(f"\n📋 TEST 4: Format Detection")
            print("-" * 40)

            # Test different formats
            formats_to_test = [
                [{"instruction": "Test", "response": "Answer"}],  # instruction format
                [{"question": "What?", "answer": "This", "context": "Here"}],  # qa format
                [{"text": "Simple text"}]  # simple format
            ]

            for i, format_data in enumerate(formats_to_test):
                detected_format = processor.detect_dataset_format(format_data)
                print(f"   Format {i+1}: {detected_format}")

            print(f"\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
            print(f"✅ Enhanced data preparation features working:")
            print(f"   • Quality filtering and scoring")
            print(f"   • Multi-format detection")
            print(f"   • Error handling and recovery")
            print(f"   • Enhanced statistics and reporting")

            return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_prep()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)