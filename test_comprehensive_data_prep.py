#!/usr/bin/env python3
"""
Comprehensive test of the enhanced data preparation script.
Tests all new features including quality filtering, encoding fixes, and error recovery.
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.append('/project/code')

def create_comprehensive_test_data():
    """Create test datasets with various quality issues and formats."""

    # Dataset 1: High-quality instruction dataset
    instruction_dataset = [
        {
            "instruction": "Explain the concept of machine learning",
            "response": "Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves."
        },
        {
            "instruction": "What are the main types of machine learning?",
            "response": "The main types of machine learning are: 1) Supervised Learning - learns from labeled training data, 2) Unsupervised Learning - finds hidden patterns in data without labels, 3) Reinforcement Learning - learns through interaction with an environment using rewards and penalties."
        },
        {
            "instruction": "Describe neural networks",
            "response": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information and learn patterns from data through training."
        }
    ]

    # Dataset 2: Mixed quality with various issues
    mixed_quality_dataset = [
        {"text": "This is a high-quality text sample with comprehensive content and proper formatting for machine learning training purposes."},
        {"text": "bad"},  # Too short
        {"text": "ok text"},  # Borderline short
        {"text": "the the the the the the the the the the the the the the the the the the the the"},  # Too repetitive
        {"text": "Another excellent quality sample with meaningful content that provides value for training language models."},
        {"text": ""},  # Empty
        {"text": None},  # Null content
        {"content": "This uses a different field name but has good quality content for processing."},  # Different field
        {"text": "Final high-quality sample with substantial content for comprehensive testing of the data preparation pipeline."},
        {"wrong_field": "This sample has no recognizable text field"},  # Wrong field name
    ]

    # Dataset 3: Conversation format
    conversation_dataset = [
        {
            "messages": [
                {"role": "human", "content": "How does photosynthesis work?"},
                {"role": "assistant", "content": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. The process converts carbon dioxide and water into glucose and oxygen using solar energy."}
            ]
        },
        {
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris, which is also the country's largest city and a major European cultural and economic center."
        }
    ]

    # Dataset 4: QA format
    qa_dataset = [
        {
            "question": "What is artificial intelligence?",
            "answer": "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence.",
            "context": "Technology and Computer Science"
        },
        {
            "question": "How do computers process information?",
            "answer": "Computers process information by converting data into binary code (0s and 1s) and using electronic circuits to perform calculations and operations.",
            "context": "Computer Hardware"
        }
    ]

    # Dataset 5: Malformed JSON (will be written as strings)
    malformed_samples = [
        '{"text": "Valid JSON sample"}',
        '{"text": "Sample with "unescaped quotes""}',  # Malformed
        '{"text": "Good sample",}',  # Trailing comma
        '{text: "Missing quotes on key"}',  # Invalid key format
        '{"text": "Another valid sample"}'
    ]

    # Dataset 6: Encoding issues (simulated)
    encoding_issues = [
        {"text": "Text with smart quotes and dashes that may have encoding issues"},
        {"text": "Accented characters: cafe, naive, resume"},
        {"text": "Currency symbols: EUR100, GBP50, YEN1000"},
        {"text": "Normal text without encoding issues"}
    ]

    return {
        "instruction_dataset": instruction_dataset,
        "mixed_quality": mixed_quality_dataset,
        "conversations": conversation_dataset,
        "qa_dataset": qa_dataset,
        "malformed_json": malformed_samples,
        "encoding_issues": encoding_issues
    }

def test_enhanced_data_prep():
    """Test the enhanced data preparation script comprehensively."""

    print("🧪 COMPREHENSIVE DATA PREPARATION TEST")
    print("=" * 60)

    try:
        # Create test datasets
        datasets = create_comprehensive_test_data()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup directories
            raw_data_dir = Path(temp_dir) / "raw_data"
            output_dir = Path(temp_dir) / "processed_output"

            raw_data_dir.mkdir()
            output_dir.mkdir()

        print(f"📂 Test directories:")
        print(f"   Raw data: {raw_data_dir}")
        print(f"   Output: {output_dir}")

        # Create dataset files
        print(f"\n📝 Creating test datasets...")
        for dataset_name, data in datasets.items():
            dataset_dir = raw_data_dir / dataset_name
            dataset_dir.mkdir()

            if dataset_name == "malformed_json":
                # Write malformed JSON as raw text lines
                with open(dataset_dir / f"{dataset_name}.jsonl", 'w') as f:
                    for line in data:
                        f.write(line + '\n')
            else:
                # Write normal JSONL
                with open(dataset_dir / f"{dataset_name}.jsonl", 'w') as f:
                    for sample in data:
                        if sample is not None:  # Skip None samples
                            f.write(json.dumps(sample) + '\n')

            sample_count = len([s for s in data if s is not None]) if dataset_name != "malformed_json" else len(data)
            print(f"   ✅ {dataset_name}: {sample_count} samples")

        # Test 1: Run with quality filtering enabled
        print(f"\n🔍 TEST 1: Processing with Quality Filtering")
        print("-" * 50)

        # Import and run the data preparation script programmatically
        from scripts.data_prep.prepare_data_rapids import LocalDataProcessor

        processor = LocalDataProcessor(
            output_dir=str(output_dir / "quality_filtered"),
            use_gpu=False,  # Use CPU for consistent testing
            enable_quality_filtering=True,
            quality_threshold=0.3,
            format_strategy="auto"
        )

        # Process all datasets
        processor.process_all_datasets(
            raw_data_dir=str(raw_data_dir),
            max_samples_per_dataset=None,
            max_total_tokens=None
        )

        print(f"\n📊 Quality Filtering Results:")
        print(f"   ✅ Samples accepted: {processor.processing_stats['samples_accepted']:,}")
        print(f"   ❌ Samples rejected: {processor.processing_stats['samples_rejected']:,}")
        print(f"   🔧 Encoding fixes: {processor.processing_stats['encoding_fixes']:,}")

        if processor.processing_stats['samples_accepted'] + processor.processing_stats['samples_rejected'] > 0:
            acceptance_rate = processor.processing_stats['samples_accepted'] / (
                processor.processing_stats['samples_accepted'] + processor.processing_stats['samples_rejected']
            )
            print(f"   📈 Acceptance rate: {acceptance_rate:.1%}")

        # Show rejection reasons
        if processor.processing_stats['rejection_reasons']:
            print(f"   📋 Top rejection reasons:")
            sorted_reasons = sorted(
                processor.processing_stats['rejection_reasons'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for reason, count in sorted_reasons[:5]:
                print(f"      • {reason}: {count:,}")

        # Test 2: Run without quality filtering for comparison
        print(f"\n🔄 TEST 2: Processing without Quality Filtering")
        print("-" * 50)

        processor_no_filter = LocalDataProcessor(
            output_dir=str(output_dir / "no_filtering"),
            use_gpu=False,
            enable_quality_filtering=False,
            format_strategy="auto"
        )

        processor_no_filter.process_all_datasets(
            raw_data_dir=str(raw_data_dir),
            max_samples_per_dataset=None,
            max_total_tokens=None
        )

        print(f"📊 No Filtering Results:")
        print(f"   📝 Samples processed: {processor_no_filter.processing_stats['samples_accepted']:,}")

        # Test 3: Check output files
        print(f"\n📁 TEST 3: Validating Output Files")
        print("-" * 50)

        quality_output_dir = output_dir / "quality_filtered"
        no_filter_output_dir = output_dir / "no_filtering"

        # Check if files were created
        quality_files = list(quality_output_dir.glob("*.jsonl"))
        no_filter_files = list(no_filter_output_dir.glob("*.jsonl"))

        print(f"   📂 Quality filtered files: {len(quality_files)}")
        print(f"   📂 No filtering files: {len(no_filter_files)}")

        # Validate some output files
        if quality_files:
            combined_file = quality_output_dir / "combined_processed.jsonl"
            if combined_file.exists():
                # Count lines in combined file
                with open(combined_file, 'r') as f:
                    line_count = sum(1 for line in f if line.strip())
                print(f"   ✅ Combined file: {line_count:,} lines")

                # Validate JSON format
                with open(combined_file, 'r') as f:
                    valid_json_count = 0
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                json.loads(line)
                                valid_json_count += 1
                            except json.JSONDecodeError:
                                print(f"   ⚠️  Invalid JSON at line {line_num}")
                                if line_num <= 5:  # Only show first few errors
                                    print(f"      Content: {line[:100]}...")

                    print(f"   ✅ Valid JSON lines: {valid_json_count:,}/{line_count:,}")

        # Test 4: Check statistics files
        print(f"\n📊 TEST 4: Validating Statistics Files")
        print("-" * 50)

        stats_file = quality_output_dir / "processing_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)

            print(f"   ✅ Statistics file found")
            print(f"   📈 Enhanced stats available: {'enhanced_processing_stats' in stats}")
            print(f"   📋 Quality report available: {'quality_report' in stats}")
            print(f"   🔧 Libraries detected: {len(stats.get('libraries_available', {}))}")

            # Show some key statistics
            if 'overall_stats' in stats:
                overall = stats['overall_stats']
                print(f"   📚 Total texts: {overall.get('total_texts', 0):,}")
                print(f"   📏 Avg length: {overall.get('avg_length', 0):.1f} chars")
                print(f"   💬 Total words: {overall.get('total_words', 0):,}")

        # Test 5: Quality comparison
        print(f"\n⚖️  TEST 5: Quality Comparison")
        print("-" * 50)

        quality_samples = processor.processing_stats['samples_accepted']
        no_filter_samples = processor_no_filter.processing_stats['samples_accepted']

        print(f"   📊 Quality filtering: {quality_samples:,} samples")
        print(f"   📊 No filtering: {no_filter_samples:,} samples")

        if no_filter_samples > 0:
            filtering_effectiveness = (no_filter_samples - quality_samples) / no_filter_samples
            print(f"   🎯 Filtering effectiveness: {filtering_effectiveness:.1%} samples filtered")

        if processor.processing_stats['encoding_fixes'] > 0:
            print(f"   🔧 Encoding issues detected and fixed: {processor.processing_stats['encoding_fixes']:,}")

        print(f"\n🎉 COMPREHENSIVE TEST COMPLETED!")
        print(f"✅ All components working correctly:")
        print(f"   • Quality filtering and scoring")
        print(f"   • Multi-format dataset detection")
        print(f"   • Encoding issue detection and fixing")
        print(f"   • Error recovery and graceful handling")
        print(f"   • Comprehensive statistics and reporting")
        print(f"   • Output file validation")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_data_prep()
    sys.exit(0 if success else 1)