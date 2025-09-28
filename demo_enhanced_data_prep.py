#!/usr/bin/env python3
"""
Demonstration of enhanced data preparation features.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append('/project/code')

def create_demo_datasets():
    """Create demo datasets with various quality issues."""

    # Dataset 1: High quality instruction dataset
    high_quality = [
        {"instruction": "Explain machine learning", "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
        {"instruction": "What is photosynthesis?", "response": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll."},
        {"instruction": "Describe the water cycle", "response": "The water cycle is the continuous movement of water on, above and below the surface of the Earth through processes like evaporation, condensation, and precipitation."}
    ]

    # Dataset 2: Mixed quality with various issues
    mixed_quality = [
        {"text": "This is a good quality text sample with proper content and length."},
        {"text": "bad"},  # Too short
        {"text": "word " * 500},  # Too repetitive
        {"text": "This text has encoding issues that need fixing."},  # Encoding issues
        {"content": "Alternative field name but good content here."},  # Different field name
        {"text": ""},  # Empty content
        {"text": "Another high quality sample with meaningful content for training."},
        {"wrong_field": "This has no text field"},  # Wrong field
        {"text": "Final good quality sample."}
    ]

    # Dataset 3: Conversation format
    conversations = [
        {"messages": [
            {"role": "human", "content": "How do computers work?"},
            {"role": "assistant", "content": "Computers work by processing data using electronic circuits that represent information as binary digits (0s and 1s)."}
        ]},
        {"input": "What is Python?", "output": "Python is a high-level programming language known for its simplicity and readability."}
    ]

    return {
        "high_quality": high_quality,
        "mixed_quality": mixed_quality,
        "conversations": conversations
    }

def demo_enhanced_processing():
    """Demonstrate the enhanced data processing capabilities."""

    print("🚀 Enhanced Data Preparation Demo")
    print("=" * 60)

    # Create demo datasets
    datasets = create_demo_datasets()

    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir) / "demo_data"
        data_dir.mkdir()

        # Save demo datasets
        for name, data in datasets.items():
            dataset_dir = data_dir / name
            dataset_dir.mkdir()

            with open(dataset_dir / f"{name}.jsonl", 'w') as f:
                for sample in data:
                    f.write(json.dumps(sample) + '\n')

        print(f"📂 Created demo datasets in: {data_dir}")
        for name, data in datasets.items():
            print(f"  • {name}: {len(data)} samples")

        # Test 1: Process with quality filtering enabled
        print(f"\n🔍 Test 1: Processing with Quality Filtering")
        print("-" * 40)

        from scripts.data_prep.prepare_data_rapids import LocalDataProcessor

        output_dir = Path(temp_dir) / "output_quality"
        processor_quality = LocalDataProcessor(
            output_dir=str(output_dir),
            use_gpu=False,
            enable_quality_filtering=True,
            quality_threshold=0.3,
            format_strategy="auto"
        )

        # Process each dataset
        for name, data in datasets.items():
            print(f"\n  📋 Processing {name} dataset:")
            texts = processor_quality.extract_text_efficiently(data)

            print(f"    ✅ Accepted: {processor_quality.processing_stats['samples_accepted']}")
            print(f"    ❌ Rejected: {processor_quality.processing_stats['samples_rejected']}")
            print(f"    🔧 Encoding fixes: {processor_quality.processing_stats['encoding_fixes']}")

            # Show rejection reasons
            if processor_quality.processing_stats['rejection_reasons']:
                print("    📊 Rejection reasons:")
                for reason, count in processor_quality.processing_stats['rejection_reasons'].items():
                    print(f"      • {reason}: {count}")

            # Reset stats for next dataset
            processor_quality.processing_stats = {
                'total_samples_processed': 0,
                'samples_accepted': 0,
                'samples_rejected': 0,
                'encoding_fixes': 0,
                'quality_improvements': 0,
                'rejection_reasons': {}
            }

        # Test 2: Process without quality filtering
        print(f"\n🔄 Test 2: Processing without Quality Filtering")
        print("-" * 40)

        output_dir_no_filter = Path(temp_dir) / "output_no_filter"
        processor_no_filter = LocalDataProcessor(
            output_dir=str(output_dir_no_filter),
            use_gpu=False,
            enable_quality_filtering=False,
            format_strategy="auto"
        )

        total_original = sum(len(data) for data in datasets.values())
        total_processed = 0

        for name, data in datasets.items():
            texts = processor_no_filter.extract_text_efficiently(data)
            total_processed += len(texts)

        print(f"  📊 Without filtering: {total_processed}/{total_original} samples processed")

        # Test 3: Format Detection Demo
        print(f"\n📋 Test 3: Format Detection Demo")
        print("-" * 40)

        for name, data in datasets.items():
            if data:  # Check if dataset is not empty
                detected_format = processor_quality.detect_dataset_format(data)
                print(f"  • {name}: {detected_format}")

        # Test 4: Quality Analysis Demo
        print(f"\n🎯 Test 4: Quality Analysis Demo")
        print("-" * 40)

        from scripts.data_prep.prepare_data_rapids import DatasetQualityAnalyzer
        analyzer = DatasetQualityAnalyzer()

        # Analyze a few samples
        test_samples = [
            {"text": "High quality comprehensive text with meaningful content."},
            {"text": "bad"},
            {"text": "This text has encoding issues."},
            {"text": "word " * 100}
        ]

        for i, sample in enumerate(test_samples):
            score, metrics = analyzer.calculate_text_quality_score(sample.get('text', ''))
            print(f"  Sample {i+1}: score={score:.2f}, reason={metrics.get('reason', 'N/A')}")

        print(f"\n✅ Demo completed successfully!")
        print(f"📈 The enhanced data preparation script provides:")
        print(f"  • Automatic quality filtering and scoring")
        print(f"  • Encoding issue detection and fixing")
        print(f"  • Multi-format dataset support")
        print(f"  • Comprehensive quality reporting")
        print(f"  • Malformed data recovery")
        print(f"  • Language detection and filtering")

if __name__ == "__main__":
    demo_enhanced_processing()