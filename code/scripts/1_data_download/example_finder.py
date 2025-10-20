#!/usr/bin/env python3
"""
Dataset Example Finder - Preview samples from datasets before downloading

This tool helps you:
- Browse available datasets by category
- Preview sample data from any dataset
- See the structure and format of each dataset
- Test text extraction before full download
- Find the right datasets for your needs

Usage:
  # List all available datasets
  python example_finder.py --list

  # List datasets by category
  python example_finder.py --list --category code

  # Preview samples from a dataset
  python example_finder.py --preview "teknium/OpenHermes-2.5" --samples 5

  # Search for datasets by keyword
  python example_finder.py --search "math"

  # Show detailed info about a dataset
  python example_finder.py --info "Anthropic/hh-rlhf"

  # Preview with story filtering
  python example_finder.py --preview "roneneldan/TinyStories" --filter-stories --samples 10
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add parent directory to path to import unified_download
sys.path.insert(0, str(Path(__file__).parent))

try:
    from unified_download import DATASETS_CONFIG, UnifiedDownloader
except ImportError:
    print("Error: Could not import from unified_download.py")
    print("Make sure unified_download.py is in the same directory.")
    sys.exit(1)


class DatasetExampleFinder:
    """Find and preview dataset examples"""

    def __init__(self):
        """Initialize the example finder"""
        # Import datasets library
        try:
            from datasets import load_dataset
            import datasets
            datasets.disable_progress_bar()
            self.load_dataset = load_dataset
        except ImportError:
            print("Installing datasets library...")
            os.system(f"{sys.executable} -m pip install datasets")
            from datasets import load_dataset
            import datasets
            datasets.disable_progress_bar()
            self.load_dataset = load_dataset

        # Create downloader instance for text extraction
        self.downloader = UnifiedDownloader()

    def list_datasets(self, category: Optional[str] = None, show_details: bool = False,
                     count_only: bool = False, show_stats: bool = False):
        """List all available datasets, optionally filtered by category"""

        # Filter by category if specified
        if category:
            filtered = {}
            for name, config in DATASETS_CONFIG.items():
                categories = config.get("categories", [])
                if category.lower() in [c.lower() for c in categories]:
                    filtered[name] = config
            datasets_to_show = filtered

            if count_only:
                print(len(datasets_to_show))
                return

            print(f"\n{'='*80}")
            print(f"DATASETS IN CATEGORY: {category.upper()}")
            print(f"{'='*80}\n")
        else:
            datasets_to_show = DATASETS_CONFIG

            if count_only:
                print(len(datasets_to_show))
                return

            print(f"\n{'='*80}")
            print(f"ALL AVAILABLE DATASETS ({len(DATASETS_CONFIG)} total)")
            print(f"{'='*80}\n")

        if not datasets_to_show:
            print(f"No datasets found in category '{category}'")
            return

        # If show_stats, just output numbers
        if show_stats:
            print(f"{'Dataset':<50} {'Examples':>15} {'Tokens (M)':>15}")
            print("=" * 80)

            total_examples = 0
            total_tokens_m = 0

            for name, config in sorted(datasets_to_show.items()):
                max_samples = config.get("max_samples")
                tokens_m = config.get("estimated_tokens_millions", 0)

                # Format examples
                if max_samples is None:
                    examples_str = "unlimited"
                elif isinstance(max_samples, int):
                    examples_str = f"{max_samples:,}"
                else:
                    examples_str = str(max_samples)

                # Format tokens
                if tokens_m and tokens_m > 0:
                    tokens_str = f"{tokens_m:,}"
                else:
                    tokens_str = "N/A"

                print(f"{name:<50} {examples_str:>15} {tokens_str:>15}")

                if isinstance(max_samples, int):
                    total_examples += max_samples
                if tokens_m > 0:
                    total_tokens_m += tokens_m

            print("=" * 80)
            print(f"{'TOTAL':<50} {total_examples:>15,} {total_tokens_m:>15,}")
            print(f"\nEstimated Total: ~{total_tokens_m/1000:.2f}B tokens")
            return

        # Group by category
        by_category = {}
        for name, config in datasets_to_show.items():
            categories = config.get("categories", ["other"])
            main_cat = categories[0]
            if main_cat not in by_category:
                by_category[main_cat] = []
            by_category[main_cat].append((name, config))

        # Display grouped datasets
        for cat, dataset_list in sorted(by_category.items()):
            print(f"📂 {cat.upper().replace('_', ' ')} ({len(dataset_list)} datasets)")
            print("-" * 80)

            for name, config in sorted(dataset_list):
                description = config.get("description", "No description")
                tokens = config.get("tokens", "unknown")
                quality = config.get("quality_score", "N/A")
                default = "⭐ DEFAULT" if config.get("default_10b", False) else ""

                print(f"  🔹 {name} {default}")
                print(f"     {description}")

                if show_details:
                    print(f"     Quality: {quality}/10 | Tokens: {tokens}")
                    print(f"     Splits: {', '.join(config.get('splits', ['train']))}")
                    print(f"     Categories: {', '.join(config.get('categories', []))}")

                print()

        print(f"\n📊 Total: {len(datasets_to_show)} datasets")

        # Show category summary
        if not category:
            print(f"\n📋 Categories:")
            for cat, dataset_list in sorted(by_category.items()):
                print(f"   • {cat}: {len(dataset_list)} datasets")

    def search_datasets(self, keyword: str, count_only: bool = False):
        """Search for datasets by keyword"""
        keyword_lower = keyword.lower()
        matches = []

        for name, config in DATASETS_CONFIG.items():
            # Search in name, description, and categories
            name_match = keyword_lower in name.lower()
            desc_match = keyword_lower in config.get("description", "").lower()
            cat_match = any(keyword_lower in cat.lower() for cat in config.get("categories", []))

            if name_match or desc_match or cat_match:
                matches.append((name, config))

        if count_only:
            print(len(matches))
            return

        print(f"\n{'='*80}")
        print(f"SEARCH RESULTS FOR: '{keyword}' ({len(matches)} matches)")
        print(f"{'='*80}\n")

        if not matches:
            print(f"No datasets found matching '{keyword}'")
            return

        for name, config in matches:
            description = config.get("description", "No description")
            categories = config.get("categories", [])
            quality = config.get("quality_score", "N/A")

            print(f"🔹 {name}")
            print(f"   {description}")
            print(f"   Quality: {quality}/10 | Categories: {', '.join(categories)}")
            print()

    def show_dataset_info(self, dataset_name: str):
        """Show detailed information about a specific dataset"""
        if dataset_name not in DATASETS_CONFIG:
            print(f"❌ Dataset '{dataset_name}' not found in configuration")
            print(f"\nAvailable datasets:")
            for name in list(DATASETS_CONFIG.keys())[:10]:
                print(f"  - {name}")
            print(f"  ... and {len(DATASETS_CONFIG) - 10} more")
            return

        config = DATASETS_CONFIG[dataset_name]

        print(f"\n{'='*80}")
        print(f"DATASET INFORMATION: {dataset_name}")
        print(f"{'='*80}\n")

        print(f"📝 Description:")
        print(f"   {config.get('description', 'No description available')}\n")

        print(f"🏷️  Categories:")
        print(f"   {', '.join(config.get('categories', ['N/A']))}\n")

        print(f"📊 Details:")
        print(f"   Quality Score: {config.get('quality_score', 'N/A')}/10")
        print(f"   Token Count: {config.get('tokens', 'unknown')}")
        print(f"   Priority: {config.get('priority', 'N/A')}")
        print(f"   Default Download: {'Yes' if config.get('default_10b', False) else 'No'}")
        print(f"   Large Dataset: {'Yes' if config.get('large', False) else 'No'}")
        print()

        print(f"📁 Splits:")
        print(f"   {', '.join(config.get('splits', ['train']))}\n")

        if config.get("subset"):
            print(f"🔧 Subset: {config['subset']}\n")

        if config.get("estimated_tokens_millions"):
            tokens_m = config["estimated_tokens_millions"]
            print(f"💾 Estimated Size:")
            print(f"   ~{tokens_m}M tokens (~{tokens_m/1000:.2f}B tokens)")
            if config.get("max_samples"):
                print(f"   Max samples: {config['max_samples']:,}")
            print()

        print(f"🔗 HuggingFace URL:")
        print(f"   https://huggingface.co/datasets/{dataset_name}\n")

    def preview_samples(self, dataset_name: str, num_samples: int = 5,
                       split: str = "train", filter_stories: bool = False):
        """Preview sample data from a dataset"""

        if dataset_name not in DATASETS_CONFIG:
            print(f"❌ Dataset '{dataset_name}' not found in configuration")
            return

        config = DATASETS_CONFIG[dataset_name]

        print(f"\n{'='*80}")
        print(f"PREVIEWING: {dataset_name}")
        print(f"Split: {split} | Samples: {num_samples}")
        if filter_stories:
            print(f"Filtering: 'once upon a time' stories only")
        print(f"{'='*80}\n")

        try:
            # Load dataset in streaming mode
            print("Loading dataset...")

            if config.get("subset"):
                dataset_args = [dataset_name, config["subset"]]
            else:
                dataset_args = [dataset_name]

            try:
                dataset = self.load_dataset(*dataset_args, split=split, streaming=True)
            except Exception as e:
                print(f"⚠️  Could not load split '{split}', trying basic parameters...")
                try:
                    dataset = self.load_dataset(dataset_name, split="train", streaming=True)
                    split = "train"
                except Exception as e2:
                    print(f"❌ Failed to load dataset: {e2}")
                    return

            print(f"✅ Dataset loaded successfully!\n")

            # Get samples
            samples_found = 0
            samples_processed = 0
            max_process = num_samples * 100  # Process up to 100x samples if filtering

            print(f"{'='*80}")
            print("SAMPLE DATA")
            print(f"{'='*80}\n")

            for idx, sample in enumerate(dataset):
                samples_processed += 1

                # Extract text using the same logic as unified_download
                text = self.downloader.extract_text(sample, dataset_name)

                # Apply story filter if requested
                if filter_stories:
                    if not text or not self.downloader.starts_with_once_upon(text):
                        if samples_processed > max_process:
                            print(f"⚠️  Processed {samples_processed} samples, found {samples_found} stories")
                            break
                        continue

                samples_found += 1

                # Display sample
                print(f"{'─'*80}")
                print(f"SAMPLE #{samples_found}")
                print(f"{'─'*80}")

                # Show raw structure
                print("📋 Raw Structure:")
                # Show keys and types
                if isinstance(sample, dict):
                    for key, value in list(sample.items())[:10]:  # Show first 10 keys
                        value_type = type(value).__name__
                        value_preview = str(value)[:50] if not isinstance(value, (list, dict)) else f"{value_type}"
                        print(f"   {key}: {value_type} = {value_preview}{'...' if len(str(value)) > 50 else ''}")
                    if len(sample) > 10:
                        print(f"   ... and {len(sample) - 10} more fields")
                else:
                    print(f"   {type(sample).__name__}: {str(sample)[:100]}")

                print()

                # Show extracted text
                print("📝 Extracted Text:")
                if text:
                    # Show first 500 characters
                    text_preview = text[:500]
                    print(f"   {text_preview}")
                    if len(text) > 500:
                        print(f"   ... (truncated, {len(text)} total chars)")

                    # Show word and line count
                    word_count = len(text.split())
                    line_count = len(text.split('\n'))
                    print(f"\n   📊 Stats: {word_count} words, {line_count} lines, {len(text)} chars")
                else:
                    print("   ⚠️  No text could be extracted")

                print()

                if samples_found >= num_samples:
                    break

                if samples_processed > max_process:
                    print(f"⚠️  Reached processing limit ({max_process} samples)")
                    break

            print(f"{'='*80}")
            print(f"Preview complete: {samples_found} samples shown")
            if filter_stories and samples_processed > samples_found:
                print(f"Processed {samples_processed} samples to find {samples_found} matching stories")
            print(f"{'='*80}\n")

        except Exception as e:
            print(f"❌ Error previewing dataset: {e}")
            import traceback
            traceback.print_exc()

    def list_categories(self):
        """List all available categories"""
        categories = set()
        for config in DATASETS_CONFIG.values():
            categories.update(config.get("categories", []))

        print(f"\n{'='*80}")
        print(f"AVAILABLE CATEGORIES ({len(categories)} total)")
        print(f"{'='*80}\n")

        # Count datasets per category
        category_counts = {}
        for cat in categories:
            count = sum(1 for config in DATASETS_CONFIG.values()
                       if cat in config.get("categories", []))
            category_counts[cat] = count

        # Display sorted by count
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  📂 {cat}: {count} datasets")

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Dataset Example Finder - Preview datasets before downloading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python example_finder.py --list

  # List datasets in a specific category
  python example_finder.py --list --category code

  # Show all categories
  python example_finder.py --categories

  # Preview samples from a dataset
  python example_finder.py --preview "teknium/OpenHermes-2.5" --samples 5

  # Preview with detailed dataset info
  python example_finder.py --list --details

  # Search for datasets
  python example_finder.py --search "math"
  python example_finder.py --search "anthropic"

  # Show detailed info about a dataset
  python example_finder.py --info "Anthropic/hh-rlhf"

  # Preview stories with filtering
  python example_finder.py --preview "roneneldan/TinyStories" --filter-stories --samples 10

  # Preview from a specific split
  python example_finder.py --preview "meta-math/MetaMathQA" --split test --samples 3
        """
    )

    # Main actions
    parser.add_argument("--list", action="store_true",
                       help="List all available datasets")
    parser.add_argument("--categories", action="store_true",
                       help="List all available categories")
    parser.add_argument("--search", type=str, metavar="KEYWORD",
                       help="Search for datasets by keyword")
    parser.add_argument("--info", type=str, metavar="DATASET",
                       help="Show detailed info about a dataset")
    parser.add_argument("--preview", type=str, metavar="DATASET",
                       help="Preview samples from a dataset")
    parser.add_argument("--count", action="store_true",
                       help="Just output the number of datasets (use with --list, --search, or --category)")
    parser.add_argument("--stats", action="store_true",
                       help="Show example count and token estimates for each dataset")

    # Options
    parser.add_argument("--category", type=str,
                       help="Filter by category (use with --list)")
    parser.add_argument("--details", action="store_true",
                       help="Show detailed information (use with --list)")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of samples to preview (default: 5)")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to preview (default: train)")
    parser.add_argument("--filter-stories", action="store_true",
                       help="Filter for 'once upon a time' stories (use with --preview)")

    args = parser.parse_args()

    # Create finder instance
    finder = DatasetExampleFinder()

    # Execute requested action
    if args.list:
        finder.list_datasets(category=args.category, show_details=args.details,
                           count_only=args.count, show_stats=args.stats)
    elif args.categories:
        if args.count:
            # Count number of categories
            categories = set()
            for config in DATASETS_CONFIG.values():
                categories.update(config.get("categories", []))
            print(len(categories))
        else:
            finder.list_categories()
    elif args.search:
        finder.search_datasets(args.search, count_only=args.count)
    elif args.info:
        finder.show_dataset_info(args.info)
    elif args.preview:
        finder.preview_samples(
            args.preview,
            num_samples=args.samples,
            split=args.split,
            filter_stories=args.filter_stories
        )
    else:
        # Default: show help
        print("\n" + "="*80)
        print("DATASET EXAMPLE FINDER")
        print("="*80)
        print("\nNo action specified. Use --help to see available options.\n")
        print("Quick start:")
        print("  • List all datasets:        python example_finder.py --list")
        print("  • List categories:          python example_finder.py --categories")
        print("  • Search datasets:          python example_finder.py --search 'math'")
        print("  • Preview dataset:          python example_finder.py --preview 'teknium/OpenHermes-2.5'")
        print("  • Show dataset info:        python example_finder.py --info 'Anthropic/hh-rlhf'")
        print()


if __name__ == "__main__":
    main()
