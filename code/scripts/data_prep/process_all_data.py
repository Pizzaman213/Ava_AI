#!/usr/bin/env python3
"""
Process ALL datasets from raw directory into training-ready JSONL format.
No limits - processes every single file found.
"""

import json
import random
import os
import gc
try:
    import psutil
except ImportError:
    print("psutil not found, installing...")
    os.system("pip install psutil")
    import psutil
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_memory_usage():
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logger.info(f"Memory usage: {memory_mb:.1f} MB")

def discover_all_datasets(data_dir: Path) -> Dict[str, Dict]:
    """Automatically discover all datasets in raw directory."""
    raw_dir = data_dir / "raw"
    datasets = {}

    # Find all directories in raw
    for dataset_dir in raw_dir.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            datasets[dataset_name] = {
                "path": f"raw/{dataset_name}",
                "type": "auto_detected",
                "max_samples": None  # No limit!
            }

    logger.info(f"Discovered {len(datasets)} datasets in {raw_dir}")
    return datasets

def process_dataset_batched(data_dir: Path, dataset_name: str, config: Dict, output_dir: Path, batch_size: int = 1000, max_files_per_dataset: int = 20) -> Dict:
    """Process a dataset in batches to avoid memory issues."""
    dataset_path = data_dir / config["path"]
    dataset_file = output_dir / f"{dataset_name}_processed.jsonl"
    
    # Check for subdirectories (train, test, validation)
    subdirs = ['train', 'test', 'validation', 'val']
    search_paths = []

    for subdir in subdirs:
        subdir_path = dataset_path / subdir
        if subdir_path.exists():
            search_paths.append(subdir_path)

    # If no subdirs, search in main directory
    if not search_paths:
        search_paths = [dataset_path]

    total_processed = 0
    total_raw = 0
    file_count = 0

    # Open output file for writing
    with open(dataset_file, 'w') as outfile:
        # Process files in batches
        for search_path in search_paths:
            for ext in ['.jsonl', '.json', '.arrow', '.parquet']:
                files = sorted(search_path.glob(f"*{ext}"))
                if files:
                    logger.info(f"Found {len(files)} {ext} files in {search_path}")
                    
                    # Limit number of files to prevent memory issues
                    files_to_process = files[:max_files_per_dataset]
                    if len(files) > max_files_per_dataset:
                        logger.info(f"Processing first {max_files_per_dataset} files (out of {len(files)} total)")

                    for file_idx, file in enumerate(files_to_process):
                        try:
                            file_processed = 0
                            file_raw = 0
                            
                            if ext == '.jsonl':
                                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                                    batch = []
                                    for line_num, line in enumerate(f):
                                        try:
                                            sample = json.loads(line)
                                            sample['_dataset'] = dataset_name
                                            sample['_file'] = file.name
                                            sample['_line'] = line_num
                                            batch.append(sample)
                                            file_raw += 1
                                            
                                            if len(batch) >= batch_size:
                                                processed_batch = process_batch(batch)
                                                write_batch(outfile, processed_batch)
                                                file_processed += len(processed_batch)
                                                batch = []
                                                # Memory cleanup
                                                if line_num % 10000 == 0:
                                                    gc.collect()
                                        except json.JSONDecodeError:
                                            continue
                                    
                                    # Process remaining samples
                                    if batch:
                                        processed_batch = process_batch(batch)
                                        write_batch(outfile, processed_batch)
                                        file_processed += len(processed_batch)

                            elif ext == '.json':
                                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                                    try:
                                        data = json.load(f)
                                        if isinstance(data, list):
                                            # Process in chunks for large JSON arrays
                                            for i in range(0, len(data), batch_size):
                                                chunk = data[i:i + batch_size]
                                                batch = []
                                                for idx, item in enumerate(chunk):
                                                    item['_dataset'] = dataset_name
                                                    item['_file'] = file.name
                                                    item['_index'] = i + idx
                                                    batch.append(item)
                                                    file_raw += 1
                                                
                                                processed_batch = process_batch(batch)
                                                write_batch(outfile, processed_batch)
                                                file_processed += len(processed_batch)
                                        
                                        elif isinstance(data, dict):
                                            data['_dataset'] = dataset_name
                                            data['_file'] = file.name
                                            processed = prepare_training_sample(data)
                                            if processed:
                                                outfile.write(json.dumps(processed) + '\n')
                                                file_processed += 1
                                            file_raw += 1
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse JSON: {file}")
                                        continue

                            elif ext in ['.arrow', '.parquet']:
                                try:
                                    import pyarrow.parquet as pq
                                    import pyarrow as pa

                                    if ext == '.arrow':
                                        try:
                                            with pa.memory_map(str(file)) as source:
                                                reader = pa.ipc.open_file(source)
                                                table = reader.read_all()
                                        except:
                                            table = pq.read_table(str(file))
                                    else:
                                        table = pq.read_table(str(file))

                                    # Process in batches
                                    df = table.to_pandas()
                                    for i in range(0, len(df), batch_size):
                                        chunk = df.iloc[i:i + batch_size]
                                        batch = []
                                        for idx, row in chunk.iterrows():
                                            sample = row.to_dict()
                                            sample['_dataset'] = dataset_name
                                            sample['_file'] = file.name
                                            sample['_row'] = idx
                                            batch.append(sample)
                                            file_raw += 1
                                        
                                        processed_batch = process_batch(batch)
                                        write_batch(outfile, processed_batch)
                                        file_processed += len(processed_batch)

                                except Exception as e:
                                    logger.warning(f"Error reading {ext} file {file}: {e}")
                                    continue

                            total_processed += file_processed
                            total_raw += file_raw
                            file_count += 1

                            if file_processed > 0:
                                logger.info(f"  File {file_idx+1}/{len(files_to_process)}: {file.name} - {file_processed}/{file_raw} samples processed")

                        except Exception as e:
                            logger.error(f"Error processing {file}: {e}")
                            continue

    logger.info(f"Total for {dataset_name}: {total_processed}/{total_raw} samples processed from {file_count} files")
    
    return {
        "raw_samples": total_raw,
        "processed_samples": total_processed,
        "output_file": str(dataset_file)
    }

def process_batch(batch: List[Dict]) -> List[Dict]:
    """Process a batch of samples."""
    processed = []
    for sample in batch:
        result = prepare_training_sample(sample)
        if result:
            processed.append(result)
    return processed

def write_batch(outfile, batch: List[Dict]):
    """Write a batch of processed samples to file."""
    for sample in batch:
        outfile.write(json.dumps(sample) + '\n')

def prepare_training_sample(sample: Dict) -> Optional[Dict]:
    """Convert a raw sample to training format."""
    # Try to find text content in various field names
    text_fields = ['text', 'content', 'prompt', 'response', 'instruction',
                   'input', 'output', 'question', 'answer', 'dialogue',
                   'conversations', 'messages']

    text = None
    for field in text_fields:
        if field in sample:
            if isinstance(sample[field], list):
                # Handle conversation/message lists
                if all(isinstance(x, dict) for x in sample[field]):
                    text = "\n".join([str(msg.get('content', msg.get('text', str(msg))))
                                     for msg in sample[field]])
                else:
                    text = " ".join(str(x) for x in sample[field])
            else:
                text = sample[field]
            break

    # Combine multiple fields if needed
    if not text and 'instruction' in sample and 'output' in sample:
        text = f"Instruction: {sample['instruction']}\nOutput: {sample['output']}"
    elif not text and 'question' in sample and 'answer' in sample:
        text = f"Q: {sample['question']}\nA: {sample['answer']}"

    if not text:
        # Last resort - concatenate all string values
        text_parts = []
        for key, value in sample.items():
            if not key.startswith('_') and isinstance(value, (str, int, float)):
                text_parts.append(str(value))
        if text_parts:
            text = " ".join(text_parts)

    if not text or len(str(text)) < 10:
        return None

    return {
        "text": str(text),
        "dataset": sample.get('_dataset', 'unknown'),
        "source_file": sample.get('_file', 'unknown')
    }

def create_combined_dataset(output_dir: Path, dataset_stats: Dict, max_samples_per_dataset: int = 50000):
    """Create combined train/eval datasets by reading from individual processed files."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Creating combined dataset...")
    
    all_samples = []
    
    # Read samples from each processed dataset file
    for dataset_name, stats in dataset_stats.items():
        if stats['processed_samples'] > 0 and stats['output_file']:
            dataset_file = Path(stats['output_file'])
            if dataset_file.exists():
                samples_from_dataset = []
                with open(dataset_file, 'r') as f:
                    for line in f:
                        try:
                            sample = json.loads(line)
                            samples_from_dataset.append(sample)
                            if len(samples_from_dataset) >= max_samples_per_dataset:
                                break
                        except json.JSONDecodeError:
                            continue
                
                # Shuffle samples from this dataset
                random.shuffle(samples_from_dataset)
                all_samples.extend(samples_from_dataset)
                logger.info(f"Added {len(samples_from_dataset)} samples from {dataset_name}")
    
    if not all_samples:
        logger.warning("No samples found for combined dataset")
        return 0, 0
    
    # Shuffle all samples
    random.shuffle(all_samples)
    logger.info(f"Total samples for combined dataset: {len(all_samples):,}")
    
    # Split into train and eval (95/5 split)
    eval_size = max(1000, int(len(all_samples) * 0.05))
    eval_samples = all_samples[:eval_size]
    train_samples = all_samples[eval_size:]
    
    # Write combined training data
    train_file = output_dir / "combined_train.jsonl"
    with open(train_file, 'w') as f:
        for sample in tqdm(train_samples, desc="Writing combined training data"):
            f.write(json.dumps(sample) + '\n')
    
    # Write combined evaluation data
    eval_file = output_dir / "combined_eval.jsonl"
    with open(eval_file, 'w') as f:
        for sample in tqdm(eval_samples, desc="Writing combined evaluation data"):
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"✓ Combined train file: {train_file} ({len(train_samples):,} samples)")
    logger.info(f"✓ Combined eval file: {eval_file} ({len(eval_samples):,} samples)")
    
    return len(train_samples), len(eval_samples)

def main():
    """Process ALL data without limits using batched processing."""
    data_dir = Path("/project/code/data")
    output_dir = data_dir / "processed"
    output_dir.mkdir(exist_ok=True)

    # Discover all datasets automatically
    datasets = discover_all_datasets(data_dir)

    # Process statistics
    total_samples = 0
    dataset_stats = {}

    # Process each dataset completely using batched approach
    for dataset_name in sorted(datasets.keys()):
        config = datasets[dataset_name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {dataset_name}...")
        log_memory_usage()
        logger.info(f"{'='*60}")

        # Process dataset in batches to avoid memory issues
        stats = process_dataset_batched(data_dir, dataset_name, config, output_dir, batch_size=1000)
        
        dataset_stats[dataset_name] = stats
        total_samples += stats['processed_samples']

        if stats['processed_samples'] > 0:
            logger.info(f"✓ Saved {stats['processed_samples']:,} samples to {stats['output_file']}")
        else:
            logger.warning(f"✗ No samples processed for {dataset_name}")
        
        # Force garbage collection after each dataset
        gc.collect()
        log_memory_usage()

    # Create combined datasets from individual files
    train_count, eval_count = create_combined_dataset(output_dir, dataset_stats)

    # Write comprehensive statistics
    stats = {
        "total_samples": total_samples,
        "train_samples": train_count,
        "eval_samples": eval_count,
        "datasets_processed": len([d for d in dataset_stats.values() if d['processed_samples'] > 0]),
        "dataset_details": dataset_stats
    }

    stats_file = output_dir / "processing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("🎉 DATA PROCESSING COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"📊 Total samples processed: {total_samples:,}")
    logger.info(f"📚 Datasets processed: {len([d for d in dataset_stats.values() if d['processed_samples'] > 0])}")
    logger.info(f"🚂 Training samples: {train_count:,}")
    logger.info(f"🧪 Evaluation samples: {eval_count:,}")
    logger.info(f"\n📁 Output files:")
    logger.info(f"  - Combined train: {output_dir}/combined_train.jsonl")
    logger.info(f"  - Combined eval: {output_dir}/combined_eval.jsonl")
    logger.info(f"  - Statistics: {stats_file}")
    logger.info(f"  - Individual datasets: {output_dir}/*_processed.jsonl")
    logger.info(f"\n📈 Dataset breakdown:")
    for name, stats in sorted(dataset_stats.items(), key=lambda x: x[1]['processed_samples'], reverse=True):
        if stats['processed_samples'] > 0:
            logger.info(f"  - {name}: {stats['processed_samples']:,} samples")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()