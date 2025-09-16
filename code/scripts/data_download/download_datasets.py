#!/usr/bin/env python3

import os
import sys
from datasets import load_dataset
import json
from pathlib import Path
import subprocess
import multiprocessing as mp
from tqdm import tqdm
import pyarrow.parquet as pq
import pyarrow as pa
import time
import hashlib
import pickle
import shutil
import logging
from typing import Optional, Dict, List, Tuple, Any
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import path utilities
from src.utils.path_utils import get_data_dir, get_scripts_dir

# Configure logging with multiple handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Only console output initially
    ]
)
logger = logging.getLogger(__name__)

# Optimize for large datasets with redundant configurations
BATCH_SIZE = 10000  # Process in batches
ALT_BATCH_SIZE = 5000  # Alternative batch size for fallback
MIN_BATCH_SIZE = 1000  # Minimum batch size for extreme fallback
NUM_PROC = mp.cpu_count()  # Use all available CPUs
FALLBACK_NUM_PROC = max(1, NUM_PROC // 2)  # Fallback with half CPUs
MIN_NUM_PROC = 1  # Minimum single process fallback

# Retry configurations with redundancy
MAX_RETRIES = 5  # Maximum number of retry attempts
INITIAL_RETRY_DELAY = 2  # Initial delay in seconds
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff factor
CONNECTION_TIMEOUT = 30  # Connection timeout in seconds
DOWNLOAD_TIMEOUT = 300  # Download timeout in seconds

datasets_to_download = [
    ("HuggingFaceTB/everyday-conversations-llama3.1-2k", "everyday-conversations", None),
    ("HuggingFaceTB/cosmopedia-100k", "cosmopedia-100k", None),
    ("Open-Orca/OpenOrca", "OpenOrca", None),
    ("llamafactory/tiny-supervised-dataset", "tiny-supervised-dataset", None),
    ("nvidia/OpenMathInstruct-1", "OpenMathInstruct-1", None),
    ("nvidia/Nemotron-Pretraining-Dataset-sample", "Nemotron-Pretraining-Dataset-sample", "Nemotron-CC-MATH"),
    # ("nvidia/AF-Think", "AF-Think", None),  # Skipping due to errors
    ("nvidia/AceReason-1.1-SFT", "AceReason-1.1-SFT", None),
    ("nvidia/HelpSteer2", "HelpSteer2", None),
    ("yahma/alpaca-cleaned", "alpaca-cleaned", None),
    ("OpenAssistant/oasst1", "oasst1", None),
    ("wikimedia/wikipedia", "wikipedia", "20220301.en")
]

def verify_dataset_integrity(output_dir: str, expected_splits: Optional[List[str]] = None) -> bool:
    """Verify dataset integrity with multiple redundant checks"""
    try:
        logger.info(f"Verifying dataset integrity for {output_dir}")
        
        # Check 1: Directory exists
        if not os.path.exists(output_dir):
            logger.error(f"Directory {output_dir} does not exist")
            return False
        
        # Check 2: Has content
        files = list(Path(output_dir).rglob('*'))
        if len(files) < 2:  # At least dataset_info.json and one data file
            logger.error(f"Directory {output_dir} has insufficient files")
            return False
        
        # Check 3: Info file exists and is valid
        info_file = Path(output_dir) / "dataset_info.json"
        if not info_file.exists():
            logger.error(f"Missing dataset_info.json in {output_dir}")
            return False
        
        try:
            with open(info_file, 'r') as f:
                info = json.load(f)
                if 'dataset_name' not in info or 'splits' not in info:
                    logger.error(f"Invalid dataset_info.json structure")
                    return False
        except Exception as e:
            logger.error(f"Cannot read dataset_info.json: {e}")
            return False
        
        # Check 4: Expected splits exist
        if expected_splits:
            for split in expected_splits:
                split_path = Path(output_dir) / split
                if not split_path.exists() and not any(f.name.startswith(f"{split}_") for f in Path(output_dir).iterdir()):
                    logger.error(f"Missing expected split: {split}")
                    return False
        
        # Check 5: File checksums (create if not exist)
        checksum_file = Path(output_dir) / ".checksums.json"
        if checksum_file.exists():
            try:
                with open(checksum_file, 'r') as f:
                    checksums = json.load(f)
                    # Verify at least one checksum
                    for file_path, expected_hash in list(checksums.items())[:1]:
                        full_path = Path(output_dir) / file_path
                        if full_path.exists():
                            actual_hash = hashlib.md5(full_path.read_bytes()).hexdigest()
                            if actual_hash != expected_hash:
                                logger.warning(f"Checksum mismatch for {file_path}")
                                return False
            except Exception as e:
                logger.warning(f"Cannot verify checksums: {e}")
        
        logger.info(f"✅ Dataset integrity verified for {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying dataset integrity: {e}")
        return False

def create_dataset_backup(output_dir: str) -> Optional[str]:
    """Create a backup of the dataset before operations"""
    try:
        backup_dir = f"{output_dir}_backup_{int(time.time())}"
        logger.info(f"Creating backup: {backup_dir}")
        shutil.copytree(output_dir, backup_dir)
        logger.info(f"✅ Backup created: {backup_dir}")
        return backup_dir
    except Exception as e:
        logger.warning(f"Could not create backup: {e}")
        return None

def restore_from_backup(backup_dir: str, output_dir: str) -> bool:
    """Restore dataset from backup"""
    try:
        logger.info(f"Restoring from backup: {backup_dir}")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.copytree(backup_dir, output_dir)
        logger.info(f"✅ Restored from backup")
        return True
    except Exception as e:
        logger.error(f"Could not restore from backup: {e}")
        return False

def save_dataset_checksums(output_dir: str) -> None:
    """Save checksums for all dataset files"""
    try:
        checksums = {}
        for file_path in Path(output_dir).rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                rel_path = file_path.relative_to(output_dir)
                checksums[str(rel_path)] = hashlib.md5(file_path.read_bytes()).hexdigest()
        
        checksum_file = Path(output_dir) / ".checksums.json"
        with open(checksum_file, 'w') as f:
            json.dump(checksums, f, indent=2)
        logger.info(f"Saved checksums for {len(checksums)} files")
    except Exception as e:
        logger.warning(f"Could not save checksums: {e}")

def download_with_retry(download_func, *args, **kwargs):
    """Execute download function with multiple retry attempts and fallback strategies"""
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Download attempt {attempt + 1}/{MAX_RETRIES}")
            
            # Adjust parameters for subsequent attempts
            if attempt > 0:
                # Reduce batch size on retries
                if 'batch_size' in kwargs:
                    kwargs['batch_size'] = max(MIN_BATCH_SIZE, kwargs['batch_size'] // 2)
                    logger.info(f"Reduced batch size to {kwargs['batch_size']}")
                
                # Reduce number of processes
                if 'num_proc' in kwargs:
                    kwargs['num_proc'] = max(MIN_NUM_PROC, kwargs['num_proc'] // 2)
                    logger.info(f"Reduced processes to {kwargs['num_proc']}")
                
                # Add retry delay with exponential backoff
                delay = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_FACTOR ** attempt)
                logger.info(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            
            # Try download
            result = download_func(*args, **kwargs)
            
            if result:
                logger.info(f"✅ Download successful on attempt {attempt + 1}")
                return result
            else:
                raise Exception("Download returned False")
                
        except Exception as e:
            last_exception = e
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            
            # Try alternative download methods on later attempts
            if attempt == 2:
                logger.info("Trying alternative download method...")
                kwargs['use_streaming'] = True
            elif attempt == 3:
                logger.info("Trying with minimal configuration...")
                kwargs['num_proc'] = 1
                kwargs['batch_size'] = MIN_BATCH_SIZE
    
    logger.error(f"All {MAX_RETRIES} attempts failed. Last error: {last_exception}")
    return False

def download_dataset(dataset_name, output_name, base_dir, config=None, max_samples=None):
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading {dataset_name} dataset")
        print(f"Downloading {dataset_name} dataset", flush=True)
        logger.info(f"{'='*60}")
        
        output_dir = os.path.join(base_dir, output_name)
        backup_dir = None
        
        # Multiple checks if dataset already exists
        if os.path.exists(output_dir):
            logger.info(f"Dataset {output_name} found in {output_dir}")
            
            # Verify integrity before skipping
            if verify_dataset_integrity(output_dir):
                logger.info(f"Dataset {output_name} verified and complete, skipping download...")
                return True
            else:
                logger.warning(f"Dataset {output_name} exists but failed integrity check")
                
                # Create backup before re-downloading
                backup_dir = create_dataset_backup(output_dir)
                
                # Remove corrupted dataset
                logger.info(f"Removing corrupted dataset and re-downloading...")
                shutil.rmtree(output_dir)
        
        # Create cache directory for partial downloads
        cache_dir = Path(base_dir) / ".cache" / output_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track download progress
        progress_file = cache_dir / "progress.json"
        progress = {}
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                logger.info(f"Resuming from previous progress: {progress}")
            except:
                progress = {}
        
        # Special handling for large datasets like OpenOrca
        if dataset_name == "Open-Orca/OpenOrca":
            logger.info(f"⚡ Using optimized settings for large dataset (OpenOrca)")
            logger.info(f"   • Primary batch size: {BATCH_SIZE}")
            logger.info(f"   • Fallback batch size: {ALT_BATCH_SIZE}")
            logger.info(f"   • Emergency batch size: {MIN_BATCH_SIZE}")
            logger.info(f"   • Primary processes: {NUM_PROC}")
            logger.info(f"   • Fallback processes: {FALLBACK_NUM_PROC}")
            if max_samples:
                logger.info(f"   • Max samples: {max_samples}")
            
            # Multiple attempts to load dataset with different strategies
            dataset = None
            load_strategies = [
                ("direct_with_limit", lambda: load_dataset(dataset_name, split=f'train[:{max_samples}]') if max_samples and max_samples < 100000 else None),
                ("streaming", lambda: load_dataset(dataset_name, streaming=True)),
                ("standard", lambda: load_dataset(dataset_name)),
                ("with_retry", lambda: load_dataset(dataset_name, download_mode="force_redownload")),
            ]
            
            for strategy_name, strategy_func in load_strategies:
                try:
                    logger.info(f"Trying load strategy: {strategy_name}")
                    result = strategy_func()
                    if result is not None:
                        if strategy_name == "direct_with_limit":
                            dataset = {'train': result}
                        else:
                            dataset = result
                        logger.info(f"✅ Successfully loaded with {strategy_name} strategy")
                        break
                except Exception as e:
                    logger.warning(f"Strategy {strategy_name} failed: {e}")
                    continue
            
            if dataset is None:
                raise Exception("All loading strategies failed")
            
            # Create output directory with multiple attempts
            for attempt in range(3):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    logger.warning(f"Failed to create directory, attempt {attempt + 1}: {e}")
                    time.sleep(1)
            
            # Process in batches for faster I/O with checkpointing
            for split_name, split_data in dataset.items():
                # Check if split was already processed
                if split_name in progress.get('completed_splits', []):
                    logger.info(f"  Split {split_name} already processed, skipping...")
                    continue
                
                output_file = f"{output_dir}/{split_name}.parquet"  # Use parquet for faster I/O
                logger.info(f"  Processing {split_name} split...")
                
                # Resume from last batch if available
                batch_data = []
                batch_num = progress.get(f'{split_name}_last_batch', 0)
                start_index = progress.get(f'{split_name}_last_index', 0)
                
                # Try different batch sizes if needed
                current_batch_size = BATCH_SIZE
                
                with tqdm(desc=f"Processing {split_name}", unit=" examples", initial=start_index) as pbar:
                    for i, item in enumerate(split_data):
                        # Skip to resume point
                        if i < start_index:
                            continue
                        
                        # Stop if we've reached max_samples
                        if max_samples and i >= max_samples:
                            break
                        
                        # Multiple attempts to append item
                        for attempt in range(3):
                            try:
                                batch_data.append(item)
                                pbar.update(1)
                                break
                            except Exception as e:
                                if attempt == 2:
                                    logger.error(f"Failed to process item {i}: {e}")
                                    continue
                                time.sleep(0.1)
                        
                        # Save batch when it reaches current_batch_size
                        if len(batch_data) >= current_batch_size:
                            # Multiple attempts to save batch
                            batch_saved = False
                            for save_attempt in range(3):
                                try:
                                    batch_file = f"{output_dir}/{split_name}_batch_{batch_num}.parquet"
                                    pa_table = pa.Table.from_pylist(batch_data)
                                    pq.write_table(pa_table, batch_file, compression='snappy')
                                    
                                    # Verify written file
                                    if Path(batch_file).exists() and Path(batch_file).stat().st_size > 0:
                                        batch_saved = True
                                        break
                                except Exception as e:
                                    logger.warning(f"Save attempt {save_attempt + 1} failed: {e}")
                                    if save_attempt < 2:
                                        time.sleep(1)
                                        # Reduce batch size for next attempt
                                        if len(batch_data) > MIN_BATCH_SIZE:
                                            current_batch_size = max(MIN_BATCH_SIZE, current_batch_size // 2)
                                            logger.info(f"Reducing batch size to {current_batch_size}")
                            
                            if batch_saved:
                                batch_data = []
                                batch_num += 1
                                
                                # Update progress
                                progress[f'{split_name}_last_batch'] = batch_num
                                progress[f'{split_name}_last_index'] = i + 1
                                with open(progress_file, 'w') as f:
                                    json.dump(progress, f)
                                
                                if batch_num % 10 == 0:
                                    processed = batch_num * current_batch_size
                                    logger.info(f"    Saved {processed} examples...")
                                    # Output progress for the progress bar
                                    if total_examples > 0:
                                        percent = (processed / total_examples) * 100
                                        print(f"Progress: {percent:.1f}% - {processed}/{total_examples} samples", flush=True)
                            else:
                                logger.error(f"Failed to save batch {batch_num} after multiple attempts")
                    
                    # Save remaining data with multiple attempts
                    if batch_data:
                        for attempt in range(3):
                            try:
                                batch_file = f"{output_dir}/{split_name}_batch_{batch_num}.parquet"
                                pa_table = pa.Table.from_pylist(batch_data)
                                pq.write_table(pa_table, batch_file, compression='snappy')
                                
                                # Verify file
                                if Path(batch_file).exists():
                                    break
                            except Exception as e:
                                if attempt == 2:
                                    logger.error(f"Failed to save final batch: {e}")
                                time.sleep(1)
                
                # Mark split as completed
                if 'completed_splits' not in progress:
                    progress['completed_splits'] = []
                progress['completed_splits'].append(split_name)
                with open(progress_file, 'w') as f:
                    json.dump(progress, f)
                
                logger.info(f"  ✅ Saved {split_name} split in {batch_num + 1} batches")
        
        else:
            # Normal processing for smaller datasets with multiple fallback attempts
            dataset = None
            
            # Try different loading configurations
            load_configs = [
                {"num_proc": NUM_PROC},
                {"num_proc": FALLBACK_NUM_PROC},
                {"num_proc": MIN_NUM_PROC},
                {}  # No parallel processing
            ]
            
            for idx, load_config in enumerate(load_configs):
                try:
                    logger.info(f"Loading attempt {idx + 1} with config: {load_config}")
                    if config:
                        dataset = load_dataset(dataset_name, config, **load_config)
                    else:
                        dataset = load_dataset(dataset_name, **load_config)
                    
                    if dataset:
                        logger.info(f"✅ Successfully loaded dataset")
                        break
                except Exception as e:
                    logger.warning(f"Load attempt {idx + 1} failed: {e}")
                    if idx < len(load_configs) - 1:
                        time.sleep(2 ** idx)  # Exponential backoff
                    else:
                        raise
            
            # Create output directory with retries
            for attempt in range(3):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    time.sleep(1)
            
            # Save dataset to disk with optimization and fallback strategies
            logger.info(f"Saving {dataset_name} to {output_dir}...")
            
            # Use save_to_disk for faster saving with multiple processes
            for split_name, split_data in dataset.items():
                split_dir = f"{output_dir}/{split_name}"
                logger.info(f"  Saving {split_name} split ({len(split_data)} examples)...")
                
                # Multiple save attempts with different configurations
                save_configs = [
                    {"num_proc": NUM_PROC, "max_shard_size": "500MB"},
                    {"num_proc": FALLBACK_NUM_PROC, "max_shard_size": "250MB"},
                    {"num_proc": MIN_NUM_PROC, "max_shard_size": "100MB"},
                    {"max_shard_size": "50MB"}  # Minimal config
                ]
                
                saved = False
                for idx, save_config in enumerate(save_configs):
                    try:
                        logger.info(f"  Save attempt {idx + 1} with config: {save_config}")
                        split_data.save_to_disk(split_dir, **save_config)
                        
                        # Verify saved data
                        if Path(split_dir).exists() and any(Path(split_dir).iterdir()):
                            saved = True
                            logger.info(f"  ✅ Successfully saved {split_name} split")
                            break
                    except Exception as e:
                        logger.warning(f"  Save attempt {idx + 1} failed: {e}")
                        if Path(split_dir).exists():
                            shutil.rmtree(split_dir)  # Clean up partial save
                        if idx < len(save_configs) - 1:
                            time.sleep(2 ** idx)
                
                if not saved:
                    raise Exception(f"Failed to save {split_name} split after all attempts")
        
        # Save dataset info with redundancy
        info_file = f"{output_dir}/dataset_info.json"
        info = {
            "dataset_name": dataset_name,
            "output_name": output_name,
            "config": config,
            "splits": list(dataset.keys()),
            "num_examples": {split: len(dataset[split]) for split in dataset.keys()},
            "download_timestamp": time.time(),
            "download_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save with multiple attempts
        for attempt in range(3):
            try:
                with open(info_file, 'w') as f:
                    json.dump(info, f, indent=2)
                
                # Also save backup copy
                backup_info_file = f"{output_dir}/.dataset_info.backup.json"
                with open(backup_info_file, 'w') as f:
                    json.dump(info, f, indent=2)
                
                break
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed to save dataset info: {e}")
                time.sleep(1)
        
        # Create checksums for verification
        save_dataset_checksums(output_dir)
        
        # Final integrity check
        if verify_dataset_integrity(output_dir, expected_splits=list(dataset.keys())):
            logger.info(f"✅ Successfully downloaded and verified {dataset_name}")
            print(f"Complete: 100% - {dataset_name} dataset saved", flush=True)
            
            # Clean up backup if successful
            if backup_dir and os.path.exists(backup_dir):
                logger.info(f"Removing backup: {backup_dir}")
                shutil.rmtree(backup_dir)
            
            # Clean up cache
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            
            return True
        else:
            logger.error(f"Dataset downloaded but failed final verification")
            
            # Restore from backup if available
            if backup_dir and os.path.exists(backup_dir):
                if restore_from_backup(backup_dir, output_dir):
                    logger.info("Restored previous version from backup")
                    return True
            
            return False
        
    except Exception as e:
        logger.error(f"❌ Error downloading {dataset_name}: {str(e)}")
        
        # Try to restore from backup if available
        if 'backup_dir' in locals() and backup_dir and os.path.exists(backup_dir):
            if restore_from_backup(backup_dir, output_dir):
                logger.info("Restored previous version from backup after error")
                return True
        
        # Save error information for debugging
        error_file = Path(base_dir) / ".errors" / f"{output_name}_error.json"
        error_file.parent.mkdir(exist_ok=True)
        error_info = {
            "dataset_name": dataset_name,
            "error": str(e),
            "timestamp": time.time(),
            "traceback": __import__('traceback').format_exc()
        }
        try:
            with open(error_file, 'w') as f:
                json.dump(error_info, f, indent=2)
        except:
            pass
        
        return False

def convert_to_unified_format(base_dir):
    """Convert all downloaded datasets to a unified format for processing"""
    print("\n" + "="*60)
    print("CONVERTING TO UNIFIED FORMAT")
    print("="*60)
    
    # Make converted directory relative to raw directory
    raw_dir = Path(base_dir)
    converted_dir = raw_dir.parent / "converted"
    os.makedirs(converted_dir, exist_ok=True)
    
    all_texts = []
    
    for dataset_dir in Path(base_dir).iterdir():
        if not dataset_dir.is_dir():
            continue
            
        print(f"Converting {dataset_dir.name}...")
        dataset_texts = []
        
        for jsonl_file in dataset_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        
                        # Extract text from various formats
                        text = None
                        if 'text' in data:
                            text = data['text']
                        elif 'content' in data:
                            text = data['content']
                        elif 'prompt' in data and 'response' in data:
                            text = f"Human: {data['prompt']}\n\nAssistant: {data['response']}"
                        elif 'instruction' in data and 'output' in data:
                            input_text = data.get('input', '')
                            if input_text:
                                text = f"Instruction: {data['instruction']}\nInput: {input_text}\nOutput: {data['output']}"
                            else:
                                text = f"Instruction: {data['instruction']}\nOutput: {data['output']}"
                        elif 'question' in data and 'answer' in data:
                            text = f"Q: {data['question']}\nA: {data['answer']}"
                        elif 'messages' in data and isinstance(data['messages'], list):
                            parts = []
                            for msg in data['messages']:
                                if isinstance(msg, dict):
                                    role = msg.get('role', '')
                                    content = msg.get('content', '')
                                    if role and content:
                                        parts.append(f"{role}: {content}")
                                    elif content:
                                        parts.append(content)
                            text = "\n\n".join(parts)
                        
                        if text and len(text.strip()) > 10:
                            dataset_texts.append({"text": text.strip()})
                            all_texts.append({"text": text.strip()})
                    except Exception as e:
                        continue
        
        # Save individual dataset
        if dataset_texts:
            dataset_file = Path(converted_dir) / f"{dataset_dir.name}.json"
            with open(dataset_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_texts, f, ensure_ascii=False)
            print(f"  Saved {len(dataset_texts)} examples")
    
    # Save combined dataset
    combined_file = Path(converted_dir) / "combined_dataset.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_texts, f, ensure_ascii=False)
    
    print(f"\n✅ Total: {len(all_texts)} examples saved to {combined_file}")
    return combined_file

def process_with_prepare_data(input_file, base_dir):
    """Process the unified dataset using prepare_data.py"""
    print("\n" + "="*60)
    print("PROCESSING DATA")
    print("="*60)
    
    import subprocess
    
    # Make processed directory relative to raw directory
    raw_dir = Path(base_dir)
    output_dir = str(raw_dir.parent / "processed")
    
    cmd = [
        "python3",
        str(get_scripts_dir("data_prep/prepare_data.py")),
        "--input-path", str(input_file),
        "--output-dir", output_dir,
        "--input-format", "json",
        "--max-length", "512",
        "--tokenizer", "gpt2"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Successfully processed data")
            print(f"Output saved to: {output_dir}")
        else:
            print(f"❌ Error processing data:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Failed to process: {e}")

def parallel_download_datasets(datasets_info: List[Tuple], base_dir: str, max_samples: Optional[int] = None, max_workers: int = 3) -> Tuple[List[str], List[str]]:
    """Download multiple datasets in parallel with thread pool"""
    successful = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_dataset = {}
        for dataset_info in datasets_info:
            dataset_name, output_name = dataset_info[:2]
            config = dataset_info[2] if len(dataset_info) > 2 else None
            
            future = executor.submit(
                download_with_retry,
                download_dataset,
                dataset_name,
                output_name,
                base_dir,
                config,
                max_samples
            )
            future_to_dataset[future] = dataset_name
        
        # Process completed downloads
        for future in as_completed(future_to_dataset):
            dataset_name = future_to_dataset[future]
            try:
                result = future.result()
                if result:
                    successful.append(dataset_name)
                    logger.info(f"✅ Completed: {dataset_name}")
                else:
                    failed.append(dataset_name)
                    logger.error(f"❌ Failed: {dataset_name}")
            except Exception as e:
                failed.append(dataset_name)
                logger.error(f"❌ Exception for {dataset_name}: {e}")
    
    return successful, failed

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process datasets with redundancy")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to download (for testing)")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Download only a specific dataset")
    parser.add_argument("--skip-large", action="store_true",
                       help="Skip large datasets like OpenOrca")
    parser.add_argument("--test-mode", action="store_true",
                       help="Test mode - download only 10k samples")
    parser.add_argument("--parallel", action="store_true",
                       help="Download datasets in parallel")
    parser.add_argument("--max-workers", type=int, default=3,
                       help="Maximum number of parallel downloads")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing datasets without downloading")
    parser.add_argument("--resume", action="store_true",
                       help="Resume interrupted downloads from cache")
    
    args = parser.parse_args()
    
    # Setup logging to file in outputs/logs directory
    log_dir = Path("/project/code/outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"download_datasets_{time.strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"Logging to: {log_file}")
    
    # Modify settings for test mode
    if args.test_mode:
        args.max_samples = 10000
        logger.info("🧪 TEST MODE: Downloading only 10,000 samples per dataset")
    
    base_dir = str(get_data_dir("pretraining/raw"))
    logger.info("="*60)
    logger.info("DATASET DOWNLOAD MANAGER WITH REDUNDANCY")
    logger.info("="*60)
    logger.info(f"Starting dataset downloads with enhanced redundancy...")
    logger.info(f"Will download {len(datasets_to_download)} datasets to {base_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Directory structure will be:")
    logger.info(f"  {base_dir}/ ← Downloaded raw datasets")
    logger.info(f"  {Path(base_dir).parent}/converted/ ← Unified format")
    logger.info(f"  {Path(base_dir).parent}/processed/ ← Tokenized data")
    logger.info(f"  {base_dir}/.cache/ ← Download cache for resume")
    logger.info(f"  {base_dir}/.errors/ ← Error logs")
    
    # Create all necessary directories with retries
    directories_to_create = [
        base_dir,
        Path(base_dir) / ".cache",
        Path(base_dir) / ".errors",
        Path(base_dir).parent / "converted",
        Path(base_dir).parent / "processed"
    ]
    
    for directory in directories_to_create:
        for attempt in range(3):
            try:
                os.makedirs(directory, exist_ok=True)
                break
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Failed to create directory {directory}: {e}")
                    raise
                time.sleep(1)
    
    # Install required libraries if not available with multiple attempts
    required_packages = ['datasets', 'pyarrow', 'tqdm', 'requests']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            logger.info(f"Installing {package} library...")
            for attempt in range(3):
                try:
                    result = subprocess.run(
                        ["pip", "install", package],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if result.returncode == 0:
                        __import__(package)
                        break
                except Exception as e:
                    if attempt == 2:
                        logger.error(f"Failed to install {package}: {e}")
                        raise
                    time.sleep(2)
    
    # Verify-only mode
    if args.verify_only:
        logger.info("Running in verify-only mode...")
        for dataset_info in datasets_to_download:
            dataset_name, output_name = dataset_info[:2]
            output_dir = os.path.join(base_dir, output_name)
            if os.path.exists(output_dir):
                if verify_dataset_integrity(output_dir):
                    logger.info(f"✅ {output_name}: Verified")
                else:
                    logger.error(f"❌ {output_name}: Failed verification")
            else:
                logger.info(f"⚠️ {output_name}: Not found")
        return
    
    # Filter datasets based on arguments
    datasets = datasets_to_download
    if args.dataset:
        datasets = [d for d in datasets_to_download if args.dataset in d[0]]
        logger.info(f"Filtered to specific dataset: {args.dataset}")
    if args.skip_large:
        datasets = [d for d in datasets if "OpenOrca" not in d[0] and "wikipedia" not in d[0]]
        logger.info("Skipping large datasets")
    
    # Download datasets
    if args.parallel and len(datasets) > 1:
        logger.info(f"Starting parallel downloads with {args.max_workers} workers...")
        successful, failed = parallel_download_datasets(
            datasets, base_dir, args.max_samples, args.max_workers
        )
    else:
        successful = []
        failed = []
        
        for dataset_info in datasets:
            dataset_name, output_name = dataset_info[:2]
            config = dataset_info[2] if len(dataset_info) > 2 else None
            
            # Use retry wrapper for sequential downloads too
            if download_with_retry(download_dataset, dataset_name, output_name, base_dir, config, args.max_samples):
                successful.append(dataset_name)
            else:
                failed.append(dataset_name)
    
    # Generate detailed summary report
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    logger.info(f"✅ Successfully downloaded: {len(successful)} datasets")
    for name in successful:
        logger.info(f"  - {name}")
    
    if failed:
        logger.info(f"\n❌ Failed to download: {len(failed)} datasets")
        for name in failed:
            logger.info(f"  - {name}")
            # Check if error log exists
            error_file = Path(base_dir) / ".errors" / f"{name.split('/')[-1]}_error.json"
            if error_file.exists():
                logger.info(f"    Error details: {error_file}")
    
    logger.info(f"\nAll datasets saved to {base_dir}")
    logger.info(f"Log file: {log_file}")
    
    # Save summary report
    summary_file = Path(base_dir) / "download_summary.json"
    summary = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "successful": successful,
        "failed": failed,
        "total_attempted": len(datasets),
        "success_rate": len(successful) / len(datasets) if datasets else 0,
        "arguments": vars(args)
    }
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to: {summary_file}")
    except Exception as e:
        logger.warning(f"Could not save summary: {e}")
    
    # Convert to unified format with redundancy
    if successful:
        try:
            logger.info("Starting unified format conversion...")
            unified_file = convert_to_unified_format(base_dir)
            
            # Verify unified file
            if unified_file and Path(unified_file).exists():
                file_size = Path(unified_file).stat().st_size
                logger.info(f"Unified file created: {unified_file} ({file_size:,} bytes)")
                
                # Process with prepare_data.py
                process_with_prepare_data(unified_file, base_dir)
            else:
                logger.error("Failed to create unified file")
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            logger.info("Datasets downloaded but post-processing failed")
    
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD PROCESS COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()