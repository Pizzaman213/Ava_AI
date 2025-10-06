"""
Enhanced Multi-Column Data Loading for Ava MoE++ Training

This module provides advanced data loading capabilities for multi-column datasets,
including support for HuggingFace datasets, multi-modal data, and flexible column mappings.

Supports datasets like:
- HuggingFaceM4/FineVision
- Instruction-tuning datasets with separate instruction/response columns
- Multi-modal datasets with text and image columns
- Structured datasets with multiple feature columns
"""

import os
import json
import math
import torch  # type: ignore[import]
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler, Sampler  # type: ignore[import]
from pathlib import Path
from typing import Optional, Iterator, Dict, List, Tuple, Any, Union, Callable
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import random
from itertools import cycle, islice
import warnings
import hashlib
import pickle
from functools import lru_cache
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import HuggingFace datasets
try:
    from datasets import load_dataset, IterableDataset as HFIterableDataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    warnings.warn("HuggingFace datasets not installed. Install with: pip install datasets")

# Try to import PIL for image handling
try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("PIL not installed. Image columns will not be supported. Install with: pip install pillow")

# Try to import h5py for HDF5 support
try:
    import h5py  # type: ignore[import]
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    logger.debug("h5py not installed. HDF5 support disabled. Install with: pip install h5py")

# Try to import tensorflow for TFRecord support
try:
    import tensorflow as tf  # type: ignore[import]
    TFRECORD_AVAILABLE = True
except ImportError:
    TFRECORD_AVAILABLE = False
    tf = None  # type: ignore[assignment]
    logger.debug("TensorFlow not installed. TFRecord support disabled. Install with: pip install tensorflow")

# Try to import distributed training support
try:
    import torch.distributed as dist  # type: ignore[import]
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    dist = None  # type: ignore[assignment]


class ColumnType(Enum):
    """Supported column data types"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    EMBEDDING = "embedding"
    JSON = "json"
    BINARY = "binary"
    TENSOR = "tensor"


@dataclass
class ColumnConfig:
    """Configuration for a single column"""
    name: str
    type: ColumnType
    role: str = "input"  # "input", "target", "auxiliary", "weight"
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    max_length: Optional[int] = None
    normalize: bool = False
    vocab: Optional[List[str]] = None  # For categorical columns
    default_value: Any = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)  # Validation constraints
    augmentation: Optional[Dict[str, Any]] = field(default_factory=dict)  # Augmentation settings
    cache_processed: bool = False  # Whether to cache processed values
    required: bool = True  # Whether column is required


@dataclass
class DatasetConfig:
    """Configuration for multi-column dataset"""
    columns: List[ColumnConfig]
    combine_strategy: str = "concatenate"  # "concatenate", "separate", "template", "custom"
    template: Optional[str] = None  # Template for combining columns
    shuffle_buffer_size: int = 10000
    max_samples: Optional[int] = None
    hf_dataset_name: Optional[str] = None  # HuggingFace dataset name
    hf_dataset_config: Optional[str] = None  # HuggingFace dataset configuration
    hf_split: Optional[str] = None  # HuggingFace dataset split
    cache_dir: Optional[str] = None  # Directory for caching processed samples
    validation_enabled: bool = True  # Whether to validate data
    augmentation_probability: float = 0.0  # Probability of applying augmentation
    custom_preprocessor: Optional[Callable] = None  # Custom preprocessing function
    distributed_rank: Optional[int] = None  # For distributed training
    distributed_world_size: Optional[int] = None  # For distributed training


class MultiColumnDataset(Dataset):
    """
    Dataset that handles multiple columns with different data types.

    Supports loading from:
    - Local files (Arrow, Parquet, CSV, JSON)
    - HuggingFace datasets
    - Custom data sources
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer=None,
        data_dir: Optional[str] = None,
        split: str = "train",
        streaming: bool = False
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.data_dir = Path(data_dir) if data_dir else None
        self.split = split
        self.streaming = streaming

        # Column name to config mapping
        self.column_map = {col.name: col for col in config.columns}

        # Validate configuration
        if config.validation_enabled:
            self._validate_config()

        # Initialize cache if enabled
        self.cache = {}
        if config.cache_dir:
            self.cache_dir = Path(config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        # Load data
        self.data = self._load_data()

        if not streaming and len(self.data) == 0:  # type: ignore[arg-type]
            warnings.warn(f"No data loaded for {split} split, using synthetic data")
            self.data = self._generate_synthetic_data()

        logger.info(f" Loaded {len(self.data) if not streaming else 'streaming'} samples for {split} split")  # type: ignore[arg-type]

    def _validate_config(self):
        """Validate dataset configuration"""
        if not self.config.columns:
            raise ValueError("No columns configured in dataset config")

        # Check for duplicate column names
        col_names = [col.name for col in self.config.columns]
        if len(col_names) != len(set(col_names)):
            raise ValueError(f"Duplicate column names found: {col_names}")

        # Validate column configurations
        for col in self.config.columns:
            if col.type == ColumnType.CATEGORICAL and not col.vocab and not col.default_value:
                warnings.warn(f"Categorical column '{col.name}' has no vocab defined")

            if col.role not in ["input", "target", "auxiliary", "weight"]:
                raise ValueError(f"Invalid role '{col.role}' for column '{col.name}'")

            if col.type == ColumnType.IMAGE and not PIL_AVAILABLE:
                raise ValueError(f"PIL not available but image column '{col.name}' configured")

    def _validate_sample(self, sample: Dict):
        """Validate a processed sample"""
        if not sample:
            raise ValueError("Empty sample encountered")

        # Check for required fields
        if 'input_ids' not in sample and not any(k.startswith('input_') for k in sample):
            raise ValueError("No input fields in processed sample")

    def _get_cache_key(self, idx: Union[int, Dict]) -> str:
        """Generate cache key for a sample"""
        if isinstance(idx, int):
            return f"{self.split}_{idx}"
        else:
            # For dict samples, create hash of content
            sample_str = json.dumps(idx, sort_keys=True)
            return hashlib.md5(sample_str.encode()).hexdigest()  # type: ignore[possibly-unbound]

    def _load_data(self) -> Union[List[Dict], Any]:
        """Load data based on configuration"""

        # Try HuggingFace datasets first if configured
        if self.config.hf_dataset_name and HF_DATASETS_AVAILABLE:
            return self._load_hf_dataset()

        # Otherwise load from local files
        if self.data_dir:
            return self._load_local_files()

        return []

    def _load_hf_dataset(self):
        """Load HuggingFace dataset"""
        try:
            print(f" Loading HuggingFace dataset: {self.config.hf_dataset_name}")

            dataset_args = {
                "path": self.config.hf_dataset_name,
                "split": self.config.hf_split or self.split,
                "streaming": self.streaming
            }

            if self.config.hf_dataset_config:
                dataset_args["name"] = self.config.hf_dataset_config

            if HF_DATASETS_AVAILABLE:
                dataset = load_dataset(**dataset_args)  # type: ignore[possibly-unbound]
            else:
                return []

            # Verify required columns exist
            if not self.streaming:
                sample = dataset[0] if len(dataset) > 0 else {}  # type: ignore[index]
            else:
                sample = next(iter(dataset), {})  # type: ignore[arg-type]

            for col_config in self.config.columns:
                if col_config.name not in sample and not col_config.default_value:
                    warnings.warn(f"Column '{col_config.name}' not found in dataset")

            return dataset

        except Exception as e:
            print(f" Failed to load HuggingFace dataset: {e}")
            return []

    def _load_local_files(self) -> List[Dict]:
        """Load data from local files"""
        data = []

        # Find data files
        files = self._find_data_files()

        for file_path in files:
            if self.config.max_samples and len(data) >= self.config.max_samples:
                break

            file_data = self._read_file(file_path)
            data.extend(file_data)

        # Limit samples if specified
        if self.config.max_samples:
            data = data[:self.config.max_samples]

        return data

    def _find_data_files(self) -> List[Path]:
        """Find all relevant data files"""
        if not self.data_dir or not self.data_dir.exists():
            return []

        files = []
        # Prioritize JSONL over JSON, and data formats over metadata
        extensions = ['.arrow', '.parquet', '.jsonl', '.csv', '.json']

        # Add HDF5 and TFRecord if available
        if HDF5_AVAILABLE:
            extensions.extend(['.h5', '.hdf5'])
        if TFRECORD_AVAILABLE:
            extensions.extend(['.tfrecord', '.tfrecords'])

        # Metadata files to exclude
        exclude_files = {
            'processing_stats.json',
            'stats.json',
            'metadata.json',
            'config.json'
        }

        for ext in extensions:
            # Look for split-specific files first
            pattern = f"{self.split}_*{ext}"
            found = sorted(self.data_dir.glob(pattern))
            files.extend(found)

            # Also check in subdirectories
            found = sorted(self.data_dir.glob(f"**/{pattern}"))
            files.extend(found)

        # If no split-specific files found, look for general files
        if not files:
            for ext in extensions:
                pattern = f"*{ext}"
                found = sorted(self.data_dir.glob(pattern))
                # Filter out metadata files
                found = [f for f in found if f.name not in exclude_files]
                files.extend(found)

        # Remove duplicates and sort by preference (JSONL first, then others)
        unique_files = list(set(files))

        # Sort by extension preference: .jsonl, .arrow, .parquet, .csv, .json
        ext_priority = {'.jsonl': 0, '.arrow': 1, '.parquet': 2, '.csv': 3, '.json': 4}
        unique_files.sort(key=lambda f: (ext_priority.get(f.suffix, 999), f.name))

        return unique_files

    def _read_file(self, file_path: Path) -> List[Dict]:
        """Read data from a file"""
        data = []

        try:
            if file_path.suffix == '.arrow':
                with pa.memory_map(str(file_path), 'r') as source:
                    batch_reader = pa.ipc.open_file(source)
                    table = batch_reader.read_all()
                    df = table.to_pandas()
                    data = df.to_dict('records')

            elif file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
                data = df.to_dict('records')

            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
                data = df.to_dict('records')

            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        data = [data]

            elif file_path.suffix == '.jsonl':
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))

            elif file_path.suffix in ['.h5', '.hdf5'] and HDF5_AVAILABLE:
                with h5py.File(file_path, 'r') as f:  # type: ignore[possibly-unbound]
                    # Assume data is stored in a group named 'data' or at root
                    if 'data' in f:
                        group = f['data']
                    else:
                        group = f

                    # Convert HDF5 datasets to dict records
                    sample_dict = {}
                    for key in group.keys():
                        sample_dict[key] = group[key][:]

                    # Transpose to list of records
                    n_samples = len(sample_dict[list(sample_dict.keys())[0]])
                    for i in range(n_samples):
                        record = {k: v[i] for k, v in sample_dict.items()}
                        data.append(record)

            elif file_path.suffix in ['.tfrecord', '.tfrecords'] and TFRECORD_AVAILABLE:
                # Parse TFRecord files
                if tf is not None:
                    raw_dataset = tf.data.TFRecordDataset(str(file_path))  # type: ignore[attr-defined]
                    for raw_record in raw_dataset:
                        # Parse the record (assuming standard TF Example format)
                        example = tf.train.Example()  # type: ignore[attr-defined]
                        example.ParseFromString(raw_record.numpy())

                        record = {}
                        for key, feature in example.features.feature.items():
                            if feature.HasField('bytes_list'):
                                record[key] = feature.bytes_list.value[0].decode('utf-8') if feature.bytes_list.value else ""
                            elif feature.HasField('float_list'):
                                record[key] = feature.float_list.value[0] if feature.float_list.value else 0.0
                            elif feature.HasField('int64_list'):
                                record[key] = feature.int64_list.value[0] if feature.int64_list.value else 0

                        if record:
                            data.append(record)

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise ValueError(f"Failed to read file {file_path}: {str(e)}")

        return data

    def _generate_synthetic_data(self) -> List[Dict]:
        """Generate synthetic data for testing"""
        synthetic_data = []

        for i in range(100):
            sample = {}

            for col_config in self.config.columns:
                if col_config.type == ColumnType.TEXT:
                    sample[col_config.name] = f"Sample {col_config.name} text {i}"
                elif col_config.type == ColumnType.NUMERIC:
                    sample[col_config.name] = random.random()
                elif col_config.type == ColumnType.CATEGORICAL:
                    vocab = col_config.vocab or ['A', 'B', 'C']
                    sample[col_config.name] = random.choice(vocab)
                else:
                    sample[col_config.name] = col_config.default_value

            synthetic_data.append(sample)

        return synthetic_data

    def __len__(self) -> int:
        if self.streaming:
            # Streaming datasets have unknown length, return very large int instead of inf
            return 2**31 - 1
        return len(self.data)  # type: ignore[arg-type]

    def __getitem__(self, idx):
        if self.streaming:
            raise NotImplementedError("Use iter() for streaming datasets")

        # Check cache if enabled
        if self.cache_dir:
            cache_key = self._get_cache_key(idx)
            cached_path = self.cache_dir / f"{cache_key}.pkl"

            if cached_path.exists():
                try:
                    with open(cached_path, 'rb') as f:
                        return pickle.load(f)
                except Exception:
                    pass  # If cache load fails, reprocess

        sample = self.data[idx]  # type: ignore[index]
        processed = self._process_sample(sample)

        # Validate processed sample
        if self.config.validation_enabled:
            self._validate_sample(processed)

        # Cache if enabled
        if self.cache_dir:
            cache_key = self._get_cache_key(idx)
            cached_path = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cached_path, 'wb') as f:
                    pickle.dump(processed, f)
            except Exception as e:
                logger.debug(f"Failed to cache sample {idx}: {e}")

        return processed

    def _process_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """Process a single sample according to column configurations"""
        processed = {}

        # Apply custom preprocessor if provided
        if self.config.custom_preprocessor:
            sample = self.config.custom_preprocessor(sample)

        # Process each configured column
        for col_config in self.config.columns:
            col_name = col_config.name

            # Get column value with default fallback
            if col_name in sample:
                value = sample[col_name]
            elif col_config.required:
                if col_config.default_value is not None:
                    value = col_config.default_value
                else:
                    raise ValueError(f"Required column '{col_name}' not found in sample and no default value provided")
            else:
                value = col_config.default_value
                if value is None:
                    continue

            # Apply augmentation if enabled
            if (col_config.augmentation and
                self.split == "train" and
                random.random() < self.config.augmentation_probability):
                value = self._apply_augmentation(value, col_config)

            # Process based on column type
            if col_config.type == ColumnType.TEXT:
                processed_value = self._process_text(value, col_config)
            elif col_config.type == ColumnType.IMAGE:
                processed_value = self._process_image(value, col_config)
            elif col_config.type == ColumnType.AUDIO:
                processed_value = self._process_audio(value, col_config)
            elif col_config.type == ColumnType.VIDEO:
                processed_value = self._process_video(value, col_config)
            elif col_config.type == ColumnType.NUMERIC:
                processed_value = self._process_numeric(value, col_config)
            elif col_config.type == ColumnType.CATEGORICAL:
                processed_value = self._process_categorical(value, col_config)
            elif col_config.type == ColumnType.EMBEDDING:
                processed_value = self._process_embedding(value, col_config)
            elif col_config.type == ColumnType.TENSOR:
                processed_value = self._process_tensor(value, col_config)
            else:
                processed_value = torch.tensor(value)

            # Apply validation rules if enabled
            if col_config.validation_rules and self.config.validation_enabled:
                self._validate_column_value(processed_value, col_config)

            # Store based on role
            if col_config.role == "input":
                processed[f"input_{col_name}"] = processed_value
            elif col_config.role == "target":
                processed[f"target_{col_name}"] = processed_value
            elif col_config.role == "weight":
                processed[f"weight_{col_name}"] = processed_value
            else:
                processed[f"aux_{col_name}"] = processed_value

        # Combine inputs based on strategy
        combined = self._combine_inputs(processed, sample)

        return combined

    def _process_text(self, text: str, config: ColumnConfig) -> torch.Tensor:
        """Process text column"""
        if self.tokenizer:
            max_length = config.max_length or 512

            encoded = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            return encoded['input_ids'].squeeze()
        else:
            # Return as is if no tokenizer
            return torch.tensor([ord(c) for c in text[:512]])  # Simple char encoding

    def _process_image(self, image_data: Any, config: ColumnConfig) -> torch.Tensor:
        """Process image column"""
        if not PIL_AVAILABLE:
            warnings.warn("PIL not available, returning dummy image tensor")
            return torch.zeros(3, 224, 224)

        try:
            if isinstance(image_data, str):
                # Assume it's a path
                if PIL_AVAILABLE:
                    image = Image.open(image_data)  # type: ignore[possibly-unbound]
                else:
                    return torch.zeros(3, 224, 224)
            elif isinstance(image_data, bytes):
                # Raw image bytes
                if PIL_AVAILABLE:
                    image = Image.open(io.BytesIO(image_data))  # type: ignore[possibly-unbound]
                else:
                    return torch.zeros(3, 224, 224)
            else:
                # Assume PIL Image
                image = image_data

            # Resize and normalize
            image = image.resize((224, 224))
            image_tensor = torch.tensor(np.array(image)).float()

            if len(image_tensor.shape) == 2:
                # Grayscale
                image_tensor = image_tensor.unsqueeze(0)
            else:
                # RGB
                image_tensor = image_tensor.permute(2, 0, 1)

            if config.normalize:
                image_tensor = image_tensor / 255.0

            return image_tensor

        except Exception as e:
            warnings.warn(f"Failed to process image: {e}")
            return torch.zeros(3, 224, 224)

    def _process_numeric(self, value: float, config: ColumnConfig) -> torch.Tensor:
        """Process numeric column"""
        tensor = torch.tensor(float(value))

        if config.normalize and 'mean' in config.preprocessing and 'std' in config.preprocessing:
            mean = config.preprocessing['mean']
            std = config.preprocessing['std']
            tensor = (tensor - mean) / std

        return tensor

    def _process_categorical(self, value: str, config: ColumnConfig) -> torch.Tensor:
        """Process categorical column"""
        if config.vocab:
            if value in config.vocab:
                idx = config.vocab.index(value)
            else:
                idx = len(config.vocab)  # Unknown category

            # One-hot encoding
            tensor = torch.zeros(len(config.vocab) + 1)
            tensor[idx] = 1.0
            return tensor
        else:
            # Simple hash encoding
            return torch.tensor(hash(str(value)) % 1000)

    def _process_audio(self, audio_data: Any, config: ColumnConfig) -> torch.Tensor:
        """Process audio column"""
        # Placeholder for audio processing
        warnings.warn("Audio processing not yet implemented, returning dummy tensor")
        return torch.zeros(1, 16000)  # 1 second at 16kHz

    def _process_video(self, video_data: Any, config: ColumnConfig) -> torch.Tensor:
        """Process video column"""
        # Placeholder for video processing
        warnings.warn("Video processing not yet implemented, returning dummy tensor")
        return torch.zeros(3, 16, 224, 224)  # 16 frames of 224x224 RGB

    def _process_embedding(self, embedding: Any, config: ColumnConfig) -> torch.Tensor:
        """Process embedding column"""
        if isinstance(embedding, torch.Tensor):
            return embedding
        elif isinstance(embedding, (list, np.ndarray)):
            return torch.tensor(embedding, dtype=torch.float32)
        else:
            raise ValueError(f"Invalid embedding type: {type(embedding)}")

    def _process_tensor(self, tensor_data: Any, config: ColumnConfig) -> torch.Tensor:
        """Process tensor column"""
        if isinstance(tensor_data, torch.Tensor):
            return tensor_data
        elif isinstance(tensor_data, np.ndarray):
            return torch.from_numpy(tensor_data)
        elif isinstance(tensor_data, list):
            return torch.tensor(tensor_data)
        else:
            raise ValueError(f"Cannot convert {type(tensor_data)} to tensor")

    def _apply_augmentation(self, value: Any, config: ColumnConfig) -> Any:
        """Apply data augmentation based on column type"""
        aug_config = config.augmentation or {}

        if config.type == ColumnType.TEXT and 'text_augmentation' in aug_config:
            # Simple text augmentation (e.g., synonym replacement, random insertion)
            if aug_config.get('text_augmentation') == 'random_mask':
                words = value.split()
                if len(words) > 1:
                    mask_idx = random.randint(0, len(words) - 1)
                    words[mask_idx] = '[MASK]'
                    value = ' '.join(words)

        elif config.type == ColumnType.IMAGE and PIL_AVAILABLE:
            # Image augmentation (rotation, flip, color jitter)
            if 'random_flip' in aug_config and random.random() < 0.5:
                # This would require PIL Image object
                pass

        elif config.type == ColumnType.NUMERIC:
            # Numeric augmentation (noise, scaling)
            if 'noise_std' in aug_config:
                noise = random.gauss(0, aug_config['noise_std'])
                value = float(value) + noise

        return value

    def _validate_column_value(self, value: torch.Tensor, config: ColumnConfig):
        """Validate column value against rules"""
        rules = config.validation_rules

        if 'min_value' in rules:
            if value.min() < rules['min_value']:
                raise ValueError(f"Column {config.name} has values below minimum {rules['min_value']}")

        if 'max_value' in rules:
            if value.max() > rules['max_value']:
                raise ValueError(f"Column {config.name} has values above maximum {rules['max_value']}")

        if 'non_empty' in rules and rules['non_empty']:
            if value.numel() == 0:
                raise ValueError(f"Column {config.name} is empty")

        if 'shape' in rules:
            expected_shape = rules['shape']
            if list(value.shape) != expected_shape:
                raise ValueError(f"Column {config.name} has shape {list(value.shape)}, expected {expected_shape}")

    def _combine_inputs(self, processed: Dict, original: Dict) -> Dict:
        """Combine processed inputs based on strategy"""
        result = {}

        if self.config.combine_strategy == "concatenate":
            # Concatenate all input columns
            input_tensors = []
            for key, value in processed.items():
                if key.startswith("input_"):
                    if value.dim() == 0:
                        value = value.unsqueeze(0)
                    if value.dim() == 1:
                        input_tensors.append(value)
                    else:
                        # Flatten multi-dimensional tensors
                        input_tensors.append(value.flatten())

            if input_tensors:
                result['input_ids'] = torch.cat(input_tensors, dim=0)
            else:
                result['input_ids'] = torch.zeros(1)

            # Handle attention mask
            result['attention_mask'] = torch.ones_like(result['input_ids'])

        elif self.config.combine_strategy == "separate":
            # Keep columns separate
            result.update(processed)

        elif self.config.combine_strategy == "template" and self.config.template:
            # Use template to combine columns
            template_text = self.config.template

            for col_config in self.config.columns:
                if col_config.name in original:
                    placeholder = f"{{{col_config.name}}}"
                    if placeholder in template_text:
                        template_text = template_text.replace(
                            placeholder,
                            str(original[col_config.name])
                        )

            # Tokenize the template result
            if self.tokenizer:
                encoded = self.tokenizer(
                    template_text,
                    max_length=512,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                result['input_ids'] = encoded['input_ids'].squeeze()
                result['attention_mask'] = encoded['attention_mask'].squeeze()
            else:
                result['input_ids'] = torch.tensor([ord(c) for c in template_text[:512]])
                result['attention_mask'] = torch.ones_like(result['input_ids'])

        # Add labels (typically same as input_ids for causal LM)
        if 'target_ids' in processed:
            result['labels'] = processed['target_ids']
        elif 'input_ids' in result:
            result['labels'] = result['input_ids'].clone()

        return result


class StreamingMultiColumnDataset(IterableDataset):
    """
    Streaming version of MultiColumnDataset for handling large datasets.
    """

    def __init__(
        self,
        config: DatasetConfig,
        tokenizer=None,
        data_dir: Optional[str] = None,
        split: str = "train",
        buffer_size: int = 1000
    ):
        self.base_dataset = MultiColumnDataset(
            config=config,
            tokenizer=tokenizer,
            data_dir=data_dir,
            split=split,
            streaming=True
        )
        self.buffer_size = buffer_size

    def __iter__(self):
        """Stream data samples"""
        # If HuggingFace streaming dataset
        if HF_DATASETS_AVAILABLE and hasattr(self.base_dataset.data, '__iter__'):
            for sample in self.base_dataset.data:
                yield self.base_dataset._process_sample(sample)

        # Otherwise stream from files
        elif self.base_dataset.data_dir:
            files = self.base_dataset._find_data_files()

            for file_path in files:
                for sample in self._stream_file(file_path):
                    yield self.base_dataset._process_sample(sample)

        # Synthetic data stream
        else:
            while True:
                synthetic = self.base_dataset._generate_synthetic_data()
                for sample in synthetic:
                    yield self.base_dataset._process_sample(sample)

    def _stream_file(self, file_path: Path) -> Iterator[Dict]:
        """Stream data from a single file"""

        try:
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)

            elif file_path.suffix in ['.arrow', '.parquet']:
                # Stream in batches
                if file_path.suffix == '.arrow':
                    with pa.memory_map(str(file_path), 'r') as source:
                        batch_reader = pa.ipc.open_file(source)
                        for i in range(batch_reader.num_record_batches):
                            batch = batch_reader.get_batch(i)
                            df = batch.to_pandas()
                            for _, row in df.iterrows():
                                yield row.to_dict()
                else:
                    parquet_file = pq.ParquetFile(file_path)
                    for batch in parquet_file.iter_batches(batch_size=self.buffer_size):
                        df = batch.to_pandas()
                        for _, row in df.iterrows():
                            yield row.to_dict()

        except Exception as e:
            print(f" Error streaming {file_path}: {e}")


class AdvancedDistributedSampler(Sampler):
    """
    Advanced distributed sampler with proper sharding, load balancing, and fault tolerance.

    Features:
    - Balanced data distribution across ranks
    - Dynamic resharding for failed ranks
    - Load balancing monitoring
    - Deterministic shuffling with proper epoch seeding
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        enable_load_balancing: bool = True,
        balancing_tolerance: float = 0.05  # 5% tolerance for load imbalance
    ):
        if num_replicas is None:
            if not DISTRIBUTED_AVAILABLE or dist is None or not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not DISTRIBUTED_AVAILABLE or dist is None or not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()  # type: ignore[assignment]
        if rank is not None and num_replicas is not None:
            if rank >= num_replicas or rank < 0:
                raise ValueError(
                    "Invalid rank {}, rank should be in the interval"
                    " [0, {}]".format(rank, num_replicas - 1))

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.enable_load_balancing = enable_load_balancing
        self.balancing_tolerance = balancing_tolerance

        # Calculate dataset size and samples per rank
        if self.num_replicas is not None:
            if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
                # Split to nearest available length that is evenly divisible
                self.num_samples = math.ceil(
                    (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
                )
            else:
                self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]

            self.total_size = self.num_samples * self.num_replicas
        else:
            self.num_samples = 0
            self.total_size = 0

        # Track load balancing statistics
        self.samples_processed = 0
        self.load_stats = {
            'samples_assigned': self.num_samples,
            'samples_processed': 0,
            'load_ratio': 0.0
        }

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # Subsample for this rank with proper sharding
        rank_indices = self._get_rank_indices(indices)

        # Update load statistics
        self.load_stats['samples_assigned'] = len(rank_indices)

        return iter(rank_indices)

    def _get_rank_indices(self, indices: List[int]) -> List[int]:
        """Get indices for this rank with advanced sharding."""
        if not self.enable_load_balancing:
            # Standard sharding
            return indices[self.rank:self.total_size:self.num_replicas]  # type: ignore[misc]

        # Advanced load-balanced sharding
        chunk_size = len(indices) // self.num_replicas  # type: ignore[operator]
        remainder = len(indices) % self.num_replicas  # type: ignore[operator]

        # Calculate start and end indices for this rank
        if self.rank < remainder:  # type: ignore[operator]
            # First `remainder` ranks get one extra sample
            start_idx = self.rank * (chunk_size + 1)  # type: ignore[operator]
            end_idx = start_idx + chunk_size + 1
        else:
            # Remaining ranks get standard chunk size
            start_idx = remainder * (chunk_size + 1) + (self.rank - remainder) * chunk_size  # type: ignore[operator]
            end_idx = start_idx + chunk_size

        rank_indices = indices[start_idx:end_idx]

        # Monitor load balance
        expected_samples = len(indices) / self.num_replicas  # type: ignore[operator]
        actual_samples = len(rank_indices)
        load_imbalance = abs(actual_samples - expected_samples) / expected_samples

        if load_imbalance > self.balancing_tolerance:
            logger.warning(
                f"Load imbalance detected on rank {self.rank}: "
                f"{actual_samples} samples vs {expected_samples:.1f} expected "
                f"(imbalance: {load_imbalance:.1%})"
            )

        self.load_stats['load_ratio'] = actual_samples / expected_samples

        return rank_indices

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for this sampler."""
        self.epoch = epoch

    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics for this rank."""
        return self.load_stats.copy()

    def coordinate_resharding(self, failed_ranks: List[int]) -> bool:
        """
        Coordinate resharding when ranks fail.

        Args:
            failed_ranks: List of failed rank IDs

        Returns:
            bool: True if resharding successful
        """
        if not failed_ranks:
            return True

        active_ranks = [r for r in range(self.num_replicas) if r not in failed_ranks]  # type: ignore[arg-type]

        if self.rank in failed_ranks:  # type: ignore[operator]
            logger.error(f"Rank {self.rank} is marked as failed - cannot reshard")
            return False

        if self.rank not in active_ranks:  # type: ignore[operator]
            logger.error(f"Rank {self.rank} not in active ranks: {active_ranks}")
            return False

        logger.info(f"Resharding data for {len(active_ranks)} active ranks (failed: {failed_ranks})")

        # Recalculate num_replicas and rank mapping for active ranks
        old_num_replicas = self.num_replicas
        old_rank = self.rank

        self.num_replicas = len(active_ranks)
        self.rank = active_ranks.index(old_rank)  # type: ignore[arg-type]

        # Recalculate samples per rank
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type,operator]
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[operator]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type,operator]

        self.total_size = self.num_samples * self.num_replicas  # type: ignore[operator]

        logger.info(
            f"Resharding complete: rank {old_rank}->{self.rank}, "
            f"replicas {old_num_replicas}->{self.num_replicas}, "
            f"samples: {self.num_samples}"
        )

        # Update load stats
        self.load_stats = {
            'samples_assigned': self.num_samples,
            'samples_processed': 0,
            'load_ratio': 0.0,
            'resharded': True,
            'failed_ranks': failed_ranks,
            'active_ranks': active_ranks
        }

        return True


def create_multi_column_dataloader(
    config: Union[DatasetConfig, Dict],
    tokenizer=None,
    batch_size: int = 32,
    data_dir: Optional[str] = None,
    split: str = "train",
    streaming: bool = False,
    num_workers: int = 0,
    distributed: Optional[bool] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    use_advanced_sampler: bool = True,
    enable_load_balancing: bool = True,
    balancing_tolerance: float = 0.05
) -> DataLoader:
    """
    Create a DataLoader for multi-column datasets with advanced features.

    This function creates a sophisticated DataLoader that can handle multiple data types
    including text, numeric, categorical, image, and tensor columns. It supports both
    file-based and HuggingFace Hub datasets, with options for streaming, distributed
    training, and advanced load balancing.

    Args:
        config: DatasetConfig or dict containing dataset configuration including:
            - columns: List of column configurations
            - combine_strategy: How to combine multiple columns
            - source_type: 'files' or 'huggingface'
            - dataset_name: HuggingFace dataset name (if using hub)
        tokenizer: Tokenizer for processing text columns (required for text data)
        batch_size: Number of samples per batch
        data_dir: Directory containing data files (for file-based datasets)
        split: Data split to load ('train', 'validation', 'test')
        streaming: Whether to use streaming mode for large datasets
        num_workers: Number of worker processes for data loading
        distributed: Whether distributed training is enabled (auto-detected if None)
        world_size: Total number of distributed processes
        rank: Current process rank in distributed setup
        use_advanced_sampler: Whether to use advanced sampling strategies
        enable_load_balancing: Whether to enable dynamic load balancing
        balancing_tolerance: Tolerance for load balancing (0.05 = 5%)

    Returns:
        DataLoader: Configured DataLoader instance ready for training

    Raises:
        ValueError: If configuration is invalid or required parameters are missing
        FileNotFoundError: If data files cannot be found
        ImportError: If required dependencies are missing

    Example:
        >>> config = {
        ...     'columns': [{'name': 'text', 'type': 'text'}],
        ...     'combine_strategy': 'concatenate',
        ...     'source_type': 'files'
        ... }
        >>> loader = create_multi_column_dataloader(
        ...     config=config,
        ...     tokenizer=tokenizer,
        ...     batch_size=32
        ... )
    """

    # Convert dict to DatasetConfig if needed
    if isinstance(config, dict):
        # Parse column configs
        columns = []
        for col_dict in config.get('columns', []):
            col_type = ColumnType(col_dict.get('type', 'text'))
            columns.append(ColumnConfig(
                name=col_dict['name'],
                type=col_type,
                role=col_dict.get('role', 'input'),
                preprocessing=col_dict.get('preprocessing', {}),
                max_length=col_dict.get('max_length'),
                normalize=col_dict.get('normalize', False),
                vocab=col_dict.get('vocab'),
                default_value=col_dict.get('default_value')
            ))

        config = DatasetConfig(
            columns=columns,
            combine_strategy=config.get('combine_strategy', 'concatenate'),
            template=config.get('template'),
            shuffle_buffer_size=config.get('shuffle_buffer_size', 10000),
            max_samples=config.get('max_samples'),
            hf_dataset_name=config.get('hf_dataset_name'),
            hf_dataset_config=config.get('hf_dataset_config'),
            hf_split=config.get('hf_split')
        )

    # Create dataset
    if streaming:
        dataset = StreamingMultiColumnDataset(
            config=config,
            tokenizer=tokenizer,
            data_dir=data_dir,
            split=split
        )
    else:
        dataset = MultiColumnDataset(
            config=config,
            tokenizer=tokenizer,
            data_dir=data_dir,
            split=split,
            streaming=False
        )

    # Auto-detect distributed training
    if distributed is None:
        distributed = DISTRIBUTED_AVAILABLE and (
            'WORLD_SIZE' in os.environ or
            (world_size is not None and world_size > 1)
        )

    # Set up distributed sampler if needed
    sampler = None
    shuffle = (not streaming and split == "train")

    if distributed and DISTRIBUTED_AVAILABLE and not streaming:
        if world_size is None:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        if rank is None:
            rank = int(os.environ.get('RANK', 0))

        if use_advanced_sampler:
            sampler = AdvancedDistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=True,  # Drop last for better load balancing
                enable_load_balancing=enable_load_balancing,
                balancing_tolerance=balancing_tolerance
            )
            print(f"🎯 Using AdvancedDistributedSampler for rank {rank}/{world_size}")
            print(f"   Load balancing: {'enabled' if enable_load_balancing else 'disabled'}")
            print(f"   Balancing tolerance: {balancing_tolerance:.1%}")
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle
            )
            print(f"📊 Using standard DistributedSampler for rank {rank}/{world_size}")

        shuffle = False  # Distributed sampler handles shuffling

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=distributed  # Drop last batch for distributed training
    )

    # Store reference to advanced sampler for monitoring
    if hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, AdvancedDistributedSampler):
        dataloader._advanced_sampler = dataloader.sampler  # type: ignore[attr-defined]

    return dataloader


def get_data_distribution_stats(dataloader: DataLoader) -> Optional[Dict[str, Any]]:
    """
    Get data distribution statistics from a DataLoader with AdvancedDistributedSampler.

    Args:
        dataloader: DataLoader instance

    Returns:
        Dictionary with distribution statistics or None if not available
    """
    if hasattr(dataloader, '_advanced_sampler'):
        sampler = dataloader._advanced_sampler  # type: ignore[attr-defined]
        return sampler.get_load_stats()
    elif hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, AdvancedDistributedSampler):
        return dataloader.sampler.get_load_stats()
    else:
        return None


def coordinate_data_resharding(dataloader: DataLoader, failed_ranks: List[int]) -> bool:
    """
    Coordinate data resharding when ranks fail.

    Args:
        dataloader: DataLoader instance
        failed_ranks: List of failed rank IDs

    Returns:
        bool: True if resharding successful
    """
    if hasattr(dataloader, '_advanced_sampler'):
        sampler = dataloader._advanced_sampler  # type: ignore[attr-defined]
        return sampler.coordinate_resharding(failed_ranks)
    elif hasattr(dataloader, 'sampler') and isinstance(dataloader.sampler, AdvancedDistributedSampler):
        return dataloader.sampler.coordinate_resharding(failed_ranks)
    else:
        logger.warning("DataLoader does not have AdvancedDistributedSampler - cannot reshard")
        return False


def set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:
    """
    Set epoch for distributed sampler to ensure proper shuffling.

    Args:
        dataloader: DataLoader instance
        epoch: Current epoch number
    """
    if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)  # type: ignore[attr-defined]
        if isinstance(dataloader.sampler, AdvancedDistributedSampler):
            logger.debug(f"Set epoch {epoch} for AdvancedDistributedSampler")
        else:
            logger.debug(f"Set epoch {epoch} for distributed sampler")