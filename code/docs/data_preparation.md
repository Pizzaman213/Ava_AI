# Data Preparation Guide

## Table of Contents
- [Overview](#overview)
- [Data Sources](#data-sources)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Quality Filtering](#quality-filtering)
- [Tokenization](#tokenization)
- [Memory-Mapped Datasets](#memory-mapped-datasets)
- [Data Mixing Strategies](#data-mixing-strategies)
- [Deduplication](#deduplication)
- [Best Practices](#best-practices)

## Overview

High-quality data is crucial for training effective MoE language models. This guide covers the complete data preparation pipeline from raw text to optimized training datasets.

### Data Pipeline Overview

```
Raw Data Sources
       ↓
Quality Filtering (BERT Scoring)
       ↓
Deduplication (MinHash LSH)
       ↓
Domain Classification
       ↓
Tokenization
       ↓
Memory-Mapped Dataset Creation
       ↓
Data Mixing & Sampling
       ↓
Training-Ready Dataset
```

## Data Sources

### Supported Formats

```python
from moe_llm.data import DataLoader

# 1. JSON Lines
loader = DataLoader.from_jsonl("data/train.jsonl")

# 2. Parquet files
loader = DataLoader.from_parquet("data/train.parquet")

# 3. CSV files
loader = DataLoader.from_csv("data/train.csv", text_column="content")

# 4. Text files
loader = DataLoader.from_text_files("data/texts/*.txt")

# 5. Hugging Face datasets
loader = DataLoader.from_huggingface("c4", split="train")

# 6. Custom format
def custom_parser(file_path):
    # Custom parsing logic
    return {"text": parsed_text}

loader = DataLoader.from_custom("data/custom/*", parser=custom_parser)
```

### Common Data Sources

```python
# Download and prepare common datasets
from moe_llm.data import download_dataset

# C4 (Colossal Clean Crawled Corpus)
download_dataset("c4", output_dir="data/c4", splits=["train", "validation"])

# The Pile
download_dataset("pile", output_dir="data/pile", subsets=["github", "arxiv"])

# RedPajama
download_dataset("redpajama", output_dir="data/redpajama")

# Custom web crawl
from moe_llm.data import WebCrawler

crawler = WebCrawler(
    domains=["example.com"],
    max_pages=10000,
    quality_threshold=0.7
)
crawler.crawl(output_dir="data/webcrawl")
```

## Data Processing Pipeline

### Complete Pipeline Example

```python
from moe_llm.data import DataPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    # Input/output
    input_path="data/raw",
    output_path="data/processed",
    
    # Processing steps
    steps=[
        "quality_filter",
        "deduplication", 
        "domain_classification",
        "tokenization",
        "mmap_creation"
    ],
    
    # Quality filtering
    min_quality_score=0.7,
    bert_model="bert-base-uncased",
    
    # Deduplication
    dedup_threshold=0.9,
    use_minhash=True,
    
    # Tokenization
    tokenizer="gpt2",
    max_length=2048,
    
    # Performance
    num_workers=32,
    batch_size=1000
)

# Create and run pipeline
pipeline = DataPipeline(config)
pipeline.run()
```

### Step-by-Step Processing

```python
# 1. Load raw data
from moe_llm.data import RawDataset

raw_data = RawDataset.from_files("data/raw/*.jsonl")
print(f"Loaded {len(raw_data)} documents")

# 2. Quality filtering
from moe_llm.data import QualityFilter

filter = QualityFilter(
    min_length=50,
    max_length=50000,
    min_quality_score=0.7,
    language="en"
)

filtered_data = filter.apply(raw_data)
print(f"Retained {len(filtered_data)} high-quality documents")

# 3. Deduplication
from moe_llm.data import Deduplicator

dedup = Deduplicator(
    method="minhash",
    threshold=0.9,
    num_perm=128
)

unique_data = dedup.apply(filtered_data)
print(f"Retained {len(unique_data)} unique documents")

# 4. Domain classification
from moe_llm.data import DomainClassifier

classifier = DomainClassifier(
    domains=["science", "code", "literature", "general"],
    model="bert-base-uncased"
)

classified_data = classifier.classify(unique_data)

# 5. Save processed data
classified_data.save("data/processed/clean_data.jsonl")
```

## Quality Filtering

### BERT-based Quality Scoring

```python
from moe_llm.data import BertScorer

# Initialize scorer
scorer = BertScorer(
    model_name="bert-base-uncased",
    device="cuda",
    batch_size=32
)

# Score individual text
text = "This is a high-quality document about machine learning..."
score = scorer.score(text)
print(f"Quality score: {score:.3f}")

# Score dataset
scores = scorer.score_dataset(texts)
high_quality = [text for text, score in zip(texts, scores) if score > 0.7]
```

### Custom Quality Metrics

```python
from moe_llm.data import QualityMetrics

# Define custom metrics
class CustomQualityScorer:
    def __init__(self):
        self.metrics = QualityMetrics()
    
    def score(self, text):
        scores = {
            "readability": self.metrics.flesch_reading_ease(text),
            "diversity": self.metrics.vocabulary_diversity(text),
            "coherence": self.metrics.sentence_coherence(text),
            "factuality": self.metrics.fact_density(text),
            "toxicity": 1 - self.metrics.toxicity_score(text)
        }
        
        # Weighted combination
        weights = {
            "readability": 0.2,
            "diversity": 0.2,
            "coherence": 0.3,
            "factuality": 0.2,
            "toxicity": 0.1
        }
        
        total_score = sum(scores[k] * weights[k] for k in scores)
        return total_score
```

### Language Detection and Filtering

```python
from moe_llm.data import LanguageFilter

# Create language filter
lang_filter = LanguageFilter(
    target_languages=["en"],
    min_confidence=0.95,
    model="xlm-roberta-base"
)

# Apply filter
english_data = lang_filter.filter(multilingual_data)

# Get language distribution
lang_dist = lang_filter.get_language_distribution(multilingual_data)
print("Language distribution:", lang_dist)
```

## Tokenization

### Basic Tokenization

```python
from moe_llm.data import Tokenizer

# Initialize tokenizer
tokenizer = Tokenizer.from_pretrained("gpt2")

# Tokenize text
text = "Hello, world!"
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Batch tokenization
texts = ["First text", "Second text", "Third text"]
batch_tokens = tokenizer.batch_encode(
    texts,
    max_length=512,
    padding=True,
    truncation=True
)
```

### Optimized Tokenization

```python
from moe_llm.data import FastTokenizer

# Create fast tokenizer with caching
tokenizer = FastTokenizer(
    model_name="gpt2",
    cache_size=100000,
    num_workers=8
)

# Parallel tokenization
from moe_llm.data import parallel_tokenize

tokenized_data = parallel_tokenize(
    texts=raw_texts,
    tokenizer=tokenizer,
    max_length=2048,
    num_workers=32,
    batch_size=1000,
    progress_bar=True
)

# Save tokenized data
tokenized_data.save("data/tokenized/train.pkl")
```

### Custom Tokenization

```python
from moe_llm.data import CustomTokenizer

# Create custom tokenizer
class CodeAwareTokenizer(CustomTokenizer):
    def __init__(self, base_tokenizer):
        self.base_tokenizer = base_tokenizer
        self.code_patterns = self.load_code_patterns()
    
    def tokenize(self, text):
        # Detect code blocks
        if self.is_code(text):
            return self.tokenize_code(text)
        else:
            return self.base_tokenizer.encode(text)
    
    def tokenize_code(self, code):
        # Special handling for code
        tokens = []
        for line in code.split('\n'):
            if line.strip().startswith('#'):
                # Comments get special tokens
                tokens.extend(self.tokenize_comment(line))
            else:
                tokens.extend(self.base_tokenizer.encode(line))
        return tokens
```

## Memory-Mapped Datasets

### Creating Memory-Mapped Files

```python
from moe_llm.data import MemoryMappedBuilder

# Create builder
builder = MemoryMappedBuilder(
    output_prefix="data/mmap/train",
    dtype="uint16",  # Use uint16 for vocabulary < 65k
    sequence_length=2048
)

# Add tokenized data
for tokens in tokenized_data:
    builder.add_sequence(tokens)

# Finalize and create index
builder.finalize()
print(f"Created memory-mapped dataset with {builder.num_sequences} sequences")
```

### Loading Memory-Mapped Data

```python
from moe_llm.data import MemoryMappedDataset

# Load dataset
dataset = MemoryMappedDataset(
    data_path="data/mmap/train.bin",
    index_path="data/mmap/train.idx",
    sequence_length=2048,
    seed=42
)

# Access data
sample = dataset[0]
print(f"Sample shape: {sample.shape}")

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

### Streaming Large Datasets

```python
from moe_llm.data import StreamingDataset

# Create streaming dataset
dataset = StreamingDataset(
    data_files=["data/train_*.bin"],
    buffer_size=10000,
    shuffle_buffer_size=10000,
    seed=42
)

# Iterate efficiently
for batch in dataset.iter_batches(batch_size=32):
    # Process batch
    pass
```

## Data Mixing Strategies

### Domain-Balanced Mixing

```python
from moe_llm.data import DataMixer

# Define domain weights
domain_weights = {
    "wikipedia": 0.3,
    "books": 0.2,
    "code": 0.15,
    "scientific": 0.15,
    "web": 0.2
}

# Create mixer
mixer = DataMixer(
    datasets={
        "wikipedia": wiki_dataset,
        "books": books_dataset,
        "code": code_dataset,
        "scientific": arxiv_dataset,
        "web": web_dataset
    },
    weights=domain_weights,
    temperature=1.0  # Sampling temperature
)

# Sample mixed batches
for epoch in range(num_epochs):
    mixer.set_epoch(epoch)  # For deterministic shuffling
    
    for batch in mixer.iter_batches(batch_size=32):
        # Batch contains mix of domains
        print(f"Batch domains: {batch['domains']}")
```

### Dynamic Mixing

```python
from moe_llm.data import DynamicDataMixer

# Create dynamic mixer that adjusts weights during training
dynamic_mixer = DynamicDataMixer(
    datasets=domain_datasets,
    initial_weights=domain_weights,
    adjustment_strategy="loss_based",  # Adjust based on domain losses
    adjustment_interval=1000  # Steps
)

# Training loop with dynamic mixing
for step, batch in enumerate(dynamic_mixer):
    loss = train_step(batch)
    
    # Update mixer with domain losses
    dynamic_mixer.update_losses({
        batch['domain']: loss.item()
    })
    
    if step % 1000 == 0:
        current_weights = dynamic_mixer.get_current_weights()
        print(f"Current mixing weights: {current_weights}")
```

### Curriculum Data Mixing

```python
from moe_llm.data import CurriculumMixer

# Define curriculum stages
curriculum = [
    {
        "steps": 10000,
        "weights": {"simple": 0.7, "medium": 0.3, "complex": 0.0}
    },
    {
        "steps": 20000,
        "weights": {"simple": 0.3, "medium": 0.5, "complex": 0.2}
    },
    {
        "steps": 30000,
        "weights": {"simple": 0.1, "medium": 0.4, "complex": 0.5}
    }
]

# Create curriculum mixer
curr_mixer = CurriculumMixer(
    datasets={
        "simple": simple_dataset,
        "medium": medium_dataset,
        "complex": complex_dataset
    },
    curriculum=curriculum
)

# Train with curriculum
for batch in curr_mixer:
    train_step(batch)
```

## Deduplication

### MinHash LSH Deduplication

```python
from moe_llm.data import MinHashDeduplicator

# Create deduplicator
dedup = MinHashDeduplicator(
    num_perm=128,  # Number of hash functions
    threshold=0.9,  # Similarity threshold
    ngram_size=5,  # Character n-grams
    num_bands=4,  # LSH bands
    rows_per_band=32  # Rows per band
)

# Build LSH index
dedup.build_index(documents)

# Find duplicates
duplicates = dedup.find_duplicates()
print(f"Found {len(duplicates)} duplicate pairs")

# Remove duplicates
unique_docs = dedup.deduplicate(documents)
print(f"Retained {len(unique_docs)} unique documents")
```

### Exact Deduplication

```python
from moe_llm.data import ExactDeduplicator

# Remove exact duplicates
exact_dedup = ExactDeduplicator(
    hash_function="sha256",
    check_substring=True,  # Also remove substring duplicates
    min_length=50  # Minimum length for substring check
)

unique_data = exact_dedup.deduplicate(data)
```

### Fuzzy Deduplication

```python
from moe_llm.data import FuzzyDeduplicator

# Fuzzy deduplication with edit distance
fuzzy_dedup = FuzzyDeduplicator(
    method="levenshtein",
    threshold=0.95,
    batch_size=1000,
    use_gpu=True
)

# Process in batches for memory efficiency
for batch in data.iter_batches(10000):
    unique_batch = fuzzy_dedup.deduplicate_batch(batch)
    save_batch(unique_batch)
```

## Best Practices

### 1. Data Quality Checklist

```python
from moe_llm.data import DataQualityReport

# Generate comprehensive quality report
reporter = DataQualityReport()
report = reporter.analyze(dataset)

print("Data Quality Report:")
print(f"- Total documents: {report.total_docs}")
print(f"- Avg document length: {report.avg_length:.1f}")
print(f"- Vocabulary size: {report.vocab_size}")
print(f"- Duplicate rate: {report.duplicate_rate:.2%}")
print(f"- Language distribution: {report.languages}")
print(f"- Domain distribution: {report.domains}")
print(f"- Quality score distribution: {report.quality_dist}")
```

### 2. Efficient Processing

```python
# Use pipeline for efficient processing
from moe_llm.data import ProcessingPipeline

pipeline = ProcessingPipeline([
    ("filter", QualityFilter(min_score=0.7)),
    ("dedup", MinHashDeduplicator(threshold=0.9)),
    ("tokenize", FastTokenizer("gpt2")),
    ("save", MemoryMappedBuilder("output/train"))
])

# Process with progress tracking
pipeline.process(
    input_files="data/raw/*.jsonl",
    num_workers=32,
    chunk_size=10000,
    progress=True
)
```

### 3. Data Validation

```python
from moe_llm.data import DataValidator

# Validate processed data
validator = DataValidator()

# Check for common issues
issues = validator.validate(dataset, checks=[
    "token_length",  # Check sequence lengths
    "vocab_coverage",  # Check vocabulary coverage
    "special_tokens",  # Verify special tokens
    "encoding_errors",  # Check for encoding issues
    "data_balance"  # Check class/domain balance
])

if issues:
    print("Validation issues found:")
    for issue in issues:
        print(f"- {issue}")
```

### 4. Monitoring Data Pipeline

```python
from moe_llm.data import PipelineMonitor

# Monitor pipeline execution
monitor = PipelineMonitor(
    log_dir="logs/data_pipeline",
    metrics=["throughput", "memory", "disk_io"],
    alert_on_error=True
)

with monitor:
    pipeline.run()

# Get performance report
perf_report = monitor.get_report()
print(f"Processing rate: {perf_report.docs_per_second:.1f} docs/sec")
print(f"Peak memory: {perf_report.peak_memory_gb:.1f} GB")
```

### 5. Data Versioning

```python
from moe_llm.data import DataVersion

# Version your datasets
versioner = DataVersion(
    base_path="data/versions",
    tracking_backend="git"  # or "mlflow", "dvc"
)

# Create new version
version_id = versioner.create_version(
    dataset=processed_data,
    metadata={
        "processing_date": "2024-01-01",
        "quality_threshold": 0.7,
        "dedup_threshold": 0.9,
        "tokenizer": "gpt2"
    }
)

print(f"Created dataset version: {version_id}")

# Load specific version
dataset_v1 = versioner.load_version(version_id)
```

For more details on specific components:
- [Training Guide](training.md)
- [API Reference](api_reference.md)
- [Memory Optimization](memory_optimization.md)