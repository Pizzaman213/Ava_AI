# Multi-Column Data Loader Usage Guide

This guide explains how to use the enhanced multi-column data loading system for training with complex datasets that contain multiple data types and formats.

## Overview

The multi-column data loader supports training on datasets with multiple columns and mixed data types:

### Supported Column Types
- **Text**: Natural language text with configurable tokenization
- **Numeric**: Numerical data with normalization support
- **Categorical**: Categorical data with vocabulary mapping and one-hot encoding
- **Image**: Image data with preprocessing pipelines (requires PIL)
- **Audio**: Audio data support (placeholder for future implementation)
- **Video**: Video data support (placeholder for future implementation)
- **Embedding**: Pre-computed embedding vectors
- **Tensor**: Raw tensor data
- **JSON**: Structured JSON data
- **Binary**: Binary data support

### Supported Dataset Formats
- **Instruction-Response** (Alpaca/Dolly style)
- **QA with Context** (SQuAD style)
- **Conversation** (OpenAssistant style)
- **Multi-Modal** (Text + Image/Audio combinations)
- **Structured Data** (Mixed text/numeric/categorical)
- **HuggingFace Datasets** (Direct integration)

## Key Features

### 🎯 Multi-Column Data Loading
Support for datasets with multiple columns of different data types, processed according to their specific requirements.

### 📊 Flexible Combine Strategies
Three different ways to combine multiple columns:

1. **Concatenate**: Merge all input columns into a single sequence
2. **Template**: Use custom templates to format multiple columns
3. **Separate**: Keep columns separate for multi-input models

### 🔧 Configurable Processing
Each column can have its own:
- Data type and role (input, target, auxiliary, weight)
- Maximum length and normalization settings
- Vocabulary for categorical data
- Preprocessing and augmentation rules
- Validation constraints

### 🚀 HuggingFace Integration
Direct loading from HuggingFace Hub with automatic column mapping and streaming support.

## Basic Usage

### 1. Command Line Interface

#### Simple Text Column
```bash
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --use-multi-column \
    --column-names text \
    --column-types text \
    --column-roles input \
    --data-dir /project/code/processed
```

#### Multiple Text Columns with Template
```bash
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --use-multi-column \
    --column-names instruction,response \
    --column-types text,text \
    --column-roles input,target \
    --combine-strategy template \
    --column-template "Instruction: {instruction}\nResponse: {response}"
```

#### Mixed Data Types
```bash
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --use-multi-column \
    --column-names text_input,category,difficulty \
    --column-types text,categorical,numeric \
    --column-roles input,auxiliary,auxiliary \
    --combine-strategy separate
```

### 2. Configuration Files

Create reusable configurations for complex datasets:

#### Instruction-Response Configuration
```yaml
# configs/multi_column_tests/instruction_response.yaml
columns:
  - name: instruction
    type: text
    role: input
    max_length: 128
    required: true
  - name: response
    type: text
    role: target
    max_length: 256
    required: true

combine_strategy: template
template: "Instruction: {instruction}\nResponse: {response}"
max_samples: 1000
validation_enabled: true
```

#### Multi-Input Configuration
```yaml
# configs/multi_column_tests/multi_input.yaml
columns:
  - name: text_input
    type: text
    role: input
    max_length: 256
    required: true
  - name: category
    type: categorical
    role: auxiliary
    vocab: ["science", "math", "history", "literature", "general"]
    required: false
  - name: difficulty
    type: numeric
    role: auxiliary
    normalize: true
    preprocessing:
      mean: 0.5
      std: 0.2
    required: false

combine_strategy: separate
max_samples: 50
validation_enabled: true
```

Use configurations with:
```bash
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --dataset-config configs/multi_column_tests/instruction_response.yaml
```

## Advanced Usage

### 1. HuggingFace Dataset Integration
```bash
# Load SQuAD dataset directly
python3 scripts/training/train.py \
    --config configs/gpu/small.yaml \
    --use-multi-column \
    --hf-dataset squad \
    --column-names context,question,answers \
    --column-types text,text,text \
    --column-roles input,input,target
```

### 2. Image + Text Multi-Modal
```yaml
# Multi-modal configuration
columns:
  - name: image
    type: image
    role: input
    preprocessing:
      resize: [224, 224]
      normalize: true
  - name: caption
    type: text
    role: target
    max_length: 128

combine_strategy: separate
```

### 3. Streaming Large Datasets
```bash
# Stream large datasets efficiently
python3 scripts/training/train.py \
    --config configs/gpu/medium.yaml \
    --use-multi-column \
    --column-names text \
    --column-types text \
    --column-roles input \
    --streaming \
    --buffer-size 5000
```

### 📝 Smart Formatting Templates

Each format uses optimized templates for better training:

**Instruction-Response Format:**
```
### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}
```

**QA with Context Format:**
```
Context: {context}

Question: {question}

Answer: {answer}
```

**Conversation Format:**
```
Human: {input}