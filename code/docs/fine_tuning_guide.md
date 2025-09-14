# Fine-Tuning Guide

## Overview

This guide covers fine-tuning pre-trained MoE++ models on your own datasets, including parameter-efficient methods like LoRA and QLoRA.

## Quick Start

### Basic Fine-Tuning
```bash
cd examples/fine_tuning
python supervised_finetuning.py \
    --checkpoint ../../outputs/checkpoints/best \
    --dataset tatsu-lab/alpaca \
    --batch-size 8 \
    --learning-rate 2e-4
```

### Parameter-Efficient Fine-Tuning (LoRA)
```bash
python supervised_finetuning.py \
    --checkpoint ../../outputs/checkpoints/best \
    --dataset tatsu-lab/alpaca \
    --use-lora \
    --batch-size 16
```

## Configuration-Based Fine-Tuning

All model configs now include fine-tuning sections:

```yaml
fine_tuning:
  # Parameter-efficient options
  use_lora: true
  use_qlora: false  # 4-bit quantization + LoRA
  freeze_base_model: false
  
  # LoRA configuration
  lora_r: 16  # Rank (higher = more parameters)
  lora_alpha: 32  # Scaling factor
  lora_dropout: 0.1
  lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  
  # Training settings
  batch_size: 8
  learning_rate: 2e-4
  num_epochs: 3
  warmup_ratio: 0.1
```

## Fine-Tuning Methods

### 1. Full Fine-Tuning
Updates all model parameters. Requires most memory but can achieve best results.

```bash
python supervised_finetuning.py \
    --checkpoint path/to/checkpoint \
    --dataset your-dataset \
    --batch-size 4 \
    --gradient-checkpointing
```

### 2. LoRA (Low-Rank Adaptation)
Only trains small adapter matrices, reducing memory by ~90%.

```bash
python supervised_finetuning.py \
    --checkpoint path/to/checkpoint \
    --dataset your-dataset \
    --use-lora \
    --batch-size 16
```

Benefits:
- 10x less memory usage
- 3x faster training
- Easy to merge/unmerge adapters

### 3. QLoRA (Quantized LoRA)
Combines 4-bit quantization with LoRA for extreme memory efficiency.

```yaml
fine_tuning:
  use_qlora: true
  lora_config:
    use_4bit: true
    bnb_4bit_compute_dtype: "float16"
    bnb_4bit_quant_type: "nf4"
```

Note: QLoRA requires `bitsandbytes` and is optimized for NVIDIA GPUs.

### 4. Freeze Base Model
Only trains the output head, keeping the base model frozen.

```bash
python supervised_finetuning.py \
    --checkpoint path/to/checkpoint \
    --dataset your-dataset \
    --freeze-base
```

## Memory Optimization Tips

### Reduce Memory Usage
1. **Enable streaming** (default in all configs now)
   ```yaml
   data:
     dataset_args:
       streaming: true
   ```

2. **Use gradient checkpointing**
   ```bash
   python supervised_finetuning.py --gradient-checkpointing
   ```

3. **Reduce batch size + increase accumulation**
   ```bash
   --batch-size 2 --gradient-accumulation-steps 16
   ```

4. **Use shorter sequences**
   ```bash
   --max-length 256
   ```

### Memory Requirements by Method
- Full fine-tuning: 4x model size
- LoRA: 1.1x model size  
- QLoRA: 0.3x model size
- Frozen base: 0.1x model size

## Supported Datasets

### Hugging Face Datasets
```bash
# Instruction tuning
--dataset tatsu-lab/alpaca
--dataset databricks/databricks-dolly-15k
--dataset OpenAssistant/oasst1

# General text
--dataset wikitext
--dataset c4
--dataset openwebtext
```

### Local Datasets
```bash
# Preprocessed tensor format
--dataset ./data/tensor/train

# JSONL format
--dataset ./data/custom.jsonl
```

### Custom Dataset Format
```json
{"text": "Your training text here"}
{"text": "Another training example"}
```

Or for instruction tuning:
```json
{"instruction": "Write a poem", "output": "Roses are red..."}
{"instruction": "Explain photosynthesis", "output": "Photosynthesis is..."}
```

## Monitoring Training

### Progress Tracking
The updated trainer shows:
- Current batch / total batches
- Percentage completion  
- Loss and average loss
- Estimated time remaining

Example output:
```
Epoch 1/3 | Total batches: 1250
Step 100 | Epoch 1/3 | Batch 100/1250 (8.0%) | Loss: 2.341 | Avg Loss: 2.567 | ETA: 45.2min
```

### Checkpointing
Checkpoints are saved automatically:
- Every N steps (configurable)
- Best model based on validation loss
- End of each epoch

### Resuming Training
```bash
python supervised_finetuning.py \
    --checkpoint path/to/checkpoint \
    --resume-from path/to/training/checkpoint
```

## Advanced Configuration

### Multi-Task Fine-Tuning
```yaml
fine_tuning:
  task_type: "multi_task"
  tasks:
    - name: "qa"
      weight: 0.3
      dataset: "squad"
    - name: "summarization"
      weight: 0.3
      dataset: "cnn_dailymail"  
    - name: "translation"
      weight: 0.4
      dataset: "wmt16"
```

### Custom Learning Rate Schedule
```yaml
fine_tuning:
  lr_scheduler_type: "polynomial"
  lr_scheduler_kwargs:
    power: 0.5
    lr_end: 1e-7
```

### Evaluation Metrics
```yaml
fine_tuning:
  eval_strategy: "steps"
  eval_steps: 100
  metric_for_best_model: "eval_loss"
  greater_is_better: false
  early_stopping: true
  early_stopping_patience: 3
```

## Best Practices

1. **Start with LoRA** for initial experiments
2. **Use streaming** to handle large datasets
3. **Monitor first 100 steps** closely
4. **Save checkpoints frequently**
5. **Use validation set** for early stopping
6. **Tune learning rate** first (most important)
7. **Increase batch size** via gradient accumulation

## Troubleshooting

### Out of Memory
- Enable gradient checkpointing
- Reduce batch size
- Use LoRA instead of full fine-tuning
- Enable data streaming (now default)

### Slow Training
- Check if using streaming from HDD (use SSD)
- Reduce number of workers if CPU-bound
- Use mixed precision training (fp16/bf16)

### Poor Results  
- Lower learning rate (try 5e-5, 2e-5, 1e-5)
- Increase warmup steps
- Train for more epochs
- Check dataset quality

### Model Not Learning
- Verify data format is correct
- Check if gradients are flowing
- Ensure not all parameters are frozen
- Try higher learning rate