# 🚀 Ava Fine-Tuning Guide

## Quick Start - Automatic Mode

The easiest way to fine-tune with the latest checkpoint and data:

```bash
cd /project/code/scripts/training
python finetune.py
```

This will automatically:
1. ✅ Find the **latest checkpoint** from `/project/code/outputs/runs/`
2. ✅ Load the **config** that was used with that checkpoint
3. ✅ Use the **latest data file** from `/project/code/data/fine-tuning/`
4. ✅ Resume training from where it left off

## Current Setup

### Latest Checkpoint
```
/project/code/outputs/runs/run_20251002_115535_4702d66b/checkpoints/step_12345/model.pt
```

### Fine-Tuning Data Available
Located in `/project/code/data/fine-tuning/`:
- sahil2801_CodeAlpaca-20k_processed.jsonl (2.7 MB) ⭐ Latest
- pubmed_qa_processed.jsonl (368 KB)
- meta-math_MetaMathQA_processed.jsonl (190 MB)
- m-a-p_CodeFeedback-Filtered-Instruction_processed.jsonl (160 MB)
- And 14 more datasets...

## Usage Examples

### 1. Simple Fine-Tuning (Recommended)
```bash
python finetune.py
```
Uses latest checkpoint + latest data file.

### 2. Fine-Tune on All Data
```bash
python finetune.py --use-all-files
```
Trains on all 18 datasets in the fine-tuning directory.

### 3. Use Specific Data
```bash
# Use latest 5 data files
python finetune.py --num-latest-files 5

# Use specific pattern
python finetune.py --file-pattern "*CodeAlpaca*"

# Use specific pattern with all matching files
python finetune.py --file-pattern "*code*" --use-all-files
```

### 4. Use Specific Checkpoint
```bash
python finetune.py --checkpoint /path/to/specific/model.pt
```
Config is auto-discovered from the checkpoint's run directory.

### 5. Override Configuration
```bash
python finetune.py --config ../../configs/gpu/small.yaml
```

### 6. Start from Scratch
```bash
python finetune.py --no-checkpoint
```
Trains a new model instead of continuing from checkpoint.

### 7. Custom Training Parameters
```bash
python finetune.py \
    --batch-size 16 \
    --learning-rate 0.0001 \
    --num-epochs 5 \
    --max-steps 10000 \
    --wandb-project my-finetuning
```

### 8. Advanced Options
```bash
python finetune.py \
    --use-all-files \
    --enable-progressive-training \
    --batch-size 16 \
    --max-length 2048 \
    --run-name "my-custom-finetune"
```

## Command-Line Options

### Data Options
- `--data-dir PATH` - Directory with fine-tuning data (default: `/project/code/data/fine-tuning`)
- `--num-latest-files N` - Use latest N files (default: 1)
- `--file-pattern PATTERN` - Glob pattern for file matching (default: `*_processed.jsonl`)
- `--use-all-files` - Use all files instead of latest N
- `--exclude-pattern PATTERN` - Exclude files matching pattern

### Checkpoint Options
- `--checkpoint PATH` - Use specific checkpoint file
- `--auto-checkpoint` - Auto-find latest checkpoint (default: True)
- `--no-checkpoint` - Start from scratch
- `--checkpoint-dir DIR` - Where to search for checkpoints (default: `/project/code/outputs/runs`)

### Config Options
- `--config PATH` - Config file (auto-discovered from checkpoint if not specified)

### Training Overrides
- `--batch-size N` - Override batch size
- `--learning-rate LR` - Override learning rate
- `--num-epochs N` - Override number of epochs
- `--max-steps N` - Override max training steps
- `--max-length N` - Override max sequence length

### Features
- `--enable-progressive-training` - Enable progressive sequence length training
- `--enable-observability` - Enable observability features (default: True)
- `--wandb-project NAME` - Weights & Biases project name

### Output
- `--run-name NAME` - Custom run name
- `--output-dir DIR` - Output directory (default: `outputs/finetune`)

## How It Works

### 1. Checkpoint Discovery
The script searches `/project/code/outputs/runs/` recursively for `model.pt` files and selects the most recently modified one:

```
/project/code/outputs/runs/
├── run_20251002_115535_4702d66b/
│   └── checkpoints/
│       └── step_12345/
│           └── model.pt ⭐ Found!
├── run_20251002_115233_0f470a51/
│   └── checkpoints/...
└── ...
```

### 2. Config Discovery
Once a checkpoint is found, the script looks for the original config:
- First: Check `run_dir/configs/*.yaml`
- Fallback: Use `/project/code/configs/gpu/small.yaml`

### 3. Data Discovery
Scans `/project/code/data/fine-tuning/` for files matching the pattern (default: `*_processed.jsonl`) and selects based on modification time.

### 4. Fine-Tuning
Passes everything to `train.py` with:
- `resume_from_checkpoint` set to the discovered checkpoint
- `data_dir` pointed to the fine-tuning directory
- All 8 enhancement phases enabled (gradient health, progressive training, etc.)

## Q&A Dataset Format

The fine-tuning script supports Q&A datasets in multiple formats:

### JSONL (Recommended)
```json
{"text": "Question: What is Python? Answer: Python is a programming language..."}
{"text": "Question: How do lists work? Answer: Lists in Python are..."}
```

Or with separate columns:
```json
{"question": "What is Python?", "answer": "Python is a programming language..."}
{"instruction": "Explain lists", "response": "Lists in Python are..."}
```

### Other Formats
- Parquet/Arrow with `question`, `answer` columns
- CSV with headers
- The data loader automatically detects and handles column formats

## Tips

1. **Start Simple**: Just run `python finetune.py` to get started
2. **Monitor Training**: Use `--wandb-project` to track metrics
3. **Memory Issues**: Reduce `--batch-size` or `--max-length`
4. **Fast Iteration**: Use `--num-latest-files 1` for quick experiments
5. **Production Run**: Use `--use-all-files` for comprehensive training

## Troubleshooting

### No checkpoint found
```bash
# Force training from scratch
python finetune.py --no-checkpoint
```

### Specific checkpoint not working
```bash
# Use explicit paths
python finetune.py \
    --checkpoint /path/to/model.pt \
    --config /path/to/config.yaml
```

### Out of memory
```bash
# Reduce batch size
python finetune.py --batch-size 4 --max-length 512
```

## Output

Training outputs are saved to:
```
outputs/finetune/run_YYYYMMDD_HHMMSS_<id>/
├── checkpoints/
│   ├── best_model.pt
│   ├── latest_model.pt
│   └── step_N/model.pt
├── logs/
│   ├── training.log
│   ├── evaluation.log
│   └── errors.log
└── metrics/
    └── training_metrics.json
```

## Next Steps

After fine-tuning completes, generate text with:
```bash
cd /project/code/scripts/generation
python generate.py --prompt "Your prompt here"
```

The generation script will automatically find and use your latest fine-tuned model!
