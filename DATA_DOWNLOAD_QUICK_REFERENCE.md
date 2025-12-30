# Data Download Quick Reference

## Dataset Sizes at a Glance

```
TinyStories       200MB  → 2GB   (Download: SECONDS)
OpenOrca          13GB   → 15GB  (Download: 10-30 MIN)
C4 (5 parts)      50GB   → 250GB (Download: 30-90 MIN)
C4 (full)         305GB  → 1.2TB (Download: DAYS)
```

## One-Liner Commands

### Quick Test (Smallest)
```bash
python code/scripts/1_data_download/unified_download.py --dataset "roneneldan/TinyStories"
# Size: 200MB → 2GB | Time: Seconds
```

### Production Training (Good Balance)
```bash
python code/scripts/1_data_download/unified_download.py --dataset "Open-Orca/OpenOrca"
# Size: 13GB → 15GB | Time: 10-30 minutes
```

### Large Scale (Multi-GPU)
```bash
python code/scripts/1_data_download/unified_download.py --dataset "allenai/c4" --max-partitions 5
# Size: 50GB → 250GB | Time: 30-90 minutes
```

### Multiple Datasets
```bash
python code/scripts/1_data_download/unified_download.py \
  --datasets roneneldan/TinyStories Open-Orca/OpenOrca
# Combined: 13.2GB → 17GB | Time: 10-30 minutes
```

## What You're Getting

### TinyStories (~200MB)
- **What:** Simple AI-generated stories
- **Quality:** Good for testing/debugging
- **Examples:** 20M short stories
- **Perfect for:** Model validation, quick iteration
- **Not for:** Real production training

### OpenOrca (~13GB)
- **What:** Instruction-following dataset (LLAMA 2, GPT-3.5, GPT-4)
- **Quality:** Excellent, production-ready
- **Examples:** 1M+ high-quality Q&A pairs
- **Perfect for:** General-purpose training, fine-tuning
- **Best for:** Single-GPU training (24-40GB VRAM)

### C4 (305GB full, or 50GB for 5 partitions)
- **What:** Massive web text corpus
- **Quality:** Mixed (web-scale data)
- **Examples:** 10+ billion tokens
- **Perfect for:** Large-scale research, scaling laws
- **Requires:** Serious hardware (multi-GPU), lots of storage
- **Not for:** Most people (except researchers)

## Disk Space Check

```bash
# Check available space
df -h /root/Ava_AI/code/data

# Minimum needed:
# TinyStories: 5GB
# OpenOrca: 25GB
# C4 (5 parts): 300GB
# C4 (full): 1.5TB
```

## Next Steps

### After Downloading
```bash
# Pre-tokenize with 16k tokenizer
python code/scripts/1_data_download/pretokenize_datasets.py \
  --dataset-path /root/Ava_AI/code/data/TinyStories \
  --output-path /root/Ava_AI/code/data/TinyStories-tokenized
```

### Start Training
```bash
# Edit config to use your dataset
# Then run:
python code/scripts/5_training/train_pipeline.py --config code/configs/moe/large.yaml
```

## Troubleshooting

**"Not enough disk space"**
- Start with TinyStories only (200MB)
- Or use `--max-partitions 1` for C4

**"Download too slow"**
- Try `--workers 8` (default is 4)
- Or download at off-peak hours
- C4 is huge - consider OpenOrca instead

**"Connection timeout"**
- Network might be unstable
- Script auto-resumes - just run again
- Use `--workers 2` on poor connections

**"Authentication errors"**
```bash
huggingface-cli login
# Paste token from https://huggingface.co/settings/tokens
```

## Dataset Links

- **TinyStories**: https://huggingface.co/datasets/roneneldan/TinyStories
- **OpenOrca**: https://huggingface.co/datasets/Open-Orca/OpenOrca
- **C4**: https://huggingface.co/datasets/allenai/c4

## Recommendation by Use Case

| Use Case | Dataset | Size | Time |
|----------|---------|------|------|
| 🧪 Testing | TinyStories | 200MB | Seconds |
| 📚 Learning | TinyStories | 200MB | Seconds |
| 🎯 Production (1 GPU) | OpenOrca | 13GB | 20 min |
| 🖥️ Multi-GPU | OpenOrca + C4(5) | 63GB | 1 hour |
| 🔬 Research | C4 full | 305GB | Days |

Start with TinyStories to validate your setup, then scale up!
