# Quick Start Guide - Training with All Data

## Status
✅ **COMPLETE**: Your training script now uses ALL 200,000 examples from both Arrow files!

---

## Key Facts

| Item | Value |
|------|-------|
| **Data Files** | 2 Arrow files (306MB total) |
| **Total Examples** | 200,000 |
| **Examples Per Epoch** | 200,000 |
| **Batches Per Epoch** | 100,000 (with batch_size=2) |
| **Data Coverage** | 100% ✅ |

---

## How It Works

The script automatically:
1. Searches for all `.arrow` files in `/project/code/data/pretokenized/`
2. Loads them ALL at once (not just the first one)
3. Falls back to synthetic data if no real files found
4. Works with ANY YAML configuration

---

## Quickest Start

```bash
# Just run it - loads all data automatically!
python code/scripts/5_training/train_100m_full.py --config code/configs/moe/minimal_working.yaml
```

---

## Common Commands

### 1. Test Run (1 minute)
```bash
timeout 60 python code/scripts/5_training/train_100m_full.py \
    --config code/configs/moe/minimal_working.yaml --epochs 1
```

### 2. Full Training (Tiny Model)
```bash
python code/scripts/5_training/train_100m_full.py \
    --config code/configs/moe/tiny_moe.yaml --epochs 5
```

### 3. Multi-GPU Training
```bash
torchrun --nproc_per_node=4 code/scripts/5_training/train_100m_full.py \
    --config code/configs/moe/small_moe.yaml --epochs 10
```

### 4. Resume from Checkpoint
```bash
python code/scripts/5_training/train_100m_full.py \
    --config code/configs/moe/minimal_working.yaml \
    --resume checkpoints/checkpoint_epoch_1_step_0.pt
```

---

## Expected Output

### Proof It's Using All Data
Look for this in the output:
```
Epoch 1:   0%|          | 0/100000 [00:00<?, ?it/s]
```

The `0/100000` means 100,000 batches = 200,000 examples ÷ 2

### Loss Should Decrease
```
Batch 0:    Loss: 10.98 (random)
Batch 100:  Loss: 8.00  (learning)
Batch 300:  Loss: 3.65  (converging)
```

Smooth descent = real training on real data ✅

---

## Data Files

```
/project/code/data/pretokenized/
├── roneneldan_TinyStories_once_upon_stories.arrow  (140MB, 100K)
├── roneneldan_TinyStories_processed.arrow          (166MB, 100K)
└── Total: 306MB, 200K examples
```

Both loaded automatically!

---

## Configurations Available

### Minimal (Ultra-Small for Testing)
```bash
--config code/configs/moe/minimal_working.yaml
# 58M params, batch_size=2, quick convergence
```

### Tiny (Single GPU Optimized)
```bash
--config code/configs/moe/tiny_moe.yaml
# 431M params, batch_size=32, good balance
```

### Small (Multi-GPU Ready)
```bash
--config code/configs/moe/small_moe.yaml
# 1.57B params, batch_size=64+, production scale
```

---

## Output Files

### Checkpoints
```
/project/code/outputs/runs/[run_name]/checkpoints/
├── checkpoint_epoch_0_step_0.pt
├── checkpoint_epoch_1_step_0.pt
└── checkpoint_epoch_2_step_0.pt
```

### Logs
```
/project/code/outputs/runs/[run_name]/logs/
├── training.log
├── metrics_summary.json
└── events.out.tfevents.*  (TensorBoard)
```

### View Training Progress
```bash
tensorboard --logdir /project/code/outputs/runs/minimal_working/logs
# Opens at http://localhost:6006
```

---

## Troubleshooting

### Script Doesn't Find Data
- Check if Arrow files exist: `ls /project/code/data/pretokenized/*.arrow`
- Script will fall back to synthetic data if files missing
- No error - just uses DummyDataset instead

### CUDA Out of Memory
- Reduce batch size: `--batch-size 1`
- Use gradient accumulation (in config)
- Use smaller config (minimal_working.yaml)

### Training Too Slow
- Check GPU utilization: `nvidia-smi`
- Increase batch size (if memory allows)
- Use `tiny_moe.yaml` (has speed optimizations)

### Loss Not Decreasing
- First few batches have random behavior (expected)
- Loss should steadily decrease after batch 100
- If not, data may not be loading correctly

---

## Next Steps

1. ✅ **Run a quick test** (1 minute):
   ```bash
   timeout 60 python code/scripts/5_training/train_100m_full.py \
       --config code/configs/moe/minimal_working.yaml --epochs 1
   ```

2. ✅ **Verify loss decreases** in output

3. ✅ **Run full training** with your chosen config

4. ✅ **Monitor with TensorBoard** (optional):
   ```bash
   tensorboard --logdir /project/code/outputs/runs/*/logs
   ```

---

## Important Notes

- ✅ **All 200K examples are used** (verified with test)
- ✅ **Works with any config file** (automatic parameter detection)
- ✅ **No changes needed** - just run it!
- ✅ **GPU training** (CUDA required)
- ✅ **Multi-GPU ready** (use torchrun)
- ✅ **Checkpoint resumable** (--resume flag)

---

**Your training system is ready to use!** 🚀
