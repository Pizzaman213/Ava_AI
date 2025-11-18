# Training Quickstart Guide

## File Location
```
/project/code/scripts/5_training/train_100m_full.py
```

## Quick Commands

### 1️⃣ **Basic Training (No Config)**
```bash
cd /project
python code/scripts/5_training/train_100m_full.py
```

### 2️⃣ **Training with Config**
```bash
python code/scripts/5_training/train_100m_full.py \
    --config code/configs/moe/tiny_moe.yaml
```

### 3️⃣ **Custom Hyperparameters**
```bash
python code/scripts/5_training/train_100m_full.py \
    --epochs 10 \
    --batch-size 32 \
    --learning-rate 5e-5 \
    --save-dir ./my_checkpoints \
    --log-dir ./my_logs
```

### 4️⃣ **Resume from Checkpoint**
```bash
python code/scripts/5_training/train_100m_full.py \
    --resume ./checkpoints/checkpoint_epoch_5_step_0.pt
```

### 5️⃣ **Multi-GPU Training (4 GPUs)**
```bash
torchrun --nproc_per_node=4 code/scripts/5_training/train_100m_full.py \
    --config code/configs/moe/tiny_moe.yaml
```

### 6️⃣ **Monitor with TensorBoard**
```bash
# Terminal 1: Run training
python code/scripts/5_training/train_100m_full.py

# Terminal 2: View metrics
tensorboard --logdir ./logs --port 6006

# Browser: http://localhost:6006
```

---

## All CLI Arguments

```
--config PATH                 Path to YAML config file
--epochs N                    Number of epochs (default: 3)
--batch-size N               Batch size per GPU (default: 8)
--learning-rate LR           Learning rate (default: 5e-5)
--save-dir PATH              Directory for checkpoints (default: ./checkpoints)
--resume PATH                Path to checkpoint to resume from
--log-dir PATH               Directory for logs (default: ./logs)
--log-interval N             Log every N batches (default: 10)
--val-interval N             Validate every N epochs (default: 1)
```

---

## Output Files

### Logs Directory (`./logs/`)
- `training_YYYYMMDD_HHMMSS.log` - Training log file
- `metrics_summary.json` - Final metrics as JSON
- `events.out.tfevents.*` - TensorBoard event files

### Checkpoints Directory (`./checkpoints/`)
- `checkpoint_epoch_X_step_Y.pt` - Saved model state
  - Keep last 3 automatically
  - Contains: model weights, optimizer state, metrics

---

## Key Features

✅ **Model**
- 92M parameters (configurable)
- 12-layer Transformer
- Supports custom architectures

✅ **Training**
- Full training loop with validation
- Mixed precision (bfloat16)
- Gradient accumulation
- Learning rate scheduling (warmup + cosine annealing)

✅ **Data**
- Dummy dataset included (replace with real data)
- Multi-worker data loading
- Support for distributed data loading

✅ **Logging**
- Console logging with timestamps
- File logging
- TensorBoard metrics
- JSON summary

✅ **Checkpointing**
- Save/load model and optimizer state
- Resume training mid-epoch
- Automatic checkpoint pruning

✅ **Distributed Training**
- Single-node multi-GPU (DDP)
- Multi-node with torchrun
- Rank-aware logging

---

## Common Recipes

### Train with 8 GPUs
```bash
torchrun --nproc_per_node=8 code/scripts/5_training/train_100m_full.py
```

### Train longer with lower memory
```bash
python code/scripts/5_training/train_100m_full.py \
    --epochs 20 \
    --batch-size 4 \
    --learning-rate 1e-5
```

### Fine-tune from checkpoint
```bash
python code/scripts/5_training/train_100m_full.py \
    --resume checkpoints/checkpoint_epoch_10_step_0.pt \
    --epochs 15 \
    --learning-rate 1e-5
```

### Quick test run
```bash
python code/scripts/5_training/train_100m_full.py \
    --epochs 1 \
    --batch-size 2 \
    --log-interval 1
```

---

## Customization Quick Links

See `TRAINING_IMPLEMENTATION_GUIDE.md` for:
- Custom dataset implementation
- Custom model architecture
- Custom optimizer/scheduler
- Weights & Biases integration
- DeepSpeed integration
- Performance tuning

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA OOM** | `--batch-size 4` or increase gradient accumulation |
| **Slow on multi-GPU** | Check GPU utilization with `nvidia-smi` |
| **DDP hanging** | Use `TORCH_DISTRIBUTED_DEBUG=DETAIL` |
| **Checkpoint issues** | Verify checkpoint with `torch.load()` |

---

## Performance Metrics (100M model)

- **Model size**: ~92M parameters
- **Memory usage**: ~8GB (batch_size=8)
- **Training speed**: ~17 batches/second (single GPU)
- **Mixed precision**: Enables bfloat16 automatically

---

## Next Steps

1. **Start training**: `python train_100m_full.py`
2. **Monitor metrics**: `tensorboard --logdir ./logs`
3. **Customize model**: Edit `TransformerModel100M` class
4. **Add your data**: Replace `DummyDataset` with your dataset
5. **Scale up**: Use `torchrun` for multi-GPU training

---

For detailed information, see `TRAINING_IMPLEMENTATION_GUIDE.md`
