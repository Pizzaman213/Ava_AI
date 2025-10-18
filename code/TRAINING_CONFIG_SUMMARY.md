# Training Configuration Summary

## Current Setup for AG News Training

### Dataset Configuration
- **Dataset**: AG News (news classification)
- **Location**: `/project/code/data/ag_news/processed`
- **Training samples**: 108,303
- **Validation samples**: 19,112
- **Total tokens**: 19M tokens
- **Categories**: 4 (World, Sports, Business, Sci/Tech)

### Model Configuration
- **Architecture**: Enhanced MoE (Mixture of Experts)
- **Hidden size**: 64
- **Layers**: 2
- **Attention heads**: 2
- **Experts**: 2 (1 active per token)
- **Vocab size**: 500 (custom tokenizer)
- **Max sequence length**: 512 tokens

### Training Hyperparameters
```yaml
batch_size: 128                    # Process 128 samples per step
gradient_accumulation_steps: 1     # No gradient accumulation
learning_rate: 0.0002              # Initial learning rate
num_epochs: 10                     # Train for 10 epochs
warmup_steps: 500                  # LR warmup for 500 steps
lr_scheduler_type: cosine          # Cosine annealing schedule
```

### Monitoring & Evaluation
```yaml
eval_steps: 100                    # Evaluate every 100 steps
logging_steps: 100                 # Log metrics every 100 steps
save_steps: 5000                   # Save checkpoint every 5000 steps
save_total_limit: 5                # Keep only 5 most recent checkpoints
```

### Training Timeline
- **Steps per epoch**: ~846 steps (108,303 ÷ 128)
- **Total steps (10 epochs)**: ~8,460 steps
- **Evaluations per epoch**: ~8-9 times
- **Total evaluations**: ~84-90 evaluations
- **Checkpoints saved**: ~1-2 per epoch (at steps 5000, 10000, etc.)

### Estimated Training Time (RTX 3090 Ti)
- **Per epoch**: ~5-10 minutes
- **Total (10 epochs)**: ~50-100 minutes (0.8-1.7 hours)
- **First evaluation**: After ~100 steps (~1-2 minutes)

### Memory Usage
- **Model size**: ~2M parameters (tiny model)
- **Batch size**: 128 samples
- **Sequence length**: 512 tokens max
- **Expected VRAM**: ~2-4 GB (plenty of headroom on 24GB GPU)

### Key Features Enabled
- ✓ Flash Attention (faster training)
- ✓ Mixed precision (bf16)
- ✓ Repetition penalties (prevent repetitive generation)
- ✓ Quality filtering (high-quality training data)
- ✓ Cosine LR schedule (better convergence)
- ✓ Frequent evaluation (catch overfitting early)

### Start Training
```bash
cd /project/code
python scripts/5_training/train.py --config configs/gpu/small.yaml
```

### Monitor Progress
Training outputs will show:
- Loss every 100 steps
- Validation metrics every 100 steps
- Checkpoint saves every 5000 steps
- WandB logging (if enabled)

### Configuration File
**Location**: `/project/code/configs/gpu/small.yaml`

**Key settings**:
- Dataset: AG News only
- Batch size: 128
- Evaluation: Every 100 steps
- Fast, frequent feedback for iterative development

---
*Last updated: 2025-10-17*
*Configuration optimized for AG News training with frequent monitoring*
