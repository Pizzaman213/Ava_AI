# Train.py Refactoring Summary

## Objective
Make `/project/code/scripts/5_training/train.py` smaller and more modular by delegating functionality to `/project/code/src/Ava/training/train` managers.

**Current State:** 5,022 lines (monolithic)
**Target State:** ~400-500 lines + modular managers
**Reduction:** ~90% smaller main script

---

## What Has Been Completed

### 1. ✅ Created DataLoaderManager (`data_loader_manager.py` - 600+ lines)
**Location:** `/project/code/src/Ava/training/train/data_loader_manager.py`

Extracted from train.py:
- `create_dataloaders()` function (~410 lines)
- `enhanced_format_detection()` function (~90 lines)
- Multi-format support (pretokenized Arrow, streaming JSONL, multi-column)
- Dataset validation and statistics logging
- Smart data directory discovery with fallback paths

**Usage:**
```python
from src.Ava.training.train.data_loader_manager import DataLoaderManager

context = TrainingContext(...)
manager = DataLoaderManager(context)
train_loader, val_loader = manager.create_dataloaders(
    training_config, tokenizer, config_dict, batch_size
)
```

### 2. ✅ Created OptimizerManager (`optimizer_manager.py` - 550+ lines)
**Location:** `/project/code/src/Ava/training/train/optimizer_manager.py`

Extracted from train.py:
- `setup_optimizer_and_lr_management()` function (~335 lines)
- Support for multiple optimizer types:
  - AdamW, Fused AdamW
  - Lion, Lion8bit
  - AdamW8bit
  - Sophia
  - AdaFactor
- Adaptive learning rate management
- Parameter grouping with weight decay exclusions
- CPU offloading support

**Usage:**
```python
from src.Ava.training.train.optimizer_manager import OptimizerManager

context = TrainingContext(...)
manager = OptimizerManager(context)
optimizer, adaptive_lr_manager = manager.setup_optimizer_and_lr(
    model, config_dict, training_config, total_steps
)
```

### 3. ✅ Created ModelManager (`model_manager.py` - 500+ lines)
**Location:** `/project/code/src/Ava/training/train/model_manager.py`

Extracted from train.py:
- `create_model_and_tokenizer()` function (~170 lines)
- `materialize_meta_model()` function (~100 lines)
- Support for OptimizedMoETransformer and EnhancedMoEModel
- Meta device initialization for memory-efficient large models
- Tokenizer loading and validation
- Vocab size mismatch detection

**Usage:**
```python
from src.Ava.training.train.model_manager import ModelManager

context = TrainingContext(...)
manager = ModelManager(context)
model, tokenizer = manager.create_model_and_tokenizer(
    config_dict, training_config
)
```

### 4. ✅ Created EvaluationManager (`evaluation_manager.py` - 380+ lines)
**Location:** `/project/code/src/Ava/training/train/evaluation_manager.py`

Extracted from train.py:
- `test_generation_quality()` function (~110 lines)
- `evaluate_model()` function (~320 lines)
- `resume_smoke_test()` function (~30 lines)
- Generation quality testing with coherence metrics
- Loss and perplexity computation
- Invalid batch detection and reporting
- Checkpoint resume validation

**Usage:**
```python
from src.Ava.training.train.evaluation_manager import EvaluationManager

context = TrainingContext(...)
manager = EvaluationManager(context)

# Test generation
results = manager.test_generation_quality(
    model, tokenizer, device, test_prompts
)

# Evaluate loss
loss, perplexity = manager.evaluate_model(
    model, val_loader, device, use_bf16=True
)

# Smoke test
passed = manager.resume_smoke_test(model, train_loader, device)
```

### 5. ✅ Created Refactored Training Script Example (`train_refactored_example.py`)
**Location:** `/project/code/scripts/5_training/train_refactored_example.py`

Shows how to use all managers together in a clean, orchestration-focused script (~200 lines).

**Key Features:**
- Configuration loading
- Manager initialization
- Model and tokenizer creation
- Dataloader setup
- Optimizer configuration
- Validation pipeline
- All in ~200 lines vs 5,022 lines

---

## What Still Needs To Be Done

### Phase 1: Essential (Required for Full Refactoring)

#### 1. Integrate AsyncCheckpointSaver into CheckpointManager
- Move `AsyncCheckpointSaver` class from train.py (~250 lines) into `CheckpointManager`
- Update checkpoint save/load methods to use async saving
- Test async checkpoint operations

**Files to modify:**
- `/project/code/src/Ava/training/train/checkpoint_manager.py`

#### 2. Enhance MonitoringManager with W&B Setup
- Extract `setup_wandb()` function from train.py (~83 lines)
- Integrate W&B initialization into MonitoringManager
- Add wandb logging helpers

**Files to modify:**
- `/project/code/src/Ava/training/train/monitoring_manager.py`

#### 3. Refactor SimplifiedEnhancedTrainer
- Break down massive `train_epoch()` function (~484 lines) into smaller methods:
  - `_process_batch()` - Single batch processing
  - `_handle_checkpoint_saving()` - Checkpoint logic
  - `_run_in_epoch_validation()` - Validation logic
  - `_handle_oom_error()` - OOM recovery
- Keep high-level orchestration in `train_epoch()`

**Files to modify:**
- `/project/code/src/Ava/training/train/trainer.py`

### Phase 2: Main Script Refactoring

#### 4. Refactor train.py Main Script
Replace the monolithic train.py with a slim orchestration script that:
1. Loads configuration
2. Initializes managers
3. Calls manager methods to create components
4. Implements high-level training loop
5. Handles distributed training setup

**Example pattern** (from train_refactored_example.py):
```python
def main(config_path, batch_size=None, learning_rate=None):
    # Setup
    context = setup_training_context(config_dict, training_config)

    # Initialize managers
    data_loader_mgr = DataLoaderManager(context)
    model_mgr = ModelManager(context)
    optimizer_mgr = OptimizerManager(context)
    eval_mgr = EvaluationManager(context)

    # Create components
    model, tokenizer = model_mgr.create_model_and_tokenizer(...)
    train_loader, val_loader = data_loader_mgr.create_dataloaders(...)
    optimizer, lr_manager = optimizer_mgr.setup_optimizer_and_lr(...)

    # Training loop (use SimplifiedEnhancedTrainer)
    trainer = SimplifiedEnhancedTrainer(context, ...)
    trainer.train(...)
```

**Expected result:** ~400-500 line train.py

### Phase 3: Optional Enhancements

#### 5. Extract Logging Infrastructure
Move logging utilities to a shared module:
- `ColoredFormatter` class (~50 lines)
- `setup_training_logger()` function (~70 lines)
- `StructuredLogger` class (~100 lines)
- `TrainingTimer` class (~50 lines)

**Suggested location:** `/project/code/src/Ava/utils/logging.py`

#### 6. Create DeepSpeedManager (Optional)
If managing DeepSpeed initialization separately is valuable:
- Extract `initialize_deepspeed()` function (~150 lines)
- Create `DeepSpeedManager` class
- Or integrate into `DistributedTrainingManager`

---

## Integration Checklist

### For Using New Managers in train.py

- [ ] Import all managers at top of train.py
- [ ] Replace `create_dataloaders()` calls with `DataLoaderManager.create_dataloaders()`
- [ ] Replace `setup_optimizer_and_lr_management()` with `OptimizerManager.setup_optimizer_and_lr()`
- [ ] Replace `create_model_and_tokenizer()` with `ModelManager.create_model_and_tokenizer()`
- [ ] Replace evaluation functions with `EvaluationManager` methods
- [ ] Remove ~2,000 lines of extracted code from train.py
- [ ] Update `main()` to use managers
- [ ] Test with various configurations
- [ ] Update documentation

### Testing

- [ ] Run with small config to verify component initialization
- [ ] Run with medium config to verify training loop
- [ ] Run with distributed training to verify DDP compatibility
- [ ] Compare metrics with original train.py to verify correctness
- [ ] Profile to ensure no performance regression

---

## Code Structure After Full Refactoring

```
/project/code/src/Ava/training/train/
├── __init__.py
├── base.py                        # TrainingContext, TrainingComponent
├── trainer.py                     # SimplifiedEnhancedTrainer (refactored)
├── distributed_manager.py         # DistributedTrainingManager
├── checkpoint_manager.py          # CheckpointManager (with AsyncSaver)
├── loss_manager.py               # LossComputationManager
├── monitoring_manager.py         # MonitoringManager (with W&B)
├── data_loader_manager.py        # DataLoaderManager (NEW)
├── optimizer_manager.py          # OptimizerManager (NEW)
├── model_manager.py              # ModelManager (NEW)
├── evaluation_manager.py         # EvaluationManager (NEW)
├── example_training.py           # Example usage
└── docs/
    ├── README.md
    ├── ARCHITECTURE.md
    └── ...

/project/code/scripts/5_training/
├── train.py                      # Refactored: ~400-500 lines
└── train_refactored_example.py   # Example: ~200 lines
```

---

## Benefits of This Refactoring

### Code Quality
- **Modularity:** Each manager has a single responsibility
- **Testability:** Managers can be tested independently
- **Maintainability:** Changes to one manager don't affect others
- **Reusability:** Managers can be used in other training scripts

### Performance
- No performance regression (same underlying code)
- Easier to optimize individual managers
- Potential for future improvements (e.g., lazy initialization)

### Developer Experience
- Clear separation of concerns
- Easy to understand data flow
- Simple to add new optimizers/dataloaders
- Better IDE support and type hints

### Size Reduction
- **train.py:** 5,022 → ~400-500 lines (90% reduction)
- **Modular framework:** 1,574 → ~3,800 lines (comprehensive but organized)
- Each manager: 300-600 lines (focused, readable)

---

## Next Steps

1. **Immediate:** Integrate AsyncCheckpointSaver into CheckpointManager
2. **Short-term:** Enhance MonitoringManager with W&B setup
3. **Mid-term:** Refactor SimplifiedEnhancedTrainer methods
4. **Long-term:** Refactor main train.py script to use managers
5. **Final:** Comprehensive testing and documentation updates

---

## Usage Example

After refactoring, users can:

```python
# Initialize managers
context = TrainingContext(config, device, dtype, rank, world_size)
models_mgr = ModelManager(context)
data_mgr = DataLoaderManager(context)
optim_mgr = OptimizerManager(context)

# Create components
model, tokenizer = models_mgr.create_model_and_tokenizer(config_dict, training_config)
train_loader, val_loader = data_mgr.create_dataloaders(training_config, tokenizer, config_dict)
optimizer, lr_mgr = optim_mgr.setup_optimizer_and_lr(model, config_dict, training_config)

# Use with trainer
trainer = SimplifiedEnhancedTrainer(context, model, optimizer, checkpoint_mgr, loss_mgr, monitoring_mgr)
trainer.train(train_loader, val_loader, num_epochs=10)
```

All complex setup is handled by managers, main script stays clean and focused.
