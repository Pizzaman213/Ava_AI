# Migration Guide: Using New Managers in train.py

This guide shows how to update `/project/code/scripts/5_training/train.py` to use the new modular managers.

## Quick Summary

The refactoring moves ~2,000 lines of functionality from monolithic `train.py` into 4 focused manager classes:
- **DataLoaderManager** - Data loading and format detection
- **ModelManager** - Model creation and initialization
- **OptimizerManager** - Optimizer setup and learning rate management
- **EvaluationManager** - Model evaluation and testing

Result: train.py shrinks from 5,022 to ~400-500 lines

---

## Step 1: Update Imports

**Remove from train.py:**
- `def materialize_meta_model()`
- `def create_model_and_tokenizer()`
- `def create_dataloaders()`
- `def enhanced_format_detection()`
- `def setup_optimizer_and_lr_management()`
- `def test_generation_quality()`
- `def evaluate_model()`
- `def resume_smoke_test()`

**Add to train.py:**
```python
from src.Ava.training.train.base import TrainingContext
from src.Ava.training.train.data_loader_manager import DataLoaderManager
from src.Ava.training.train.model_manager import ModelManager
from src.Ava.training.train.optimizer_manager import OptimizerManager
from src.Ava.training.train.evaluation_manager import EvaluationManager
```

---

## Step 2: Create TrainingContext in main()

```python
# Create training context
context = TrainingContext(
    config=training_config,
    raw_config_dict=config_dict,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.bfloat16,
    rank=int(os.environ.get("RANK", 0)),
    world_size=int(os.environ.get("WORLD_SIZE", 1)),
)
```

---

## Step 3: Replace Function Calls

### Model and Tokenizer Creation

**Before:**
```python
model, tokenizer = create_model_and_tokenizer(config_dict, training_config)
```

**After:**
```python
model_manager = ModelManager(context)
model, tokenizer = model_manager.create_model_and_tokenizer(config_dict, training_config)
```

### Dataloader Creation

**Before:**
```python
train_loader, val_loader = create_dataloaders(
    training_config, tokenizer, config_dict, batch_size
)
```

**After:**
```python
data_loader_manager = DataLoaderManager(context)
train_loader, val_loader = data_loader_manager.create_dataloaders(
    training_config, tokenizer, config_dict, batch_size
)
```

### Optimizer Setup

**Before:**
```python
optimizer, adaptive_lr_manager = setup_optimizer_and_lr_management(
    model, config_dict, training_config, total_steps
)
```

**After:**
```python
optimizer_manager = OptimizerManager(context)
optimizer, adaptive_lr_manager = optimizer_manager.setup_optimizer_and_lr(
    model, config_dict, training_config, total_steps
)
```

### Evaluation

**Before:**
```python
gen_results = test_generation_quality(model, tokenizer, device, test_prompts)
val_loss, val_perplexity = evaluate_model(model, val_loader, device, use_bf16=True)
smoke_passed = resume_smoke_test(model, train_loader, device, num_steps=3)
```

**After:**
```python
evaluation_manager = EvaluationManager(context)
gen_results = evaluation_manager.test_generation_quality(
    model, tokenizer, context.device, test_prompts
)
val_loss, val_perplexity = evaluation_manager.evaluate_model(
    model, val_loader, context.device, use_bf16=True
)
smoke_passed = evaluation_manager.resume_smoke_test(
    model, train_loader, context.device, num_steps=3
)
```

---

## Complete Refactored main() Example

```python
def main(config_path: str, batch_size: Optional[int] = None, learning_rate: Optional[float] = None):
    """Main training orchestration using managers (slim version)."""

    # Setup logging
    setup_training_logger()
    logger = get_logger()

    # Load config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    training_config = TrainingConfigManager.from_dict(config_dict)

    # Apply overrides
    if batch_size is not None:
        training_config.training.batch_size = batch_size
    if learning_rate is not None:
        training_config.training.learning_rate = learning_rate

    # Create context
    context = TrainingContext(
        config=training_config,
        raw_config_dict=config_dict,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.bfloat16,
        rank=int(os.environ.get("RANK", 0)),
        world_size=int(os.environ.get("WORLD_SIZE", 1)),
    )

    # Initialize managers
    data_loader_manager = DataLoaderManager(context)
    model_manager = ModelManager(context)
    optimizer_manager = OptimizerManager(context)
    evaluation_manager = EvaluationManager(context)

    # Create components
    logger.info("Creating model and tokenizer...")
    model, tokenizer = model_manager.create_model_and_tokenizer(config_dict, training_config)

    logger.info("Creating dataloaders...")
    train_loader, val_loader = data_loader_manager.create_dataloaders(
        training_config, tokenizer, config_dict, batch_size
    )

    # Calculate total steps
    samples_per_epoch = training_config.data.buffer_size or 10000
    batch_size_effective = training_config.training.batch_size * \
                          getattr(training_config.training, 'gradient_accumulation_steps', 1)
    steps_per_epoch = samples_per_epoch // batch_size_effective
    total_steps = steps_per_epoch * training_config.training.num_epochs

    logger.info("Setting up optimizer...")
    optimizer, adaptive_lr_manager = optimizer_manager.setup_optimizer_and_lr(
        model, config_dict, training_config, total_steps
    )

    # Quick validation tests
    logger.info("Running initialization tests...")
    test_prompts = ["The future of AI is", "Machine learning helps", "Neural networks can"]
    gen_results = evaluation_manager.test_generation_quality(
        model, tokenizer, context.device, test_prompts, max_length=50
    )
    logger.info(f"Generated {len(gen_results['generated_texts'])} samples")

    val_loss, val_perplexity = evaluation_manager.evaluate_model(
        model, val_loader, context.device, use_bf16=True, max_batches=10
    )
    if val_loss is not None:
        logger.info(f"Validation loss: {val_loss:.4f}")

    smoke_passed = evaluation_manager.resume_smoke_test(
        model, train_loader, context.device, num_steps=3
    )

    logger.info("✅ All components initialized!")

    return {
        "context": context,
        "model": model,
        "tokenizer": tokenizer,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "adaptive_lr_manager": adaptive_lr_manager,
    }
```

---

## Benefits

✅ **Smaller train.py:** 5,022 → ~400 lines (90% reduction)
✅ **Better maintainability:** Each manager handles one responsibility
✅ **Easier testing:** Managers can be tested independently
✅ **Improved reusability:** Managers can be used in other scripts
✅ **Backward compatible:** Works with existing YAML configs

---

## See Also

- `/project/REFACTORING_SUMMARY.md` - Complete refactoring details
- `/project/code/scripts/5_training/train_refactored_example.py` - Full working example
- `DataLoaderManager` docstring - Usage details
- `ModelManager` docstring - Usage details
- `OptimizerManager` docstring - Usage details
- `EvaluationManager` docstring - Usage details
