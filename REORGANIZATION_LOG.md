# Codebase Reorganization Log

## Date: 2026-01-01

## Summary of Changes

### New Structure
```
code/src/ava/
├── config/          # Configuration (unchanged)
├── core/            # Core utilities (unchanged)
├── cuda/            # CUDA + Kernels (merged)
├── data/            # Data loading (unchanged)
├── models/          # Models + NN layers (merged)
├── optimizations/   # Optimizations + LR managers (merged)
└── training/        # Training + Eval + Strategies (merged)
```

### Removed Directories (merged into others)
- `nn/` → merged into `models/`
- `kernels/` → merged into `cuda/`
- `eval/` → merged into `training/`
- `strategies/` → merged into `training/`
- `optim/` → merged into `optimizations/`

---

## Detailed Path Changes

### MERGE 1: nn/ → models/
| Old Path | New Path |
|----------|----------|
| `code/src/ava/nn/experts.py` | `code/src/ava/models/experts.py` |
| `code/src/ava/nn/routing.py` | `code/src/ava/models/routing.py` |

### MERGE 2: kernels/ → cuda/
| Old Path | New Path |
|----------|----------|
| `code/src/ava/kernels/activations.py` | `code/src/ava/cuda/kernel_activations.py` |
| `code/src/ava/kernels/fused_experts.py` | `code/src/ava/cuda/fused_experts.py` |
| `code/src/ava/kernels/moe.py` | `code/src/ava/cuda/moe_kernels.py` |

### MERGE 3: eval/ → training/
| Old Path | New Path |
|----------|----------|
| `code/src/ava/eval/coherence.py` | `code/src/ava/training/coherence.py` |

### MERGE 4: strategies/ → training/
| Old Path | New Path |
|----------|----------|
| `code/src/ava/strategies/progressive.py` | `code/src/ava/training/progressive.py` |

### MERGE 5: optim/ → optimizations/
| Old Path | New Path |
|----------|----------|
| `code/src/ava/optim/lr_managers.py` | `code/src/ava/optimizations/lr_managers.py` |

---

## Import Updates Required

The following import statements need to be updated across the codebase:

| Old Import | New Import |
|------------|------------|
| `from ava.nn.experts import ...` | `from ava.models.experts import ...` |
| `from ava.nn.routing import ...` | `from ava.models.routing import ...` |
| `from ava.kernels.activations import ...` | `from ava.cuda.kernel_activations import ...` |
| `from ava.kernels.fused_experts import ...` | `from ava.cuda.fused_experts import ...` |
| `from ava.kernels.moe import ...` | `from ava.cuda.moe_kernels import ...` |
| `from ava.eval.coherence import ...` | `from ava.training.coherence import ...` |
| `from ava.strategies.progressive import ...` | `from ava.training.progressive import ...` |
| `from ava.optim.lr_managers import ...` | `from ava.optimizations.lr_managers import ...` |

---

## Files Updated

### Import Updates Applied To:
1. `code/src/ava/models/moe_layer.py` - Updated nn imports to local
2. `code/src/ava/models/moe.py` - Updated nn imports to local
3. `code/src/ava/models/experts.py` - Updated kernels imports to cuda
4. `code/src/ava/models/routing.py` - Updated kernels imports to cuda
5. `code/src/ava/training/generation.py` - Updated eval.coherence to training.coherence
6. `code/scripts/5_training/finetune.py` - Updated optim and strategies imports
7. `code/tests/test_all.py` - Updated all old package imports
8. `code/docs/08_API_REFERENCE.md` - Updated documentation examples

### __init__.py Files Updated:
1. `code/src/ava/models/__init__.py` - Added experts, routing exports
2. `code/src/ava/cuda/__init__.py` - Added kernel exports
3. `code/src/ava/training/__init__.py` - Added coherence, progressive exports
4. `code/src/ava/optimizations/__init__.py` - Added lr_managers exports

### Directories Removed:
1. `code/src/ava/nn/` - Merged into models/
2. `code/src/ava/kernels/` - Merged into cuda/
3. `code/src/ava/eval/` - Merged into training/
4. `code/src/ava/strategies/` - Merged into training/
5. `code/src/ava/optim/` - Merged into optimizations/

---

## Final Structure

```
code/src/ava/
├── config/           # Configuration (4 files)
│   ├── constants.py
│   ├── training_config.py
│   ├── validator.py
│   └── yaml_loader.py
│
├── core/             # Core utilities (8 files)
│   ├── activations.py
│   ├── checkpoint.py
│   ├── data_utils.py
│   ├── error_tracking.py
│   ├── logging.py
│   ├── mixed_precision.py
│   ├── paths.py
│   ├── script_utils.py
│   └── wandb_logger.py
│
├── cuda/             # CUDA + Kernels (7 files)
│   ├── buffers.py
│   ├── fused_experts.py      # (from kernels/)
│   ├── kernel_activations.py # (from kernels/)
│   ├── metrics.py
│   ├── moe_kernels.py        # (from kernels/)
│   ├── profiler.py
│   └── streams.py
│
├── data/             # Data loading (11 files)
│   └── ... (unchanged)
│
├── models/           # Models + NN layers (6 files)
│   ├── experts.py            # (from nn/)
│   ├── moe.py
│   ├── moe_layer.py
│   └── routing.py            # (from nn/)
│
├── optimizations/    # Optimizations + LR (9 files)
│   ├── batch_controller.py
│   ├── checkpointing.py
│   ├── fp8.py
│   ├── gradients.py
│   ├── hybrid_cache.py
│   ├── lr_managers.py        # (from optim/)
│   ├── oom_recovery.py
│   ├── overlapped_recomputation.py
│   ├── prefetch.py
│   └── quantization.py
│
└── training/         # Training + Eval + Strategies (15 files)
    ├── coherence.py          # (from eval/)
    ├── context.py
    ├── data_manager.py
    ├── deepspeed.py
    ├── diagnostics.py
    ├── distributed.py
    ├── generation.py
    ├── loop.py
    ├── model_builder.py
    ├── optimizer.py
    ├── pipeline.py
    ├── progressive.py        # (from strategies/)
    ├── run_manager.py
    ├── state_guard.py
    └── validation.py
```

## Summary

- **Directories reduced**: 12 → 7 (removed 5 single-purpose directories)
- **Total files moved**: 8
- **Import updates**: 8 source files + 4 __init__.py files + 1 documentation file
- **All tests should pass with updated imports**

