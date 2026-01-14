"""
DeepSpeed utilities and configuration builder for Ava training framework.

This module provides:
- Engine detection and utilities
- Checkpoint path resolution
- State gathering (ZeRO-3)
- Compatibility checks
- YAML to DeepSpeed JSON config conversion
- ZeRO-1/2/3 optimization support
- CPU/NVMe offloading support
"""

from typing import Any, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# DeepSpeed Engine Utilities
# =============================================================================

def is_deepspeed_engine(model: Any) -> bool:
    """
    Check if model is a DeepSpeed engine.

    Args:
        model: Model or engine to check

    Returns:
        True if model is DeepSpeed engine

    Example:
        >>> if is_deepspeed_engine(model):
        ...     model.backward(loss)  # Use engine methods
        ... else:
        ...     loss.backward()  # Use standard PyTorch
    """
    return (
        hasattr(model, 'save_checkpoint') and
        hasattr(model, 'load_checkpoint') and
        hasattr(model, 'zero_optimization_stage') and
        hasattr(model, 'backward') and
        hasattr(model, 'step')
    )


def get_zero_stage(model: Any) -> int:
    """
    Get ZeRO optimization stage from DeepSpeed engine.

    Args:
        model: DeepSpeed engine

    Returns:
        ZeRO stage (0, 1, 2, or 3)

    Example:
        >>> stage = get_zero_stage(model)
        >>> if stage == 3:
        ...     # Use ZeRO-3 specific logic
        ...     pass
    """
    if not is_deepspeed_engine(model):
        return 0

    return model.zero_optimization_stage()


def is_zero3(model: Any) -> bool:
    """
    Check if model uses ZeRO-3 optimization.

    Args:
        model: DeepSpeed engine

    Returns:
        True if using ZeRO-3

    Example:
        >>> if is_zero3(model):
        ...     # Gather parameters for saving
        ...     with deepspeed.zero.GatheredParameters(model.module.parameters()):
        ...         state_dict = model.module.state_dict()
    """
    return get_zero_stage(model) == 3


def resolve_checkpoint_path(
    checkpoint_path: Path,
    prefer_deepspeed: bool = True,
) -> Path:
    """
    Resolve checkpoint path (handles both DeepSpeed and PyTorch formats).

    Args:
        checkpoint_path: Path to checkpoint (dir or .pt file)
        prefer_deepspeed: If True, prefer DeepSpeed format when both exist

    Returns:
        Resolved checkpoint path

    Example:
        >>> path = resolve_checkpoint_path(Path('checkpoints/latest_model.pt'))
        >>> # Returns DeepSpeed dir if exists, else PyTorch file
    """
    if checkpoint_path.is_dir():
        # DeepSpeed format - return as-is
        return checkpoint_path
    elif checkpoint_path.suffix == '.pt':
        # PyTorch format
        if prefer_deepspeed:
            # Check if corresponding DeepSpeed checkpoint exists
            # Convert checkpoint_epoch_X_step_Y.pt → deepspeed_epoch_X_step_Y/
            ds_dir = checkpoint_path.parent / checkpoint_path.stem.replace('checkpoint_', 'deepspeed_')
            if ds_dir.is_dir():
                logger.info(f"Found DeepSpeed checkpoint: {ds_dir}")
                return ds_dir

        return checkpoint_path
    else:
        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")


def check_deepspeed_available() -> bool:
    """
    Check if DeepSpeed is available.

    Returns:
        True if DeepSpeed can be imported

    Example:
        >>> if not check_deepspeed_available():
        ...     raise RuntimeError("DeepSpeed not installed")
    """
    try:
        import deepspeed
        return True
    except ImportError:
        return False


def get_deepspeed_version() -> Optional[str]:
    """
    Get DeepSpeed version string.

    Returns:
        Version string or None if not installed

    Example:
        >>> version = get_deepspeed_version()
        >>> print(f"DeepSpeed {version}")
    """
    try:
        import deepspeed
        return deepspeed.__version__
    except ImportError:
        return None


def log_deepspeed_info(engine: Any, rank: int = 0) -> None:
    """
    Log DeepSpeed engine configuration info.

    Args:
        engine: DeepSpeed engine
        rank: Process rank (only rank 0 logs)

    Example:
        >>> log_deepspeed_info(model_engine, rank=0)
        # Logs ZeRO stage, batch size, precision, offloading, etc.
    """
    if rank != 0:
        return

    if not is_deepspeed_engine(engine):
        logger.warning("Not a DeepSpeed engine - cannot log info")
        return

    zero_stage = engine.zero_optimization_stage()

    # Extract config
    ds_config = engine.config

    logger.info("="*60)
    logger.info("DeepSpeed Engine Configuration")
    logger.info("="*60)
    logger.info(f"ZeRO Stage: {zero_stage}")
    logger.info(f"Train Batch Size: {ds_config.get('train_batch_size')}")
    logger.info(f"Micro Batch Size: {ds_config.get('train_micro_batch_size_per_gpu')}")
    logger.info(f"Gradient Accumulation: {ds_config.get('gradient_accumulation_steps')}")

    # Precision
    if ds_config.get('fp16', {}).get('enabled'):
        logger.info(f"Precision: FP16")
    elif ds_config.get('bf16', {}).get('enabled'):
        logger.info(f"Precision: BF16")
    else:
        logger.info(f"Precision: FP32")

    # Offloading
    zero_opt = ds_config.get('zero_optimization', {})
    offload_opt = zero_opt.get('offload_optimizer')
    offload_param = zero_opt.get('offload_param')

    if offload_opt:
        device = offload_opt.get('device', 'cpu')
        logger.info(f"Optimizer Offload: {device}")
        if device == 'nvme':
            logger.info(f"  NVMe Path: {offload_opt.get('nvme_path')}")

    if offload_param:
        device = offload_param.get('device', 'cpu')
        logger.info(f"Parameter Offload: {device}")
        if device == 'nvme':
            logger.info(f"  NVMe Path: {offload_param.get('nvme_path')}")

    logger.info("="*60)


def extract_deepspeed_tag_from_dir(checkpoint_dir: Path) -> str:
    """
    Extract tag from DeepSpeed checkpoint directory.

    Args:
        checkpoint_dir: DeepSpeed checkpoint directory

    Returns:
        Tag string (e.g., "epoch1_step1000")

    Example:
        >>> tag = extract_deepspeed_tag_from_dir(Path('deepspeed_epoch_1_step_1000'))
        >>> # Returns: "epoch1_step1000"
    """
    # DeepSpeed saves with structure: checkpoint_dir/tag/...
    # Try to find tag from metadata or directory name
    tag_file = checkpoint_dir / 'latest'
    if tag_file.exists():
        return tag_file.read_text().strip()

    # Fallback: extract from directory name
    # Format: deepspeed_epoch_X_step_Y → tag: epochX_stepY
    name = checkpoint_dir.name
    if 'epoch' in name and 'step' in name:
        parts = name.split('_')
        try:
            epoch_idx = parts.index('epoch') + 1
            step_idx = parts.index('step') + 1
            return f"epoch{parts[epoch_idx]}_step{parts[step_idx]}"
        except (ValueError, IndexError):
            pass

    return "latest"


def gather_deepspeed_state_dict(engine: Any) -> dict:
    """
    Gather full model state from DeepSpeed engine (handles ZeRO-3 sharding).

    For ZeRO-3, parameters are sharded across ranks. This function gathers them.

    Args:
        engine: DeepSpeed engine

    Returns:
        Full model state_dict (all parameters gathered)

    Example:
        >>> state_dict = gather_deepspeed_state_dict(model_engine)
        >>> torch.save(state_dict, 'checkpoint.pt')
    """
    import deepspeed

    zero_stage = engine.zero_optimization_stage()

    if zero_stage == 3:
        # ZeRO-3: Parameters are sharded - use DeepSpeed API to gather
        with deepspeed.zero.GatheredParameters(engine.module.parameters()):
            state_dict = engine.module.state_dict()
    else:
        # ZeRO-1/2: Parameters are replicated on all ranks
        state_dict = engine.module.state_dict()

    return state_dict


# =============================================================================
# DeepSpeed Configuration Builder
# =============================================================================

def build_deepspeed_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build DeepSpeed JSON config from Ava YAML config.

    Args:
        config: Full Ava config dictionary

    Returns:
        DeepSpeed config dictionary (ready for deepspeed.initialize())

    Example:
        >>> config = load_yaml_with_path_resolution('config.yaml')
        >>> ds_config = build_deepspeed_config(config)
        >>> model_engine, optimizer, _, scheduler = deepspeed.initialize(
        ...     model=model, config=ds_config
        ... )
    """
    # Support both v2.0 path (distributed.deepspeed) and legacy path (deepspeed)
    distributed_cfg = config.get('distributed', {})
    deepspeed_cfg = distributed_cfg.get('deepspeed', {})
    if not deepspeed_cfg:
        deepspeed_cfg = config.get('deepspeed', {})

    training_cfg = config.get('training', {})

    # Get batch configuration from nested structure (v2.0) or flat structure (legacy)
    batching_cfg = training_cfg.get('batching', {})
    batch_size = int(batching_cfg.get('batch_size') or training_cfg.get('batch_size', 32))

    # Gradient accumulation steps - check DeepSpeed config first, then training config
    grad_accum = int(
        deepspeed_cfg.get('gradient_accumulation_steps') or
        batching_cfg.get('gradient_accumulation_steps') or
        training_cfg.get('gradient_accumulation_steps', 1)
    )

    # Micro batch size is the per-GPU batch size
    micro_batch_size = int(deepspeed_cfg.get('micro_batch_size') or batch_size)

    # train_batch_size = micro_batch_size * grad_accum * world_size
    # DeepSpeed does NOT support 'auto' for train_batch_size - it must be an integer
    train_batch_size_raw = deepspeed_cfg.get('train_batch_size')
    if train_batch_size_raw is not None:
        train_batch_size = int(train_batch_size_raw)
    else:
        # Calculate based on world_size (get from torch.distributed if available)
        import torch.distributed as dist
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        train_batch_size = micro_batch_size * grad_accum * world_size
        logger.info(
            f"DeepSpeed train_batch_size calculated: {train_batch_size} "
            f"(micro_batch_size={micro_batch_size} × grad_accum={grad_accum} × world_size={world_size})"
        )

    ds_config = {
        'train_batch_size': train_batch_size,
        'train_micro_batch_size_per_gpu': micro_batch_size,
        'gradient_accumulation_steps': grad_accum,
        'steps_per_print': 100,
        'wall_clock_breakdown': deepspeed_cfg.get('wall_clock_breakdown', False),
    }

    # Gradient clipping
    grad_clip = deepspeed_cfg.get('gradient_clipping') or training_cfg.get('max_gradient_norm') or training_cfg.get('max_grad_norm')
    if grad_clip:
        ds_config['gradient_clipping'] = grad_clip

    # Mixed precision (FP16/BF16)
    precision = deepspeed_cfg.get('precision_type', training_cfg.get('mixed_precision', 'fp16'))
    if precision == 'fp16':
        ds_config['fp16'] = {
            'enabled': True,
            'loss_scale': 0,  # Dynamic loss scaling
            'initial_scale_power': 16,
            'loss_scale_window': 1000,
            'hysteresis': 2,
            'min_loss_scale': 1,
        }
    elif precision == 'bf16':
        ds_config['bf16'] = {'enabled': True}

    # ZeRO optimization
    zero_stage = deepspeed_cfg.get('zero_stage', 2)
    ds_config['zero_optimization'] = build_zero_config(deepspeed_cfg, zero_stage)

    # Activation checkpointing
    if deepspeed_cfg.get('activation_checkpointing', False):
        ds_config['activation_checkpointing'] = {
            'partition_activations': deepspeed_cfg.get('partition_activations', False),
            'cpu_checkpointing': deepspeed_cfg.get('cpu_checkpointing', False),
            'contiguous_memory_optimization': deepspeed_cfg.get('contiguous_memory_optimization', False),
            'synchronize_checkpoint_boundary': deepspeed_cfg.get('synchronize_checkpoint_boundary', False),
        }

    # Optimizer (DeepSpeed manages it)
    # Map unsupported optimizer types to DeepSpeed equivalents
    # 8-bit optimizers (bitsandbytes) are not supported by DeepSpeed
    DEEPSPEED_OPTIMIZER_MAP = {
        'adamw_8bit': 'AdamW',
        'adam_8bit': 'Adam',
        'adamw': 'AdamW',
        'adam': 'Adam',
        'sgd': 'SGD',
    }

    optimizer_cfg = training_cfg.get('optimizer', {})
    if isinstance(optimizer_cfg, dict):
        opt_type = optimizer_cfg.get('type', 'AdamW')
        opt_lr = optimizer_cfg.get('learning_rate') or training_cfg.get('learning_rate', 1e-4)
        opt_betas = optimizer_cfg.get('betas', [0.9, 0.999])
        opt_eps = optimizer_cfg.get('eps', 1e-8)
        opt_wd = optimizer_cfg.get('weight_decay') or training_cfg.get('weight_decay', 0.01)
    else:
        # optimizer is a string like 'adamw'
        opt_type = str(optimizer_cfg) if optimizer_cfg else 'AdamW'
        opt_lr = training_cfg.get('learning_rate', 1e-4)
        opt_betas = [0.9, 0.999]
        opt_eps = 1e-8
        opt_wd = training_cfg.get('weight_decay', 0.01)

    # Normalize to DeepSpeed-compatible optimizer type
    original_opt_type = opt_type
    opt_type_lower = opt_type.lower()
    if opt_type_lower in DEEPSPEED_OPTIMIZER_MAP:
        opt_type = DEEPSPEED_OPTIMIZER_MAP[opt_type_lower]
        if '8bit' in opt_type_lower:
            logger.warning(
                f"Optimizer '{original_opt_type}' is not compatible with DeepSpeed. "
                f"Using '{opt_type}' instead. Note: ZeRO optimization already provides "
                f"significant memory savings similar to 8-bit optimizers."
            )

    ds_config['optimizer'] = {
        'type': opt_type,
        'params': {
            'lr': opt_lr,
            'betas': opt_betas,
            'eps': opt_eps,
            'weight_decay': opt_wd,
        }
    }

    # Scheduler (DeepSpeed manages it)
    scheduler_cfg = training_cfg.get('schedule', {}) or training_cfg.get('scheduler', {})
    warmup_steps = scheduler_cfg.get('warmup_steps', training_cfg.get('warmup_steps', 1000))

    # FIX: Calculate total_steps properly instead of using broken default
    # Priority: 1) explicit max_steps, 2) explicit total_steps, 3) estimate from epochs
    total_steps = training_cfg.get('max_steps') or scheduler_cfg.get('total_steps')

    if total_steps is None:
        # Estimate from num_epochs - use reasonable defaults
        num_epochs = scheduler_cfg.get('num_epochs', training_cfg.get('num_epochs', 1))

        # Check for explicit steps_per_epoch first
        steps_per_epoch = scheduler_cfg.get('steps_per_epoch') or training_cfg.get('steps_per_epoch')

        if steps_per_epoch is None:
            # Use a conservative fallback and warn the user
            logger.warning(
                "DeepSpeed scheduler: steps_per_epoch not set. "
                "Consider setting 'training.schedule.max_steps' or 'training.schedule.steps_per_epoch' explicitly. "
                "Using fallback estimate of 10,000 steps/epoch."
            )
            steps_per_epoch = 10000

        # Get gradient accumulation for calculation
        batching_cfg = training_cfg.get('batching', {})
        grad_accum_for_calc = (
            batching_cfg.get('gradient_accumulation_steps') or
            training_cfg.get('gradient_accumulation_steps', 1)
        )

        # Calculate: total_steps = num_epochs * steps_per_epoch / grad_accum
        total_steps = (num_epochs * steps_per_epoch) // max(grad_accum_for_calc, 1)

        logger.info(
            f"DeepSpeed scheduler: estimated total_steps={total_steps} "
            f"(epochs={num_epochs}, steps_per_epoch={steps_per_epoch}, grad_accum={grad_accum_for_calc})"
        )

    # Get min_lr for end of training
    min_lr = scheduler_cfg.get('min_lr', training_cfg.get('min_lr', opt_lr * 0.1))

    ds_config['scheduler'] = {
        'type': 'WarmupDecayLR',
        'params': {
            'warmup_min_lr': min_lr,  # End at min_lr, not 0
            'warmup_max_lr': opt_lr,
            'warmup_num_steps': warmup_steps,
            'total_num_steps': total_steps,
        }
    }

    logger.info(f"DeepSpeed scheduler config: warmup={warmup_steps}, total={total_steps}, "
                f"lr={opt_lr:.2e} -> {min_lr:.2e}")

    return ds_config


def build_zero_config(deepspeed_cfg: Dict[str, Any], zero_stage: int) -> Dict[str, Any]:
    """
    Build ZeRO optimization config for given stage.

    Supports:
    - ZeRO-1: Optimizer state sharding
    - ZeRO-2: Optimizer + gradient sharding
    - ZeRO-3: Optimizer + gradient + parameter sharding
    - CPU offloading for optimizer/parameters
    - NVMe offloading for optimizer/parameters

    Args:
        deepspeed_cfg: DeepSpeed section from Ava config
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3)

    Returns:
        ZeRO configuration dictionary
    """
    zero_config = {
        'stage': zero_stage,
        'allgather_partitions': deepspeed_cfg.get('allgather_partitions', True),
        'allgather_bucket_size': deepspeed_cfg.get('zero_allgather_bucket_size', 500000000),
        'overlap_comm': deepspeed_cfg.get('zero_overlap_comm', True),
        'reduce_scatter': deepspeed_cfg.get('zero_reduce_scatter', True),
        'reduce_bucket_size': deepspeed_cfg.get('zero_reduce_bucket_size', 500000000),
        'contiguous_gradients': deepspeed_cfg.get('zero_contiguous_gradients', True),
    }

    # CPU offloading
    cpu_offload = deepspeed_cfg.get('cpu_offload', False)
    if cpu_offload:
        if zero_stage >= 2:
            # ZeRO-2/3: Can offload optimizer
            zero_config['offload_optimizer'] = {
                'device': 'cpu',
                'pin_memory': True,
            }

        if zero_stage == 3:
            # ZeRO-3: Can also offload parameters
            zero_config['offload_param'] = {
                'device': 'cpu',
                'pin_memory': True,
            }

    # NVMe offloading (ZeRO-Infinity)
    nvme_offload = deepspeed_cfg.get('nvme_offload', False)
    nvme_path = deepspeed_cfg.get('nvme_path', '/local_nvme')

    if nvme_offload:
        if zero_stage >= 2:
            # Offload optimizer to NVMe
            zero_config['offload_optimizer'] = {
                'device': 'nvme',
                'nvme_path': nvme_path,
                'pin_memory': True,
                'buffer_count': 4,
                'fast_init': False,
            }

        if zero_stage == 3:
            # Offload parameters to NVMe
            zero_config['offload_param'] = {
                'device': 'nvme',
                'nvme_path': nvme_path,
                'pin_memory': True,
                'buffer_count': 5,
                'buffer_size': int(1e8),
                'max_in_cpu': int(1e9),
            }

    # ZeRO-3 specific settings
    if zero_stage == 3:
        zero_config['stage3_prefetch_bucket_size'] = deepspeed_cfg.get(
            'zero_stage3_prefetch_bucket_size', 500000000
        )
        zero_config['stage3_param_persistence_threshold'] = deepspeed_cfg.get(
            'zero_stage3_param_persistence_threshold', 1000000
        )
        zero_config['stage3_max_live_parameters'] = deepspeed_cfg.get(
            'zero_stage3_max_live_parameters', 1000000000
        )
        zero_config['stage3_max_reuse_distance'] = deepspeed_cfg.get(
            'zero_stage3_max_reuse_distance', 1000000000
        )
        zero_config['stage3_gather_16bit_weights_on_model_save'] = True

    return zero_config


def validate_deepspeed_config(ds_config: Dict[str, Any]) -> None:
    """
    Validate DeepSpeed config for common issues.

    Args:
        ds_config: DeepSpeed configuration dictionary

    Raises:
        ValueError: If config has incompatible settings
    """
    zero_stage = ds_config.get('zero_optimization', {}).get('stage', 0)

    # Check batch size consistency
    train_bs = ds_config.get('train_batch_size')
    micro_bs = ds_config.get('train_micro_batch_size_per_gpu')
    grad_accum = ds_config.get('gradient_accumulation_steps', 1)

    if train_bs and micro_bs and grad_accum:
        # Note: train_batch_size = micro_batch_size * grad_accum * world_size
        # We can't check world_size here, so just warn if mismatch
        if train_bs != micro_bs * grad_accum:
            logger.warning(
                f"DeepSpeed batch size mismatch: train_batch_size={train_bs} != "
                f"micro_batch_size_per_gpu={micro_bs} * gradient_accumulation_steps={grad_accum}. "
                f"This may be intentional if using multiple GPUs."
            )

    # Check ZeRO-3 + activation checkpointing compatibility
    if zero_stage == 3:
        activation_checkpointing = ds_config.get('activation_checkpointing', {}).get('enabled', False)
        if activation_checkpointing:
            logger.info("ZeRO-3 + activation checkpointing enabled (good for large models)")

    # Check offloading compatibility
    zero_opt = ds_config.get('zero_optimization', {})
    offload_optimizer = zero_opt.get('offload_optimizer')
    offload_param = zero_opt.get('offload_param')

    if offload_optimizer and zero_stage < 2:
        raise ValueError("Optimizer offloading requires ZeRO stage 2 or higher")

    if offload_param and zero_stage < 3:
        raise ValueError("Parameter offloading requires ZeRO stage 3")

    logger.info(f"DeepSpeed config validated: ZeRO-{zero_stage}")


def validate_model_for_deepspeed(model: Any, config: Dict[str, Any]) -> None:
    """
    Validate model configuration for DeepSpeed compatibility.

    Args:
        model: The PyTorch model
        config: Full Ava config dictionary

    Raises:
        ValueError: If model has incompatible settings for DeepSpeed
    """
    # Check for tie_word_embeddings which is incompatible with ZeRO
    # Weight tying causes "parameter has already been reduced" errors
    # because DeepSpeed tries to reduce the shared parameter twice
    model_cfg = config.get('model', {})
    tie_word_embeddings = model_cfg.get('tie_word_embeddings', False)

    # Also check the model directly if it has the attribute
    if hasattr(model, '_tie_word_embeddings'):
        tie_word_embeddings = model._tie_word_embeddings
    elif hasattr(model, 'module') and hasattr(model.module, '_tie_word_embeddings'):
        tie_word_embeddings = model.module._tie_word_embeddings

    if tie_word_embeddings:
        raise ValueError(
            "tie_word_embeddings=True is INCOMPATIBLE with DeepSpeed ZeRO.\n"
            "Weight tying causes 'The parameter X has already been reduced' errors "
            "because DeepSpeed tries to reduce the shared parameter twice.\n"
            "Fix: Set 'model.tie_word_embeddings: false' in your config file."
        )

    # Warn about gradient checkpointing compatibility
    # PyTorch's gradient checkpointing with use_reentrant=True causes the same issue
    if model_cfg.get('gradient_checkpointing', False):
        logger.info(
            "gradient_checkpointing + DeepSpeed ZeRO detected. "
            "Ensure use_reentrant=False in torch.utils.checkpoint.checkpoint() calls "
            "to avoid 'parameter already reduced' errors."
        )
