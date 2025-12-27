"""
DeepSpeed configuration builder for Ava training framework.

Converts Ava YAML config to DeepSpeed JSON format with full support for:
- ZeRO-1/2/3 optimization stages
- CPU offloading (optimizer, parameters)
- NVMe offloading (optimizer, parameters)
- Gradient accumulation
- Mixed precision (FP16/BF16/FP32)
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


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
    deepspeed_cfg = config.get('deepspeed', {})
    training_cfg = config.get('training', {})

    # Base config
    train_batch_size = deepspeed_cfg.get('train_batch_size')
    if train_batch_size is None:
        train_batch_size = (training_cfg.get('batch_size', 32) *
                           training_cfg.get('gradient_accumulation_steps', 1))

    micro_batch_size = deepspeed_cfg.get('micro_batch_size') or training_cfg.get('batch_size', 32)
    grad_accum = deepspeed_cfg.get('gradient_accumulation_steps') or training_cfg.get('gradient_accumulation_steps', 1)

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
    optimizer_cfg = training_cfg.get('optimizer', {})
    if isinstance(optimizer_cfg, dict):
        opt_type = optimizer_cfg.get('type', 'AdamW')
        opt_lr = optimizer_cfg.get('learning_rate') or training_cfg.get('learning_rate', 1e-4)
        opt_betas = optimizer_cfg.get('betas', [0.9, 0.999])
        opt_eps = optimizer_cfg.get('eps', 1e-8)
        opt_wd = optimizer_cfg.get('weight_decay') or training_cfg.get('weight_decay', 0.01)
    else:
        # optimizer is a string like 'adamw'
        opt_type = 'AdamW'
        opt_lr = training_cfg.get('learning_rate', 1e-4)
        opt_betas = [0.9, 0.999]
        opt_eps = 1e-8
        opt_wd = training_cfg.get('weight_decay', 0.01)

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
    total_steps = training_cfg.get('max_steps', training_cfg.get('total_steps', 100000))

    ds_config['scheduler'] = {
        'type': 'WarmupDecayLR',
        'params': {
            'warmup_min_lr': scheduler_cfg.get('warmup_min_lr', training_cfg.get('min_lr', 0)),
            'warmup_max_lr': opt_lr,
            'warmup_num_steps': warmup_steps,
            'total_num_steps': total_steps,
        }
    }

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
