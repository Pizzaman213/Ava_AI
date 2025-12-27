"""
DeepSpeed utilities for Ava training framework.

Provides helper functions for:
- Engine detection
- Checkpoint path resolution
- State gathering (ZeRO-3)
- Compatibility checks
"""

from typing import Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
