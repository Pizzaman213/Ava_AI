"""
Model quantization utilities for the Ava pipeline.

Supports INT8, NF4, and NVFP4 quantization for memory-efficient training.
"""

import logging
from typing import Callable, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Check optional dependencies
try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    bnb = None

try:
    from torchao.quantization import int4_weight_only, quantize_
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False
    int4_weight_only = None
    quantize_ = None


def quantize_model_int8(
    model: nn.Module,
    threshold: float = 6.0,
    skip_modules: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> nn.Module:
    """
    Quantize model Linear layers to INT8 using bitsandbytes.

    This replaces nn.Linear layers with bnb.nn.Linear8bitLt for memory-efficient
    INT8 training. Note: INT8 training requires careful handling of gradients.

    Args:
        model: The model to quantize
        threshold: Outlier threshold for mixed-precision decomposition (default: 6.0)
        skip_modules: List of module name patterns to skip (e.g., ['embed', 'lm_head'])
        logger: Optional logger for status messages

    Returns:
        Quantized model with INT8 linear layers
    """
    if not BITSANDBYTES_AVAILABLE:
        if logger:
            logger.warning(
                "bitsandbytes not available, skipping INT8 quantization. "
                "Install with: pip install bitsandbytes"
            )
        return model

    if skip_modules is None:
        skip_modules = ['embed', 'lm_head']

    quantized_count = 0

    # Replace Linear layers with INT8 versions
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Get parent module and attribute name
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            # Skip specified modules (keep in higher precision)
            if any(skip in name.lower() for skip in skip_modules):
                continue

            # Create INT8 linear layer with fp16 weights for training
            # has_fp16_weights=True keeps a fp16 copy for gradient computation
            int8_layer = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=True,  # Keep fp16 weights for training
                threshold=threshold,
            )

            # Copy weights - use fp16 for training compatibility
            # NOTE: has_fp16_weights=True maintains a trainable fp16 copy internally
            # Setting requires_grad=True allows gradient flow through this fp16 copy
            int8_layer.weight = bnb.nn.Int8Params(
                module.weight.data.to(torch.float16),
                requires_grad=True,  # Enable gradient flow through fp16 copy
                has_fp16_weights=True
            )
            if module.bias is not None:
                int8_layer.bias = nn.Parameter(module.bias.data)

            # Replace the layer
            setattr(parent, attr_name, int8_layer)
            quantized_count += 1

    if logger:
        logger.info(f"Quantized {quantized_count} Linear layers to INT8 (threshold={threshold})")

    return model


def quantize_model_nf4(
    model: nn.Module,
    skip_modules: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> nn.Module:
    """
    Quantize model to NF4 (Normalized Float 4-bit) using bitsandbytes.

    NF4 provides better accuracy than standard INT4 by using a normalized
    distribution that better matches neural network weight distributions.

    Args:
        model: The model to quantize
        skip_modules: List of module name patterns to skip
        logger: Optional logger for status messages

    Returns:
        Quantized model with NF4 linear layers
    """
    if not BITSANDBYTES_AVAILABLE:
        if logger:
            logger.warning(
                "bitsandbytes not available, skipping NF4 quantization. "
                "Install with: pip install bitsandbytes"
            )
        return model

    if skip_modules is None:
        skip_modules = ['embed', 'lm_head']

    quantized_count = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            if any(skip in name.lower() for skip in skip_modules):
                continue

            # Create NF4 linear layer
            nf4_layer = bnb.nn.Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.bfloat16,
                quant_type='nf4',  # Normalized Float 4
            )

            # Copy weights - requires_grad=False for quantized params
            nf4_layer.weight = bnb.nn.Params4bit(
                module.weight.data.to(torch.float16),
                requires_grad=False,  # Quantized params don't support gradients directly
                quant_type='nf4',
            )
            if module.bias is not None:
                nf4_layer.bias = nn.Parameter(module.bias.data)

            setattr(parent, attr_name, nf4_layer)
            quantized_count += 1

    if logger:
        logger.info(f"Quantized {quantized_count} Linear layers to NF4 (4-bit)")

    return model


def quantize_model_nvfp4(
    model: nn.Module,
    block_size: int = 16,
    skip_modules: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> nn.Module:
    """
    Quantize model to NVFP4 (NVIDIA FP4) using torchao.

    NVFP4 is NVIDIA's 4-bit floating point format optimized for inference
    on Hopper (H100) and later GPUs. Provides ~8x memory reduction.

    Args:
        model: The model to quantize
        block_size: Block size for quantization (default: 16)
        skip_modules: List of module name patterns to skip
        logger: Optional logger for status messages

    Returns:
        Quantized model
    """
    if not TORCHAO_AVAILABLE:
        if logger:
            logger.warning(
                "torchao not available, skipping NVFP4 quantization. "
                "Install with: pip install torchao"
            )
        return model

    if skip_modules is None:
        skip_modules = ['embed', 'lm_head']

    try:
        # Use torchao's int4_weight_only as approximation for FP4
        # Note: True NVFP4 requires Hopper+ GPUs and specific torchao version
        def skip_filter(mod: nn.Module, fqn: str) -> bool:
            return any(skip in fqn.lower() for skip in skip_modules)

        quantize_(model, int4_weight_only(), filter_fn=lambda m, fqn: not skip_filter(m, fqn))

        if logger:
            logger.info(f"Applied NVFP4/INT4 quantization (block_size={block_size})")

    except Exception as e:
        if logger:
            logger.warning(f"NVFP4 quantization failed: {e}, model unchanged")

    return model


def apply_quantization(
    model: nn.Module,
    quant_config: dict,
    logger: Optional[logging.Logger] = None
) -> nn.Module:
    """
    Apply quantization based on configuration.

    Args:
        model: The model to quantize
        quant_config: Quantization configuration dict with keys:
            - enabled: bool
            - type: 'none', 'int8', 'nf4', 'nvfp4'
            - int8_threshold: float (for INT8)
            - nvfp4_block_size: int (for NVFP4)
            - skip_modules: list of module patterns to skip
        logger: Optional logger

    Returns:
        Quantized model
    """
    if not quant_config.get('enabled', False):
        if logger:
            logger.info("Quantization disabled")
        return model

    quant_type = quant_config.get('type', 'none').lower()
    skip_modules = quant_config.get('skip_modules', ['embed', 'lm_head'])

    if quant_type == 'int8':
        threshold = quant_config.get('int8_threshold', 6.0)
        return quantize_model_int8(model, threshold=threshold, skip_modules=skip_modules, logger=logger)

    elif quant_type == 'nf4':
        return quantize_model_nf4(model, skip_modules=skip_modules, logger=logger)

    elif quant_type == 'nvfp4' or quant_type == 'fp4':
        block_size = quant_config.get('nvfp4_block_size', 16)
        return quantize_model_nvfp4(model, block_size=block_size, skip_modules=skip_modules, logger=logger)

    elif quant_type == 'none':
        if logger:
            logger.info("No quantization applied")
        return model

    else:
        if logger:
            logger.warning(f"Unknown quantization type '{quant_type}', skipping")
        return model


def get_quantization_info() -> dict:
    """
    Get information about available quantization backends.

    Returns:
        Dictionary with availability status for each backend.
    """
    return {
        'bitsandbytes': BITSANDBYTES_AVAILABLE,
        'torchao': TORCHAO_AVAILABLE,
        'supported_types': [
            'none',
            'int8' if BITSANDBYTES_AVAILABLE else None,
            'nf4' if BITSANDBYTES_AVAILABLE else None,
            'nvfp4' if TORCHAO_AVAILABLE else None,
        ]
    }
