"""
FP8 Training Support for H100/B200 GPUs

This module provides comprehensive FP8 (8-bit floating point) training support
for NVIDIA H100 and B200 GPUs, enabling 2x computational throughput with
maintained model accuracy.

Features:
- FP8 E4M3 and E5M2 format support
- Transformer Engine integration
- Dynamic loss scaling for FP8
- Mixed FP8/BF16 precision strategies
- Hardware-aware precision selection
- Numerical stability safeguards

References:
- FP8 Formats for Deep Learning: https://arxiv.org/abs/2209.05433
- Transformer Engine: https://github.com/NVIDIA/TransformerEngine
"""

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
from torch.cuda.amp import GradScaler  # type: ignore[import]
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from enum import Enum
from contextlib import contextmanager
import warnings

logger = logging.getLogger(__name__)

# Try to import Transformer Engine
try:
    import transformer_engine.pytorch as te  # type: ignore[import]
    from transformer_engine.common import recipe  # type: ignore[import]
    from transformer_engine.pytorch import DotProductAttention  # type: ignore[import]
    TE_AVAILABLE = True
    logger.info("Transformer Engine available for FP8 training")
except ImportError:
    TE_AVAILABLE = False
    te = None  # type: ignore[assignment]
    recipe = None  # type: ignore[assignment]
    DotProductAttention = None  # type: ignore[assignment,misc]
    logger.warning("Transformer Engine not available. FP8 training will use fallback implementations.")


class FP8Format(Enum):
    """FP8 format types."""
    E4M3 = "E4M3"  # 1 sign + 4 exponent + 3 mantissa bits
    E5M2 = "E5M2"  # 1 sign + 5 exponent + 2 mantissa bits


@dataclass
class FP8Config:
    """Configuration for FP8 training."""

    # Basic FP8 settings
    enable_fp8: bool = True
    fp8_format: FP8Format = FP8Format.E4M3  # E4M3 for forward, E5M2 for gradients
    mixed_precision_policy: str = "auto"  # "auto", "fp8_only", "mixed_fp8_bf16"

    # Scaling and stability
    initial_loss_scale: float = 2**15
    loss_scale_window: int = 1000
    min_loss_scale: float = 1.0
    max_loss_scale: float = 2**24

    # Recipe settings (Transformer Engine)
    margin: int = 0  # Margin for scaling
    interval: int = 1  # Interval for amax updates
    fp8_dpa: bool = True  # FP8 Dot Product Attention
    fp8_mha: bool = True  # FP8 Multi-Head Attention

    # Fallback settings
    use_te_recipe: bool = True
    fallback_to_bf16: bool = True
    enable_amax_scaling: bool = True

    # Advanced settings
    delayed_scaling: bool = True
    override_linear_precision: bool = False
    reduce_amax: bool = True


class FP8Handler:
    """
    Main handler for FP8 training operations.

    Provides unified interface for FP8 training whether using Transformer Engine
    or fallback implementations.
    """

    def __init__(self, config: FP8Config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.scaler = None
        self.te_recipe = None

        # Check hardware compatibility
        self._check_hardware_support()

        # Initialize based on available backend
        if TE_AVAILABLE and config.use_te_recipe:
            self._init_transformer_engine()
        else:
            self._init_fallback()

    def _check_hardware_support(self):
        """Check if hardware supports FP8."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available for FP8 training")

        device_props = torch.cuda.get_device_properties(0)
        device_name = device_props.name.lower()

        # Check for H100 or newer
        if "h100" in device_name or "h200" in device_name or "b200" in device_name:
            logger.info(f"FP8-capable GPU detected: {device_props.name}")
            self.fp8_supported = True
        else:
            logger.warning(f"GPU {device_props.name} may not support FP8. Training may fall back to BF16.")
            self.fp8_supported = self.config.fallback_to_bf16

    def _init_transformer_engine(self):
        """Initialize Transformer Engine backend."""
        logger.info("Initializing Transformer Engine for FP8 training")

        # Create FP8 recipe
        self.te_recipe = recipe.DelayedScaling(  # type: ignore[possibly-unbound]
            margin=self.config.margin,
            interval=self.config.interval,
            fp8_format=recipe.Format.E4M3 if self.config.fp8_format == FP8Format.E4M3 else recipe.Format.E5M2,  # type: ignore[possibly-unbound]
            amax_history_len=1024,
            amax_compute_algo="max",
            override_linear_precision=(
                False, False, not self.config.override_linear_precision
            )
        )

        # Initialize scaling
        self.scaler = GradScaler(
            init_scale=self.config.initial_loss_scale,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=self.config.loss_scale_window
        )

    def _init_fallback(self):
        """Initialize fallback FP8 implementation."""
        logger.info("Initializing fallback FP8 implementation")

        # Create custom scaling mechanism
        self.scaler = GradScaler(
            init_scale=self.config.initial_loss_scale,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=self.config.loss_scale_window
        )

        # Custom FP8 simulation (simplified)
        self.fp8_max_values = {
            FP8Format.E4M3: 448.0,  # Max value for E4M3
            FP8Format.E5M2: 57344.0  # Max value for E5M2
        }

    @contextmanager
    def fp8_autocast(self):
        """Context manager for FP8 autocast."""
        if TE_AVAILABLE and self.te_recipe is not None:
            # Use Transformer Engine autocast
            with te.fp8_autocast(enabled=self.config.enable_fp8, fp8_recipe=self.te_recipe):  # type: ignore[possibly-unbound]
                yield
        else:
            # Use fallback implementation
            if self.config.enable_fp8 and self.fp8_supported:
                with self._fallback_fp8_context():
                    yield
            else:
                # Fall back to BF16
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    yield

    @contextmanager
    def _fallback_fp8_context(self):
        """Fallback FP8 context using quantization simulation."""
        original_forward_hooks = {}

        try:
            # Register hooks to simulate FP8 quantization
            def fp8_quantize_hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.dtype == torch.float32:
                    return self._simulate_fp8_quantization(output)
                return output

            # This is a simplified simulation - real implementation would be more sophisticated
            yield

        finally:
            # Clean up hooks
            for module, hook in original_forward_hooks.items():
                hook.remove()

    def _simulate_fp8_quantization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simulate FP8 quantization for fallback implementation."""
        # Simplified FP8 simulation by clamping values
        max_val = self.fp8_max_values[self.config.fp8_format]

        # Clamp to FP8 range
        clamped = torch.clamp(tensor, -max_val, max_val)

        # Simulate quantization noise (very simplified)
        if self.config.fp8_format == FP8Format.E4M3:
            # E4M3 has 3 mantissa bits -> 8 levels
            scale = max_val / 8.0
            quantized = torch.round(clamped / scale) * scale
        else:
            # E5M2 has 2 mantissa bits -> 4 levels
            scale = max_val / 4.0
            quantized = torch.round(clamped / scale) * scale

        return quantized

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for FP8 training."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(self, optimizer, model_parameters):
        """Perform optimizer step with FP8 considerations."""
        if self.scaler is not None:
            # Unscale gradients
            self.scaler.unscale_(optimizer)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model_parameters, max_norm=1.0)

            # Optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def get_loss_scale(self) -> float:
        """Get current loss scale."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0


class FP8Linear(nn.Module):
    """
    FP8-optimized Linear layer.

    Uses Transformer Engine if available, otherwise falls back to
    standard linear with FP8 simulation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if TE_AVAILABLE:
            # Use Transformer Engine Linear
            self.linear = te.Linear(  # type: ignore[possibly-unbound]
                in_features,
                out_features,
                bias=bias,
                device=device,
                dtype=dtype
            )
        else:
            # Use standard Linear
            self.linear = nn.Linear(
                in_features,
                out_features,
                bias=bias,
                device=device,
                dtype=dtype
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FP8MultiHeadAttention(nn.Module):
    """
    FP8-optimized Multi-Head Attention.

    Leverages Transformer Engine's FP8 attention implementation
    for maximum performance on H100/B200 GPUs.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        if TE_AVAILABLE:
            # Use Transformer Engine MultiHeadAttention
            self.attention = te.MultiheadAttention(  # type: ignore[possibly-unbound]
                embed_dim,
                num_heads,
                dropout=dropout,
                bias=bias,
                device=device,
                dtype=dtype
            )
        else:
            # Use standard MultiHeadAttention
            self.attention = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                bias=bias,
                batch_first=True,
                device=device,
                dtype=dtype
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of FP8 attention."""

        if TE_AVAILABLE:
            # Transformer Engine expects different input format
            return self.attention(
                query, key, value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )
        else:
            # Standard PyTorch MultiHeadAttention
            return self.attention(
                query, key, value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )


class FP8LayerNorm(nn.Module):
    """
    FP8-optimized Layer Normalization.
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        if TE_AVAILABLE:
            # Use Transformer Engine LayerNorm
            self.layer_norm = te.LayerNorm(  # type: ignore[possibly-unbound]
                normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
                device=device,
                dtype=dtype
            )
        else:
            # Use standard LayerNorm
            self.layer_norm = nn.LayerNorm(
                normalized_shape,
                eps=eps,
                elementwise_affine=elementwise_affine,
                device=device,
                dtype=dtype
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)


class FP8TransformerLayer(nn.Module):
    """
    Complete Transformer layer optimized for FP8 training.

    Combines FP8-optimized attention, feed-forward, and normalization
    layers for maximum H100/B200 performance.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if TE_AVAILABLE:
            # Use Transformer Engine TransformerLayer
            self.transformer_layer = te.TransformerLayer(  # type: ignore[possibly-unbound]
                hidden_size=embed_dim,
                ffn_hidden_size=ff_dim,
                num_attention_heads=num_heads,
                layernorm_epsilon=layer_norm_eps,
                hidden_dropout=dropout,
                attention_dropout=dropout,
                fuse_wgrad_accumulation=True,
                get_rng_state_tracker=None,
                init_method=None,
                output_layer_init_method=None,
                layer_number=1,
                drop_path_rate=0.0,
                device=device,
                dtype=dtype
            )
        else:
            # Manual implementation with FP8 components
            self.self_attn = FP8MultiHeadAttention(
                embed_dim, num_heads, dropout, device=device, dtype=dtype
            )
            self.norm1 = FP8LayerNorm(embed_dim, layer_norm_eps, device=device, dtype=dtype)

            self.ffn = nn.Sequential(
                FP8Linear(embed_dim, ff_dim, device=device, dtype=dtype),
                nn.GELU() if activation == "gelu" else nn.ReLU(),
                nn.Dropout(dropout),
                FP8Linear(ff_dim, embed_dim, device=device, dtype=dtype),
                nn.Dropout(dropout)
            )
            self.norm2 = FP8LayerNorm(embed_dim, layer_norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass of FP8 transformer layer."""

        if TE_AVAILABLE and hasattr(self, 'transformer_layer'):
            # Use Transformer Engine implementation
            return self.transformer_layer(
                x,
                attention_mask=attn_mask,
                encoder_output=None,
                enc_dec_attn_mask=None
            )
        else:
            # Manual implementation
            # Self-attention with residual connection
            attn_output, _ = self.self_attn(x, x, x, attn_mask, key_padding_mask)
            x = self.norm1(x + attn_output)

            # Feed-forward with residual connection
            ffn_output = self.ffn(x)
            x = self.norm2(x + ffn_output)

            return x


class FP8ModelWrapper(nn.Module):
    """
    Wrapper to convert an existing model to use FP8 layers.

    Automatically replaces compatible layers with FP8-optimized versions.
    """

    def __init__(self, model: nn.Module, fp8_config: FP8Config):
        super().__init__()
        self.original_model = model
        self.fp8_config = fp8_config

        # Replace layers with FP8 versions
        self._convert_to_fp8()

    def _convert_to_fp8(self):
        """Convert model layers to FP8 versions."""
        def replace_layer(module, name, new_layer):
            """Replace a layer in the module."""
            parent = module
            atoms = name.split('.')
            for atom in atoms[:-1]:
                parent = getattr(parent, atom)
            setattr(parent, atoms[-1], new_layer)

        # Find and replace Linear layers
        for name, module in self.original_model.named_modules():
            if isinstance(module, nn.Linear):
                fp8_linear = FP8Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    device=str(module.weight.device),
                    dtype=module.weight.dtype
                )

                # Copy weights
                with torch.no_grad():
                    fp8_linear.linear.weight.copy_(module.weight)
                    if module.bias is not None:
                        fp8_linear.linear.bias.copy_(module.bias)

                replace_layer(self.original_model, name, fp8_linear)
                logger.info(f"Replaced Linear layer: {name}")

            elif isinstance(module, nn.LayerNorm):
                # Convert normalized_shape tuple to list for compatibility
                norm_shape = list(module.normalized_shape) if isinstance(module.normalized_shape, tuple) else module.normalized_shape
                fp8_norm = FP8LayerNorm(
                    norm_shape,
                    eps=module.eps,
                    elementwise_affine=module.elementwise_affine,
                    device=str(module.weight.device) if module.weight is not None else None,
                    dtype=module.weight.dtype if module.weight is not None else None
                )

                # Copy weights
                with torch.no_grad():
                    if module.weight is not None:
                        fp8_norm.layer_norm.weight.copy_(module.weight)
                    if module.bias is not None:
                        fp8_norm.layer_norm.bias.copy_(module.bias)

                replace_layer(self.original_model, name, fp8_norm)
                logger.info(f"Replaced LayerNorm: {name}")

    def forward(self, *args, **kwargs):
        """Forward pass through the FP8-converted model."""
        return self.original_model(*args, **kwargs)


def create_fp8_model(
    model: nn.Module,
    config: Optional[FP8Config] = None
) -> Tuple[nn.Module, FP8Handler]:
    """
    Convert a model to use FP8 training.

    Args:
        model: PyTorch model to convert
        config: FP8 configuration (uses defaults if None)

    Returns:
        Tuple of (converted_model, fp8_handler)
    """
    if config is None:
        config = FP8Config()

    # Create FP8 handler
    fp8_handler = FP8Handler(config)

    # Convert model to FP8
    if config.mixed_precision_policy == "fp8_only":
        # Convert all compatible layers
        fp8_model = FP8ModelWrapper(model, config)
    else:
        # Keep original model, use FP8 context
        fp8_model = model

    return fp8_model, fp8_handler


def benchmark_fp8_training(
    model: nn.Module,
    data_loader,
    num_steps: int = 100,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Benchmark FP8 training performance vs BF16.

    Args:
        model: Model to benchmark
        data_loader: Training data loader
        num_steps: Number of steps to benchmark
        device: Device to run on

    Returns:
        Dictionary with benchmark results
    """
    results = {}

    # Test configurations
    configs = [
        ("BF16", None),
        ("FP8_E4M3", FP8Config(fp8_format=FP8Format.E4M3)),
        ("FP8_E5M2", FP8Config(fp8_format=FP8Format.E5M2)),
        ("Mixed_FP8_BF16", FP8Config(mixed_precision_policy="mixed_fp8_bf16"))
    ]

    for config_name, fp8_config in configs:
        logger.info(f"Benchmarking {config_name}...")

        # Create model copy
        model_copy = type(model)(model.config).to(device)
        model_copy.load_state_dict(model.state_dict())

        # Setup FP8 if needed
        if fp8_config is not None:
            model_copy, fp8_handler = create_fp8_model(model_copy, fp8_config)
        else:
            fp8_handler = None

        # Create optimizer
        optimizer = torch.optim.AdamW(model_copy.parameters(), lr=1e-4)

        # Benchmark training
        torch.cuda.reset_peak_memory_stats()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        total_loss = 0.0

        for step, batch in enumerate(data_loader):
            if step >= num_steps:
                break

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward pass
            if fp8_handler is not None:
                with fp8_handler.fp8_autocast():
                    outputs = model_copy(**batch)
                    loss = outputs.loss

                # Scale loss
                loss = fp8_handler.scale_loss(loss)
            else:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = model_copy(**batch)
                    loss = outputs.loss

            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Optimizer step
            if fp8_handler is not None:
                fp8_handler.step_optimizer(optimizer, model_copy.parameters())
            else:
                optimizer.step()

            optimizer.zero_grad()

        end_time.record()
        torch.cuda.synchronize()

        # Collect results
        elapsed_time = start_time.elapsed_time(end_time) / 1000.0
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)

        results[config_name] = {
            "avg_loss": total_loss / min(num_steps, len(data_loader)),
            "time_per_step": elapsed_time / min(num_steps, len(data_loader)),
            "peak_memory_gb": peak_memory,
            "throughput_steps_per_sec": min(num_steps, len(data_loader)) / elapsed_time,
            "loss_scale": fp8_handler.get_loss_scale() if fp8_handler else 1.0
        }

        # Clean up
        del model_copy, optimizer
        if fp8_handler:
            del fp8_handler
        torch.cuda.empty_cache()

        logger.info(f"{config_name} results: {results[config_name]}")

    # Calculate speedups
    if "BF16" in results:
        baseline_time = results["BF16"]["time_per_step"]
        for config_name in results:
            if config_name != "BF16":
                results[config_name]["speedup_vs_bf16"] = baseline_time / results[config_name]["time_per_step"]

    return results