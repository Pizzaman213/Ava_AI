"""
Model Manager for Ava Training Pipeline

Handles all model creation and initialization including:
- Standard and optimized MoE model variants
- Meta device materialization for large models
- Tokenizer loading and configuration
- Model parameter validation
"""

from typing import Any, Tuple, Union

import torch
from transformers import AutoTokenizer

from src.Ava.config import EnhancedTrainingConfig
from src.Ava.models.moe_model import (
    EnhancedMoEConfig,
    EnhancedMoEModel,
    OptimizedMoEConfig,
    OptimizedMoETransformer,
)

from .base import TrainingComponent, TrainingContext


class ModelManager(TrainingComponent):
    """Manages model creation, initialization, and configuration."""

    def __init__(self, context: TrainingContext):
        """Initialize model manager.

        Args:
            context: Training context with configuration
        """
        super().__init__(context)
        self.model = None
        self.tokenizer = None
        self.model_config = None

    def create_model_and_tokenizer(
        self,
        config_dict: dict,
        training_config: EnhancedTrainingConfig,
    ) -> Tuple:
        """Create model and tokenizer from configuration.

        Args:
            config_dict: Raw configuration dictionary
            training_config: Enhanced training configuration

        Returns:
            Tuple of (model, tokenizer)
        """
        from src.Ava.utils.logging import get_logger

        model_config_dict = config_dict.get("model", {})
        use_optimized_moe = model_config_dict.get("use_optimized_moe", False)

        # Create enhanced model config with feature flags
        enhanced_model_config = model_config_dict.copy()
        if not use_optimized_moe:
            enhanced_model_config.update(
                {
                    "use_moh": training_config.architecture.use_moh,
                    "use_moa": training_config.architecture.use_moa,
                    "use_cross_attention": training_config.architecture.use_cross_attention,
                    "use_alibi": training_config.architecture.use_alibi,
                    "router_type": training_config.architecture.expert_routing_type,
                }
            )

        # Filter and validate config fields
        filtered_config = self._filter_model_config(
            enhanced_model_config, use_optimized_moe, training_config
        )

        # Create model
        model = self._create_model(
            filtered_config, use_optimized_moe, config_dict, training_config
        )

        # Create tokenizer
        tokenizer = self._create_tokenizer(config_dict, training_config)

        # Validate tokenizer vocab size
        self._validate_tokenizer_vocab(model, model.config, tokenizer)

        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config

        return model, tokenizer

    def _filter_model_config(
        self, model_config_dict: dict, use_optimized_moe: bool, training_config: Any
    ) -> dict:
        """Filter and validate model configuration.

        Args:
            model_config_dict: Raw model config dictionary
            use_optimized_moe: Whether using optimized MoE
            training_config: Training configuration

        Returns:
            Filtered configuration dictionary
        """
        from dataclasses import fields

        config_class = OptimizedMoEConfig if use_optimized_moe else EnhancedMoEConfig
        valid_fields = {f.name: f.type for f in fields(config_class)}

        filtered_config = {}
        for k, v in model_config_dict.items():
            if k in valid_fields and v is not None:
                field_type = valid_fields[k]

                # Convert to proper type
                if field_type == float or "float" in str(field_type):
                    try:
                        v = float(v)
                    except (ValueError, TypeError):
                        continue
                elif field_type == int or "int" in str(field_type):
                    try:
                        v = int(v)
                    except (ValueError, TypeError):
                        continue
                elif field_type == bool or "bool" in str(field_type):
                    if isinstance(v, str):
                        v = v.lower() in ("true", "yes", "1")

                filtered_config[k] = v

        # Add defaults for critical fields
        defaults = {
            "vocab_size": getattr(
                training_config.model,
                "default_vocab_size",
                50257,
            ) if hasattr(training_config, "model") else 50257,
            "hidden_size": getattr(
                training_config.model,
                "default_hidden_size",
                768,
            ) if hasattr(training_config, "model") else 768,
            "num_layers": getattr(
                training_config.model,
                "default_num_layers",
                12,
            ) if hasattr(training_config, "model") else 12,
            "num_attention_heads": getattr(
                training_config.model,
                "default_num_attention_heads",
                12,
            ) if hasattr(training_config, "model") else 12,
        }

        for k, default_v in defaults.items():
            if k not in filtered_config:
                filtered_config[k] = default_v

        return filtered_config

    def _create_model(
        self,
        filtered_config: dict,
        use_optimized_moe: bool,
        config_dict: dict,
        training_config: Any,
    ) -> torch.nn.Module:
        """Create model with appropriate initialization strategy.

        Args:
            filtered_config: Filtered model configuration
            use_optimized_moe: Whether to use optimized MoE
            config_dict: Raw configuration dictionary
            training_config: Training configuration

        Returns:
            Initialized model
        """
        from src.Ava.utils.logging import get_logger

        if use_optimized_moe:
            return self._create_optimized_moe(filtered_config, config_dict)
        else:
            return self._create_standard_moe(filtered_config)

    def _create_optimized_moe(self, filtered_config: dict, config_dict: dict) -> torch.nn.Module:
        """Create OptimizedMoETransformer with meta device support.

        Args:
            filtered_config: Filtered model configuration
            config_dict: Raw configuration dictionary

        Returns:
            Initialized model
        """
        from src.Ava.utils.logging import get_logger

        get_logger().info("Using OptimizedMoETransformer (high-performance MoE)")
        model_config = OptimizedMoEConfig(**filtered_config)

        # Check for expert offloading and torch.compile
        use_offloading = (
            filtered_config.get("use_expert_offloading")
            or config_dict.get("memory_optimization", {}).get("use_expert_offloading")
            or config_dict.get("optimization", {}).get("use_expert_offloading", False)
        )

        enable_compile = (
            config_dict.get("performance", {}).get("enable_torch_compile")
            or config_dict.get("hardware", {}).get("compile", True)
        )

        if use_offloading or enable_compile:
            # Create directly on GPU (meta device incompatible)
            reasons = []
            if use_offloading:
                reasons.append("expert offloading")
            if enable_compile:
                reasons.append("torch.compile")
            reason_str = " and ".join(reasons)
            get_logger().info(
                f"{reason_str.capitalize()} enabled - creating model directly on GPU..."
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = OptimizedMoETransformer(model_config).to(
                device=device, dtype=torch.bfloat16
            )
            get_logger().info(f"Model created on {device} in bf16 dtype")
        else:
            # Use meta device for memory-efficient initialization
            get_logger().info("Initializing model on meta device (low RAM usage)...")
            with torch.device("meta"):
                model = OptimizedMoETransformer(model_config)

            # Materialize weights directly on GPU
            get_logger().info("Materializing model weights on GPU in bf16...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = self.materialize_meta_model(model, device=device, dtype=torch.bfloat16)
            get_logger().info(f"Model materialized on {device} in bf16 dtype")

        self._log_model_parameters(model)
        return model

    def _create_standard_moe(self, filtered_config: dict) -> torch.nn.Module:
        """Create standard EnhancedMoEModel.

        Args:
            filtered_config: Filtered model configuration

        Returns:
            Initialized model
        """
        from src.Ava.utils.logging import get_logger

        get_logger().info("Using EnhancedMoEModel (standard MoE)")
        model_config = EnhancedMoEConfig(**filtered_config)
        model = EnhancedMoEModel(model_config)

        self._log_model_parameters(model)
        return model

    def _create_tokenizer(
        self, config_dict: dict, training_config: Any
    ) -> AutoTokenizer:
        """Create and configure tokenizer.

        Args:
            config_dict: Raw configuration dictionary
            training_config: Training configuration

        Returns:
            Configured tokenizer
        """
        from src.Ava.utils.logging import get_logger

        # Determine tokenizer name with fallback
        default_tokenizer = (
            getattr(training_config.data, "default_tokenizer_name", "Qwen/Qwen2.5-0.5B")
            if hasattr(training_config, "data")
            else "Qwen/Qwen2.5-0.5B"
        )
        tokenizer_name = (
            config_dict.get("data", {}).get("tokenizer_name")
            or config_dict.get("tokenizer", {}).get("name")
            or default_tokenizer
        )

        get_logger().info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        get_logger().info(f"Tokenizer loaded: vocab_size={len(tokenizer)}")

        return tokenizer

    def _validate_tokenizer_vocab(
        self, model: torch.nn.Module, model_config: Any, tokenizer: AutoTokenizer
    ) -> None:
        """Validate that tokenizer vocab size matches model.

        Args:
            model: Model instance
            model_config: Model configuration
            tokenizer: Tokenizer instance

        Raises:
            ValueError: If vocab sizes don't match
        """
        from src.Ava.utils.logging import get_logger

        model_vocab_size = model_config.vocab_size
        tokenizer_vocab_size = len(tokenizer)

        if model_vocab_size != tokenizer_vocab_size:
            error_msg = (
                f"CRITICAL ERROR: Tokenizer vocab size mismatch!\n"
                f"  Model expects: {model_vocab_size} tokens\n"
                f"  Tokenizer has: {tokenizer_vocab_size} tokens\n"
                f"  This will cause 'CUDA index out of bounds' errors during training.\n"
                f"  Please either:\n"
                f"    1. Use a tokenizer with {model_vocab_size} tokens, or\n"
                f"    2. Update model config vocab_size to {tokenizer_vocab_size}"
            )
            get_logger().error(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def _log_model_parameters(model: torch.nn.Module) -> None:
        """Log model parameter statistics.

        Args:
            model: Model instance
        """
        from src.Ava.utils.logging import get_logger

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        get_logger().info(" Model Parameters:")
        get_logger().info(
            f"  Total: {total_params:,} ({total_params/1e6:.1f}M / {total_params/1e9:.2f}B)"
        )
        get_logger().info(
            f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.1f}M / {trainable_params/1e9:.2f}B)"
        )

        if total_params != trainable_params:
            frozen_params = total_params - trainable_params
            get_logger().info(
                f"  Frozen: {frozen_params:,} ({frozen_params/1e6:.1f}M / {frozen_params/1e9:.2f}B)"
            )

    @staticmethod
    def materialize_meta_model(
        model: torch.nn.Module,
        device: Union[str, torch.device] = "cuda",
        dtype=torch.bfloat16,
    ) -> torch.nn.Module:
        """Materialize a meta-device model directly on target device.

        This avoids the CPU RAM bottleneck when initializing large models by
        creating parameters directly on GPU without intermediate CPU storage.

        Args:
            model: Model created with torch.device("meta")
            device: Target device for materialization
            dtype: Target dtype for parameters

        Returns:
            Model with parameters materialized on target device
        """
        import torch.nn as nn

        # Normalize device to string
        if isinstance(device, torch.device):
            device = str(device)

        def init_fn(module):
            """Initialize function to convert meta tensors to real tensors."""
            # Process parameters
            for name, param in list(module.named_parameters(recurse=False)):
                if param.device.type == "meta":
                    new_param = nn.Parameter(
                        torch.empty(param.shape, device=device, dtype=dtype)
                    )

                    # Initialize with layer-appropriate strategy
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(new_param, mean=0.0, std=0.02)
                    elif isinstance(module, nn.Embedding):
                        nn.init.normal_(new_param, mean=0.0, std=0.02)
                    else:
                        nn.init.normal_(new_param, mean=0.0, std=0.02)

                    module._parameters[name] = new_param

            # Process buffers
            for name, buffer in list(module.named_buffers(recurse=False)):
                if buffer is not None and buffer.device.type == "meta":
                    new_buffer = torch.empty(buffer.shape, device=device, dtype=dtype)
                    module._buffers[name] = new_buffer

        model.apply(init_fn)
        return model
