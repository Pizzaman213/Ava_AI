"""Device detection and configuration utilities."""

import torch
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

def detect_device() -> Tuple[str, Dict[str, Any]]:
    """
    Detect available compute device and return device string with capabilities.
    
    Returns:
        Tuple of (device_string, device_info_dict)
    """
    device_info = {}
    
    if torch.cuda.is_available():
        device = "cuda"
        device_info = {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "compute_capability": torch.cuda.get_device_capability(0),
            "supports_bf16": torch.cuda.is_bf16_supported(),
            "supports_flash_attn": True,  # Assume true for modern GPUs
            "supports_fused_kernels": True,
            "supports_torch_compile": hasattr(torch, "compile"),
        }
    elif torch.backends.mps.is_available():
        device = "mps"
        device_info = {
            "type": "mps",
            "name": "Apple Silicon",
            "count": 1,
            "memory_gb": 32,  # Default, can be overridden
            "supports_bf16": False,
            "supports_flash_attn": False,
            "supports_fused_kernels": False,
            "supports_torch_compile": False,
            "use_metal_performance_shaders": True,
        }
    else:
        device = "cpu"
        device_info = {
            "type": "cpu",
            "name": "CPU",
            "count": os.cpu_count(),
            "supports_bf16": False,
            "supports_flash_attn": False,
            "supports_fused_kernels": False,
            "supports_torch_compile": hasattr(torch, "compile"),
        }
    
    return device, device_info


def apply_device_config(config: Dict[str, Any], device: Optional[str] = None) -> Dict[str, Any]:
    """
    Apply device-specific configuration overrides.
    
    Args:
        config: Base configuration dictionary
        device: Device type (cuda/mps/cpu) or None for auto-detect
    
    Returns:
        Updated configuration with device-specific settings
    """
    if device is None:
        device, device_info = detect_device()
    else:
        device_info = {}
    
    # Check for device_config section
    if "device_config" in config and config["device_config"].get("auto_detect", False):
        device_overrides = config["device_config"].get(device, {})
        
        # Apply overrides to various config sections
        for key, value in device_overrides.items():
            if key == "batch_size" and "training" in config:
                config["training"]["batch_size"] = value
            elif key == "gradient_accumulation_steps" and "training" in config:
                config["training"]["gradient_accumulation_steps"] = value
            elif key == "mixed_precision" and "training" in config:
                config["training"]["mixed_precision"] = value
            elif key == "num_workers" and "training" in config:
                config["training"]["num_workers"] = value
            elif key == "pin_memory" and "training" in config:
                config["training"]["pin_memory"] = value
            elif key == "use_flash_attention" and "model" in config:
                config["model"]["use_flash_attn"] = value
            elif key == "use_fused_kernels" and "model" in config:
                config["model"]["use_fused_kernels"] = value
            elif key == "use_torch_compile" and "model" in config:
                config["model"]["use_torch_compile"] = value
    
    # Handle 'auto' values throughout config
    config = resolve_auto_values(config, device, device_info)
    
    # Set device in training config
    if "training" in config:
        if config["training"].get("device") == "auto":
            config["training"]["device"] = device
    
    return config


def resolve_auto_values(config: Dict[str, Any], device: str, device_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve 'auto' values in configuration based on device capabilities.
    
    Args:
        config: Configuration dictionary
        device: Device type
        device_info: Device information dictionary
    
    Returns:
        Updated configuration with resolved auto values
    """
    # Model configuration
    if "model" in config:
        model_config = config["model"]
        
        if model_config.get("use_flash_attn") == "auto":
            model_config["use_flash_attn"] = device_info.get("supports_flash_attn", False)
        
        if model_config.get("use_mixed_precision") == "auto":
            if device == "cuda":
                model_config["use_mixed_precision"] = "bf16" if device_info.get("supports_bf16") else "fp16"
            else:
                model_config["use_mixed_precision"] = False
        
        if model_config.get("use_torch_compile") == "auto":
            model_config["use_torch_compile"] = device_info.get("supports_torch_compile", False)
        
        if model_config.get("use_fused_kernels") == "auto":
            model_config["use_fused_kernels"] = device_info.get("supports_fused_kernels", False)
        
        if model_config.get("gradient_checkpointing") == "auto":
            # Enable for GPUs with less than 16GB memory
            model_config["gradient_checkpointing"] = device_info.get("memory_gb", 8) < 16
        
        if model_config.get("use_memory_efficient_attention") == "auto":
            model_config["use_memory_efficient_attention"] = device_info.get("memory_gb", 8) < 16
    
    # Training configuration
    if "training" in config:
        training_config = config["training"]
        
        if training_config.get("batch_size") == "auto":
            if device == "cuda":
                memory_gb = device_info.get("memory_gb", 8)
                training_config["batch_size"] = min(32, int(memory_gb * 2))
            elif device == "mps":
                training_config["batch_size"] = 8
            else:
                training_config["batch_size"] = 4
        
        if training_config.get("gradient_accumulation_steps") == "auto":
            batch_size = training_config.get("batch_size", 8)
            target_batch = 32
            training_config["gradient_accumulation_steps"] = max(1, target_batch // batch_size)
        
        if training_config.get("mixed_precision") == "auto":
            if device == "cuda":
                training_config["mixed_precision"] = "bf16" if device_info.get("supports_bf16") else "fp16"
            elif device == "mps":
                training_config["mixed_precision"] = "fp32"
            else:
                training_config["mixed_precision"] = "fp32"
        
        if training_config.get("num_workers") == "auto":
            if device == "cuda":
                training_config["num_workers"] = min(4, os.cpu_count() or 1)
            elif device == "mps":
                training_config["num_workers"] = 0  # MPS works best with 0
            else:
                training_config["num_workers"] = min(2, os.cpu_count() or 1)
        
        if training_config.get("pin_memory") == "auto":
            training_config["pin_memory"] = device == "cuda"
        
        if training_config.get("persistent_workers") == "auto":
            training_config["persistent_workers"] = device == "cuda" and training_config.get("num_workers", 0) > 0
        
        if training_config.get("gradient_checkpointing") == "auto":
            training_config["gradient_checkpointing"] = device_info.get("memory_gb", 8) < 16
    
    # Data configuration
    if "data" in config and "dataset_args" in config["data"]:
        dataset_args = config["data"]["dataset_args"]
        
        if dataset_args.get("num_proc") == "auto":
            if device == "mps":
                dataset_args["num_proc"] = 1  # MPS works best with single process
            else:
                dataset_args["num_proc"] = min(4, os.cpu_count() or 1)
    
    # Distributed configuration
    if "distributed" in config:
        if config.get("distributed") == "auto":
            config["distributed"] = device == "cuda" and device_info.get("count", 1) > 1
        
        if config.get("num_gpus") == "auto":
            config["num_gpus"] = device_info.get("count", 0) if device == "cuda" else 0
        
        if config.get("device_map") == "auto":
            if device == "cuda" and device_info.get("count", 1) > 1:
                config["device_map"] = "auto"
            else:
                config["device_map"] = device
    
    # Inference configuration
    if "inference" in config:
        inference_config = config["inference"]
        
        if inference_config.get("use_flash_attn") == "auto":
            inference_config["use_flash_attn"] = device_info.get("supports_flash_attn", False)
        
        if inference_config.get("torch_compile") == "auto":
            inference_config["torch_compile"] = device_info.get("supports_torch_compile", False)
        
        if inference_config.get("dtype") == "auto":
            if device == "cuda":
                inference_config["dtype"] = "bfloat16" if device_info.get("supports_bf16") else "float16"
            else:
                inference_config["dtype"] = "float32"
    
    # Memory optimization
    if "memory_optimization" in config:
        mem_config = config["memory_optimization"]
        
        for key in ["gradient_checkpointing", "activation_checkpointing", "memory_efficient_attention"]:
            if mem_config.get(key) == "auto":
                mem_config[key] = device_info.get("memory_gb", 8) < 16
    
    return config


def load_config_with_device(config_path: str, device: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration file and apply device-specific settings.
    
    Args:
        config_path: Path to configuration YAML file
        device: Device type or None for auto-detect
    
    Returns:
        Configuration dictionary with device-specific settings applied
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return apply_device_config(config, device)


def get_optimal_batch_size(model_size: str, device: str, memory_gb: Optional[float] = None) -> int:
    """
    Get optimal batch size based on model size and available memory.
    
    Args:
        model_size: Model size (small/medium/large)
        device: Device type
        memory_gb: Available memory in GB
    
    Returns:
        Recommended batch size
    """
    if device == "cuda":
        if memory_gb is None:
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        size_multipliers = {"small": 4, "medium": 2, "large": 1}
        base_batch = int(memory_gb * size_multipliers.get(model_size, 2))
        return min(64, max(1, base_batch))
    
    elif device == "mps":
        size_map = {"small": 16, "medium": 8, "large": 4}
        return size_map.get(model_size, 8)
    
    else:  # CPU
        size_map = {"small": 8, "medium": 4, "large": 2}
        return size_map.get(model_size, 4)


def print_device_info():
    """Print detailed device information."""
    device, info = detect_device()
    
    print("=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    print(f"Device Type: {device.upper()}")
    
    for key, value in info.items():
        if key != "type":
            key_display = key.replace("_", " ").title()
            print(f"{key_display}: {value}")
    
    print("=" * 60)