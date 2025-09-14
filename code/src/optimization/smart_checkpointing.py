"""
Smart Checkpointing and Memory Optimization

Implements intelligent gradient checkpointing policies that adapt
based on memory pressure, compute budget, and layer importance.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
import psutil
import GPUtil
import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)


@dataclass
class CheckpointingConfig:
    """Configuration for smart checkpointing"""
    # Memory thresholds
    memory_pressure_threshold: float = 0.8  # Start checkpointing at 80% memory
    critical_memory_threshold: float = 0.95  # Aggressive checkpointing at 95%
    
    # Compute budget
    max_recompute_ratio: float = 1.5  # Max 50% extra compute for checkpointing
    
    # Layer importance
    important_layer_names: List[str] = None
    checkpoint_probability: Dict[str, float] = None
    
    # Adaptive settings
    enable_adaptive_checkpointing: bool = True
    profile_interval: int = 100  # Profile every N steps
    
    # Selective checkpointing
    checkpoint_attention: bool = True
    checkpoint_mlp: bool = True
    checkpoint_experts: bool = True
    min_checkpoint_size_mb: float = 10.0  # Don't checkpoint small ops
    
    def __post_init__(self):
        if self.important_layer_names is None:
            self.important_layer_names = [
                "attention", "moe_block", "mlp"
            ]
            
        if self.checkpoint_probability is None:
            # Default probabilities based on layer type
            self.checkpoint_probability = {
                "attention": 0.8,
                "moe_block": 0.9,
                "mlp": 0.7,
                "embedding": 0.0,
                "norm": 0.0
            }


class MemoryMonitor:
    """Monitors system memory usage"""
    
    def __init__(self):
        self.history = []
        self.gpu_available = torch.cuda.is_available()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        stats = {}
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        stats["cpu_used_gb"] = (cpu_memory.total - cpu_memory.available) / 1e9
        stats["cpu_total_gb"] = cpu_memory.total / 1e9
        stats["cpu_percent"] = cpu_memory.percent / 100.0
        
        # GPU memory
        if self.gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Primary GPU
                    stats["gpu_used_gb"] = gpu.memoryUsed / 1000.0
                    stats["gpu_total_gb"] = gpu.memoryTotal / 1000.0
                    stats["gpu_percent"] = gpu.memoryUsed / gpu.memoryTotal
                    
                    # PyTorch specific
                    stats["pytorch_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                    stats["pytorch_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
            except Exception as e:
                logger.warning(f"Failed to get GPU memory stats: {e}")
                
        return stats
        
    def get_memory_pressure(self) -> float:
        """Get overall memory pressure (0-1)"""
        stats = self.get_memory_usage()
        
        if "gpu_percent" in stats:
            # Use GPU memory as primary indicator
            return stats["gpu_percent"]
        else:
            # Fall back to CPU memory
            return stats["cpu_percent"]


class ComputeBudgetTracker:
    """Tracks computational budget for recomputation"""
    
    def __init__(self, max_recompute_ratio: float = 1.5):
        self.max_recompute_ratio = max_recompute_ratio
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.recompute_time = 0.0
        
    def start_timer(self) -> float:
        """Start timing an operation"""
        return time.perf_counter()
        
    def end_timer(self, start_time: float, operation: str):
        """End timing and record"""
        elapsed = time.perf_counter() - start_time
        
        if operation == "forward":
            self.forward_time += elapsed
        elif operation == "backward":
            self.backward_time += elapsed
        elif operation == "recompute":
            self.recompute_time += elapsed
            
    def get_recompute_ratio(self) -> float:
        """Get current recomputation ratio"""
        total_compute = self.forward_time + self.backward_time
        if total_compute == 0:
            return 0.0
            
        return (total_compute + self.recompute_time) / total_compute
        
    def has_budget(self) -> bool:
        """Check if we have compute budget for more checkpointing"""
        return self.get_recompute_ratio() < self.max_recompute_ratio
        
    def reset(self):
        """Reset timers"""
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.recompute_time = 0.0


class SmartCheckpointPolicy:
    """
    Intelligent checkpointing policy that adapts based on
    memory pressure and compute budget
    """
    
    def __init__(self, config: CheckpointingConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor()
        self.compute_tracker = ComputeBudgetTracker(config.max_recompute_ratio)
        
        # Layer statistics
        self.layer_memory_usage = {}
        self.layer_compute_time = {}
        self.layer_importance = {}
        
        # Checkpointing decisions
        self.checkpoint_decisions = {}
        self.adaptation_history = []
        
    def should_checkpoint(
        self,
        module: nn.Module,
        layer_name: str,
        layer_idx: int,
        input_size: Tuple[int, ...]
    ) -> bool:
        """Decide whether to checkpoint a specific layer"""
        
        # Check if adaptive checkpointing is enabled
        if not self.config.enable_adaptive_checkpointing:
            return self._static_checkpoint_decision(module, layer_name)
            
        # Get current system state
        memory_pressure = self.memory_monitor.get_memory_pressure()
        has_compute_budget = self.compute_tracker.has_budget()
        
        # Estimate layer memory usage
        estimated_memory_mb = self._estimate_activation_memory(module, input_size)
        
        # Make decision based on multiple factors
        decision_score = 0.0
        
        # Factor 1: Memory pressure
        if memory_pressure > self.config.critical_memory_threshold:
            decision_score += 1.0  # Always checkpoint under critical pressure
        elif memory_pressure > self.config.memory_pressure_threshold:
            decision_score += 0.5
            
        # Factor 2: Layer importance
        layer_type = self._get_layer_type(module, layer_name)
        if layer_type in self.config.checkpoint_probability:
            decision_score += self.config.checkpoint_probability[layer_type] * 0.3
            
        # Factor 3: Memory size
        if estimated_memory_mb > self.config.min_checkpoint_size_mb:
            decision_score += min(estimated_memory_mb / 100.0, 0.3)
            
        # Factor 4: Compute budget
        if not has_compute_budget:
            decision_score -= 0.5
            
        # Factor 5: Historical performance
        if layer_name in self.layer_importance:
            decision_score += self.layer_importance[layer_name] * 0.2
            
        # Make final decision
        should_checkpoint = decision_score > 0.5
        
        # Record decision
        self.checkpoint_decisions[layer_name] = {
            "checkpoint": should_checkpoint,
            "score": decision_score,
            "memory_pressure": memory_pressure,
            "estimated_memory_mb": estimated_memory_mb
        }
        
        return should_checkpoint
        
    def _static_checkpoint_decision(self, module: nn.Module, layer_name: str) -> bool:
        """Static checkpointing decision based on configuration"""
        layer_type = self._get_layer_type(module, layer_name)
        
        if layer_type == "attention" and self.config.checkpoint_attention:
            return True
        elif layer_type == "mlp" and self.config.checkpoint_mlp:
            return True
        elif layer_type == "moe_block" and self.config.checkpoint_experts:
            return True
            
        return False
        
    def _get_layer_type(self, module: nn.Module, layer_name: str) -> str:
        """Identify layer type from module and name"""
        module_type = type(module).__name__.lower()
        
        if "attention" in module_type or "attention" in layer_name.lower():
            return "attention"
        elif "moe" in module_type or "expert" in layer_name.lower():
            return "moe_block"
        elif "mlp" in module_type or "feedforward" in layer_name.lower():
            return "mlp"
        elif "embed" in module_type or "embed" in layer_name.lower():
            return "embedding"
        elif "norm" in module_type or "norm" in layer_name.lower():
            return "norm"
        else:
            return "other"
            
    def _estimate_activation_memory(
        self,
        module: nn.Module,
        input_size: Tuple[int, ...]
    ) -> float:
        """Estimate activation memory in MB"""
        # Simple estimation based on output size
        # This is a heuristic and can be improved
        
        total_elements = 1
        for dim in input_size:
            total_elements *= dim
            
        # Estimate based on module type
        if hasattr(module, "hidden_size"):
            output_elements = total_elements * module.hidden_size // input_size[-1]
        else:
            # Conservative estimate
            output_elements = total_elements * 4
            
        # 4 bytes per float32 element
        memory_bytes = output_elements * 4
        memory_mb = memory_bytes / (1024 * 1024)
        
        return memory_mb
        
    def update_layer_statistics(
        self,
        layer_name: str,
        compute_time: float,
        memory_usage: float,
        gradient_norm: float
    ):
        """Update layer statistics for adaptive decisions"""
        # Update compute time
        if layer_name not in self.layer_compute_time:
            self.layer_compute_time[layer_name] = []
        self.layer_compute_time[layer_name].append(compute_time)
        
        # Update memory usage
        if layer_name not in self.layer_memory_usage:
            self.layer_memory_usage[layer_name] = []
        self.layer_memory_usage[layer_name].append(memory_usage)
        
        # Update importance based on gradient norm
        if layer_name not in self.layer_importance:
            self.layer_importance[layer_name] = 0.0
            
        # Exponential moving average of gradient norm
        alpha = 0.1
        self.layer_importance[layer_name] = (
            alpha * gradient_norm + 
            (1 - alpha) * self.layer_importance[layer_name]
        )
        
    def adapt_policy(self):
        """Adapt checkpointing policy based on collected statistics"""
        if not self.config.enable_adaptive_checkpointing:
            return
            
        # Analyze recent performance
        recent_memory = self.memory_monitor.get_memory_pressure()
        recompute_ratio = self.compute_tracker.get_recompute_ratio()
        
        # Adjust checkpoint probabilities
        if recent_memory > 0.9 and recompute_ratio < self.config.max_recompute_ratio:
            # Increase checkpointing
            for layer_type in self.config.checkpoint_probability:
                self.config.checkpoint_probability[layer_type] = min(
                    1.0,
                    self.config.checkpoint_probability[layer_type] * 1.1
                )
        elif recent_memory < 0.6 and recompute_ratio > 1.2:
            # Decrease checkpointing
            for layer_type in self.config.checkpoint_probability:
                self.config.checkpoint_probability[layer_type] = max(
                    0.0,
                    self.config.checkpoint_probability[layer_type] * 0.9
                )
                
        # Record adaptation
        self.adaptation_history.append({
            "memory_pressure": recent_memory,
            "recompute_ratio": recompute_ratio,
            "checkpoint_probabilities": dict(self.config.checkpoint_probability)
        })
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of checkpointing decisions"""
        total_checkpointed = sum(
            1 for d in self.checkpoint_decisions.values() 
            if d["checkpoint"]
        )
        total_layers = len(self.checkpoint_decisions)
        
        return {
            "total_layers": total_layers,
            "checkpointed_layers": total_checkpointed,
            "checkpoint_ratio": total_checkpointed / max(total_layers, 1),
            "memory_pressure": self.memory_monitor.get_memory_pressure(),
            "recompute_ratio": self.compute_tracker.get_recompute_ratio(),
            "decisions": self.checkpoint_decisions
        }


def smart_checkpoint(
    func: Callable,
    policy: SmartCheckpointPolicy,
    layer_name: str,
    layer_idx: int
) -> Callable:
    """Decorator for smart checkpointing"""
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        module = args[0]  # Assuming first arg is self (module)
        inputs = args[1] if len(args) > 1 else None
        
        if inputs is not None and hasattr(inputs, "shape"):
            input_size = inputs.shape
        else:
            input_size = (1, 1, 1)  # Default
            
        # Decide whether to checkpoint
        should_checkpoint = policy.should_checkpoint(
            module, layer_name, layer_idx, input_size
        )
        
        if should_checkpoint and module.training:
            # Use gradient checkpointing
            import torch.utils.checkpoint as checkpoint
            
            # Time the recomputation
            start_time = policy.compute_tracker.start_timer()
            output = checkpoint.checkpoint(func, *args, **kwargs)
            policy.compute_tracker.end_timer(start_time, "recompute")
        else:
            # Normal forward
            start_time = policy.compute_tracker.start_timer()
            output = func(*args, **kwargs)
            policy.compute_tracker.end_timer(start_time, "forward")
            
        return output
        
    return wrapper


class ActivationOffloader:
    """
    Offloads activations to CPU/disk when memory pressure is high
    """
    
    def __init__(self, offload_device: str = "cpu", threshold_gb: float = 10.0):
        self.offload_device = offload_device
        self.threshold_gb = threshold_gb
        self.offloaded_tensors = {}
        
    def maybe_offload(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Offload tensor if it's large enough"""
        size_gb = tensor.element_size() * tensor.nelement() / 1e9
        
        if size_gb > self.threshold_gb:
            # Offload to CPU
            cpu_tensor = tensor.to(self.offload_device)
            self.offloaded_tensors[name] = {
                "device": tensor.device,
                "shape": tensor.shape,
                "dtype": tensor.dtype
            }
            logger.info(f"Offloaded {name} ({size_gb:.2f} GB) to {self.offload_device}")
            return cpu_tensor
            
        return tensor
        
    def maybe_reload(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Reload tensor if it was offloaded"""
        if name in self.offloaded_tensors:
            info = self.offloaded_tensors[name]
            gpu_tensor = tensor.to(info["device"])
            del self.offloaded_tensors[name]
            logger.info(f"Reloaded {name} to {info['device']}")
            return gpu_tensor
            
        return tensor