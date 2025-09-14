"""
Expert Health Monitoring System

Monitors expert health, detects issues like expert collapse,
and provides diagnostic information for MoE models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    """Container for expert health metrics"""
    usage_frequency: Dict[int, float] = field(default_factory=dict)
    gradient_norm: Dict[int, float] = field(default_factory=dict)
    weight_drift: Dict[int, float] = field(default_factory=dict)
    output_variance: Dict[int, float] = field(default_factory=dict)
    routing_entropy: float = 0.0
    load_balance_score: float = 0.0
    expert_similarity: np.ndarray = None
    dead_experts: Set[int] = field(default_factory=set)
    
    
@dataclass
class HealthConfig:
    """Configuration for health monitoring"""
    # Monitoring intervals
    check_interval: int = 100  # Check health every N steps
    detailed_check_interval: int = 1000  # Detailed analysis interval
    
    # Thresholds
    min_usage_threshold: float = 0.01  # Below this is considered "dead"
    max_gradient_norm: float = 10.0  # Above this indicates instability
    weight_drift_threshold: float = 0.5  # Significant weight change
    min_output_variance: float = 0.01  # Below this indicates collapsed expert
    
    # History tracking
    history_window: int = 1000  # Number of steps to track
    
    # Alerting
    enable_alerts: bool = True
    alert_on_dead_experts: bool = True
    alert_on_collapse: bool = True
    alert_on_imbalance: bool = True
    
    # Visualization
    enable_visualization: bool = True
    visualization_interval: int = 5000


class ExpertHealthMonitor:
    """
    Comprehensive health monitoring for MoE models
    """
    
    def __init__(self, model: nn.Module, config: HealthConfig):
        self.model = model
        self.config = config
        
        # Find all MoE layers
        self.moe_layers = self._find_moe_layers(model)
        self.num_experts_per_layer = {
            name: len(layer.experts) 
            for name, layer in self.moe_layers.items()
        }
        
        # Initialize tracking
        self.step_count = 0
        self.health_history = defaultdict(lambda: deque(maxlen=config.history_window))
        self.alerts = []
        
        # Current metrics
        self.current_metrics = {
            layer_name: HealthMetrics()
            for layer_name in self.moe_layers.keys()
        }
        
        # Hooks for monitoring
        self._register_hooks()
        
    def _find_moe_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Find all MoE layers in the model"""
        moe_layers = {}
        
        for name, module in model.named_modules():
            # Check if this is an MoE layer
            if hasattr(module, 'experts') and hasattr(module, 'router'):
                moe_layers[name] = module
                logger.info(f"Found MoE layer: {name} with {len(module.experts)} experts")
                
        return moe_layers
        
    def _register_hooks(self):
        """Register hooks for monitoring"""
        self.hooks = []
        
        for layer_name, layer in self.moe_layers.items():
            # Router hook for usage tracking
            hook = layer.router.register_forward_hook(
                self._create_router_hook(layer_name)
            )
            self.hooks.append(hook)
            
            # Expert hooks for gradient and output tracking
            for expert_idx, expert in enumerate(layer.experts):
                # Gradient hook
                hook = expert.register_backward_hook(
                    self._create_gradient_hook(layer_name, expert_idx)
                )
                self.hooks.append(hook)
                
                # Output hook
                hook = expert.register_forward_hook(
                    self._create_output_hook(layer_name, expert_idx)
                )
                self.hooks.append(hook)
                
    def _create_router_hook(self, layer_name: str):
        """Create hook for router monitoring"""
        def hook(module, input, output):
            # Track routing decisions
            if isinstance(output, tuple):
                router_probs = output[0]  # Assuming first element is probabilities
            else:
                router_probs = output
                
            self._update_usage_frequency(layer_name, router_probs)
            self._update_routing_entropy(layer_name, router_probs)
            
        return hook
        
    def _create_gradient_hook(self, layer_name: str, expert_idx: int):
        """Create hook for gradient monitoring"""
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                grad_norm = grad_output[0].norm().item()
                self.current_metrics[layer_name].gradient_norm[expert_idx] = grad_norm
                
        return hook
        
    def _create_output_hook(self, layer_name: str, expert_idx: int):
        """Create hook for output monitoring"""
        def hook(module, input, output):
            if output is not None:
                output_var = output.var().item()
                self.current_metrics[layer_name].output_variance[expert_idx] = output_var
                
        return hook
        
    def _update_usage_frequency(self, layer_name: str, router_probs: torch.Tensor):
        """Update expert usage frequency"""
        # Average usage across batch and sequence
        usage = router_probs.mean(dim=[0, 1]) if router_probs.dim() > 2 else router_probs.mean(dim=0)
        
        for expert_idx, freq in enumerate(usage.tolist()):
            self.current_metrics[layer_name].usage_frequency[expert_idx] = freq
            
    def _update_routing_entropy(self, layer_name: str, router_probs: torch.Tensor):
        """Update routing entropy (diversity measure)"""
        # Compute entropy of routing distribution
        entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=-1)
        avg_entropy = entropy.mean().item()
        
        self.current_metrics[layer_name].routing_entropy = avg_entropy
        
    def step(self):
        """Called after each training step"""
        self.step_count += 1
        
        # Regular health check
        if self.step_count % self.config.check_interval == 0:
            self.check_health()
            
        # Detailed analysis
        if self.step_count % self.config.detailed_check_interval == 0:
            self.detailed_analysis()
            
        # Visualization
        if self.config.enable_visualization and \
           self.step_count % self.config.visualization_interval == 0:
            self.visualize_health()
            
    def check_health(self):
        """Perform health check"""
        for layer_name in self.moe_layers:
            metrics = self.current_metrics[layer_name]
            
            # Check for dead experts
            dead_experts = self._check_dead_experts(layer_name, metrics)
            if dead_experts and self.config.alert_on_dead_experts:
                self._raise_alert(
                    f"Dead experts detected in {layer_name}: {dead_experts}",
                    severity="warning"
                )
                
            # Check for expert collapse
            if self._check_expert_collapse(layer_name, metrics):
                if self.config.alert_on_collapse:
                    self._raise_alert(
                        f"Expert collapse detected in {layer_name}",
                        severity="critical"
                    )
                    
            # Check load balance
            load_balance = self._compute_load_balance(metrics)
            metrics.load_balance_score = load_balance
            
            if load_balance < 0.5 and self.config.alert_on_imbalance:
                self._raise_alert(
                    f"Load imbalance in {layer_name}: {load_balance:.3f}",
                    severity="warning"
                )
                
            # Update history
            self._update_history(layer_name, metrics)
            
    def _check_dead_experts(
        self,
        layer_name: str,
        metrics: HealthMetrics
    ) -> Set[int]:
        """Check for dead (unused) experts"""
        dead_experts = set()
        
        for expert_idx, usage in metrics.usage_frequency.items():
            if usage < self.config.min_usage_threshold:
                dead_experts.add(expert_idx)
                
        metrics.dead_experts = dead_experts
        return dead_experts
        
    def _check_expert_collapse(
        self,
        layer_name: str,
        metrics: HealthMetrics
    ) -> bool:
        """Check if experts have collapsed (producing similar outputs)"""
        if len(metrics.output_variance) < 2:
            return False
            
        # Check if all experts have low variance
        variances = list(metrics.output_variance.values())
        avg_variance = np.mean(variances)
        
        if avg_variance < self.config.min_output_variance:
            return True
            
        # Check if outputs are too similar across experts
        # This requires comparing expert outputs directly
        # Simplified check based on variance similarity
        variance_std = np.std(variances)
        if variance_std < self.config.min_output_variance * 0.1:
            return True
            
        return False
        
    def _compute_load_balance(self, metrics: HealthMetrics) -> float:
        """Compute load balance score (0-1, higher is better)"""
        if not metrics.usage_frequency:
            return 0.0
            
        usage_values = list(metrics.usage_frequency.values())
        
        # Perfect balance would be uniform distribution
        num_experts = len(usage_values)
        ideal_usage = 1.0 / num_experts
        
        # Compute deviation from ideal
        deviations = [abs(usage - ideal_usage) for usage in usage_values]
        avg_deviation = np.mean(deviations)
        
        # Convert to 0-1 score
        load_balance = 1.0 - min(avg_deviation / ideal_usage, 1.0)
        
        return load_balance
        
    def detailed_analysis(self):
        """Perform detailed health analysis"""
        logger.info("Performing detailed expert health analysis...")
        
        for layer_name, layer in self.moe_layers.items():
            metrics = self.current_metrics[layer_name]
            
            # Compute expert similarity
            similarity_matrix = self._compute_expert_similarity(layer)
            metrics.expert_similarity = similarity_matrix
            
            # Check for weight drift
            weight_drift = self._check_weight_drift(layer_name, layer)
            for expert_idx, drift in weight_drift.items():
                metrics.weight_drift[expert_idx] = drift
                
            # Diagnose issues
            issues = self._diagnose_issues(layer_name, metrics)
            if issues:
                logger.warning(f"Issues in {layer_name}: {issues}")
                
    def _compute_expert_similarity(self, layer: nn.Module) -> np.ndarray:
        """Compute pairwise similarity between experts"""
        num_experts = len(layer.experts)
        similarity_matrix = np.zeros((num_experts, num_experts))
        
        for i in range(num_experts):
            for j in range(i, num_experts):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Compute cosine similarity between expert parameters
                    sim = self._cosine_similarity_experts(
                        layer.experts[i],
                        layer.experts[j]
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
                    
        return similarity_matrix
        
    def _cosine_similarity_experts(
        self,
        expert1: nn.Module,
        expert2: nn.Module
    ) -> float:
        """Compute cosine similarity between two experts"""
        # Flatten all parameters
        params1 = torch.cat([p.flatten() for p in expert1.parameters()])
        params2 = torch.cat([p.flatten() for p in expert2.parameters()])
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(
            params1.unsqueeze(0),
            params2.unsqueeze(0)
        ).item()
        
        return similarity
        
    def _check_weight_drift(
        self,
        layer_name: str,
        layer: nn.Module
    ) -> Dict[int, float]:
        """Check how much weights have drifted from initialization"""
        # This requires storing initial weights
        # Simplified version - check weight magnitude
        weight_drift = {}
        
        for expert_idx, expert in enumerate(layer.experts):
            total_norm = 0.0
            param_count = 0
            
            for param in expert.parameters():
                total_norm += param.norm().item()
                param_count += param.numel()
                
            avg_norm = total_norm / max(param_count, 1)
            weight_drift[expert_idx] = avg_norm
            
        return weight_drift
        
    def _diagnose_issues(
        self,
        layer_name: str,
        metrics: HealthMetrics
    ) -> List[str]:
        """Diagnose potential issues"""
        issues = []
        
        # Check for dead experts
        if metrics.dead_experts:
            issues.append(f"Dead experts: {metrics.dead_experts}")
            
        # Check for low entropy (lack of diversity)
        if metrics.routing_entropy < 0.5:
            issues.append(f"Low routing entropy: {metrics.routing_entropy:.3f}")
            
        # Check for high gradient norms
        for expert_idx, grad_norm in metrics.gradient_norm.items():
            if grad_norm > self.config.max_gradient_norm:
                issues.append(f"High gradient norm in expert {expert_idx}: {grad_norm:.3f}")
                
        # Check for expert similarity
        if metrics.expert_similarity is not None:
            # Check if any experts are too similar
            similarity_threshold = 0.95
            high_similarity_pairs = []
            
            for i in range(metrics.expert_similarity.shape[0]):
                for j in range(i+1, metrics.expert_similarity.shape[1]):
                    if metrics.expert_similarity[i, j] > similarity_threshold:
                        high_similarity_pairs.append((i, j))
                        
            if high_similarity_pairs:
                issues.append(f"Highly similar experts: {high_similarity_pairs}")
                
        return issues
        
    def _update_history(self, layer_name: str, metrics: HealthMetrics):
        """Update historical metrics"""
        # Store key metrics
        self.health_history[f"{layer_name}_usage"].append(
            list(metrics.usage_frequency.values())
        )
        self.health_history[f"{layer_name}_entropy"].append(
            metrics.routing_entropy
        )
        self.health_history[f"{layer_name}_load_balance"].append(
            metrics.load_balance_score
        )
        
    def _raise_alert(self, message: str, severity: str = "warning"):
        """Raise an alert"""
        alert = {
            "timestamp": datetime.now(),
            "step": self.step_count,
            "message": message,
            "severity": severity
        }
        
        self.alerts.append(alert)
        
        if self.config.enable_alerts:
            if severity == "critical":
                logger.error(f"CRITICAL: {message}")
            elif severity == "warning":
                logger.warning(f"WARNING: {message}")
            else:
                logger.info(f"INFO: {message}")
                
    def visualize_health(self):
        """Create visualizations of expert health"""
        if not self.config.enable_visualization:
            return
            
        num_layers = len(self.moe_layers)
        fig, axes = plt.subplots(num_layers, 3, figsize=(15, 5 * num_layers))
        
        if num_layers == 1:
            axes = axes.reshape(1, -1)
            
        for idx, (layer_name, metrics) in enumerate(self.current_metrics.items()):
            # Usage frequency heatmap
            self._plot_usage_frequency(axes[idx, 0], layer_name, metrics)
            
            # Expert similarity matrix
            self._plot_expert_similarity(axes[idx, 1], layer_name, metrics)
            
            # Health metrics over time
            self._plot_health_timeline(axes[idx, 2], layer_name)
            
        plt.tight_layout()
        plt.savefig(f"expert_health_step_{self.step_count}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
    def _plot_usage_frequency(self, ax, layer_name: str, metrics: HealthMetrics):
        """Plot expert usage frequency"""
        expert_ids = list(range(self.num_experts_per_layer[layer_name]))
        usage_freq = [metrics.usage_frequency.get(i, 0) for i in expert_ids]
        
        # Color dead experts differently
        colors = ['red' if i in metrics.dead_experts else 'blue' for i in expert_ids]
        
        ax.bar(expert_ids, usage_freq, color=colors)
        ax.set_xlabel("Expert ID")
        ax.set_ylabel("Usage Frequency")
        ax.set_title(f"{layer_name} - Expert Usage")
        
        # Add threshold line
        ax.axhline(
            y=self.config.min_usage_threshold,
            color='red',
            linestyle='--',
            label='Death threshold'
        )
        ax.legend()
        
    def _plot_expert_similarity(self, ax, layer_name: str, metrics: HealthMetrics):
        """Plot expert similarity matrix"""
        if metrics.expert_similarity is not None:
            sns.heatmap(
                metrics.expert_similarity,
                ax=ax,
                cmap='coolwarm',
                center=0.5,
                vmin=0,
                vmax=1,
                square=True,
                cbar_kws={'label': 'Cosine Similarity'}
            )
            ax.set_title(f"{layer_name} - Expert Similarity")
            ax.set_xlabel("Expert ID")
            ax.set_ylabel("Expert ID")
            
    def _plot_health_timeline(self, ax, layer_name: str):
        """Plot health metrics over time"""
        # Get historical data
        entropy_history = self.health_history[f"{layer_name}_entropy"]
        balance_history = self.health_history[f"{layer_name}_load_balance"]
        
        if entropy_history:
            steps = list(range(len(entropy_history)))
            ax.plot(steps, entropy_history, label='Routing Entropy', color='green')
            ax.plot(steps, balance_history, label='Load Balance', color='blue')
            
            ax.set_xlabel("Steps")
            ax.set_ylabel("Score")
            ax.set_title(f"{layer_name} - Health Metrics Timeline")
            ax.legend()
            ax.set_ylim(0, 1)
            
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of current health status"""
        summary = {
            "step": self.step_count,
            "layers": {}
        }
        
        for layer_name, metrics in self.current_metrics.items():
            layer_summary = {
                "num_dead_experts": len(metrics.dead_experts),
                "avg_usage": np.mean(list(metrics.usage_frequency.values())) if metrics.usage_frequency else 0,
                "routing_entropy": metrics.routing_entropy,
                "load_balance": metrics.load_balance_score,
                "alerts": [a for a in self.alerts if layer_name in a["message"]]
            }
            summary["layers"][layer_name] = layer_summary
            
        return summary
        
    def cleanup(self):
        """Remove hooks and cleanup"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def save_report(self, filepath: str):
        """Save detailed health report"""
        report = {
            "summary": self.get_health_summary(),
            "alerts": self.alerts,
            "detailed_metrics": {
                layer: {
                    "usage_frequency": metrics.usage_frequency,
                    "gradient_norms": metrics.gradient_norm,
                    "weight_drift": metrics.weight_drift,
                    "output_variance": metrics.output_variance
                }
                for layer, metrics in self.current_metrics.items()
            }
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)