"""
Model Pruning for MoE++ Models

Implements various pruning strategies to reduce model size
while maintaining performance, with special focus on expert pruning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import copy
import logging

logger = logging.getLogger(__name__)


@dataclass
class PruningConfig:
    """Configuration for model pruning"""
    # Pruning targets
    target_sparsity: float = 0.5  # Target 50% sparsity
    prune_experts: bool = True
    prune_attention: bool = True
    prune_ffn: bool = True
    
    # Expert pruning
    min_experts_per_layer: int = 4  # Keep at least 4 experts
    expert_importance_metric: str = "usage"  # usage, gradient, performance
    expert_merge_threshold: float = 0.95  # Merge experts with >95% similarity
    
    # Structured vs unstructured
    pruning_type: str = "structured"  # structured, unstructured, hybrid
    structured_granularity: str = "channel"  # channel, head, layer
    
    # Importance metrics
    importance_metric: str = "magnitude"  # magnitude, gradient, taylor, fisher
    
    # Fine-tuning after pruning
    finetune_epochs: int = 5
    finetune_lr: float = 1e-4
    
    # Iterative pruning
    iterative_steps: int = 1  # Number of pruning iterations
    pruning_schedule: str = "linear"  # linear, exponential, constant
    
    # Evaluation
    eval_metric: str = "perplexity"
    acceptable_performance_drop: float = 0.05  # 5% drop acceptable


class ExpertPruner:
    """
    Specialized pruning for MoE experts
    """
    
    def __init__(self, model: nn.Module, config: PruningConfig):
        self.model = model
        self.config = config
        
        # Find MoE layers
        self.moe_layers = self._find_moe_layers(model)
        
        # Expert statistics
        self.expert_importance = defaultdict(dict)
        self.expert_similarity = defaultdict(dict)
        self.expert_usage = defaultdict(lambda: defaultdict(float))
        
    def _find_moe_layers(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Find all MoE layers in model"""
        moe_layers = {}
        for name, module in model.named_modules():
            if hasattr(module, 'experts') and hasattr(module, 'router'):
                moe_layers[name] = module
        return moe_layers
        
    def compute_expert_importance(
        self,
        dataloader: Any,
        num_samples: int = 1000
    ) -> Dict[str, Dict[int, float]]:
        """Compute importance scores for all experts"""
        logger.info("Computing expert importance scores...")
        
        # Reset statistics
        self.expert_usage.clear()
        gradient_importance = defaultdict(lambda: defaultdict(float))
        
        # Collect statistics
        sample_count = 0
        for batch in dataloader:
            if sample_count >= num_samples:
                break
                
            # Forward pass to get routing decisions
            with torch.no_grad():
                outputs = self.model(**batch, output_router_probs=True)
                
            # Track usage
            for layer_name, layer in self.moe_layers.items():
                if hasattr(outputs, 'router_probs') and layer_name in outputs.router_probs:
                    router_probs = outputs.router_probs[layer_name]
                    for expert_idx in range(layer.num_experts):
                        usage = router_probs[..., expert_idx].mean().item()
                        self.expert_usage[layer_name][expert_idx] += usage
                        
            # Backward pass for gradient importance
            if self.config.expert_importance_metric in ["gradient", "taylor"]:
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                loss.backward()
                
                # Compute gradient-based importance
                for layer_name, layer in self.moe_layers.items():
                    for expert_idx, expert in enumerate(layer.experts):
                        grad_norm = 0.0
                        for param in expert.parameters():
                            if param.grad is not None:
                                grad_norm += param.grad.norm().item() ** 2
                        gradient_importance[layer_name][expert_idx] += np.sqrt(grad_norm)
                        
                self.model.zero_grad()
                
            sample_count += batch['input_ids'].size(0)
            
        # Normalize and combine metrics
        for layer_name in self.moe_layers:
            num_experts = len(self.moe_layers[layer_name].experts)
            
            for expert_idx in range(num_experts):
                # Usage-based importance
                usage_score = self.expert_usage[layer_name][expert_idx] / max(sample_count, 1)
                
                # Gradient-based importance
                grad_score = gradient_importance[layer_name][expert_idx] / max(sample_count, 1)
                
                # Combine metrics
                if self.config.expert_importance_metric == "usage":
                    importance = usage_score
                elif self.config.expert_importance_metric == "gradient":
                    importance = grad_score
                else:  # Combined
                    importance = 0.7 * usage_score + 0.3 * grad_score
                    
                self.expert_importance[layer_name][expert_idx] = importance
                
        return self.expert_importance
        
    def compute_expert_similarity(self) -> Dict[str, np.ndarray]:
        """Compute pairwise similarity between experts"""
        logger.info("Computing expert similarity...")
        
        for layer_name, layer in self.moe_layers.items():
            num_experts = len(layer.experts)
            similarity_matrix = np.zeros((num_experts, num_experts))
            
            for i in range(num_experts):
                for j in range(i, num_experts):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        # Compute parameter similarity
                        sim = self._compute_parameter_similarity(
                            layer.experts[i],
                            layer.experts[j]
                        )
                        similarity_matrix[i, j] = sim
                        similarity_matrix[j, i] = sim
                        
            self.expert_similarity[layer_name] = similarity_matrix
            
        return self.expert_similarity
        
    def _compute_parameter_similarity(
        self,
        expert1: nn.Module,
        expert2: nn.Module
    ) -> float:
        """Compute cosine similarity between expert parameters"""
        params1 = torch.cat([p.flatten() for p in expert1.parameters()])
        params2 = torch.cat([p.flatten() for p in expert2.parameters()])
        
        similarity = F.cosine_similarity(
            params1.unsqueeze(0),
            params2.unsqueeze(0)
        ).item()
        
        return similarity
        
    def prune_experts(
        self,
        importance_scores: Optional[Dict[str, Dict[int, float]]] = None
    ) -> Dict[str, List[int]]:
        """Prune least important experts"""
        if importance_scores is None:
            importance_scores = self.expert_importance
            
        pruned_experts = {}
        
        for layer_name, layer in self.moe_layers.items():
            num_experts = len(layer.experts)
            num_to_keep = max(
                self.config.min_experts_per_layer,
                int(num_experts * (1 - self.config.target_sparsity))
            )
            
            # Sort experts by importance
            expert_scores = [
                (idx, importance_scores[layer_name].get(idx, 0))
                for idx in range(num_experts)
            ]
            expert_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Keep top experts
            keep_indices = [idx for idx, _ in expert_scores[:num_to_keep]]
            prune_indices = [idx for idx, _ in expert_scores[num_to_keep:]]
            
            # Actually remove experts
            self._remove_experts(layer, prune_indices, keep_indices)
            
            pruned_experts[layer_name] = prune_indices
            logger.info(f"Pruned {len(prune_indices)} experts from {layer_name}")
            
        return pruned_experts
        
    def merge_similar_experts(
        self,
        similarity_threshold: Optional[float] = None
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Merge highly similar experts"""
        if similarity_threshold is None:
            similarity_threshold = self.config.expert_merge_threshold
            
        # Compute similarities if not already done
        if not self.expert_similarity:
            self.compute_expert_similarity()
            
        merged_pairs = {}
        
        for layer_name, layer in self.moe_layers.items():
            similarity_matrix = self.expert_similarity[layer_name]
            layer_merged = []
            merged_set = set()
            
            # Find expert pairs to merge
            for i in range(len(layer.experts)):
                if i in merged_set:
                    continue
                    
                for j in range(i + 1, len(layer.experts)):
                    if j in merged_set:
                        continue
                        
                    if similarity_matrix[i, j] > similarity_threshold:
                        # Merge j into i based on importance
                        if self.expert_importance[layer_name][i] >= \
                           self.expert_importance[layer_name][j]:
                            self._merge_experts(layer, i, j)
                            layer_merged.append((i, j))
                            merged_set.add(j)
                        else:
                            self._merge_experts(layer, j, i)
                            layer_merged.append((j, i))
                            merged_set.add(i)
                            
            merged_pairs[layer_name] = layer_merged
            
            if layer_merged:
                logger.info(f"Merged {len(layer_merged)} expert pairs in {layer_name}")
                
        return merged_pairs
        
    def _remove_experts(
        self,
        layer: nn.Module,
        prune_indices: List[int],
        keep_indices: List[int]
    ):
        """Remove pruned experts from layer"""
        # Create new expert list with only kept experts
        new_experts = nn.ModuleList([
            layer.experts[idx] for idx in keep_indices
        ])
        
        # Update router to match new number of experts
        old_router = layer.router
        new_router = nn.Linear(
            old_router.in_features,
            len(new_experts),
            bias=old_router.bias is not None
        )
        
        # Copy weights for kept experts
        with torch.no_grad():
            for new_idx, old_idx in enumerate(keep_indices):
                new_router.weight[new_idx] = old_router.weight[old_idx]
                if new_router.bias is not None:
                    new_router.bias[new_idx] = old_router.bias[old_idx]
                    
        # Update layer
        layer.experts = new_experts
        layer.router = new_router
        layer.num_experts = len(new_experts)
        
    def _merge_experts(self, layer: nn.Module, keep_idx: int, merge_idx: int):
        """Merge one expert into another"""
        expert_keep = layer.experts[keep_idx]
        expert_merge = layer.experts[merge_idx]
        
        # Average parameters
        with torch.no_grad():
            for param_keep, param_merge in zip(
                expert_keep.parameters(),
                expert_merge.parameters()
            ):
                param_keep.data = (param_keep.data + param_merge.data) / 2
                
        # Update router weights
        layer.router.weight.data[keep_idx] = (
            layer.router.weight.data[keep_idx] +
            layer.router.weight.data[merge_idx]
        ) / 2
        
        if layer.router.bias is not None:
            layer.router.bias.data[keep_idx] = (
                layer.router.bias.data[keep_idx] +
                layer.router.bias.data[merge_idx]
            ) / 2


class StructuredPruner:
    """
    Structured pruning for attention heads and FFN channels
    """
    
    def __init__(self, model: nn.Module, config: PruningConfig):
        self.model = model
        self.config = config
        
    def prune_attention_heads(
        self,
        importance_scores: Dict[str, torch.Tensor]
    ) -> Dict[str, List[int]]:
        """Prune attention heads"""
        pruned_heads = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention) or \
               (hasattr(module, 'num_heads') and hasattr(module, 'head_dim')):
                if name not in importance_scores:
                    continue
                    
                head_importance = importance_scores[name]
                num_heads = len(head_importance)
                num_to_prune = int(num_heads * self.config.target_sparsity)
                
                # Get indices of heads to prune
                _, prune_indices = torch.topk(
                    head_importance,
                    num_to_prune,
                    largest=False
                )
                
                # Prune heads
                self._prune_heads(module, prune_indices.tolist())
                pruned_heads[name] = prune_indices.tolist()
                
        return pruned_heads
        
    def _prune_heads(self, module: nn.Module, head_indices: List[int]):
        """Actually prune attention heads"""
        # This is simplified - actual implementation depends on attention type
        if hasattr(module, 'num_heads'):
            remaining_heads = [
                i for i in range(module.num_heads)
                if i not in head_indices
            ]
            module.num_heads = len(remaining_heads)
            
            # Update projection layers accordingly
            # This would need proper dimension adjustment
            

class UnstructuredPruner:
    """
    Unstructured weight pruning (magnitude-based)
    """
    
    def __init__(self, model: nn.Module, config: PruningConfig):
        self.model = model
        self.config = config
        
    def compute_importance_scores(self) -> Dict[str, torch.Tensor]:
        """Compute importance scores for all parameters"""
        importance_scores = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if self.config.importance_metric == "magnitude":
                    scores = param.abs()
                elif self.config.importance_metric == "gradient":
                    scores = param.grad.abs() if param.grad is not None else param.abs()
                else:
                    scores = param.abs()  # Default to magnitude
                    
                importance_scores[name] = scores
                
        return importance_scores
        
    def apply_pruning_mask(
        self,
        importance_scores: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply pruning masks based on importance scores"""
        pruning_masks = {}
        
        for name, scores in importance_scores.items():
            if name.replace('.weight', '') in pruning_masks:
                continue
                
            # Compute threshold
            num_params = scores.numel()
            num_to_prune = int(num_params * self.config.target_sparsity)
            
            if num_to_prune > 0:
                threshold = torch.kthvalue(
                    scores.flatten(),
                    num_to_prune
                ).values
                
                # Create mask
                mask = (scores > threshold).float()
                pruning_masks[name] = mask
                
                # Apply mask
                param = dict(self.model.named_parameters())[name]
                param.data *= mask
                
        return pruning_masks


class ModelPruner:
    """
    Main pruning orchestrator
    """
    
    def __init__(self, model: nn.Module, config: PruningConfig):
        self.model = model
        self.config = config
        
        # Initialize pruners
        self.expert_pruner = ExpertPruner(model, config)
        self.structured_pruner = StructuredPruner(model, config)
        self.unstructured_pruner = UnstructuredPruner(model, config)
        
        # Track pruning history
        self.pruning_history = []
        
    def prune(
        self,
        dataloader: Any,
        eval_fn: Callable[[nn.Module], float]
    ) -> Dict[str, Any]:
        """Execute pruning pipeline"""
        logger.info("Starting pruning pipeline...")
        
        # Baseline evaluation
        baseline_metric = eval_fn(self.model)
        logger.info(f"Baseline {self.config.eval_metric}: {baseline_metric:.4f}")
        
        results = {
            "baseline_metric": baseline_metric,
            "pruning_stages": []
        }
        
        # Iterative pruning
        for iteration in range(self.config.iterative_steps):
            logger.info(f"Pruning iteration {iteration + 1}/{self.config.iterative_steps}")
            
            # Adjust sparsity for this iteration
            if self.config.pruning_schedule == "linear":
                current_sparsity = self.config.target_sparsity * (iteration + 1) / self.config.iterative_steps
            elif self.config.pruning_schedule == "exponential":
                current_sparsity = self.config.target_sparsity * (1 - np.exp(-2 * iteration))
            else:
                current_sparsity = self.config.target_sparsity
                
            stage_results = {}
            
            # Expert pruning
            if self.config.prune_experts:
                expert_importance = self.expert_pruner.compute_expert_importance(
                    dataloader
                )
                
                # Merge similar experts first
                merged_experts = self.expert_pruner.merge_similar_experts()
                stage_results["merged_experts"] = merged_experts
                
                # Then prune low-importance experts
                pruned_experts = self.expert_pruner.prune_experts(
                    expert_importance
                )
                stage_results["pruned_experts"] = pruned_experts
                
            # Structured pruning
            if self.config.pruning_type in ["structured", "hybrid"]:
                # Compute head importance
                head_importance = self._compute_head_importance(dataloader)
                
                if self.config.prune_attention:
                    pruned_heads = self.structured_pruner.prune_attention_heads(
                        head_importance
                    )
                    stage_results["pruned_heads"] = pruned_heads
                    
            # Unstructured pruning
            if self.config.pruning_type in ["unstructured", "hybrid"]:
                importance_scores = self.unstructured_pruner.compute_importance_scores()
                pruning_masks = self.unstructured_pruner.apply_pruning_mask(
                    importance_scores
                )
                stage_results["pruning_masks"] = {
                    k: v.sum().item() / v.numel()
                    for k, v in pruning_masks.items()
                }
                
            # Fine-tune after pruning
            if self.config.finetune_epochs > 0:
                logger.info(f"Fine-tuning for {self.config.finetune_epochs} epochs...")
                self._finetune(dataloader)
                
            # Evaluate
            current_metric = eval_fn(self.model)
            performance_drop = (baseline_metric - current_metric) / baseline_metric
            
            stage_results.update({
                "iteration": iteration,
                "sparsity": current_sparsity,
                "metric": current_metric,
                "performance_drop": performance_drop
            })
            
            results["pruning_stages"].append(stage_results)
            
            logger.info(f"Current {self.config.eval_metric}: {current_metric:.4f} "
                       f"(drop: {performance_drop:.2%})")
            
            # Stop if performance drop is too large
            if performance_drop > self.config.acceptable_performance_drop:
                logger.warning("Performance drop exceeds threshold, stopping pruning")
                break
                
        # Final statistics
        results["final_sparsity"] = self._compute_model_sparsity()
        results["parameter_reduction"] = self._compute_parameter_reduction()
        
        return results
        
    def _compute_head_importance(self, dataloader: Any) -> Dict[str, torch.Tensor]:
        """Compute importance scores for attention heads"""
        # Simplified - would need proper implementation
        head_importance = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'num_heads'):
                # Placeholder importance scores
                head_importance[name] = torch.randn(module.num_heads)
                
        return head_importance
        
    def _finetune(self, dataloader: Any):
        """Fine-tune model after pruning"""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.finetune_lr
        )
        
        self.model.train()
        for epoch in range(self.config.finetune_epochs):
            for batch in dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
    def _compute_model_sparsity(self) -> float:
        """Compute overall model sparsity"""
        total_params = 0
        zero_params = 0
        
        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
            
        return zero_params / total_params
        
    def _compute_parameter_reduction(self) -> Dict[str, float]:
        """Compute parameter reduction statistics"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Count remaining experts
        remaining_experts = 0
        for layer in self.expert_pruner.moe_layers.values():
            remaining_experts += len(layer.experts)
            
        return {
            "total_parameters": total_params,
            "remaining_experts": remaining_experts,
            "sparsity": self._compute_model_sparsity()
        }