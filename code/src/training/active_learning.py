"""
Active Learning for MoE++ Models

Implements active learning strategies to select the most
informative samples for training, improving data efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
import heapq
import logging

logger = logging.getLogger(__name__)


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning"""
    # Selection strategy
    strategy: str = "uncertainty"  # uncertainty, diversity, hybrid, badge
    selection_ratio: float = 0.1  # Select top 10% of samples
    
    # Uncertainty methods
    uncertainty_method: str = "entropy"  # entropy, margin, variation_ratio
    mc_samples: int = 10  # Monte Carlo samples for uncertainty
    
    # Diversity methods
    diversity_method: str = "coreset"  # coreset, kmeans, determinantal
    num_clusters: int = 100
    
    # Hybrid settings
    uncertainty_weight: float = 0.7
    diversity_weight: float = 0.3
    
    # Pool settings
    pool_size: int = 10000  # Maximum unlabeled pool size
    batch_size: int = 32  # Samples to select per round
    
    # Expert-specific
    use_expert_uncertainty: bool = True  # Use expert routing uncertainty
    
    # Performance tracking
    track_performance: bool = True
    validation_interval: int = 100


class ActiveLearningSelector:
    """
    Selects most informative samples for training
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ActiveLearningConfig,
        labeled_pool: Optional[Any] = None,
        unlabeled_pool: Optional[Any] = None
    ):
        self.model = model
        self.config = config
        self.labeled_pool = labeled_pool or []
        self.unlabeled_pool = unlabeled_pool or []
        
        # Selection strategies
        self.selection_strategies = {
            "uncertainty": self._uncertainty_sampling,
            "diversity": self._diversity_sampling,
            "hybrid": self._hybrid_sampling,
            "badge": self._badge_sampling,
            "random": self._random_sampling
        }
        
        # Performance tracking
        self.selection_history = []
        self.performance_history = []
        
    def select_batch(
        self,
        unlabeled_data: Optional[List[Dict[str, torch.Tensor]]] = None,
        num_samples: Optional[int] = None
    ) -> Tuple[List[int], Dict[str, float]]:
        """
        Select a batch of samples from unlabeled pool
        
        Returns:
            indices: Indices of selected samples
            metrics: Selection metrics
        """
        if unlabeled_data is None:
            unlabeled_data = self.unlabeled_pool
            
        if num_samples is None:
            num_samples = self.config.batch_size
            
        # Ensure we don't select more than available
        num_samples = min(num_samples, len(unlabeled_data))
        
        # Apply selection strategy
        strategy_fn = self.selection_strategies[self.config.strategy]
        selected_indices, scores = strategy_fn(unlabeled_data, num_samples)
        
        # Track selection
        self.selection_history.append({
            "indices": selected_indices,
            "scores": scores,
            "strategy": self.config.strategy
        })
        
        # Compute metrics
        metrics = self._compute_selection_metrics(selected_indices, scores)
        
        return selected_indices, metrics
        
    def _uncertainty_sampling(
        self,
        unlabeled_data: List[Dict[str, torch.Tensor]],
        num_samples: int
    ) -> Tuple[List[int], List[float]]:
        """Select samples with highest uncertainty"""
        uncertainties = []
        
        self.model.eval()
        with torch.no_grad():
            for idx, sample in enumerate(unlabeled_data):
                uncertainty = self._compute_uncertainty(sample)
                uncertainties.append((idx, uncertainty))
                
        # Sort by uncertainty (descending)
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        
        # Select top-k
        selected_indices = [idx for idx, _ in uncertainties[:num_samples]]
        scores = [score for _, score in uncertainties[:num_samples]]
        
        return selected_indices, scores
        
    def _compute_uncertainty(
        self,
        sample: Dict[str, torch.Tensor]
    ) -> float:
        """Compute uncertainty for a single sample"""
        if self.config.uncertainty_method == "entropy":
            return self._entropy_uncertainty(sample)
        elif self.config.uncertainty_method == "margin":
            return self._margin_uncertainty(sample)
        elif self.config.uncertainty_method == "variation_ratio":
            return self._variation_ratio_uncertainty(sample)
        else:
            raise ValueError(f"Unknown uncertainty method: {self.config.uncertainty_method}")
            
    def _entropy_uncertainty(
        self,
        sample: Dict[str, torch.Tensor]
    ) -> float:
        """Compute entropy-based uncertainty"""
        # Get predictions with MC dropout
        predictions = []
        
        for _ in range(self.config.mc_samples):
            self.model.train()  # Enable dropout
            with torch.no_grad():
                output = self.model(**sample)
                logits = output.logits if hasattr(output, 'logits') else output
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)
                
        # Stack predictions
        predictions = torch.stack(predictions)
        
        # Compute mean prediction
        mean_probs = predictions.mean(dim=0)
        
        # Compute entropy
        entropy_val = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        
        # Also consider expert uncertainty if available
        if self.config.use_expert_uncertainty and hasattr(output, 'router_logits'):
            expert_entropy = self._compute_expert_entropy(output.router_logits)
            entropy_val = 0.7 * entropy_val + 0.3 * expert_entropy
            
        return entropy_val.mean().item()
        
    def _margin_uncertainty(
        self,
        sample: Dict[str, torch.Tensor]
    ) -> float:
        """Compute margin-based uncertainty (difference between top 2 probs)"""
        with torch.no_grad():
            output = self.model(**sample)
            logits = output.logits if hasattr(output, 'logits') else output
            probs = F.softmax(logits, dim=-1)
            
        # Get top 2 probabilities
        top2_probs, _ = torch.topk(probs, k=2, dim=-1)
        
        # Margin is difference between highest and second highest
        margin = top2_probs[..., 0] - top2_probs[..., 1]
        
        # Lower margin = higher uncertainty
        uncertainty = 1.0 - margin.mean().item()
        
        return uncertainty
        
    def _variation_ratio_uncertainty(
        self,
        sample: Dict[str, torch.Tensor]
    ) -> float:
        """Compute variation ratio uncertainty"""
        predictions = []
        
        for _ in range(self.config.mc_samples):
            self.model.train()
            with torch.no_grad():
                output = self.model(**sample)
                logits = output.logits if hasattr(output, 'logits') else output
                pred_class = logits.argmax(dim=-1)
                predictions.append(pred_class)
                
        predictions = torch.stack(predictions)
        
        # Compute variation ratio (1 - fraction of mode)
        mode_count = torch.mode(predictions, dim=0).values
        variation_ratio = 1.0 - (mode_count.float() / self.config.mc_samples)
        
        return variation_ratio.mean().item()
        
    def _compute_expert_entropy(
        self,
        router_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute entropy of expert routing"""
        router_probs = F.softmax(router_logits, dim=-1)
        expert_entropy = -torch.sum(
            router_probs * torch.log(router_probs + 1e-8),
            dim=-1
        )
        return expert_entropy.mean()
        
    def _diversity_sampling(
        self,
        unlabeled_data: List[Dict[str, torch.Tensor]],
        num_samples: int
    ) -> Tuple[List[int], List[float]]:
        """Select diverse samples"""
        if self.config.diversity_method == "kmeans":
            return self._kmeans_sampling(unlabeled_data, num_samples)
        elif self.config.diversity_method == "coreset":
            return self._coreset_sampling(unlabeled_data, num_samples)
        else:
            raise ValueError(f"Unknown diversity method: {self.config.diversity_method}")
            
    def _kmeans_sampling(
        self,
        unlabeled_data: List[Dict[str, torch.Tensor]],
        num_samples: int
    ) -> Tuple[List[int], List[float]]:
        """K-means based diverse sampling"""
        # Get representations
        representations = self._get_representations(unlabeled_data)
        
        # Run k-means
        kmeans = KMeans(n_clusters=min(num_samples, len(unlabeled_data)))
        cluster_labels = kmeans.fit_predict(representations)
        
        # Select samples closest to cluster centers
        selected_indices = []
        scores = []
        
        for cluster_id in range(kmeans.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            # Find sample closest to cluster center
            cluster_reps = representations[cluster_indices]
            distances = np.linalg.norm(
                cluster_reps - kmeans.cluster_centers_[cluster_id],
                axis=1
            )
            closest_idx = cluster_indices[np.argmin(distances)]
            
            selected_indices.append(closest_idx)
            scores.append(1.0 / (distances.min() + 1e-8))  # Inverse distance as score
            
        return selected_indices, scores
        
    def _coreset_sampling(
        self,
        unlabeled_data: List[Dict[str, torch.Tensor]],
        num_samples: int
    ) -> Tuple[List[int], List[float]]:
        """Coreset-based diverse sampling (greedy)"""
        representations = self._get_representations(unlabeled_data)
        
        # Initialize with random sample
        selected_indices = [np.random.randint(len(unlabeled_data))]
        scores = [1.0]
        
        # Greedily add samples that maximize minimum distance
        while len(selected_indices) < num_samples:
            selected_reps = representations[selected_indices]
            
            # Compute minimum distances to selected set
            min_distances = []
            for i in range(len(representations)):
                if i not in selected_indices:
                    distances = np.linalg.norm(
                        representations[i] - selected_reps,
                        axis=1
                    )
                    min_distances.append((i, distances.min()))
                    
            # Select sample with maximum minimum distance
            next_idx = max(min_distances, key=lambda x: x[1])[0]
            selected_indices.append(next_idx)
            scores.append(min_distances[next_idx][1])
            
        return selected_indices, scores
        
    def _get_representations(
        self,
        data: List[Dict[str, torch.Tensor]]
    ) -> np.ndarray:
        """Get feature representations for samples"""
        representations = []
        
        self.model.eval()
        with torch.no_grad():
            for sample in data:
                output = self.model(**sample, output_hidden_states=True)
                
                # Use last hidden state
                hidden_states = output.hidden_states[-1]
                
                # Pool to get sample representation
                pooled = hidden_states.mean(dim=1).squeeze()
                representations.append(pooled.cpu().numpy())
                
        return np.array(representations)
        
    def _hybrid_sampling(
        self,
        unlabeled_data: List[Dict[str, torch.Tensor]],
        num_samples: int
    ) -> Tuple[List[int], List[float]]:
        """Combine uncertainty and diversity sampling"""
        # Get uncertainty scores
        uncertainties = []
        for idx, sample in enumerate(unlabeled_data):
            uncertainty = self._compute_uncertainty(sample)
            uncertainties.append(uncertainty)
            
        uncertainties = np.array(uncertainties)
        
        # Get diversity scores
        representations = self._get_representations(unlabeled_data)
        
        # Normalize scores
        norm_uncertainties = (uncertainties - uncertainties.min()) / \
                           (uncertainties.max() - uncertainties.min() + 1e-8)
                           
        # Iteratively select samples
        selected_indices = []
        scores = []
        
        for _ in range(num_samples):
            # Compute diversity scores relative to selected set
            if selected_indices:
                selected_reps = representations[selected_indices]
                diversity_scores = []
                
                for i in range(len(representations)):
                    if i not in selected_indices:
                        distances = np.linalg.norm(
                            representations[i] - selected_reps,
                            axis=1
                        )
                        diversity_scores.append(distances.min())
                    else:
                        diversity_scores.append(0.0)
                        
                diversity_scores = np.array(diversity_scores)
                norm_diversity = (diversity_scores - diversity_scores.min()) / \
                               (diversity_scores.max() - diversity_scores.min() + 1e-8)
            else:
                norm_diversity = np.ones_like(norm_uncertainties)
                
            # Combine scores
            combined_scores = (
                self.config.uncertainty_weight * norm_uncertainties +
                self.config.diversity_weight * norm_diversity
            )
            
            # Mask already selected
            for idx in selected_indices:
                combined_scores[idx] = -np.inf
                
            # Select best
            best_idx = np.argmax(combined_scores)
            selected_indices.append(best_idx)
            scores.append(combined_scores[best_idx])
            
        return selected_indices, scores
        
    def _badge_sampling(
        self,
        unlabeled_data: List[Dict[str, torch.Tensor]],
        num_samples: int
    ) -> Tuple[List[int], List[float]]:
        """BADGE sampling (gradient embeddings + k-means++)"""
        # Get gradient embeddings
        gradient_embeddings = self._get_gradient_embeddings(unlabeled_data)
        
        # Use k-means++ initialization to select diverse gradients
        selected_indices = []
        scores = []
        
        # First sample random
        first_idx = np.random.randint(len(gradient_embeddings))
        selected_indices.append(first_idx)
        scores.append(1.0)
        
        # Iteratively select based on distance to nearest selected
        while len(selected_indices) < num_samples:
            distances = []
            
            for i in range(len(gradient_embeddings)):
                if i not in selected_indices:
                    # Distance to nearest selected gradient
                    selected_grads = gradient_embeddings[selected_indices]
                    dists = np.linalg.norm(
                        gradient_embeddings[i] - selected_grads,
                        axis=1
                    )
                    distances.append((i, dists.min()))
                    
            # Probabilistically select based on distances
            indices, dists = zip(*distances)
            dists = np.array(dists)
            probs = dists / dists.sum()
            
            next_idx = np.random.choice(indices, p=probs)
            selected_indices.append(next_idx)
            scores.append(dists[indices.index(next_idx)])
            
        return selected_indices, scores
        
    def _get_gradient_embeddings(
        self,
        data: List[Dict[str, torch.Tensor]]
    ) -> np.ndarray:
        """Get gradient embeddings for BADGE"""
        gradient_embeddings = []
        
        self.model.train()
        for sample in data:
            # Forward pass
            output = self.model(**sample)
            logits = output.logits if hasattr(output, 'logits') else output
            
            # Compute hypothetical labels (max prediction)
            pseudo_labels = logits.argmax(dim=-1)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                pseudo_labels.view(-1),
                reduction='mean'
            )
            
            # Get gradients of last layer
            self.model.zero_grad()
            loss.backward()
            
            # Extract gradient embedding (last layer gradients)
            grad_embedding = []
            for name, param in self.model.named_parameters():
                if 'output' in name or 'head' in name:  # Last layer
                    if param.grad is not None:
                        grad_embedding.append(param.grad.flatten())
                        
            if grad_embedding:
                grad_embedding = torch.cat(grad_embedding)
                gradient_embeddings.append(grad_embedding.cpu().numpy())
            else:
                # Fallback to zero embedding
                gradient_embeddings.append(np.zeros(100))
                
        return np.array(gradient_embeddings)
        
    def _random_sampling(
        self,
        unlabeled_data: List[Dict[str, torch.Tensor]],
        num_samples: int
    ) -> Tuple[List[int], List[float]]:
        """Random sampling baseline"""
        indices = np.random.choice(
            len(unlabeled_data),
            size=num_samples,
            replace=False
        ).tolist()
        
        scores = [1.0] * num_samples  # Uniform scores
        
        return indices, scores
        
    def _compute_selection_metrics(
        self,
        selected_indices: List[int],
        scores: List[float]
    ) -> Dict[str, float]:
        """Compute metrics about the selection"""
        return {
            "num_selected": len(selected_indices),
            "avg_score": np.mean(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores)
        }
        
    def update_pools(
        self,
        new_labeled_indices: List[int],
        new_unlabeled_data: Optional[List[Dict[str, torch.Tensor]]] = None
    ):
        """Update labeled and unlabeled pools after selection"""
        # Move selected samples from unlabeled to labeled
        for idx in sorted(new_labeled_indices, reverse=True):
            if idx < len(self.unlabeled_pool):
                sample = self.unlabeled_pool.pop(idx)
                self.labeled_pool.append(sample)
                
        # Add new unlabeled data if provided
        if new_unlabeled_data:
            self.unlabeled_pool.extend(new_unlabeled_data)
            
        # Trim pool if too large
        if len(self.unlabeled_pool) > self.config.pool_size:
            # Keep most recent samples
            self.unlabeled_pool = self.unlabeled_pool[-self.config.pool_size:]
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get active learning statistics"""
        return {
            "labeled_pool_size": len(self.labeled_pool),
            "unlabeled_pool_size": len(self.unlabeled_pool),
            "total_selections": len(self.selection_history),
            "strategy": self.config.strategy,
            "performance_history": self.performance_history
        }