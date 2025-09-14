"""
Curriculum Learning for MoE++ Models

Implements various curriculum learning strategies to improve training efficiency
and model quality by gradually increasing task difficulty.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning"""
    warmup_steps: int = 10000
    max_sequence_length: int = 2048
    min_sequence_length: int = 128
    
    # Difficulty levels
    difficulty_stages: List[str] = None
    
    # Expert curriculum
    start_num_experts: int = 16
    final_num_experts: int = 64
    expert_growth_steps: int = 50000
    
    # Attention curriculum  
    start_attention_window: int = 256
    final_attention_window: int = 2048
    
    # Data mixing curriculum
    start_data_mix: Dict[str, float] = None
    final_data_mix: Dict[str, float] = None
    
    # Loss weighting curriculum
    start_loss_weights: Dict[str, float] = None
    final_loss_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.difficulty_stages is None:
            self.difficulty_stages = ["easy", "medium", "hard", "expert"]
            
        if self.start_data_mix is None:
            self.start_data_mix = {
                "wikipedia": 0.5,
                "books": 0.3,
                "web": 0.2
            }
            
        if self.final_data_mix is None:
            self.final_data_mix = {
                "wikipedia": 0.2,
                "books": 0.2,
                "web": 0.2,
                "code": 0.2,
                "math": 0.1,
                "science": 0.1
            }
            
        if self.start_loss_weights is None:
            self.start_loss_weights = {
                "lm_loss": 1.0,
                "auxiliary_loss": 0.01
            }
            
        if self.final_loss_weights is None:
            self.final_loss_weights = {
                "lm_loss": 1.0,
                "auxiliary_loss": 0.1,
                "regularization_loss": 0.01
            }


class CurriculumScheduler:
    """
    Main curriculum scheduler that coordinates all curriculum strategies
    """
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_step = 0
        self.metrics_history = defaultdict(list)
        
    def step(self) -> Dict[str, Any]:
        """Get current curriculum configuration"""
        self.current_step += 1
        
        return {
            "sequence_length": self._get_sequence_length(),
            "difficulty": self._get_difficulty(),
            "num_active_experts": self._get_num_experts(),
            "attention_window": self._get_attention_window(),
            "data_mix": self._get_data_mix(),
            "loss_weights": self._get_loss_weights(),
            "learning_rate_mult": self._get_lr_multiplier(),
            "dropout_rate": self._get_dropout_rate()
        }
        
    def _get_sequence_length(self) -> int:
        """Gradually increase sequence length"""
        if self.current_step >= self.config.warmup_steps:
            return self.config.max_sequence_length
            
        progress = self.current_step / self.config.warmup_steps
        length_range = self.config.max_sequence_length - self.config.min_sequence_length
        current_length = self.config.min_sequence_length + int(progress * length_range)
        
        return min(current_length, self.config.max_sequence_length)
        
    def _get_difficulty(self) -> str:
        """Get current difficulty stage"""
        stages = self.config.difficulty_stages
        stage_duration = self.config.warmup_steps / len(stages)
        stage_idx = min(int(self.current_step / stage_duration), len(stages) - 1)
        return stages[stage_idx]
        
    def _get_num_experts(self) -> int:
        """Gradually increase number of active experts"""
        if self.current_step >= self.config.expert_growth_steps:
            return self.config.final_num_experts
            
        progress = self.current_step / self.config.expert_growth_steps
        expert_range = self.config.final_num_experts - self.config.start_num_experts
        current_experts = self.config.start_num_experts + int(progress * expert_range)
        
        # Round to nearest power of 2 for efficiency
        return int(2 ** round(np.log2(current_experts)))
        
    def _get_attention_window(self) -> int:
        """Gradually increase attention window size"""
        if self.current_step >= self.config.warmup_steps:
            return self.config.final_attention_window
            
        progress = self.current_step / self.config.warmup_steps
        window_range = self.config.final_attention_window - self.config.start_attention_window
        current_window = self.config.start_attention_window + int(progress * window_range)
        
        return min(current_window, self.config.final_attention_window)
        
    def _get_data_mix(self) -> Dict[str, float]:
        """Gradually shift data mixture"""
        if self.current_step >= self.config.warmup_steps:
            return self.config.final_data_mix
            
        progress = self.current_step / self.config.warmup_steps
        current_mix = {}
        
        # Interpolate between start and final mix
        all_domains = set(self.config.start_data_mix.keys()) | set(self.config.final_data_mix.keys())
        
        for domain in all_domains:
            start_weight = self.config.start_data_mix.get(domain, 0.0)
            final_weight = self.config.final_data_mix.get(domain, 0.0)
            current_mix[domain] = start_weight + progress * (final_weight - start_weight)
            
        # Normalize to sum to 1
        total = sum(current_mix.values())
        return {k: v / total for k, v in current_mix.items()}
        
    def _get_loss_weights(self) -> Dict[str, float]:
        """Gradually adjust loss weights"""
        if self.current_step >= self.config.warmup_steps:
            return self.config.final_loss_weights
            
        progress = self.current_step / self.config.warmup_steps
        current_weights = {}
        
        all_losses = set(self.config.start_loss_weights.keys()) | set(self.config.final_loss_weights.keys())
        
        for loss_name in all_losses:
            start_weight = self.config.start_loss_weights.get(loss_name, 0.0)
            final_weight = self.config.final_loss_weights.get(loss_name, 0.0)
            current_weights[loss_name] = start_weight + progress * (final_weight - start_weight)
            
        return current_weights
        
    def _get_lr_multiplier(self) -> float:
        """Learning rate warmup multiplier"""
        if self.current_step >= self.config.warmup_steps:
            return 1.0
            
        # Linear warmup
        return self.current_step / self.config.warmup_steps
        
    def _get_dropout_rate(self) -> float:
        """Gradually increase dropout for regularization"""
        if self.current_step < self.config.warmup_steps // 2:
            return 0.0  # No dropout initially
        elif self.current_step >= self.config.warmup_steps:
            return 0.1  # Full dropout
        else:
            # Linear increase from 0 to 0.1
            progress = (self.current_step - self.config.warmup_steps // 2) / (self.config.warmup_steps // 2)
            return 0.1 * progress
            
    def update_metrics(self, metrics: Dict[str, float]):
        """Update scheduler with training metrics for adaptive curriculum"""
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
            
        # Adaptive curriculum based on performance
        self._adapt_curriculum()
        
    def _adapt_curriculum(self):
        """Adapt curriculum based on training metrics"""
        if len(self.metrics_history["loss"]) < 100:
            return
            
        # Check if loss is plateauing
        recent_losses = self.metrics_history["loss"][-100:]
        loss_variance = np.var(recent_losses)
        
        if loss_variance < 0.001:  # Loss is stable
            # Accelerate curriculum
            logger.info("Loss plateau detected, accelerating curriculum")
            self.config.warmup_steps = int(self.config.warmup_steps * 0.9)
            self.config.expert_growth_steps = int(self.config.expert_growth_steps * 0.9)


class DifficultyScorer:
    """
    Scores data samples by difficulty for curriculum learning
    """
    
    def __init__(self, model=None):
        self.model = model
        self.cache = {}
        
    def score_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Score a batch of samples by difficulty"""
        input_ids = batch["input_ids"]
        
        # Multiple difficulty metrics
        length_score = self._length_difficulty(input_ids)
        vocab_score = self._vocabulary_difficulty(input_ids)
        
        if self.model is not None:
            perplexity_score = self._perplexity_difficulty(batch)
        else:
            perplexity_score = torch.zeros_like(length_score)
            
        # Combine scores
        difficulty_score = (
            0.3 * length_score +
            0.3 * vocab_score +
            0.4 * perplexity_score
        )
        
        return difficulty_score
        
    def _length_difficulty(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Score based on sequence length"""
        seq_lengths = (input_ids != 0).sum(dim=1).float()
        # Normalize to [0, 1]
        return seq_lengths / input_ids.size(1)
        
    def _vocabulary_difficulty(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Score based on vocabulary diversity"""
        batch_size = input_ids.size(0)
        vocab_scores = []
        
        for i in range(batch_size):
            tokens = input_ids[i]
            unique_tokens = len(torch.unique(tokens[tokens != 0]))
            total_tokens = (tokens != 0).sum().item()
            
            # Higher diversity = higher difficulty
            diversity = unique_tokens / max(total_tokens, 1)
            vocab_scores.append(diversity)
            
        return torch.tensor(vocab_scores, device=input_ids.device)
        
    def _perplexity_difficulty(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Score based on model perplexity"""
        with torch.no_grad():
            outputs = self.model(**batch)
            
            # Calculate perplexity
            loss = outputs.loss
            perplexity = torch.exp(loss)
            
            # Normalize to [0, 1] range (assuming max perplexity of 1000)
            normalized_perplexity = torch.clamp(perplexity / 1000, 0, 1)
            
        return normalized_perplexity.expand(batch["input_ids"].size(0))


class CurriculumDataSampler:
    """
    Data sampler that implements curriculum-based sampling
    """
    
    def __init__(
        self,
        dataset,
        difficulty_scorer: DifficultyScorer,
        curriculum_scheduler: CurriculumScheduler,
        batch_size: int = 32
    ):
        self.dataset = dataset
        self.difficulty_scorer = difficulty_scorer
        self.curriculum_scheduler = curriculum_scheduler
        self.batch_size = batch_size
        
        # Pre-compute difficulty scores for efficiency
        self._precompute_difficulties()
        
    def _precompute_difficulties(self):
        """Pre-compute difficulty scores for all samples"""
        logger.info("Pre-computing difficulty scores...")
        self.difficulty_scores = []
        
        # Process in batches
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = range(i, min(i + self.batch_size, len(self.dataset)))
            batch = {
                "input_ids": torch.stack([
                    torch.tensor(self.dataset[j]["input_ids"]) 
                    for j in batch_indices
                ])
            }
            
            scores = self.difficulty_scorer.score_batch(batch)
            self.difficulty_scores.extend(scores.tolist())
            
        self.difficulty_scores = torch.tensor(self.difficulty_scores)
        logger.info(f"Computed difficulty scores for {len(self.difficulty_scores)} samples")
        
    def get_batch_indices(self) -> List[int]:
        """Get batch indices based on current curriculum"""
        curriculum_state = self.curriculum_scheduler.step()
        difficulty = curriculum_state["difficulty"]
        
        # Define difficulty thresholds
        thresholds = {
            "easy": (0.0, 0.3),
            "medium": (0.2, 0.6),
            "hard": (0.5, 0.9),
            "expert": (0.7, 1.0)
        }
        
        min_diff, max_diff = thresholds[difficulty]
        
        # Find samples in difficulty range
        mask = (self.difficulty_scores >= min_diff) & (self.difficulty_scores <= max_diff)
        valid_indices = torch.where(mask)[0].tolist()
        
        if len(valid_indices) < self.batch_size:
            # Fall back to random sampling if not enough samples
            valid_indices = list(range(len(self.dataset)))
            
        # Sample batch
        batch_indices = np.random.choice(
            valid_indices,
            size=self.batch_size,
            replace=len(valid_indices) < self.batch_size
        )
        
        return batch_indices.tolist()


class ExpertCurriculum:
    """
    Manages progressive expert activation during training
    """
    
    def __init__(self, total_experts: int, curriculum_config: CurriculumConfig):
        self.total_experts = total_experts
        self.config = curriculum_config
        self.active_experts = set(range(curriculum_config.start_num_experts))
        
    def get_active_experts(self, step: int) -> List[int]:
        """Get list of currently active expert indices"""
        target_num_experts = self._compute_target_experts(step)
        
        # Activate more experts if needed
        while len(self.active_experts) < target_num_experts:
            # Add the next expert
            new_expert_idx = len(self.active_experts)
            if new_expert_idx < self.total_experts:
                self.active_experts.add(new_expert_idx)
                logger.info(f"Activated expert {new_expert_idx} at step {step}")
                
        return sorted(list(self.active_experts))
        
    def _compute_target_experts(self, step: int) -> int:
        """Compute target number of experts for current step"""
        if step >= self.config.expert_growth_steps:
            return min(self.config.final_num_experts, self.total_experts)
            
        progress = step / self.config.expert_growth_steps
        expert_range = self.config.final_num_experts - self.config.start_num_experts
        target = self.config.start_num_experts + int(progress * expert_range)
        
        return min(target, self.total_experts)
        
    def should_freeze_expert(self, expert_idx: int, step: int) -> bool:
        """Check if an expert should be frozen (not trained)"""
        # Freeze inactive experts
        if expert_idx not in self.active_experts:
            return True
            
        # Optionally freeze early experts after warmup
        if step > self.config.warmup_steps and expert_idx < 4:
            return True  # Freeze first 4 experts after warmup
            
        return False