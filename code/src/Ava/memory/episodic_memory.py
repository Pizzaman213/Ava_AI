"""
Episodic Memory Bank for Continual Learning

Implements episodic memory mechanisms to prevent catastrophic forgetting
and enable continual learning through selective experience replay.
"""

import torch  # type: ignore[import]
import torch.nn as nn  # type: ignore[import]
import torch.nn.functional as F  # type: ignore[import]
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from collections import deque
import random
import math


@dataclass
class MemoryEntry:
    """Single memory entry containing input, output, and metadata."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor]
    hidden_states: torch.Tensor
    task_id: int
    importance_score: float
    timestamp: int
    gradient_norm: Optional[float] = None
    loss_value: Optional[float] = None


class MemoryRetriever(nn.Module):
    """Retrieves relevant memories based on current input."""

    def __init__(
        self,
        hidden_size: int,
        memory_size: int,
        retrieval_method: str = "cosine"
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.retrieval_method = retrieval_method

        # Query projection for memory retrieval
        self.query_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query_states: torch.Tensor,
        memory_bank: List[MemoryEntry],
        k: int = 10
    ) -> Tuple[List[MemoryEntry], torch.Tensor]:
        """
        Retrieve k most relevant memories.

        Args:
            query_states: Current hidden states [batch_size, seq_len, hidden_size]
            memory_bank: List of memory entries
            k: Number of memories to retrieve

        Returns:
            retrieved_memories: List of relevant memory entries
            similarity_scores: Similarity scores for retrieved memories
        """
        if not memory_bank:
            return [], torch.tensor([])

        # Project query
        query = self.query_proj(query_states.mean(dim=1))  # [batch_size, hidden_size]
        batch_size = query.size(0)

        # Compute similarities with all memories
        similarities = []
        for memory in memory_bank:
            memory_repr = memory.hidden_states.mean(dim=1)  # [1, hidden_size]

            if self.retrieval_method == "cosine":
                sim = F.cosine_similarity(
                    query.unsqueeze(1),
                    memory_repr.unsqueeze(0),
                    dim=-1
                ).mean()
            elif self.retrieval_method == "euclidean":
                sim = -torch.norm(query - memory_repr, dim=-1).mean()
            else:  # dot product
                sim = torch.mm(query, memory_repr.t()).mean()

            similarities.append(sim.item())

        # Get top k memories
        similarities = torch.tensor(similarities)
        top_k_indices = torch.topk(similarities, min(k, len(memory_bank))).indices

        retrieved_memories = [memory_bank[i] for i in top_k_indices]
        retrieved_scores = similarities[top_k_indices]

        return retrieved_memories, retrieved_scores


class EpisodicMemoryBank(nn.Module):
    """
    Episodic Memory Bank for storing and retrieving important experiences.

    Implements various memory management strategies including:
    - Importance-based storage
    - Gradient-based selection
    - Task-aware memory organization
    """

    def __init__(
        self,
        capacity: int = 1000,
        hidden_size: int = 768,
        selection_strategy: str = "importance",
        importance_threshold: float = 0.5,
        retrieval_method: str = "cosine"
    ):
        super().__init__()
        self.capacity = capacity
        self.hidden_size = hidden_size
        self.selection_strategy = selection_strategy
        self.importance_threshold = importance_threshold

        # Memory storage
        self.memories: List[MemoryEntry] = []
        self.memory_index = 0

        # Memory retriever
        self.retriever = MemoryRetriever(
            hidden_size=hidden_size,
            memory_size=capacity,
            retrieval_method=retrieval_method
        )

        # Task-specific memory counters
        self.task_memory_counts: Dict[int, int] = {}

    def compute_importance(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        loss_value: Optional[float] = None,
        gradient_norm: Optional[float] = None
    ) -> float:
        """Compute importance score for a memory entry."""
        importance = 0.0

        # Gradient-based importance
        if gradient_norm is not None:
            importance += gradient_norm * 0.3

        # Loss-based importance
        if loss_value is not None:
            importance += loss_value * 0.3

        # Activation-based importance (entropy of hidden states)
        with torch.no_grad():
            activation_entropy = -torch.sum(
                F.softmax(hidden_states.mean(dim=1), dim=-1) *
                F.log_softmax(hidden_states.mean(dim=1), dim=-1),
                dim=-1
            ).mean().item()
            importance += activation_entropy * 0.4

        return importance

    def should_store_memory(
        self,
        importance_score: float,
        task_id: int
    ) -> bool:
        """Determine if a memory should be stored."""
        if self.selection_strategy == "importance":
            return importance_score > self.importance_threshold
        elif self.selection_strategy == "random":
            return random.random() > 0.5
        elif self.selection_strategy == "task_balanced":
            # Ensure balanced representation across tasks
            task_count = self.task_memory_counts.get(task_id, 0)
            avg_count = sum(self.task_memory_counts.values()) / max(len(self.task_memory_counts), 1)
            return task_count < avg_count or importance_score > self.importance_threshold
        else:
            return True

    def add_memory(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        task_id: int,
        labels: Optional[torch.Tensor] = None,
        loss_value: Optional[float] = None,
        gradient_norm: Optional[float] = None
    ):
        """Add a new memory entry."""
        # Compute importance score
        importance_score = self.compute_importance(
            hidden_states, labels, loss_value, gradient_norm
        )

        # Check if memory should be stored
        if not self.should_store_memory(importance_score, task_id):
            return

        # Create memory entry
        memory_entry = MemoryEntry(
            input_ids=input_ids.detach().cpu(),
            attention_mask=attention_mask.detach().cpu(),
            labels=labels.detach().cpu() if labels is not None else None,
            hidden_states=hidden_states.detach().cpu(),
            task_id=task_id,
            importance_score=importance_score,
            timestamp=self.memory_index,
            gradient_norm=gradient_norm,
            loss_value=loss_value
        )

        # Add to memory bank
        if len(self.memories) < self.capacity:
            self.memories.append(memory_entry)
        else:
            # Replace least important memory
            min_importance_idx = min(
                range(len(self.memories)),
                key=lambda i: self.memories[i].importance_score
            )
            if importance_score > self.memories[min_importance_idx].importance_score:
                old_task = self.memories[min_importance_idx].task_id
                self.task_memory_counts[old_task] = max(0, self.task_memory_counts.get(old_task, 1) - 1)
                self.memories[min_importance_idx] = memory_entry

        # Update counters
        self.task_memory_counts[task_id] = self.task_memory_counts.get(task_id, 0) + 1
        self.memory_index += 1

    def retrieve_memories(
        self,
        query_states: torch.Tensor,
        k: int = 10,
        task_id: Optional[int] = None
    ) -> Tuple[List[MemoryEntry], torch.Tensor]:
        """Retrieve relevant memories."""
        memory_bank = self.memories

        # Filter by task if specified
        if task_id is not None:
            memory_bank = [m for m in self.memories if m.task_id == task_id]

        return self.retriever(query_states, memory_bank, k)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory bank."""
        if not self.memories:
            return {"total_memories": 0}

        importance_scores = [m.importance_score for m in self.memories]

        return {
            "total_memories": len(self.memories),
            "capacity_utilization": len(self.memories) / self.capacity,
            "task_distribution": dict(self.task_memory_counts),
            "avg_importance": np.mean(importance_scores),
            "max_importance": max(importance_scores),
            "min_importance": min(importance_scores)
        }


class AdaptiveMemoryManager(nn.Module):
    """
    Adaptive memory manager that adjusts memory parameters based on performance.
    """

    def __init__(
        self,
        memory_bank: EpisodicMemoryBank,
        adaptation_rate: float = 0.01,
        performance_window: int = 100
    ):
        super().__init__()
        self.memory_bank = memory_bank
        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window

        # Performance tracking
        self.recent_losses = deque(maxlen=performance_window)
        self.recent_accuracies = deque(maxlen=performance_window)

    def update_performance(self, loss: float, accuracy: float):
        """Update performance metrics."""
        self.recent_losses.append(loss)
        self.recent_accuracies.append(accuracy)

        # Adapt memory parameters based on performance
        if len(self.recent_losses) >= self.performance_window:
            self._adapt_memory_parameters()

    def _adapt_memory_parameters(self):
        """Adapt memory parameters based on recent performance."""
        avg_loss = np.mean(self.recent_losses)
        avg_accuracy = np.mean(self.recent_accuracies)

        # Adjust importance threshold based on performance
        if avg_accuracy < 0.7:  # Poor performance
            # Lower threshold to store more memories
            self.memory_bank.importance_threshold *= (1 - self.adaptation_rate)
        elif avg_accuracy > 0.9:  # Good performance
            # Raise threshold to be more selective
            self.memory_bank.importance_threshold *= (1 + self.adaptation_rate)

        # Clamp threshold
        self.memory_bank.importance_threshold = max(0.1, min(1.0, self.memory_bank.importance_threshold))


class ExperienceReplay(nn.Module):
    """
    Experience replay mechanism for continual learning.
    """

    def __init__(
        self,
        memory_bank: EpisodicMemoryBank,
        replay_ratio: float = 0.2,
        replay_strategy: str = "random"
    ):
        super().__init__()
        self.memory_bank = memory_bank
        self.replay_ratio = replay_ratio
        self.replay_strategy = replay_strategy

    def get_replay_batch(
        self,
        current_batch_size: int,
        current_hidden_states: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get a batch of memories for replay.

        Returns:
            input_ids: Replayed input tokens
            attention_mask: Attention masks for replayed inputs
            labels: Labels for replayed inputs (if available)
        """
        replay_size = max(1, int(current_batch_size * self.replay_ratio))

        if self.replay_strategy == "random":
            # Random sampling
            sampled_memories = random.sample(
                self.memory_bank.memories,
                min(replay_size, len(self.memory_bank.memories))
            )
        elif self.replay_strategy == "importance":
            # Importance-weighted sampling
            memories = self.memory_bank.memories
            importance_scores = [m.importance_score for m in memories]
            probs = F.softmax(torch.tensor(importance_scores), dim=0).numpy()

            indices = np.random.choice(
                len(memories),
                size=min(replay_size, len(memories)),
                replace=False,
                p=probs
            )
            sampled_memories = [memories[i] for i in indices]
        elif self.replay_strategy == "similarity" and current_hidden_states is not None:
            # Similarity-based sampling
            sampled_memories, _ = self.memory_bank.retrieve_memories(
                current_hidden_states, k=replay_size
            )
        else:
            # Fallback to random
            sampled_memories = random.sample(
                self.memory_bank.memories,
                min(replay_size, len(self.memory_bank.memories))
            )

        if not sampled_memories:
            return None, None, None

        # Batch the memories
        input_ids = torch.stack([m.input_ids.squeeze(0) for m in sampled_memories])
        attention_mask = torch.stack([m.attention_mask.squeeze(0) for m in sampled_memories])

        labels = None
        if all(m.labels is not None for m in sampled_memories):
            labels = torch.stack([m.labels.squeeze(0) for m in sampled_memories])  # type: ignore[union-attr]

        return input_ids, attention_mask, labels

    def compute_replay_loss(
        self,
        model: nn.Module,
        replay_batch: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        device: torch.device
    ) -> torch.Tensor:
        """Compute loss for replay batch."""
        input_ids, attention_mask, labels = replay_batch

        if input_ids is None:
            return torch.tensor(0.0, device=device)

        # Move to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs.loss if hasattr(outputs, 'loss') else torch.tensor(0.0, device=device)