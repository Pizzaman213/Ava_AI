"""
Episodic Memory for Continual Learning in Ava Training.

This module provides experience replay functionality for continual learning,
implementing prioritized experience replay with importance sampling.

Key Features:
- Priority-based sampling: Samples with higher loss are replayed more frequently
- Importance sampling: Corrects for sampling bias with importance weights
- Efficient buffer: Circular buffer with O(1) add and O(k) sample operations
- Training integration: Protocol-based integration with TrainingLoopManager

Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │                     EpisodicMemoryManager                         │
    │  ┌────────────────┐  ┌─────────────────────────────────────────┐ │
    │  │  MemoryBuffer  │  │           Training Loop Integration      │ │
    │  │  ┌──────────┐  │  │  - augment_batch(): Mix replay samples  │ │
    │  │  │ Entry 1  │  │  │  - get_sample_weights(): IS weights     │ │
    │  │  │ Entry 2  │  │  │  - store_batch(): Add new experiences   │ │
    │  │  │   ...    │  │  │  - update_priorities(): Update losses   │ │
    │  │  │ Entry N  │  │  │  - get_metrics(): Buffer statistics     │ │
    │  │  └──────────┘  │  └─────────────────────────────────────────┘ │
    │  └────────────────┘                                               │
    └──────────────────────────────────────────────────────────────────┘

Usage:
    from ava.config import EpisodicMemoryConfig
    from ava.training.episodic_memory import EpisodicMemoryManager

    # Create manager from config
    config = EpisodicMemoryConfig(use_episodic_memory=True, memory_capacity=10000)
    manager = EpisodicMemoryManager(config, device='cuda')

    # In training loop integration
    loop_manager.set_components(episodic_memory_manager=manager)

References:
    - Prioritized Experience Replay (Schaul et al., 2015)
    - Experience Replay for Continual Learning (Rolnick et al., 2019)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """
    Single entry in the episodic memory buffer.

    Stores all information needed to replay a training sample,
    including the original batch data, computed loss, and priority.

    Attributes:
        input_ids: Tokenized input sequence [seq_len]
        attention_mask: Attention mask [seq_len]
        labels: Target labels [seq_len]
        loss: Per-sample loss when this entry was stored
        priority: Sampling priority (updated based on loss)
        position_ids: Optional position IDs for sequence packing
        aux_info: Optional auxiliary info (MoE routing, etc.)
        metadata: Additional metadata (step, epoch, etc.)
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    loss: float
    priority: float = 1.0
    position_ids: Optional[torch.Tensor] = None
    aux_info: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_device(self, device: torch.device) -> "MemoryEntry":
        """Move tensors to specified device."""
        return MemoryEntry(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
            loss=self.loss,
            priority=self.priority,
            position_ids=self.position_ids.to(device) if self.position_ids is not None else None,
            aux_info=self.aux_info,
            metadata=self.metadata,
        )


class EpisodicMemoryBuffer:
    """
    Priority-based experience replay buffer with importance sampling.

    Implements prioritized experience replay where samples with higher
    loss are replayed more frequently. Uses importance sampling weights
    to correct for the sampling bias.

    Key Features:
        - Circular buffer with O(1) insertion
        - Prioritized sampling based on loss magnitude
        - Importance sampling (IS) weight computation
        - Efficient numpy-based implementation

    Priority Calculation:
        P(i) = priority_i^alpha / sum(priority_j^alpha)
        where alpha (priority_exponent) controls prioritization strength

    IS Weights:
        w_i = (N * P(i))^(-beta) / max(w)
        where beta (importance_weight_exponent) anneals from 0 to 1

    Args:
        capacity: Maximum number of entries in buffer
        priority_exponent: Alpha for priority sampling (0 = uniform, 1 = proportional)
        importance_weight_exponent: Beta for IS weights (0 = no correction, 1 = full)
        selection_strategy: 'importance' (prioritized) or 'uniform' (random)
    """

    def __init__(
        self,
        capacity: int = 10000,
        priority_exponent: float = 0.6,
        importance_weight_exponent: float = 0.4,
        selection_strategy: str = "importance",
    ):
        self.capacity = capacity
        self.priority_exponent = priority_exponent
        self.importance_weight_exponent = importance_weight_exponent
        self.selection_strategy = selection_strategy

        # Buffer storage
        self.buffer: List[MemoryEntry] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0  # Write position for circular buffer
        self.max_priority = 1.0

        # Statistics
        self._total_added = 0
        self._total_sampled = 0

    def add(self, entry: MemoryEntry) -> None:
        """
        Add an entry to the buffer with maximum priority.

        New entries get max priority to ensure they're sampled at least once.
        Priority is updated after replay based on actual loss.

        Args:
            entry: Memory entry to add
        """
        entry.priority = self.max_priority

        if len(self.buffer) < self.capacity:
            # Buffer not full - append
            self.buffer.append(entry)
            self.priorities[len(self.buffer) - 1] = self.max_priority
        else:
            # Buffer full - circular replacement
            if self.selection_strategy == "importance":
                # Replace lowest priority entry
                min_idx = int(np.argmin(self.priorities[: len(self.buffer)]))
                self.buffer[min_idx] = entry
                self.priorities[min_idx] = self.max_priority
            else:
                # Simple circular buffer
                self.buffer[self.position] = entry
                self.priorities[self.position] = self.max_priority
                self.position = (self.position + 1) % self.capacity

        self._total_added += 1

    def sample(
        self, batch_size: int
    ) -> Tuple[List[MemoryEntry], np.ndarray, np.ndarray]:
        """
        Sample a batch from the buffer with prioritized selection.

        Args:
            batch_size: Number of samples to return

        Returns:
            entries: List of MemoryEntry objects
            indices: Array of indices in buffer (for priority update)
            weights: Importance sampling weights (normalized to max 1.0)
        """
        if len(self.buffer) == 0:
            return [], np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        n = min(batch_size, len(self.buffer))

        if self.selection_strategy == "importance":
            # Prioritized sampling
            probs = self.priorities[: len(self.buffer)] ** self.priority_exponent
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs = probs / probs_sum
            else:
                probs = np.ones(len(self.buffer)) / len(self.buffer)

            # Sample without replacement
            indices = np.random.choice(
                len(self.buffer), size=n, replace=False, p=probs
            )

            # Compute importance sampling weights
            # w_i = (N * P(i))^(-beta) / max(w)
            weights = (len(self.buffer) * probs[indices]) ** (
                -self.importance_weight_exponent
            )
            weights = weights / weights.max()  # Normalize to [0, 1]
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), size=n, replace=False)
            weights = np.ones(n, dtype=np.float32)

        entries = [self.buffer[i] for i in indices]
        self._total_sampled += n

        return entries, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, losses: np.ndarray) -> None:
        """
        Update priorities based on loss values.

        Higher loss = higher priority = more likely to be sampled.

        Args:
            indices: Buffer indices to update
            losses: New loss values for each index
        """
        for idx, loss in zip(indices, losses):
            # Priority = |loss| + epsilon for stability
            new_priority = abs(float(loss)) + 1e-6
            self.priorities[idx] = new_priority
            self.max_priority = max(self.max_priority, new_priority)

            # Update entry's stored priority
            if idx < len(self.buffer):
                self.buffer[idx].priority = new_priority

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def get_stats(self) -> Dict[str, Any]:
        """Return buffer statistics."""
        if len(self.buffer) == 0:
            return {
                "buffer_size": 0,
                "buffer_capacity": self.capacity,
                "fill_ratio": 0.0,
                "total_added": self._total_added,
                "total_sampled": self._total_sampled,
            }

        valid_priorities = self.priorities[: len(self.buffer)]
        return {
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.capacity,
            "fill_ratio": len(self.buffer) / self.capacity,
            "total_added": self._total_added,
            "total_sampled": self._total_sampled,
            "mean_priority": float(np.mean(valid_priorities)),
            "max_priority": float(np.max(valid_priorities)),
            "min_priority": float(np.min(valid_priorities)),
            "priority_std": float(np.std(valid_priorities)),
        }


class EpisodicMemoryManager:
    """
    Manager for episodic memory integration with the training loop.

    Implements the EpisodicMemoryProtocol interface expected by
    TrainingLoopManager for seamless integration.

    Architecture:
        1. augment_batch(): Called before forward pass
           - Samples replay experiences from buffer
           - Concatenates replay samples with current batch
           - Returns augmented batch + replay indices

        2. get_sample_weights(): Called after forward pass
           - Returns importance sampling weights for loss computation
           - Current batch gets weight 1.0, replay gets IS weights

        3. store_batch(): Called after backward pass
           - Stores current batch samples in buffer
           - Uses loss values for initial priority

        4. update_replay_priorities(): Called after loss computation
           - Updates priorities for replayed samples
           - Based on actual loss during replay

    Args:
        config: EpisodicMemoryConfig with buffer settings
        device: Target device for tensors
    """

    def __init__(self, config: Any, device: torch.device = torch.device("cuda")):
        """
        Initialize episodic memory manager.

        Args:
            config: EpisodicMemoryConfig dataclass
            device: Target device for tensor operations
        """
        self.config = config
        self.device = device
        self.enabled = getattr(config, "use_episodic_memory", False)

        if not self.enabled:
            self.buffer = None
            logger.debug("Episodic memory disabled")
            return

        # Extract config values with fallbacks
        capacity = getattr(config, "memory_capacity", 1000)
        priority_exp = getattr(config, "priority_exponent", 0.6)
        importance_exp = getattr(config, "importance_weight_exponent", 0.4)
        selection_strategy = getattr(config, "memory_selection_strategy", "importance")

        self.buffer = EpisodicMemoryBuffer(
            capacity=capacity,
            priority_exponent=priority_exp,
            importance_weight_exponent=importance_exp,
            selection_strategy=selection_strategy,
        )

        self.replay_ratio = getattr(config, "memory_replay_ratio", 0.2)
        self.warmup_steps = getattr(config, "buffer_warmup_steps", 100)
        self.store_aux_info = getattr(config, "store_aux_info", False)
        self.silent_mode = getattr(config, "silent_mode", False)

        # State tracking
        self._global_step = 0
        self._last_replay_indices: Optional[np.ndarray] = None
        self._last_replay_weights: Optional[np.ndarray] = None
        self._current_batch_size = 0

        if not self.silent_mode:
            logger.info(
                f"Episodic memory initialized: capacity={capacity}, "
                f"replay_ratio={self.replay_ratio}, warmup={self.warmup_steps}"
            )

    def augment_batch(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        global_step: int,
        current_phase: str = "training",
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
        """
        Augment current batch with replay samples from memory.

        Args:
            batch: Current training batch (already on GPU)
            batch_idx: Current batch index
            global_step: Current global training step
            current_phase: 'training' or 'accumulating'

        Returns:
            augmented_batch: Batch with replay samples appended
            replay_indices: Indices of replay samples in buffer (for priority update)
        """
        self._global_step = global_step

        if not self.enabled or self.buffer is None:
            return batch, np.array([], dtype=np.int64)

        # Skip replay during warmup (let buffer fill first)
        if global_step < self.warmup_steps or len(self.buffer) < 10:
            self._current_batch_size = batch["input_ids"].size(0)
            self._last_replay_indices = np.array([], dtype=np.int64)
            self._last_replay_weights = np.array([], dtype=np.float32)
            return batch, np.array([], dtype=np.int64)

        current_batch_size = batch["input_ids"].size(0)
        self._current_batch_size = current_batch_size

        # Calculate replay size
        replay_size = max(1, int(current_batch_size * self.replay_ratio))

        # Sample from buffer
        entries, indices, weights = self.buffer.sample(replay_size)

        if len(entries) == 0:
            self._last_replay_indices = np.array([], dtype=np.int64)
            self._last_replay_weights = np.array([], dtype=np.float32)
            return batch, np.array([], dtype=np.int64)

        # Store for later use
        self._last_replay_indices = indices
        self._last_replay_weights = weights

        # Stack replay samples and move to device
        replay_input_ids = torch.stack([e.input_ids for e in entries]).to(self.device)
        replay_attention_mask = torch.stack([e.attention_mask for e in entries]).to(
            self.device
        )
        replay_labels = torch.stack([e.labels for e in entries]).to(self.device)

        # Handle position_ids if present in current batch
        replay_position_ids = None
        if "position_ids" in batch and batch["position_ids"] is not None:
            # Check if entries have position_ids
            if entries[0].position_ids is not None:
                replay_position_ids = torch.stack(
                    [e.position_ids for e in entries]
                ).to(self.device)
            else:
                # Generate default position_ids for replay samples
                seq_len = replay_input_ids.size(1)
                replay_position_ids = (
                    torch.arange(seq_len, device=self.device)
                    .unsqueeze(0)
                    .expand(len(entries), -1)
                )

        # Concatenate with current batch
        augmented_batch = {
            "input_ids": torch.cat([batch["input_ids"], replay_input_ids], dim=0),
            "attention_mask": torch.cat(
                [batch["attention_mask"], replay_attention_mask], dim=0
            ),
            "labels": torch.cat([batch["labels"], replay_labels], dim=0),
        }

        if "position_ids" in batch and batch["position_ids"] is not None:
            augmented_batch["position_ids"] = torch.cat(
                [batch["position_ids"], replay_position_ids], dim=0
            )

        # Add sample weights to batch for weighted loss computation
        current_weights = torch.ones(current_batch_size, device=self.device)
        replay_weights_tensor = torch.from_numpy(weights).float().to(self.device)
        augmented_batch["sample_weights"] = torch.cat(
            [current_weights, replay_weights_tensor], dim=0
        )

        return augmented_batch, indices

    def get_sample_weights(
        self,
        batch_idx: int,
        outputs: Dict[str, Any],
        aux_info: List[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        """
        Get per-sample weights for loss computation.

        Returns importance sampling weights where:
        - Current batch samples get weight 1.0
        - Replay samples get IS weights from prioritized sampling

        Args:
            batch_idx: Current batch index
            outputs: Model outputs dict
            aux_info: Per-layer auxiliary info

        Returns:
            Tensor of shape [batch_size] with sample weights, or None if not applicable
        """
        if not self.enabled or self.buffer is None:
            return None

        if (
            self._last_replay_indices is None
            or len(self._last_replay_indices) == 0
        ):
            return None

        # Construct weights: [current_batch (1.0), replay_batch (IS weights)]
        current_weights = torch.ones(self._current_batch_size, device=self.device)
        replay_weights = torch.from_numpy(self._last_replay_weights).float().to(
            self.device
        )

        return torch.cat([current_weights, replay_weights], dim=0)

    def store_batch(
        self,
        batch: Dict[str, torch.Tensor],
        losses: torch.Tensor,
        aux_info: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Store current batch samples in the memory buffer.

        Only stores the original batch samples (not replay samples).
        Uses per-sample loss for initial priority assignment.

        Args:
            batch: Original batch (before augmentation)
            losses: Per-sample loss values [batch_size] or scalar
            aux_info: Optional auxiliary info to store with samples
        """
        if not self.enabled or self.buffer is None:
            return

        # Handle scalar loss (expand to per-sample)
        if losses.dim() == 0 or losses.numel() == 1:
            batch_size = batch["input_ids"].size(0)
            losses = losses.expand(batch_size).clone()

        # Only store original batch, not replay portion
        store_size = self._current_batch_size
        if store_size <= 0:
            store_size = batch["input_ids"].size(0)

        # OPTIMIZATION: Batch all GPU->CPU transfers before the loop
        # This reduces N GPU synchronizations to just 1, providing 10-30ms savings
        loss_vals = losses[:store_size].detach().tolist()  # Single GPU->CPU sync
        input_ids_cpu = batch["input_ids"][:store_size].cpu()  # Single transfer
        attention_mask_cpu = batch["attention_mask"][:store_size].cpu()
        labels_cpu = batch["labels"][:store_size].cpu()
        position_ids_cpu = (
            batch["position_ids"][:store_size].cpu()
            if "position_ids" in batch and batch["position_ids"] is not None
            else None
        )

        for i in range(store_size):
            loss_val = loss_vals[i] if i < len(loss_vals) else sum(loss_vals) / len(loss_vals)

            entry = MemoryEntry(
                input_ids=input_ids_cpu[i].clone(),  # Clone from CPU tensor (no GPU sync)
                attention_mask=attention_mask_cpu[i].clone(),
                labels=labels_cpu[i].clone(),
                loss=loss_val,
                priority=abs(loss_val) + 1e-6,
                position_ids=(
                    position_ids_cpu[i].clone()
                    if position_ids_cpu is not None
                    else None
                ),
                aux_info=aux_info[i] if aux_info and self.store_aux_info else None,
                metadata={"step": self._global_step},
            )
            self.buffer.add(entry)

    def update_replay_priorities(
        self, indices: np.ndarray, losses: torch.Tensor
    ) -> None:
        """
        Update priorities for replayed samples based on new loss.

        Called after forward pass to update priority based on
        actual loss during replay (not stored loss).

        Args:
            indices: Buffer indices of replayed samples
            losses: New loss values from replay forward pass
        """
        if not self.enabled or self.buffer is None:
            return

        if len(indices) == 0:
            return

        # Convert losses to numpy
        if isinstance(losses, torch.Tensor):
            losses_np = losses.detach().cpu().numpy()
        else:
            losses_np = np.array(losses)

        self.buffer.update_priorities(indices, losses_np)

    def get_metrics(self) -> Dict[str, float]:
        """
        Get episodic memory metrics for logging.

        Returns:
            Dictionary with buffer statistics and replay metrics
        """
        if not self.enabled or self.buffer is None:
            return {}

        stats = self.buffer.get_stats()

        return {
            "episodic/buffer_size": stats["buffer_size"],
            "episodic/fill_ratio": stats["fill_ratio"],
            "episodic/total_added": stats["total_added"],
            "episodic/total_sampled": stats["total_sampled"],
            "episodic/mean_priority": stats.get("mean_priority", 0.0),
            "episodic/max_priority": stats.get("max_priority", 0.0),
        }

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, epoch: int) -> None:
        """Called at the end of each epoch."""
        if not self.enabled or self.buffer is None:
            return

        if not self.silent_mode:
            stats = self.buffer.get_stats()
            logger.info(
                f"Episodic memory epoch {epoch}: "
                f"buffer_size={stats['buffer_size']}, "
                f"total_sampled={stats['total_sampled']}"
            )


def create_episodic_memory_manager(
    config: Any, device: torch.device = torch.device("cuda")
) -> Optional[EpisodicMemoryManager]:
    """
    Factory function to create an EpisodicMemoryManager if enabled.

    Args:
        config: Configuration object with episodic memory settings
        device: Target device

    Returns:
        EpisodicMemoryManager if enabled, None otherwise
    """
    # Check if episodic memory is enabled
    if hasattr(config, "memory") and hasattr(config.memory, "use_episodic_memory"):
        mem_config = config.memory
    elif hasattr(config, "use_episodic_memory"):
        mem_config = config
    else:
        return None

    if not getattr(mem_config, "use_episodic_memory", False):
        return None

    return EpisodicMemoryManager(mem_config, device)
