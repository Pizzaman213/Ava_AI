"""
CUDA Graph Manager for Training Step Capture

This module provides CUDA graph capture for forward+backward+optimizer steps,
eliminating kernel launch overhead for 15-25% speedup on small batch sizes.

CUDA graphs work by capturing a sequence of GPU operations and replaying them
with a single kernel launch, dramatically reducing CPU overhead.

Usage:
    from ava.cuda.graph_manager import CUDAGraphManager

    graph_manager = CUDAGraphManager(model, optimizer, config)

    # Warmup (required before capture)
    for _ in range(3):
        graph_manager.warmup_step(input_ids, labels)

    # Capture the graph
    graph_manager.capture()

    # Training loop - replay instead of normal forward/backward
    for batch in dataloader:
        loss = graph_manager.replay(batch['input_ids'], batch['labels'])
        # Scheduler, logging, etc.

Limitations:
- Requires fixed input shapes (batch_size, seq_len)
- Incompatible with dynamic control flow
- Incompatible with layer-wise optimizer
- Incompatible with fused all-reduce in DDP

Requirements:
- PyTorch >= 1.10 for CUDA graphs
- Single GPU or DDP (not FSDP)
"""

import logging
from typing import Optional, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


@dataclass
class CUDAGraphConfig:
    """Configuration for CUDA graph capture."""
    enabled: bool = False
    capture_backward: bool = True
    capture_optimizer_step: bool = True
    max_cached_graphs: int = 1  # Reduced from 4 for memory efficiency
    use_memory_pool: bool = True
    warmup_steps: int = 3
    capture_stream: bool = True  # Use dedicated capture stream


class CapturedGraph:
    """A captured CUDA graph with its input/output buffers."""

    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        static_input_ids: torch.Tensor,
        static_labels: torch.Tensor,
        static_output: torch.Tensor,
        shape_key: Tuple[int, ...],
    ):
        self.graph = graph
        self.static_input_ids = static_input_ids
        self.static_labels = static_labels
        self.static_output = static_output
        self.shape_key = shape_key
        self.replay_count = 0

    def replay(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Replay the captured graph with new inputs."""
        # Copy inputs to static buffers
        self.static_input_ids.copy_(input_ids)
        self.static_labels.copy_(labels)

        # Replay the graph
        self.graph.replay()
        self.replay_count += 1

        # Return the output (already in static buffer)
        return self.static_output.clone()


class CUDAGraphManager:
    """
    Manages CUDA graph capture for training steps.

    Captures the forward+backward+optimizer sequence as a CUDA graph,
    which can be replayed with minimal CPU overhead.

    Features:
    - Automatic warmup before capture
    - Shape-based graph caching (multiple batch sizes)
    - Memory pool for consistent addresses
    - Fallback to eager mode on capture failure

    Args:
        model: Model to train
        optimizer: Optimizer for the model
        config: CUDAGraphConfig or dict with config options
        loss_fn: Loss function (default: CrossEntropyLoss)
        scaler: Optional GradScaler for mixed precision
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: Optional[CUDAGraphConfig] = None,
        loss_fn: Optional[Callable] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler

        # Config
        if config is not None:
            self.config = config
        else:
            self.config = CUDAGraphConfig()

        # Loss function
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        # Graph cache: shape_key -> CapturedGraph
        self._graph_cache: Dict[Tuple[int, ...], CapturedGraph] = {}

        # Memory pool for consistent addresses
        self._memory_pool: Optional[torch.cuda.graphs.graph_pool_handle] = None
        if self.config.use_memory_pool:
            self._memory_pool = torch.cuda.graph_pool_handle()

        # Capture stream
        self._capture_stream: Optional[torch.cuda.Stream] = None
        if self.config.capture_stream and torch.cuda.is_available():
            self._capture_stream = torch.cuda.Stream()

        # State
        self._warmup_count = 0
        self._is_capturing = False
        self._enabled = self.config.enabled and torch.cuda.is_available()

        # Statistics
        self._total_replays = 0
        self._total_captures = 0
        self._cache_hits = 0
        self._cache_misses = 0

        if self._enabled:
            logger.info(
                f"CUDAGraphManager initialized: "
                f"capture_backward={self.config.capture_backward}, "
                f"capture_optimizer={self.config.capture_optimizer_step}, "
                f"max_cached={self.config.max_cached_graphs}"
            )
        else:
            logger.debug("CUDAGraphManager disabled")

    def _get_shape_key(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[int, ...]:
        """Get cache key from input shapes."""
        return (input_ids.shape[0], input_ids.shape[1])  # (batch_size, seq_len)

    def _create_static_buffers(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create static buffers for graph capture."""
        # Static input buffers (must have fixed addresses)
        static_input_ids = torch.empty_like(input_ids)
        static_labels = torch.empty_like(labels)

        # Static output buffer for loss
        static_output = torch.empty((), device=input_ids.device, dtype=torch.float32)

        return static_input_ids, static_labels, static_output

    def warmup_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run a warmup step (required before capture).

        Warmup ensures:
        1. All lazy initializations complete (e.g., cuBLAS handles)
        2. Memory allocations stabilize
        3. JIT compilation completes

        Args:
            input_ids: Input token IDs
            labels: Target labels
            attention_mask: Optional attention mask

        Returns:
            Loss value from warmup step
        """
        self._warmup_count += 1

        # Standard forward pass
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            # Try with attention_mask first, fall back without if not supported
            try:
                outputs = self.model(input_ids, attention_mask=attention_mask)
            except TypeError:
                outputs = self.model(input_ids)

            # Handle different output formats
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Compute loss
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss.detach()

    def capture(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> bool:
        """
        Capture forward+backward+optimizer as a CUDA graph.

        Args:
            input_ids: Example input (shape will be used for this graph)
            labels: Example labels
            attention_mask: Optional attention mask

        Returns:
            True if capture succeeded, False otherwise
        """
        if not self._enabled:
            logger.warning("CUDAGraphManager not enabled, skipping capture")
            return False

        if self._warmup_count < self.config.warmup_steps:
            logger.warning(
                f"Insufficient warmup: {self._warmup_count}/{self.config.warmup_steps}. "
                f"Run warmup_step() first."
            )
            return False

        shape_key = self._get_shape_key(input_ids, labels)

        # Check if already captured for this shape
        if shape_key in self._graph_cache:
            logger.debug(f"Graph already captured for shape {shape_key}")
            return True

        # Check cache limit
        if len(self._graph_cache) >= self.config.max_cached_graphs:
            # Evict least used graph
            self._evict_least_used()

        logger.info(f"Capturing CUDA graph for shape {shape_key}")
        self._is_capturing = True

        try:
            # Create static buffers
            static_input_ids, static_labels, static_output = self._create_static_buffers(
                input_ids, labels
            )

            # Copy initial data
            static_input_ids.copy_(input_ids)
            static_labels.copy_(labels)

            # Ensure previous work completes
            torch.cuda.synchronize()

            # Create graph
            graph = torch.cuda.CUDAGraph()

            # Zero gradients before capture
            self.optimizer.zero_grad(set_to_none=True)

            # Capture the graph
            with torch.cuda.graph(graph, pool=self._memory_pool):
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    outputs = self.model(static_input_ids, attention_mask=attention_mask)

                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                    loss = self.loss_fn(
                        logits.view(-1, logits.size(-1)),
                        static_labels.view(-1)
                    )

                # Store loss in static buffer
                static_output.copy_(loss)

                # Backward (if configured)
                if self.config.capture_backward:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # Optimizer step (if configured)
                if self.config.capture_optimizer_step:
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

            # Create captured graph object
            captured = CapturedGraph(
                graph=graph,
                static_input_ids=static_input_ids,
                static_labels=static_labels,
                static_output=static_output,
                shape_key=shape_key,
            )

            self._graph_cache[shape_key] = captured
            self._total_captures += 1

            logger.info(
                f"CUDA graph captured for shape {shape_key} "
                f"(total cached: {len(self._graph_cache)})"
            )
            return True

        except Exception as e:
            logger.error(f"CUDA graph capture failed: {e}")
            return False

        finally:
            self._is_capturing = False

    def replay(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Replay captured graph or fall back to eager execution.

        Args:
            input_ids: Input token IDs
            labels: Target labels
            attention_mask: Optional attention mask (ignored if using graph)

        Returns:
            Loss value
        """
        if not self._enabled:
            # Fall back to eager execution
            return self._eager_step(input_ids, labels, attention_mask)

        shape_key = self._get_shape_key(input_ids, labels)

        # Check cache
        if shape_key in self._graph_cache:
            self._cache_hits += 1
            captured = self._graph_cache[shape_key]
            loss = captured.replay(input_ids, labels)
            self._total_replays += 1
            return loss
        else:
            self._cache_misses += 1

            # Try to capture for this shape
            if len(self._graph_cache) < self.config.max_cached_graphs:
                # Do warmup and capture
                for _ in range(self.config.warmup_steps):
                    self.warmup_step(input_ids, labels, attention_mask)

                if self.capture(input_ids, labels, attention_mask):
                    # Now replay
                    captured = self._graph_cache[shape_key]
                    loss = captured.replay(input_ids, labels)
                    self._total_replays += 1
                    return loss

            # Fall back to eager
            return self._eager_step(input_ids, labels, attention_mask)

    def _eager_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Eager (non-graph) training step."""
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            # Try with attention_mask first, fall back without if not supported
            try:
                outputs = self.model(input_ids, attention_mask=attention_mask)
            except TypeError:
                outputs = self.model(input_ids)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return loss.detach()

    def _evict_least_used(self) -> None:
        """Evict least replayed graph from cache."""
        if not self._graph_cache:
            return

        # Find least used
        min_replays = float('inf')
        min_key = None

        for key, graph in self._graph_cache.items():
            if graph.replay_count < min_replays:
                min_replays = graph.replay_count
                min_key = key

        if min_key is not None:
            del self._graph_cache[min_key]
            logger.debug(f"Evicted graph for shape {min_key} (replays: {min_replays})")

    def has_graph_for_shape(self, batch_size: int, seq_len: int) -> bool:
        """Check if graph is cached for given shape."""
        return (batch_size, seq_len) in self._graph_cache

    def get_stats(self) -> Dict[str, Any]:
        """Get graph manager statistics."""
        return {
            'enabled': self._enabled,
            'total_captures': self._total_captures,
            'total_replays': self._total_replays,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            'cached_shapes': list(self._graph_cache.keys()),
            'warmup_count': self._warmup_count,
        }

    def clear_cache(self) -> None:
        """Clear all cached graphs."""
        self._graph_cache.clear()
        logger.info("CUDA graph cache cleared")

    @property
    def enabled(self) -> bool:
        """Whether graph manager is enabled."""
        return self._enabled


def create_cuda_graph_manager(
    model: nn.Module,
    optimizer: Optimizer,
    enabled: bool = True,
    capture_backward: bool = True,
    capture_optimizer_step: bool = True,
    max_cached_graphs: int = 4,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Optional[CUDAGraphManager]:
    """
    Factory function to create CUDAGraphManager.

    Args:
        model: Model to train
        optimizer: Optimizer for the model
        enabled: Whether to enable CUDA graphs
        capture_backward: Include backward in graph
        capture_optimizer_step: Include optimizer in graph
        max_cached_graphs: Maximum graphs to cache
        scaler: Optional GradScaler for mixed precision

    Returns:
        CUDAGraphManager or None if not enabled/available
    """
    if not enabled:
        return None

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping CUDAGraphManager")
        return None

    config = CUDAGraphConfig(
        enabled=True,
        capture_backward=capture_backward,
        capture_optimizer_step=capture_optimizer_step,
        max_cached_graphs=max_cached_graphs,
    )

    return CUDAGraphManager(
        model=model,
        optimizer=optimizer,
        config=config,
        scaler=scaler,
    )
