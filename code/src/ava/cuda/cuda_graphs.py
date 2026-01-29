"""
CUDA Graph Caching and Management.

This module provides shape-aware CUDA graph caching for improved training
throughput with variable batch sizes.

Features:
    - Shape-based graph caching (multiple graphs for different shapes)
    - Canonical shape normalization (reduces graph count)
    - LRU eviction for memory management
    - Memory pool management for efficient allocation

Usage:
    cache = CUDAGraphCache(max_graphs=8)

    # Get or capture graph for a shape
    graph_data = cache.get_or_capture(
        shape_key=(batch_size, seq_len),
        capture_fn=lambda: capture_training_step(...),
    )

    # Replay the graph
    cache.replay(shape_key, batch_data)
"""

import gc
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CachedGraph:
    """Cached CUDA graph with associated static buffers."""

    graph: torch.cuda.CUDAGraph
    static_inputs: Dict[str, torch.Tensor]
    static_loss: torch.Tensor
    batch_size: int
    seq_len: int
    capture_time: float = 0.0

    def copy_inputs(self, batch: Dict[str, torch.Tensor]) -> None:
        """Copy batch data into static buffers."""
        for key in self.static_inputs:
            if key in batch:
                self.static_inputs[key].copy_(batch[key])

    def replay(self) -> torch.Tensor:
        """Replay the graph and return the loss."""
        self.graph.replay()
        return self.static_loss

    def reset(self) -> None:
        """Reset and free the graph resources."""
        try:
            self.graph.reset()
        except Exception:
            pass


class CUDAGraphCache:
    """
    LRU cache for CUDA graphs with shape-based lookup.

    This cache maintains multiple captured CUDA graphs, one for each
    unique input shape encountered during training. Shapes can optionally
    be normalized to canonical bucket sizes to reduce the number of graphs.

    Memory Management:
        - Maximum graph count limits memory usage
        - LRU eviction removes least-recently-used graphs
        - Memory pool can be shared across graphs

    Shape Normalization:
        When enabled, batch sizes are rounded to canonical buckets
        (e.g., 4, 8, 12, 16, ...) to reduce graph proliferation.

    Example:
        cache = CUDAGraphCache(max_graphs=8, normalize_shapes=True)

        # During training loop
        shape = (batch_size, seq_len)
        if cache.has_graph(shape):
            loss = cache.replay(shape, gpu_batch)
        else:
            capture_fn = lambda: model(batch)
            cache.capture(shape, capture_fn, model, optimizer, ...)
    """

    def __init__(
        self,
        max_graphs: int = 8,
        normalize_shapes: bool = True,
        batch_size_bucket: int = 4,
        seq_len_bucket: int = 64,
    ):
        """
        Initialize the CUDA graph cache.

        Args:
            max_graphs: Maximum number of graphs to keep in cache
            normalize_shapes: Whether to normalize shapes to bucket sizes
            batch_size_bucket: Batch size normalization bucket (e.g., 4 -> 4, 8, 12, ...)
            seq_len_bucket: Sequence length normalization bucket (e.g., 64 -> 64, 128, ...)
        """
        self.max_graphs = max_graphs
        self.normalize_shapes = normalize_shapes
        self.batch_size_bucket = batch_size_bucket
        self.seq_len_bucket = seq_len_bucket

        # LRU cache using OrderedDict
        self._cache: OrderedDict[Tuple[int, int], CachedGraph] = OrderedDict()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.captures = 0
        self.evictions = 0

    def _normalize_shape(self, batch_size: int, seq_len: int) -> Tuple[int, int]:
        """
        Normalize shape to canonical bucket sizes.

        This reduces the number of unique graphs needed by rounding
        shapes to standard bucket sizes.

        Args:
            batch_size: Actual batch size
            seq_len: Actual sequence length

        Returns:
            Normalized (batch_size, seq_len) tuple
        """
        if not self.normalize_shapes:
            return (batch_size, seq_len)

        # Round up to next bucket
        norm_batch = ((batch_size + self.batch_size_bucket - 1) // self.batch_size_bucket) * self.batch_size_bucket
        norm_seq = ((seq_len + self.seq_len_bucket - 1) // self.seq_len_bucket) * self.seq_len_bucket

        return (norm_batch, norm_seq)

    def get_shape_key(self, batch: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        """
        Get the shape key for a batch.

        Args:
            batch: Input batch with 'input_ids' tensor

        Returns:
            Shape key tuple (batch_size, seq_len)
        """
        input_ids = batch.get('input_ids')
        if input_ids is None:
            raise ValueError("Batch must contain 'input_ids' tensor")

        batch_size, seq_len = input_ids.shape[:2]
        return self._normalize_shape(batch_size, seq_len)

    def has_graph(self, shape_key: Tuple[int, int]) -> bool:
        """Check if a graph exists for the given shape."""
        normalized_key = self._normalize_shape(*shape_key)
        return normalized_key in self._cache

    def get_graph(self, shape_key: Tuple[int, int]) -> Optional[CachedGraph]:
        """
        Get a cached graph, updating LRU order.

        Args:
            shape_key: (batch_size, seq_len) tuple

        Returns:
            CachedGraph if found, None otherwise
        """
        normalized_key = self._normalize_shape(*shape_key)

        if normalized_key not in self._cache:
            self.misses += 1
            return None

        self.hits += 1

        # Move to end (most recently used)
        self._cache.move_to_end(normalized_key)
        return self._cache[normalized_key]

    def _create_static_buffers(
        self,
        sample_batch: Dict[str, torch.Tensor],
        target_batch_size: int,
        target_seq_len: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Create static input buffers for graph capture.

        If normalizing shapes, creates buffers with the canonical size
        which may be larger than the actual batch.

        Args:
            sample_batch: Sample batch for shape/dtype reference
            target_batch_size: Target batch size (may be normalized)
            target_seq_len: Target sequence length (may be normalized)

        Returns:
            Dictionary of static tensors
        """
        device = sample_batch['input_ids'].device
        dtype = sample_batch['input_ids'].dtype

        static_inputs = {}

        # Create input_ids buffer
        static_inputs['input_ids'] = torch.zeros(
            (target_batch_size, target_seq_len),
            dtype=dtype,
            device=device,
        )

        # Create labels buffer (same shape as input_ids for causal LM)
        static_inputs['labels'] = torch.zeros(
            (target_batch_size, target_seq_len),
            dtype=dtype,
            device=device,
        )

        # Create attention_mask buffer if present in sample
        if 'attention_mask' in sample_batch:
            static_inputs['attention_mask'] = torch.ones(
                (target_batch_size, target_seq_len),
                dtype=sample_batch['attention_mask'].dtype,
                device=device,
            )

        # Create position_ids buffer if present (for sequence packing)
        if 'position_ids' in sample_batch:
            static_inputs['position_ids'] = torch.zeros(
                (target_batch_size, target_seq_len),
                dtype=sample_batch['position_ids'].dtype,
                device=device,
            )

        return static_inputs

    def capture(
        self,
        shape_key: Tuple[int, int],
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        sample_batch: Dict[str, torch.Tensor],
        amp_dtype: torch.dtype = torch.bfloat16,
        use_amp: bool = True,
        use_scaler: bool = False,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        gradient_accumulation_steps: int = 1,
    ) -> Optional[CachedGraph]:
        """
        Capture a new CUDA graph for the given shape.

        Args:
            shape_key: (batch_size, seq_len) tuple
            model: Model to capture
            optimizer: Optimizer
            sample_batch: Sample batch for buffer creation
            amp_dtype: AMP data type
            use_amp: Whether to use automatic mixed precision
            use_scaler: Whether to use gradient scaler
            scaler: Gradient scaler instance
            gradient_accumulation_steps: Gradient accumulation steps

        Returns:
            CachedGraph if capture succeeded, None otherwise
        """
        import time

        normalized_key = self._normalize_shape(*shape_key)
        target_bs, target_seq = normalized_key

        # Evict if at capacity
        while len(self._cache) >= self.max_graphs:
            # Remove least recently used (first item)
            evicted_key, evicted_graph = self._cache.popitem(last=False)
            evicted_graph.reset()
            self.evictions += 1
            logger.debug(f"Evicted CUDA graph for shape {evicted_key}")

        try:
            start_time = time.time()

            # Create static buffers
            static_inputs = self._create_static_buffers(sample_batch, target_bs, target_seq)

            # Copy sample data into static buffers (with potential padding)
            actual_bs = sample_batch['input_ids'].size(0)
            actual_seq = sample_batch['input_ids'].size(1)

            # Copy actual data (may not fill entire buffer if normalized)
            static_inputs['input_ids'][:actual_bs, :actual_seq].copy_(sample_batch['input_ids'])
            if 'labels' in sample_batch:
                static_inputs['labels'][:actual_bs, :actual_seq].copy_(sample_batch['labels'])
            else:
                static_inputs['labels'][:actual_bs, :actual_seq].copy_(sample_batch['input_ids'])
            if 'attention_mask' in static_inputs and 'attention_mask' in sample_batch:
                static_inputs['attention_mask'][:actual_bs, :actual_seq].copy_(sample_batch['attention_mask'])
                # Zero out padding region
                if actual_seq < target_seq:
                    static_inputs['attention_mask'][:, actual_seq:] = 0
            if 'position_ids' in static_inputs and 'position_ids' in sample_batch:
                static_inputs['position_ids'][:actual_bs, :actual_seq].copy_(sample_batch['position_ids'])

            # Static loss tensor
            device = sample_batch['input_ids'].device
            static_loss = torch.zeros(1, device=device)

            # Warmup pass (prime memory allocations)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=static_inputs['input_ids'],
                    attention_mask=static_inputs.get('attention_mask'),
                    labels=static_inputs['labels'],
                    position_ids=static_inputs.get('position_ids'),
                )
                loss = outputs['loss'] / gradient_accumulation_steps
            if use_scaler and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            optimizer.zero_grad(set_to_none=True)

            # Synchronize before capture
            torch.cuda.synchronize()

            # Capture the graph
            graph = torch.cuda.CUDAGraph()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.graph(graph):
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    outputs = model(
                        input_ids=static_inputs['input_ids'],
                        attention_mask=static_inputs.get('attention_mask'),
                        labels=static_inputs['labels'],
                        position_ids=static_inputs.get('position_ids'),
                    )
                    loss = outputs['loss'] / gradient_accumulation_steps

                if use_scaler and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Store loss reference
                static_loss = loss.detach()

            # Create cached graph
            cached = CachedGraph(
                graph=graph,
                static_inputs=static_inputs,
                static_loss=static_loss,
                batch_size=target_bs,
                seq_len=target_seq,
                capture_time=time.time() - start_time,
            )

            self._cache[normalized_key] = cached
            self.captures += 1

            logger.info(
                f"CUDA graph captured for shape {normalized_key} "
                f"in {cached.capture_time:.2f}s (cache size: {len(self._cache)}/{self.max_graphs})"
            )

            return cached

        except Exception as e:
            logger.warning(f"CUDA graph capture failed for shape {normalized_key}: {e}")
            return None

    def replay(
        self,
        shape_key: Tuple[int, int],
        batch: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Replay a cached graph with new batch data.

        Args:
            shape_key: (batch_size, seq_len) tuple
            batch: New batch data

        Returns:
            Loss tensor if successful, None if graph not found
        """
        cached = self.get_graph(shape_key)
        if cached is None:
            return None

        # Copy batch data into static buffers
        actual_bs = batch['input_ids'].size(0)
        actual_seq = batch['input_ids'].size(1)

        # Handle potential size mismatch with normalized shapes
        cached.static_inputs['input_ids'][:actual_bs, :actual_seq].copy_(batch['input_ids'])
        if 'labels' in batch:
            cached.static_inputs['labels'][:actual_bs, :actual_seq].copy_(batch['labels'])
        else:
            cached.static_inputs['labels'][:actual_bs, :actual_seq].copy_(batch['input_ids'])
        if 'attention_mask' in cached.static_inputs and 'attention_mask' in batch:
            cached.static_inputs['attention_mask'][:actual_bs, :actual_seq].copy_(batch['attention_mask'])
        if 'position_ids' in cached.static_inputs and 'position_ids' in batch:
            cached.static_inputs['position_ids'][:actual_bs, :actual_seq].copy_(batch['position_ids'])

        # Replay
        return cached.replay()

    def clear(self) -> None:
        """Clear all cached graphs and free memory."""
        for cached in self._cache.values():
            cached.reset()
        self._cache.clear()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("CUDA graph cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "max_graphs": self.max_graphs,
            "hits": self.hits,
            "misses": self.misses,
            "captures": self.captures,
            "evictions": self.evictions,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0,
            "shapes": list(self._cache.keys()),
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __del__(self):
        """Clean up on deletion."""
        self.clear()
