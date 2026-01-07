"""
Hybrid Caching System for Training and Generation

This module provides two distinct caching strategies:

1. ActivationCache (Training Mode):
   - Caches intermediate activations during gradient checkpointing
   - Reduces recomputation cost by storing activations from forward pass
   - Cleared after each batch to prevent stale data
   - Uses batch-aware keys to prevent cache collisions

2. KVCacheManager (Generation Mode):
   - Manages KV cache eviction for very long sequences
   - Implements StreamingLLM-style eviction (sink tokens + recent tokens)
   - Integrates with existing model KV cache (past_key_values)
   - Supports sliding window and hybrid eviction policies

Based on research from:
- arXiv:2501.01792: Efficient Caching for Transformer Training
- arXiv:2309.17453: Efficient Streaming Language Models with Attention Sinks
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import logging
import time

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class ActivationCacheConfig:
    """Configuration for training activation cache."""
    enabled: bool = True
    max_size_gb: float = 2.0
    eviction_policy: str = 'lru'  # 'lru', 'lfu', 'hybrid'
    cache_layers: Optional[List[int]] = None  # Which layers to cache (None = all)


@dataclass
class KVCacheConfig:
    """Configuration for generation KV cache management."""
    enabled: bool = True
    max_size_gb: float = 4.0
    eviction_policy: str = 'hybrid'  # 'sliding_window', 'hybrid'
    sliding_window: Optional[int] = None  # Window size for sliding window
    sink_tokens: int = 4  # StreamingLLM: keep first N tokens as "sinks"
    recent_tokens: int = 256  # Keep last N tokens (working memory)


@dataclass
class HybridCacheConfig:
    """Master configuration for hybrid caching system."""
    enabled: bool = True
    mode: str = 'auto'  # 'training', 'generation', 'auto'
    activation_cache: ActivationCacheConfig = field(default_factory=ActivationCacheConfig)
    kv_cache: KVCacheConfig = field(default_factory=KVCacheConfig)


# =============================================================================
# Cache Entry for Activation Cache
# =============================================================================

@dataclass
class CacheEntry:
    """Single cache entry with metadata for eviction scoring."""
    key: str
    value: torch.Tensor
    score: float = 0.0
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    size_bytes: int = 0

    def update_score(self, policy: str = "lru") -> float:
        """Update score based on eviction policy."""
        current_time = time.time()
        recency = 1.0 / (current_time - self.last_access_time + 1e-6)
        frequency = self.access_count

        if policy == "lru":
            self.score = recency
        elif policy == "lfu":
            self.score = frequency
        elif policy == "hybrid":
            # Combine recency (70%) and frequency (30%)
            self.score = 0.7 * recency + 0.3 * frequency

        return self.score


# =============================================================================
# Activation Cache (Training Mode)
# =============================================================================

class ActivationCache:
    """
    Activation cache for gradient checkpointing optimization.

    During gradient checkpointing, the forward pass is computed twice:
    1. During forward pass (activations discarded for memory)
    2. During backward pass (recomputation for gradients)

    This cache strategically stores activations from (1) to reduce
    recomputation in (2), trading memory for time.

    Key Design:
    - Keys: f"layer_{layer_idx}_{phase}_batch_{batch_ptr}" for unique identification
    - Values: Intermediate activations (hidden_states after attention, after FFN)
    - Cleared: After each backward pass (batch boundary)

    Example:
        >>> cache = ActivationCache(ActivationCacheConfig(max_size_gb=2.0))
        >>> cache.on_batch_start({'input_ids': input_ids})
        >>>
        >>> # During forward pass
        >>> cache.store(layer_idx=0, activation=hidden_states, phase='post_attn')
        >>>
        >>> # During recomputation (backward)
        >>> cached = cache.get(layer_idx=0, phase='post_attn')
        >>> if cached is not None:
        ...     hidden_states = cached  # Skip recomputation
        >>>
        >>> cache.on_batch_end()  # Clear for next batch
    """

    def __init__(self, config: ActivationCacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Batch tracking
        self._batch_ptr: int = 0
        self._forward_pass_count: int = 0

        # Memory tracking
        self.max_bytes = int(config.max_size_gb * 1e9)
        self.current_bytes = 0

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        logger.info(f"ActivationCache initialized: {config.max_size_gb:.1f}GB max")

    def _make_key(self, layer_idx: int, phase: str) -> str:
        """Generate unique cache key for an activation."""
        return f"layer_{layer_idx}_{phase}_batch_{self._batch_ptr}_fwd_{self._forward_pass_count}"

    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """Get tensor size in bytes."""
        return tensor.element_size() * tensor.nelement()

    def _evict_if_needed(self, required_bytes: int):
        """Evict entries if needed to make room."""
        if self.current_bytes + required_bytes <= self.max_bytes:
            return

        # Update all scores
        for entry in self.cache.values():
            entry.update_score(self.config.eviction_policy)

        # Sort by score (lowest first)
        entries = sorted(self.cache.items(), key=lambda x: x[1].score)

        # Evict until we have room
        bytes_to_free = (self.current_bytes + required_bytes) - self.max_bytes
        freed_bytes = 0

        for key, entry in entries:
            if freed_bytes >= bytes_to_free:
                break
            del self.cache[key]
            freed_bytes += entry.size_bytes
            self.current_bytes -= entry.size_bytes
            self.evictions += 1

        logger.debug(f"ActivationCache: evicted {freed_bytes/1e6:.1f}MB")

    def store(self, layer_idx: int, activation: torch.Tensor, phase: str = 'post_attn'):
        """
        Store activation with memory management.

        Args:
            layer_idx: Layer index
            activation: Activation tensor to cache
            phase: Phase identifier ('post_attn', 'post_ffn', etc.)
        """
        if not self.config.enabled:
            return

        # Check if layer should be cached
        if self.config.cache_layers is not None:
            if layer_idx not in self.config.cache_layers:
                return

        key = self._make_key(layer_idx, phase)
        size_bytes = self._get_tensor_size(activation)

        # Evict if needed
        self._evict_if_needed(size_bytes)

        # Store detached tensor (no gradients)
        entry = CacheEntry(
            key=key,
            value=activation.detach(),
            size_bytes=size_bytes,
        )
        entry.update_score(self.config.eviction_policy)

        self.cache[key] = entry
        self.current_bytes += size_bytes

    def get(self, layer_idx: int, phase: str = 'post_attn') -> Optional[torch.Tensor]:
        """
        Retrieve cached activation if available.

        Args:
            layer_idx: Layer index
            phase: Phase identifier

        Returns:
            Cached tensor or None if not found
        """
        if not self.config.enabled:
            return None

        key = self._make_key(layer_idx, phase)

        if key in self.cache:
            self.hits += 1
            entry = self.cache[key]
            entry.access_count += 1
            entry.last_access_time = time.time()
            entry.update_score(self.config.eviction_policy)
            self.cache.move_to_end(key)
            return entry.value
        else:
            self.misses += 1
            return None

    def on_batch_start(self, batch: Dict[str, torch.Tensor]):
        """
        Called at batch start to set batch identifier.

        Args:
            batch: Batch dictionary containing input tensors
        """
        # Use data pointer as batch identifier
        if 'input_ids' in batch:
            self._batch_ptr = batch['input_ids'].data_ptr()
        else:
            # Fallback to time-based ID
            self._batch_ptr = int(time.time() * 1000) % (2**31)
        self._forward_pass_count = 0

    def on_forward_start(self):
        """Called at start of each forward pass (for gradient checkpointing tracking)."""
        self._forward_pass_count += 1

    def on_batch_end(self):
        """Called at batch end to clear per-batch cache."""
        self.clear()

    def clear(self):
        """Clear all cached activations."""
        self.cache.clear()
        self.current_bytes = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0.0

        return {
            'hit_rate': hit_rate,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'total_entries': len(self.cache),
            'current_size_gb': self.current_bytes / 1e9,
            'max_size_gb': self.max_bytes / 1e9,
        }

    def log_stats(self):
        """Log cache statistics."""
        stats = self.get_stats()
        logger.info(f"ActivationCache: hit_rate={stats['hit_rate']:.1%}, "
                   f"entries={stats['total_entries']}, "
                   f"size={stats['current_size_gb']:.2f}GB")


# =============================================================================
# KV Cache Manager (Generation Mode)
# =============================================================================

class KVCacheManager:
    """
    Enhanced KV cache manager for generation with intelligent eviction.

    The model's MultiHeadAttention already handles KV caching via past_key_value tuples.
    This manager provides:
    1. Memory-aware eviction for very long sequences
    2. Sliding window support
    3. StreamingLLM-style sink token preservation

    Integration: This does NOT wrap attention. It provides utilities that the
    generation loop and model forward() call to manage KV cache tensors.

    Based on "Efficient Streaming Language Models with Attention Sinks"
    (arXiv:2309.17453), the first few tokens act as "attention sinks"
    and should be preserved alongside recent tokens for quality.

    Example:
        >>> kv_manager = KVCacheManager(KVCacheConfig(max_size_gb=4.0))
        >>>
        >>> # In model forward during generation
        >>> if kv_manager.should_evict(past_key_values):
        ...     past_key_values = kv_manager.evict(past_key_values)
    """

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.max_bytes = int(config.max_size_gb * 1e9)

        # Statistics
        self.eviction_count = 0
        self.total_tokens_evicted = 0

        logger.info(f"KVCacheManager initialized: {config.max_size_gb:.1f}GB max, "
                   f"policy={config.eviction_policy}, "
                   f"sink_tokens={config.sink_tokens}, "
                   f"recent_tokens={config.recent_tokens}")

    def _get_kv_size(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]) -> int:
        """Calculate total size of KV cache in bytes."""
        total = 0
        for kv in past_key_values:
            if kv is not None and kv[0] is not None:
                k, v = kv
                total += k.element_size() * k.numel()
                total += v.element_size() * v.numel()
        return total

    def _get_seq_len(self, past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]) -> int:
        """Get sequence length from KV cache."""
        if past_key_values and past_key_values[0] is not None:
            k = past_key_values[0][0]
            if k is not None:
                return k.size(2)  # Shape: [batch, heads, seq_len, head_dim]
        return 0

    def should_evict(self, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]) -> bool:
        """
        Check if KV cache exceeds memory budget.

        Args:
            past_key_values: List of (k, v) tuples from model

        Returns:
            True if eviction is needed
        """
        if past_key_values is None or not self.config.enabled:
            return False

        total_bytes = self._get_kv_size(past_key_values)
        return total_bytes > self.max_bytes

    def evict(
        self,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        attention_scores: Optional[List[torch.Tensor]] = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Evict tokens from KV cache to stay within memory budget.

        Args:
            past_key_values: List of (k, v) tuples, shape [batch, heads, seq, head_dim]
            attention_scores: Optional attention scores for attention-based eviction

        Returns:
            Evicted KV cache (same format, fewer tokens)
        """
        if self.config.eviction_policy == 'sliding_window':
            return self._sliding_window_evict(past_key_values)
        else:  # 'hybrid' (default)
            return self._hybrid_evict(past_key_values, attention_scores)

    def _sliding_window_evict(
        self,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Keep only last sliding_window tokens.

        Simple but effective for models that don't need long-range attention.
        """
        window = self.config.sliding_window
        if window is None:
            # Compute window based on memory budget
            seq_len = self._get_seq_len(past_key_values)
            if seq_len == 0:
                return past_key_values

            bytes_per_token = self._get_kv_size(past_key_values) / seq_len
            window = max(256, int(self.max_bytes / bytes_per_token))

        new_past = []
        tokens_before = self._get_seq_len(past_key_values)

        for kv in past_key_values:
            if kv is None:
                new_past.append(None)
                continue

            k, v = kv
            if k is None:
                new_past.append((None, None))
                continue

            seq_len = k.size(2)
            if seq_len > window:
                # Keep last 'window' tokens
                new_past.append((
                    k[:, :, -window:, :].contiguous(),
                    v[:, :, -window:, :].contiguous()
                ))
            else:
                new_past.append((k, v))

        tokens_after = self._get_seq_len(new_past)
        self.eviction_count += 1
        self.total_tokens_evicted += (tokens_before - tokens_after)

        logger.debug(f"KVCache sliding window: {tokens_before} -> {tokens_after} tokens")

        return new_past

    def _hybrid_evict(
        self,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        attention_scores: Optional[List[torch.Tensor]] = None
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        StreamingLLM-style eviction: keep sink tokens + recent tokens.

        Based on "Efficient Streaming Language Models with Attention Sinks"
        (arXiv:2309.17453), the first few tokens act as "attention sinks"
        that accumulate attention and should be preserved.

        Strategy:
        - Keep first 'sink_tokens' tokens (attention sinks)
        - Keep last 'recent_tokens' tokens (working memory)
        - Discard middle tokens

        This preserves both global context (via sinks) and local context (recent).
        """
        sink = self.config.sink_tokens
        recent = self.config.recent_tokens

        new_past = []
        tokens_before = self._get_seq_len(past_key_values)

        for kv in past_key_values:
            if kv is None:
                new_past.append(None)
                continue

            k, v = kv
            if k is None:
                new_past.append((None, None))
                continue

            seq_len = k.size(2)

            if seq_len > sink + recent:
                # Keep first 'sink' tokens + last 'recent' tokens
                k_sink = k[:, :, :sink, :]
                k_recent = k[:, :, -recent:, :]
                k_new = torch.cat([k_sink, k_recent], dim=2).contiguous()

                v_sink = v[:, :, :sink, :]
                v_recent = v[:, :, -recent:, :]
                v_new = torch.cat([v_sink, v_recent], dim=2).contiguous()

                new_past.append((k_new, v_new))
            else:
                # Sequence shorter than sink + recent, keep all
                new_past.append((k, v))

        tokens_after = self._get_seq_len(new_past)
        self.eviction_count += 1
        self.total_tokens_evicted += (tokens_before - tokens_after)

        logger.debug(f"KVCache hybrid evict: {tokens_before} -> {tokens_after} tokens "
                    f"(kept {sink} sink + {min(recent, tokens_before - sink)} recent)")

        return new_past

    def get_stats(self) -> Dict[str, Any]:
        """Get cache manager statistics."""
        return {
            'eviction_count': self.eviction_count,
            'total_tokens_evicted': self.total_tokens_evicted,
            'policy': self.config.eviction_policy,
            'sink_tokens': self.config.sink_tokens,
            'recent_tokens': self.config.recent_tokens,
            'max_size_gb': self.max_bytes / 1e9,
        }

    def log_stats(self):
        """Log cache manager statistics."""
        stats = self.get_stats()
        logger.info(f"KVCacheManager: evictions={stats['eviction_count']}, "
                   f"tokens_evicted={stats['total_tokens_evicted']}, "
                   f"policy={stats['policy']}")


# =============================================================================
# Hybrid Cache Manager (Facade)
# =============================================================================

class HybridCacheManager:
    """
    Central manager for hybrid caching across training and generation modes.

    This class acts as a facade that provides access to appropriate cache
    implementations based on the current mode (training vs generation).

    Usage:
        >>> config = HybridCacheConfig(mode='auto')
        >>> manager = HybridCacheManager(config)
        >>>
        >>> # Attach to model
        >>> model._hybrid_cache_manager = manager
        >>>
        >>> # Set activation cache on layers (training)
        >>> for idx, layer in enumerate(model.layers):
        ...     layer.set_activation_cache(manager.activation_cache)
        >>>
        >>> # Set KV cache manager (generation)
        >>> model._kv_cache_manager = manager.kv_cache_manager
    """

    def __init__(self, config: HybridCacheConfig):
        self.config = config
        self._mode = config.mode

        # Initialize sub-caches based on config
        if config.activation_cache.enabled:
            self.activation_cache = ActivationCache(config.activation_cache)
        else:
            self.activation_cache = None

        if config.kv_cache.enabled:
            self.kv_cache_manager = KVCacheManager(config.kv_cache)
        else:
            self.kv_cache_manager = None

        logger.info(f"HybridCacheManager initialized: mode={config.mode}, "
                   f"activation_cache={config.activation_cache.enabled}, "
                   f"kv_cache={config.kv_cache.enabled}")

    @property
    def mode(self) -> str:
        """Get current caching mode."""
        return self._mode

    def set_mode(self, mode: str):
        """
        Set caching mode: 'training', 'generation', or 'auto'.

        Args:
            mode: Caching mode
        """
        if mode not in ('training', 'generation', 'auto'):
            raise ValueError(f"Invalid mode: {mode}. Must be 'training', 'generation', or 'auto'")
        self._mode = mode
        logger.debug(f"HybridCacheManager: mode set to {mode}")

    def infer_mode(self, model: nn.Module) -> str:
        """
        Auto-detect mode from model.training state.

        Args:
            model: PyTorch model

        Returns:
            'training' if model.training else 'generation'
        """
        return 'training' if model.training else 'generation'

    def get_effective_mode(self, model: Optional[nn.Module] = None) -> str:
        """
        Get effective mode (resolves 'auto' using model state).

        Args:
            model: Optional model for auto-detection

        Returns:
            'training' or 'generation'
        """
        if self._mode == 'auto' and model is not None:
            return self.infer_mode(model)
        elif self._mode == 'auto':
            return 'training'  # Default to training if no model
        return self._mode

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics from all caches."""
        stats = {
            'mode': self._mode,
            'activation_cache': None,
            'kv_cache': None,
        }

        if self.activation_cache:
            stats['activation_cache'] = self.activation_cache.get_stats()

        if self.kv_cache_manager:
            stats['kv_cache'] = self.kv_cache_manager.get_stats()

        return stats

    def log_stats(self):
        """Log statistics from all caches."""
        logger.info("=" * 60)
        logger.info("Hybrid Cache Statistics")
        logger.info("=" * 60)
        logger.info(f"Mode: {self._mode}")

        if self.activation_cache:
            self.activation_cache.log_stats()

        if self.kv_cache_manager:
            self.kv_cache_manager.log_stats()

        logger.info("=" * 60)


# =============================================================================
# Helper Functions
# =============================================================================

def build_hybrid_cache_config(config_dict: Dict[str, Any]) -> HybridCacheConfig:
    """
    Build HybridCacheConfig from a dictionary (e.g., from YAML).

    Args:
        config_dict: Configuration dictionary with hybrid_caching keys

    Returns:
        HybridCacheConfig instance
    """
    activation_dict = config_dict.get('activation_cache', {})
    kv_dict = config_dict.get('kv_cache', {})

    activation_config = ActivationCacheConfig(
        enabled=activation_dict.get('enabled', True),
        max_size_gb=activation_dict.get('max_size_gb', 2.0),
        eviction_policy=activation_dict.get('eviction_policy', 'lru'),
        cache_layers=activation_dict.get('cache_layers'),
    )

    kv_config = KVCacheConfig(
        enabled=kv_dict.get('enabled', True),
        max_size_gb=kv_dict.get('max_size_gb', 4.0),
        eviction_policy=kv_dict.get('eviction_policy', 'hybrid'),
        sliding_window=kv_dict.get('sliding_window'),
        sink_tokens=kv_dict.get('sink_tokens', 4),
        recent_tokens=kv_dict.get('recent_tokens', 256),
    )

    return HybridCacheConfig(
        enabled=config_dict.get('enabled', True),
        mode=config_dict.get('mode', 'auto'),
        activation_cache=activation_config,
        kv_cache=kv_config,
    )


# =============================================================================
# Backwards Compatibility - Keep old names as aliases (deprecated)
# =============================================================================

# These are kept for backwards compatibility with existing model checkpoints
# that may have been saved with the old CachedAttention wrapper

class _DeprecatedHybridCacheConfig:
    """Deprecated: Use HybridCacheConfig instead."""
    def __init__(self, **kwargs):
        import warnings
        warnings.warn(
            "The old HybridCacheConfig is deprecated. Use the new config structure.",
            DeprecationWarning,
            stacklevel=2
        )


def apply_hybrid_caching(model: nn.Module, config=None, target_modules=None):
    """
    Deprecated: Hybrid caching is now applied via HybridCacheManager.

    This function is kept for backwards compatibility but does nothing.
    The new approach attaches a HybridCacheManager to the model instead
    of wrapping attention modules.
    """
    import warnings
    warnings.warn(
        "apply_hybrid_caching() is deprecated. Hybrid caching is now applied "
        "via HybridCacheManager attached to the model. This function is a no-op.",
        DeprecationWarning,
        stacklevel=2
    )
    return model, None


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Hybrid Cache System...")
    print()

    # Test ActivationCache
    print("1. Testing ActivationCache...")
    activation_config = ActivationCacheConfig(max_size_gb=0.01)  # 10MB for testing
    activation_cache = ActivationCache(activation_config)

    # Simulate batch
    fake_batch = {'input_ids': torch.randint(0, 1000, (4, 128))}
    activation_cache.on_batch_start(fake_batch)

    # Store some activations
    for layer in range(4):
        activation = torch.randn(4, 128, 256)
        activation_cache.store(layer, activation, 'post_attn')

    # Retrieve
    for layer in range(4):
        cached = activation_cache.get(layer, 'post_attn')
        assert cached is not None, f"Layer {layer} should be cached"

    activation_cache.log_stats()
    activation_cache.on_batch_end()

    # After batch end, cache should be empty
    cached = activation_cache.get(0, 'post_attn')
    assert cached is None, "Cache should be empty after batch end"

    print("   ActivationCache: PASSED")
    print()

    # Test KVCacheManager
    print("2. Testing KVCacheManager...")
    kv_config = KVCacheConfig(
        max_size_gb=0.001,  # 1MB for testing
        eviction_policy='hybrid',
        sink_tokens=4,
        recent_tokens=10,
    )
    kv_manager = KVCacheManager(kv_config)

    # Create fake past_key_values (16 layers, batch=2, heads=4, seq=100, head_dim=64)
    past_kv = []
    for _ in range(16):
        k = torch.randn(2, 4, 100, 64)
        v = torch.randn(2, 4, 100, 64)
        past_kv.append((k, v))

    # Check if eviction needed
    should_evict = kv_manager.should_evict(past_kv)
    print(f"   Should evict: {should_evict}")

    if should_evict:
        new_past_kv = kv_manager.evict(past_kv)
        new_seq_len = new_past_kv[0][0].size(2)
        print(f"   Evicted: 100 -> {new_seq_len} tokens")
        assert new_seq_len == 14, f"Expected 14 (4 sink + 10 recent), got {new_seq_len}"

    kv_manager.log_stats()
    print("   KVCacheManager: PASSED")
    print()

    # Test HybridCacheManager
    print("3. Testing HybridCacheManager...")
    hybrid_config = HybridCacheConfig(
        mode='auto',
        activation_cache=activation_config,
        kv_cache=kv_config,
    )
    manager = HybridCacheManager(hybrid_config)

    assert manager.activation_cache is not None
    assert manager.kv_cache_manager is not None

    # Test mode detection
    class FakeModel(nn.Module):
        pass

    model = FakeModel()
    model.train()
    assert manager.infer_mode(model) == 'training'

    model.eval()
    assert manager.infer_mode(model) == 'generation'

    manager.log_stats()
    print("   HybridCacheManager: PASSED")
    print()

    # Test config builder
    print("4. Testing config builder...")
    yaml_dict = {
        'enabled': True,
        'mode': 'auto',
        'activation_cache': {
            'enabled': True,
            'max_size_gb': 2.0,
            'eviction_policy': 'lru',
        },
        'kv_cache': {
            'enabled': True,
            'max_size_gb': 4.0,
            'eviction_policy': 'hybrid',
            'sink_tokens': 4,
            'recent_tokens': 256,
        },
    }
    config = build_hybrid_cache_config(yaml_dict)
    assert config.enabled == True
    assert config.mode == 'auto'
    assert config.activation_cache.max_size_gb == 2.0
    assert config.kv_cache.sink_tokens == 4
    print("   Config builder: PASSED")
    print()

    print("All tests passed!")
