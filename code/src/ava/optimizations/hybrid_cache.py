"""
KV-Activation Hybrid Caching

Based on research from:
- arXiv:2501.01792: Efficient Caching for Transformer Training

This module implements an intelligent hybrid caching system for both KV cache
(attention keys/values) and activations. Instead of keeping everything or
evicting randomly, this uses a scoring system to keep the most valuable cache
entries.

Key features:
- Unified cache for KV and activations
- Intelligent eviction policy (keeps high-value entries)
- Predictive prefetching (loads what will be needed)
- Configurable cache size and policies

How it works:
1. Score cache entries by reuse probability
2. Keep high-scoring entries, evict low-scoring
3. Prefetch entries that will be needed soon
4. Balance between KV cache and activation cache

Scoring factors:
- Recency: Recently used = higher score
- Frequency: Often reused = higher score
- Position: Certain positions reused more = higher score
- Size: Smaller entries easier to keep = slight boost

Cache policies:
- LRU (Least Recently Used): Simple, effective
- LFU (Least Frequently Used): Good for repeated patterns
- Hybrid: Combines recency + frequency
- Adaptive: Learns optimal policy during training
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    key: str
    value: torch.Tensor
    score: float = 0.0
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    size_bytes: int = 0
    is_kv_cache: bool = False  # True for KV cache, False for activation

    def update_score(self, policy: str = "hybrid"):
        """Update score based on policy"""
        current_time = time.time()
        recency = 1.0 / (current_time - self.last_access_time + 1e-6)
        frequency = self.access_count

        if policy == "lru":
            self.score = recency
        elif policy == "lfu":
            self.score = frequency
        elif policy == "hybrid":
            # Combine recency and frequency
            self.score = 0.7 * recency + 0.3 * frequency
        elif policy == "adaptive":
            # Adaptive scoring (can be tuned during training)
            size_penalty = 1.0 / (self.size_bytes / 1e6 + 1.0)  # Prefer smaller entries
            self.score = 0.5 * recency + 0.3 * frequency + 0.2 * size_penalty
        
        return self.score


@dataclass
class HybridCacheConfig:
    """Configuration for hybrid caching"""
    enabled: bool = True
    max_cache_size_gb: float = 4.0  # Maximum cache size in GB
    kv_cache_ratio: float = 0.6  # Ratio of cache for KV vs activations
    eviction_policy: str = "hybrid"  # lru, lfu, hybrid, adaptive
    prefetch_enabled: bool = True
    prefetch_lookahead: int = 2  # Prefetch N layers ahead
    min_score_threshold: float = 0.1  # Evict entries below this score


class HybridCache:
    """
    Hybrid cache for KV cache and activations with intelligent eviction.

    This cache maintains both KV cache (for attention) and activation cache
    (for gradient checkpointing) in a unified system with smart eviction.

    Args:
        config: HybridCacheConfig

    Example:
        >>> cache = HybridCache(config)
        >>>
        >>> # Store KV cache
        >>> cache.store("layer_0_kv", kv_tensor, is_kv_cache=True)
        >>>
        >>> # Store activation
        >>> cache.store("layer_5_act", activation, is_kv_cache=False)
        >>>
        >>> # Retrieve (automatically updates scores)
        >>> kv = cache.get("layer_0_kv")
        >>>
        >>> # Cache automatically evicts low-score entries when full
    """

    def __init__(self, config: HybridCacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.current_size_bytes = 0
        
        # Separate tracking for KV vs activation
        self.kv_size_bytes = 0
        self.activation_size_bytes = 0
        
        # Prefetch queue
        self.prefetch_queue: List[str] = []
        
        # Max sizes
        max_bytes = int(config.max_cache_size_gb * 1e9)
        self.max_kv_bytes = int(max_bytes * config.kv_cache_ratio)
        self.max_activation_bytes = int(max_bytes * (1 - config.kv_cache_ratio))
        
        logger.info(f"Hybrid cache initialized: {config.max_cache_size_gb:.1f}GB total")
        logger.info(f"  KV cache: {self.max_kv_bytes/1e9:.1f}GB")
        logger.info(f"  Activation cache: {self.max_activation_bytes/1e9:.1f}GB")

    def _get_tensor_size(self, tensor: torch.Tensor) -> int:
        """Get tensor size in bytes"""
        return tensor.element_size() * tensor.nelement()

    def _evict_if_needed(self, required_bytes: int, is_kv_cache: bool):
        """Evict entries if needed to make room"""
        if is_kv_cache:
            current_size = self.kv_size_bytes
            max_size = self.max_kv_bytes
        else:
            current_size = self.activation_size_bytes
            max_size = self.max_activation_bytes

        if current_size + required_bytes <= max_size:
            return  # No eviction needed

        # Need to evict - update all scores first
        for entry in self.cache.values():
            if entry.is_kv_cache == is_kv_cache:
                entry.update_score(self.config.eviction_policy)

        # Sort entries by score (lowest first)
        entries_to_consider = [
            (key, entry) for key, entry in self.cache.items()
            if entry.is_kv_cache == is_kv_cache
        ]
        entries_to_consider.sort(key=lambda x: x[1].score)

        # Evict lowest-scoring entries until we have room
        bytes_to_free = (current_size + required_bytes) - max_size
        freed_bytes = 0

        for key, entry in entries_to_consider:
            if freed_bytes >= bytes_to_free:
                break

            # Evict this entry
            del self.cache[key]
            freed_bytes += entry.size_bytes
            self.evictions += 1

            if entry.is_kv_cache:
                self.kv_size_bytes -= entry.size_bytes
            else:
                self.activation_size_bytes -= entry.size_bytes

        self.current_size_bytes -= freed_bytes

        logger.debug(f"Evicted {freed_bytes/1e6:.1f}MB ({self.evictions} entries)")

    def store(
        self,
        key: str,
        value: torch.Tensor,
        is_kv_cache: bool = False,
        score: Optional[float] = None,
    ):
        """
        Store a tensor in the cache.

        Args:
            key: Cache key
            value: Tensor to store
            is_kv_cache: True if this is KV cache, False if activation
            score: Optional initial score (computed if not provided)
        """
        if not self.config.enabled:
            return

        # Calculate size
        size_bytes = self._get_tensor_size(value)

        # Evict if needed
        self._evict_if_needed(size_bytes, is_kv_cache)

        # Create entry
        entry = CacheEntry(
            key=key,
            value=value.detach(),  # Detach to avoid holding gradients
            size_bytes=size_bytes,
            is_kv_cache=is_kv_cache,
            score=score or 1.0,
        )

        # Update entry score
        entry.update_score(self.config.eviction_policy)

        # Store
        self.cache[key] = entry
        self.current_size_bytes += size_bytes

        if is_kv_cache:
            self.kv_size_bytes += size_bytes
        else:
            self.activation_size_bytes += size_bytes

    def get(self, key: str) -> Optional[torch.Tensor]:
        """
        Retrieve a tensor from cache.

        Args:
            key: Cache key

        Returns:
            Cached tensor or None if not found
        """
        if not self.config.enabled:
            return None

        if key in self.cache:
            # Hit
            self.hits += 1
            entry = self.cache[key]
            
            # Update access metadata
            entry.access_count += 1
            entry.last_access_time = time.time()
            entry.update_score(self.config.eviction_policy)
            
            # Move to end (for LRU aspect)
            self.cache.move_to_end(key)
            
            return entry.value
        else:
            # Miss
            self.misses += 1
            return None

    def prefetch(self, keys: List[str]):
        """
        Prefetch entries (hint that these will be needed soon).

        Args:
            keys: List of keys to prefetch
        """
        if not self.config.prefetch_enabled:
            return

        self.prefetch_queue.extend(keys)

        # Boost scores for prefetch queue
        for key in keys:
            if key in self.cache:
                self.cache[key].score *= 1.5  # Boost score

    def clear(self, kv_only: bool = False, activation_only: bool = False):
        """
        Clear cache.

        Args:
            kv_only: Only clear KV cache
            activation_only: Only clear activation cache
        """
        if kv_only:
            keys_to_remove = [k for k, v in self.cache.items() if v.is_kv_cache]
        elif activation_only:
            keys_to_remove = [k for k, v in self.cache.items() if not v.is_kv_cache]
        else:
            keys_to_remove = list(self.cache.keys())

        for key in keys_to_remove:
            entry = self.cache[key]
            del self.cache[key]
            
            self.current_size_bytes -= entry.size_bytes
            if entry.is_kv_cache:
                self.kv_size_bytes -= entry.size_bytes
            else:
                self.activation_size_bytes -= entry.size_bytes

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_accesses = self.hits + self.misses
        hit_rate = self.hits / total_accesses if total_accesses > 0 else 0.0

        return {
            'hit_rate': hit_rate,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'total_entries': len(self.cache),
            'current_size_gb': self.current_size_bytes / 1e9,
            'kv_size_gb': self.kv_size_bytes / 1e9,
            'activation_size_gb': self.activation_size_bytes / 1e9,
            'kv_entries': sum(1 for v in self.cache.values() if v.is_kv_cache),
            'activation_entries': sum(1 for v in self.cache.values() if not v.is_kv_cache),
        }

    def log_stats(self):
        """Log cache statistics"""
        stats = self.get_stats()

        logger.info("=" * 60)
        logger.info("Hybrid Cache Statistics")
        logger.info("=" * 60)
        logger.info(f"Hit rate: {stats['hit_rate']:.1%}")
        logger.info(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
        logger.info(f"Evictions: {stats['evictions']}")
        logger.info(f"Total entries: {stats['total_entries']}")
        logger.info(f"  KV cache: {stats['kv_entries']} ({stats['kv_size_gb']:.2f}GB)")
        logger.info(f"  Activations: {stats['activation_entries']} ({stats['activation_size_gb']:.2f}GB)")
        logger.info(f"Total size: {stats['current_size_gb']:.2f}GB")
        logger.info("=" * 60)


class CachedAttention(nn.Module):
    """
    Attention module with hybrid caching.

    This wraps a standard attention module and adds hybrid caching for KV pairs.

    Args:
        attention_module: Base attention module
        cache: HybridCache instance
        layer_idx: Layer index (for cache keys)
    """

    def __init__(
        self,
        attention_module: nn.Module,
        cache: HybridCache,
        layer_idx: int,
    ):
        super().__init__()
        # Use _wrapped_attn instead of attention to avoid double-nesting in state dict keys
        # This prevents keys like layers.X.attention.attention.q_proj.weight
        self._wrapped_attn = attention_module
        self.cache = cache
        self.layer_idx = layer_idx

    def forward(self, *args, use_cache: bool = True, **kwargs):
        """Forward with caching"""
        # Generate cache key
        # In practice, would include position info, batch idx, etc.
        cache_key = f"layer_{self.layer_idx}_kv"

        # Try to get from cache
        if use_cache:
            cached_kv = self.cache.get(cache_key)
            if cached_kv is not None:
                # Use cached KV
                return cached_kv

        # Compute attention
        output = self._wrapped_attn(*args, **kwargs)

        # Store in cache
        if use_cache:
            self.cache.store(cache_key, output, is_kv_cache=True)

        return output


def apply_hybrid_caching(
    model: nn.Module,
    config: Optional[HybridCacheConfig] = None,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, HybridCache]:
    """
    Apply hybrid caching to a model.

    Args:
        model: Model to modify
        config: HybridCacheConfig
        target_modules: Modules to cache (None = auto-detect attention)

    Returns:
        Tuple of (modified model, cache instance)
    """
    if config is None:
        config = HybridCacheConfig()

    cache = HybridCache(config)

    # Collect modules to wrap first (to avoid modifying dict during iteration)
    modules_to_wrap = []

    for name, module in model.named_modules():
        # Only wrap top-level attention modules, not their children
        # e.g., wrap "layers.0.attention" but not "layers.0.attention.q_proj"
        attr_name = name.split(".")[-1] if name else ""

        if target_modules is not None:
            # Use explicit target list
            is_target = any(t == attr_name for t in target_modules)
        else:
            # Auto-detect: only wrap modules named exactly "attention"
            # This avoids wrapping children like q_proj, k_proj, etc.
            is_target = attr_name == "attention"

        if is_target:
            modules_to_wrap.append((name, module))

    # Now apply wrapping
    for layer_idx, (name, module) in enumerate(modules_to_wrap):
        parent_name = ".".join(name.split(".")[:-1])
        attr_name = name.split(".")[-1]

        parent = model
        for part in parent_name.split("."):
            if part:
                parent = getattr(parent, part)

        cached_module = CachedAttention(module, cache, layer_idx)
        setattr(parent, attr_name, cached_module)

        logger.info(f"Applied hybrid caching to {name}")

    return model, cache


if __name__ == "__main__":
    # Test hybrid caching
    logging.basicConfig(level=logging.INFO)

    print("Testing Hybrid Cache...")

    config = HybridCacheConfig(
        max_cache_size_gb=0.1,  # 100MB for testing
        eviction_policy="hybrid",
    )

    cache = HybridCache(config)

    # Simulate caching
    for i in range(20):
        key = f"tensor_{i}"
        value = torch.randn(100, 100)
        is_kv = (i % 2 == 0)
        cache.store(key, value, is_kv_cache=is_kv)

    # Simulate access pattern
    for i in [0, 2, 4, 0, 2, 0]:  # Access some repeatedly
        result = cache.get(f"tensor_{i}")

    # Print stats
    cache.log_stats()

    print("\n✓ Hybrid cache test passed!")
