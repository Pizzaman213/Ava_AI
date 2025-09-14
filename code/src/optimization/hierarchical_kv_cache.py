"""
Hierarchical KV Cache Storage System

Implements a multi-tier caching system for KV pairs with automatic migration
between GPU, CPU, and NVMe storage based on access patterns.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
import numpy as np
from collections import OrderedDict, deque
import time
import os
import mmap
import pickle
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for hierarchical KV cache"""
    # Memory tiers
    gpu_cache_size_gb: float = 8.0
    cpu_cache_size_gb: float = 32.0
    nvme_cache_size_gb: float = 128.0
    
    # Cache settings
    block_size: int = 16  # tokens per block
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    
    # Performance settings
    prefetch_distance: int = 4  # blocks to prefetch
    eviction_policy: str = "lru"  # lru, lfu, or adaptive
    async_transfers: bool = True
    num_async_workers: int = 4
    
    # Thresholds
    gpu_memory_threshold: float = 0.9
    cpu_memory_threshold: float = 0.9
    promotion_threshold: int = 3  # accesses before promotion
    
    # File paths
    nvme_cache_dir: str = "/tmp/kv_cache"


class CacheBlock(NamedTuple):
    """Single cache block metadata"""
    block_id: int
    layer_id: int
    seq_id: int
    position: int
    size: int
    tier: str  # "gpu", "cpu", "nvme"
    last_access: float
    access_count: int
    data_ptr: Optional[Any] = None


class TierStorage:
    """Base class for storage tiers"""
    
    def __init__(self, name: str, capacity_bytes: int):
        self.name = name
        self.capacity_bytes = capacity_bytes
        self.used_bytes = 0
        self.blocks: Dict[int, CacheBlock] = {}
        self.lock = threading.Lock()
        
    def can_allocate(self, size_bytes: int) -> bool:
        """Check if can allocate given size"""
        return self.used_bytes + size_bytes <= self.capacity_bytes
    
    def allocate(self, block: CacheBlock, data: torch.Tensor) -> bool:
        """Allocate storage for block"""
        raise NotImplementedError
        
    def deallocate(self, block_id: int):
        """Deallocate storage for block"""
        raise NotImplementedError
        
    def read(self, block_id: int) -> Optional[torch.Tensor]:
        """Read data from storage"""
        raise NotImplementedError


class GPUStorage(TierStorage):
    """GPU memory storage tier"""
    
    def __init__(self, capacity_bytes: int, device: str = "cuda"):
        super().__init__("gpu", capacity_bytes)
        self.device = torch.device(device)
        self.data_pool: Dict[int, torch.Tensor] = {}
        
    def allocate(self, block: CacheBlock, data: torch.Tensor) -> bool:
        """Allocate GPU memory for block"""
        with self.lock:
            size_bytes = data.numel() * data.element_size()
            if not self.can_allocate(size_bytes):
                return False
                
            # Store data on GPU
            self.data_pool[block.block_id] = data.to(self.device)
            self.blocks[block.block_id] = block
            self.used_bytes += size_bytes
            return True
            
    def deallocate(self, block_id: int):
        """Free GPU memory"""
        with self.lock:
            if block_id in self.data_pool:
                data = self.data_pool[block_id]
                size_bytes = data.numel() * data.element_size()
                del self.data_pool[block_id]
                del self.blocks[block_id]
                self.used_bytes -= size_bytes
                
    def read(self, block_id: int) -> Optional[torch.Tensor]:
        """Read data from GPU"""
        with self.lock:
            return self.data_pool.get(block_id)


class CPUStorage(TierStorage):
    """CPU memory storage tier with pinned memory"""
    
    def __init__(self, capacity_bytes: int):
        super().__init__("cpu", capacity_bytes)
        self.data_pool: Dict[int, torch.Tensor] = {}
        
    def allocate(self, block: CacheBlock, data: torch.Tensor) -> bool:
        """Allocate CPU memory for block"""
        with self.lock:
            size_bytes = data.numel() * data.element_size()
            if not self.can_allocate(size_bytes):
                return False
                
            # Store data in pinned CPU memory for fast transfers
            cpu_data = torch.empty_like(data, device="cpu", pin_memory=True)
            cpu_data.copy_(data)
            
            self.data_pool[block.block_id] = cpu_data
            self.blocks[block.block_id] = block
            self.used_bytes += size_bytes
            return True
            
    def deallocate(self, block_id: int):
        """Free CPU memory"""
        with self.lock:
            if block_id in self.data_pool:
                data = self.data_pool[block_id]
                size_bytes = data.numel() * data.element_size()
                del self.data_pool[block_id]
                del self.blocks[block_id]
                self.used_bytes -= size_bytes
                
    def read(self, block_id: int) -> Optional[torch.Tensor]:
        """Read data from CPU"""
        with self.lock:
            return self.data_pool.get(block_id)


class NVMeStorage(TierStorage):
    """NVMe/SSD storage tier with memory mapping"""
    
    def __init__(self, capacity_bytes: int, cache_dir: str):
        super().__init__("nvme", capacity_bytes)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Memory mapped files for fast access
        self.mmap_files: Dict[int, mmap.mmap] = {}
        self.file_metadata: Dict[int, Dict[str, Any]] = {}
        
    def allocate(self, block: CacheBlock, data: torch.Tensor) -> bool:
        """Write block to NVMe storage"""
        with self.lock:
            size_bytes = data.numel() * data.element_size()
            if not self.can_allocate(size_bytes):
                return False
                
            # Create file path
            filename = f"block_{block.block_id}.bin"
            filepath = os.path.join(self.cache_dir, filename)
            
            # Save tensor data
            tensor_bytes = data.cpu().numpy().tobytes()
            metadata = {
                "shape": list(data.shape),
                "dtype": str(data.dtype),
                "size_bytes": len(tensor_bytes)
            }
            
            # Write to file
            with open(filepath, "wb") as f:
                # Write metadata
                pickle.dump(metadata, f)
                metadata_size = f.tell()
                # Write tensor data
                f.write(tensor_bytes)
            
            # Memory map the file
            with open(filepath, "r+b") as f:
                mmapped = mmap.mmap(f.fileno(), 0)
                self.mmap_files[block.block_id] = mmapped
                self.file_metadata[block.block_id] = {
                    **metadata,
                    "metadata_size": metadata_size,
                    "filepath": filepath
                }
            
            self.blocks[block.block_id] = block
            self.used_bytes += size_bytes
            return True
            
    def deallocate(self, block_id: int):
        """Remove block from NVMe storage"""
        with self.lock:
            if block_id in self.mmap_files:
                # Close memory map
                self.mmap_files[block_id].close()
                del self.mmap_files[block_id]
                
                # Remove file
                metadata = self.file_metadata[block_id]
                if os.path.exists(metadata["filepath"]):
                    os.remove(metadata["filepath"])
                    
                self.used_bytes -= metadata["size_bytes"]
                del self.file_metadata[block_id]
                del self.blocks[block_id]
                
    def read(self, block_id: int) -> Optional[torch.Tensor]:
        """Read data from NVMe storage"""
        with self.lock:
            if block_id not in self.mmap_files:
                return None
                
            mmapped = self.mmap_files[block_id]
            metadata = self.file_metadata[block_id]
            
            # Seek past metadata
            mmapped.seek(metadata["metadata_size"])
            
            # Read tensor data
            tensor_bytes = mmapped.read(metadata["size_bytes"])
            
            # Reconstruct tensor
            np_array = np.frombuffer(tensor_bytes, dtype=np.float16)
            np_array = np_array.reshape(metadata["shape"])
            tensor = torch.from_numpy(np_array.copy())
            
            return tensor


class HierarchicalKVCache:
    """
    Hierarchical KV Cache with automatic tier management
    
    Features:
    - Three-tier storage hierarchy (GPU -> CPU -> NVMe)
    - Automatic promotion/demotion based on access patterns
    - Asynchronous transfers between tiers
    - Prefetching for sequential access
    - Adaptive eviction policies
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        
        # Calculate tier capacities
        bytes_per_element = 2  # float16
        elements_per_block = (
            config.block_size * config.num_heads * config.head_dim * 2  # K and V
        )
        bytes_per_block = elements_per_block * bytes_per_element
        
        # Initialize storage tiers
        self.gpu_storage = GPUStorage(
            int(config.gpu_cache_size_gb * 1024**3)
        )
        self.cpu_storage = CPUStorage(
            int(config.cpu_cache_size_gb * 1024**3)
        )
        self.nvme_storage = NVMeStorage(
            int(config.nvme_cache_size_gb * 1024**3),
            config.nvme_cache_dir
        )
        
        # Block tracking
        self.block_registry: Dict[int, CacheBlock] = {}
        self.access_history: deque = deque(maxlen=10000)
        self.next_block_id = 0
        
        # Tier management
        self.promotion_queue: deque = deque()
        self.demotion_queue: deque = deque()
        
        # Async transfer management
        if config.async_transfers:
            self.transfer_executor = ThreadPoolExecutor(
                max_workers=config.num_async_workers
            )
            self.pending_transfers: Dict[int, Any] = {}
        
        # Statistics
        self.stats = {
            "hits": {"gpu": 0, "cpu": 0, "nvme": 0},
            "misses": 0,
            "promotions": 0,
            "demotions": 0,
            "evictions": 0,
        }
        
        # Start background management thread
        self.management_thread = threading.Thread(
            target=self._management_loop, daemon=True
        )
        self.management_thread.start()
    
    def allocate_kv_block(
        self,
        layer_id: int,
        seq_id: int,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> int:
        """Allocate a new KV cache block"""
        # Combine K and V tensors
        kv_data = torch.stack([key, value], dim=0)
        size_bytes = kv_data.numel() * kv_data.element_size()
        
        # Create block metadata
        block = CacheBlock(
            block_id=self.next_block_id,
            layer_id=layer_id,
            seq_id=seq_id,
            position=position,
            size=size_bytes,
            tier="gpu",  # Always start in GPU
            last_access=time.time(),
            access_count=1
        )
        self.next_block_id += 1
        
        # Try to allocate in GPU first
        if self.gpu_storage.allocate(block, kv_data):
            self.block_registry[block.block_id] = block
            return block.block_id
            
        # GPU full - evict and retry
        self._evict_blocks("gpu", size_bytes)
        
        if self.gpu_storage.allocate(block, kv_data):
            self.block_registry[block.block_id] = block
            return block.block_id
            
        raise RuntimeError("Failed to allocate KV cache block")
    
    def read_kv_block(self, block_id: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Read KV block from cache"""
        if block_id not in self.block_registry:
            self.stats["misses"] += 1
            return None
            
        block = self.block_registry[block_id]
        
        # Update access metadata
        block = block._replace(
            last_access=time.time(),
            access_count=block.access_count + 1
        )
        self.block_registry[block_id] = block
        self.access_history.append((block_id, time.time()))
        
        # Read from appropriate tier
        if block.tier == "gpu":
            data = self.gpu_storage.read(block_id)
            self.stats["hits"]["gpu"] += 1
        elif block.tier == "cpu":
            data = self.cpu_storage.read(block_id)
            self.stats["hits"]["cpu"] += 1
            # Schedule promotion if accessed frequently
            if block.access_count >= self.config.promotion_threshold:
                self.promotion_queue.append(block_id)
        elif block.tier == "nvme":
            data = self.nvme_storage.read(block_id)
            self.stats["hits"]["nvme"] += 1
            # Always promote from NVMe
            self.promotion_queue.append(block_id)
        else:
            return None
            
        if data is None:
            return None
            
        # Split K and V
        key, value = data[0], data[1]
        
        # Prefetch next blocks if sequential access
        self._prefetch_sequential(block)
        
        return key, value
    
    def _evict_blocks(self, tier: str, required_bytes: int):
        """Evict blocks from tier to make space"""
        storage = self._get_storage(tier)
        
        # Get eviction candidates
        candidates = self._get_eviction_candidates(tier)
        
        evicted_bytes = 0
        for block_id in candidates:
            if evicted_bytes >= required_bytes:
                break
                
            block = self.block_registry[block_id]
            
            # Move to next tier
            if tier == "gpu":
                self._demote_block(block_id, "cpu")
            elif tier == "cpu":
                self._demote_block(block_id, "nvme")
            else:
                # Evict from NVMe (remove completely)
                self._evict_block(block_id)
                
            evicted_bytes += block.size
            self.stats["evictions"] += 1
    
    def _get_eviction_candidates(self, tier: str) -> List[int]:
        """Get blocks to evict based on policy"""
        storage = self._get_storage(tier)
        blocks = list(storage.blocks.values())
        
        if self.config.eviction_policy == "lru":
            # Least recently used
            blocks.sort(key=lambda b: b.last_access)
        elif self.config.eviction_policy == "lfu":
            # Least frequently used
            blocks.sort(key=lambda b: b.access_count)
        elif self.config.eviction_policy == "adaptive":
            # Adaptive policy based on access patterns
            blocks.sort(key=lambda b: b.access_count / (time.time() - b.last_access + 1))
            
        return [b.block_id for b in blocks]
    
    def _promote_block(self, block_id: int, target_tier: str):
        """Promote block to higher tier"""
        if block_id not in self.block_registry:
            return
            
        block = self.block_registry[block_id]
        current_storage = self._get_storage(block.tier)
        target_storage = self._get_storage(target_tier)
        
        # Read data from current tier
        data = current_storage.read(block_id)
        if data is None:
            return
            
        # Allocate in target tier
        new_block = block._replace(tier=target_tier)
        if target_storage.allocate(new_block, data):
            # Deallocate from current tier
            current_storage.deallocate(block_id)
            self.block_registry[block_id] = new_block
            self.stats["promotions"] += 1
    
    def _demote_block(self, block_id: int, target_tier: str):
        """Demote block to lower tier"""
        self._promote_block(block_id, target_tier)  # Same operation
        self.stats["demotions"] += 1
    
    def _evict_block(self, block_id: int):
        """Completely evict block from cache"""
        if block_id not in self.block_registry:
            return
            
        block = self.block_registry[block_id]
        storage = self._get_storage(block.tier)
        storage.deallocate(block_id)
        del self.block_registry[block_id]
    
    def _prefetch_sequential(self, block: CacheBlock):
        """Prefetch sequential blocks"""
        # Simple sequential prefetching
        for i in range(1, self.config.prefetch_distance + 1):
            next_position = block.position + i * self.config.block_size
            # Check if next block exists and is not in GPU
            for block_id, b in self.block_registry.items():
                if (b.layer_id == block.layer_id and 
                    b.seq_id == block.seq_id and
                    b.position == next_position and
                    b.tier != "gpu"):
                    self.promotion_queue.append(block_id)
                    break
    
    def _get_storage(self, tier: str) -> TierStorage:
        """Get storage object for tier"""
        if tier == "gpu":
            return self.gpu_storage
        elif tier == "cpu":
            return self.cpu_storage
        elif tier == "nvme":
            return self.nvme_storage
        else:
            raise ValueError(f"Unknown tier: {tier}")
    
    def _management_loop(self):
        """Background thread for tier management"""
        while True:
            try:
                # Process promotions
                while self.promotion_queue:
                    block_id = self.promotion_queue.popleft()
                    if block_id in self.block_registry:
                        block = self.block_registry[block_id]
                        if block.tier == "nvme":
                            self._promote_block(block_id, "cpu")
                        elif block.tier == "cpu":
                            self._promote_block(block_id, "gpu")
                
                # Check memory pressure
                gpu_usage = self.gpu_storage.used_bytes / self.gpu_storage.capacity_bytes
                cpu_usage = self.cpu_storage.used_bytes / self.cpu_storage.capacity_bytes
                
                # Demote blocks if needed
                if gpu_usage > self.config.gpu_memory_threshold:
                    candidates = self._get_eviction_candidates("gpu")[:10]
                    for block_id in candidates:
                        self._demote_block(block_id, "cpu")
                        
                if cpu_usage > self.config.cpu_memory_threshold:
                    candidates = self._get_eviction_candidates("cpu")[:10]
                    for block_id in candidates:
                        self._demote_block(block_id, "nvme")
                
                time.sleep(0.1)  # 100ms interval
                
            except Exception as e:
                logger.error(f"Error in cache management: {e}")
                time.sleep(1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(self.stats["hits"].values())
        total_accesses = total_hits + self.stats["misses"]
        
        return {
            **self.stats,
            "hit_rate": total_hits / max(total_accesses, 1),
            "tier_distribution": {
                "gpu": len(self.gpu_storage.blocks),
                "cpu": len(self.cpu_storage.blocks),
                "nvme": len(self.nvme_storage.blocks),
            },
            "memory_usage": {
                "gpu": f"{self.gpu_storage.used_bytes / 1024**3:.2f}GB",
                "cpu": f"{self.cpu_storage.used_bytes / 1024**3:.2f}GB",
                "nvme": f"{self.nvme_storage.used_bytes / 1024**3:.2f}GB",
            }
        }
    
    def clear_cache(self):
        """Clear all cache data"""
        # Clear all tiers
        for block_id in list(self.block_registry.keys()):
            self._evict_block(block_id)
            
        self.block_registry.clear()
        self.access_history.clear()
        self.promotion_queue.clear()
        self.demotion_queue.clear()
        
        # Reset stats
        self.stats = {
            "hits": {"gpu": 0, "cpu": 0, "nvme": 0},
            "misses": 0,
            "promotions": 0,
            "demotions": 0,
            "evictions": 0,
        }