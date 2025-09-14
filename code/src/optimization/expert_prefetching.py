"""
Expert Prefetching and Caching System

Implements predictive expert loading to minimize latency
by prefetching experts that are likely to be used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set, Deque
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, deque
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrefetchConfig:
    """Configuration for expert prefetching"""
    # Prediction model
    use_lstm_predictor: bool = True
    predictor_hidden_size: int = 256
    prediction_horizon: int = 5  # Predict next N tokens
    
    # Caching
    cache_size: int = 32  # Number of experts to keep in fast memory
    cache_device: str = "cuda"
    offload_device: str = "cpu"
    
    # Prefetching
    prefetch_queue_size: int = 8
    num_prefetch_threads: int = 2
    prefetch_threshold: float = 0.1  # Min probability to prefetch
    
    # Usage tracking
    track_usage_history: bool = True
    history_window: int = 1000
    
    # Adaptive settings
    adaptive_threshold: bool = True
    min_hit_rate: float = 0.7


class ExpertUsagePredictor(nn.Module):
    """
    LSTM-based model to predict which experts will be needed
    """
    
    def __init__(self, num_experts: int, config: PrefetchConfig):
        super().__init__()
        self.num_experts = num_experts
        self.config = config
        
        # Input: recent expert usage pattern
        self.embedding = nn.Embedding(num_experts, config.predictor_hidden_size)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=config.predictor_hidden_size,
            hidden_size=config.predictor_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Output: probability distribution over experts
        self.output_proj = nn.Linear(
            config.predictor_hidden_size,
            num_experts * config.prediction_horizon
        )
        
        # Attention mechanism for importance weighting
        self.attention = nn.MultiheadAttention(
            embed_dim=config.predictor_hidden_size,
            num_heads=4,
            batch_first=True
        )
        
    def forward(
        self,
        recent_experts: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict future expert usage
        
        Args:
            recent_experts: [batch_size, sequence_length] - recent expert IDs
            hidden_state: Optional LSTM hidden state
            
        Returns:
            predictions: [batch_size, prediction_horizon, num_experts] - probabilities
            hidden_state: Updated LSTM state
        """
        # Embed expert IDs
        embedded = self.embedding(recent_experts)
        
        # Apply self-attention
        attended, _ = self.attention(embedded, embedded, embedded)
        
        # LSTM forward
        lstm_out, hidden_state = self.lstm(attended, hidden_state)
        
        # Project to expert probabilities
        logits = self.output_proj(lstm_out[:, -1])  # Use last timestep
        logits = logits.view(-1, self.config.prediction_horizon, self.num_experts)
        
        # Convert to probabilities
        predictions = F.softmax(logits, dim=-1)
        
        return predictions, hidden_state


class ExpertCache:
    """
    Manages expert caching with LRU eviction
    """
    
    def __init__(self, cache_size: int, cache_device: str, offload_device: str):
        self.cache_size = cache_size
        self.cache_device = cache_device
        self.offload_device = offload_device
        
        # Cache storage
        self.cached_experts: Dict[int, nn.Module] = {}
        self.access_times: Dict[int, float] = {}
        self.access_counts: Dict[int, int] = defaultdict(int)
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
    def get(self, expert_id: int) -> Optional[nn.Module]:
        """Get expert from cache"""
        with self.lock:
            if expert_id in self.cached_experts:
                # Update access time and count
                self.access_times[expert_id] = time.time()
                self.access_counts[expert_id] += 1
                return self.cached_experts[expert_id]
        return None
        
    def put(self, expert_id: int, expert: nn.Module) -> None:
        """Put expert in cache, evicting if necessary"""
        with self.lock:
            # Check if eviction is needed
            if len(self.cached_experts) >= self.cache_size and expert_id not in self.cached_experts:
                self._evict_lru()
                
            # Move expert to cache device
            if str(expert.device) != self.cache_device:
                expert = expert.to(self.cache_device)
                
            # Store in cache
            self.cached_experts[expert_id] = expert
            self.access_times[expert_id] = time.time()
            self.access_counts[expert_id] += 1
            
    def _evict_lru(self) -> int:
        """Evict least recently used expert"""
        # Find LRU expert
        lru_id = min(self.access_times.keys(), key=self.access_times.get)
        
        # Move to offload device
        expert = self.cached_experts[lru_id]
        expert.to(self.offload_device)
        
        # Remove from cache
        del self.cached_experts[lru_id]
        del self.access_times[lru_id]
        
        logger.debug(f"Evicted expert {lru_id} from cache")
        return lru_id
        
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        with self.lock:
            total_accesses = sum(self.access_counts.values())
            cache_hits = sum(
                count for expert_id, count in self.access_counts.items()
                if expert_id in self.cached_experts
            )
            
            return {
                "cache_size": len(self.cached_experts),
                "total_accesses": total_accesses,
                "hit_rate": cache_hits / max(total_accesses, 1),
                "avg_access_count": np.mean(list(self.access_counts.values())) if self.access_counts else 0
            }


class PrefetchQueue:
    """
    Manages asynchronous prefetching queue
    """
    
    def __init__(self, max_size: int, num_workers: int):
        self.max_size = max_size
        self.queue: Deque[Tuple[int, float]] = deque(maxlen=max_size)
        self.in_progress: Set[int] = set()
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.lock = threading.Lock()
        
    def add(self, expert_id: int, priority: float) -> bool:
        """Add expert to prefetch queue"""
        with self.lock:
            if expert_id in self.in_progress:
                return False
                
            # Add to queue with priority
            self.queue.append((expert_id, priority))
            
            # Sort by priority
            self.queue = deque(sorted(self.queue, key=lambda x: x[1], reverse=True))
            
            # Trim to max size
            while len(self.queue) > self.max_size:
                self.queue.pop()
                
            return True
            
    def get_next(self) -> Optional[int]:
        """Get next expert to prefetch"""
        with self.lock:
            if self.queue:
                expert_id, _ = self.queue.popleft()
                self.in_progress.add(expert_id)
                return expert_id
        return None
        
    def mark_done(self, expert_id: int) -> None:
        """Mark expert as prefetched"""
        with self.lock:
            self.in_progress.discard(expert_id)
            
    def shutdown(self) -> None:
        """Shutdown prefetch workers"""
        self.executor.shutdown(wait=True)


class ExpertPrefetcher:
    """
    Main expert prefetching system
    """
    
    def __init__(
        self,
        experts: List[nn.Module],
        config: PrefetchConfig
    ):
        self.experts = experts
        self.num_experts = len(experts)
        self.config = config
        
        # Initialize components
        self.cache = ExpertCache(
            config.cache_size,
            config.cache_device,
            config.offload_device
        )
        
        self.predictor = ExpertUsagePredictor(self.num_experts, config)
        self.prefetch_queue = PrefetchQueue(
            config.prefetch_queue_size,
            config.num_prefetch_threads
        )
        
        # Usage tracking
        self.usage_history: Deque[int] = deque(maxlen=config.history_window)
        self.prediction_accuracy: Deque[float] = deque(maxlen=100)
        
        # Start prefetch workers
        self._start_prefetch_workers()
        
    def _start_prefetch_workers(self):
        """Start background prefetching threads"""
        def worker():
            while True:
                expert_id = self.prefetch_queue.get_next()
                if expert_id is None:
                    time.sleep(0.001)  # Brief sleep if queue is empty
                    continue
                    
                try:
                    self._prefetch_expert(expert_id)
                    self.prefetch_queue.mark_done(expert_id)
                except Exception as e:
                    logger.error(f"Prefetch error for expert {expert_id}: {e}")
                    self.prefetch_queue.mark_done(expert_id)
                    
        # Start worker threads
        for _ in range(self.config.num_prefetch_threads):
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            
    def _prefetch_expert(self, expert_id: int):
        """Prefetch a specific expert"""
        # Check if already cached
        if self.cache.get(expert_id) is not None:
            return
            
        # Load expert and put in cache
        expert = self.experts[expert_id]
        self.cache.put(expert_id, expert)
        logger.debug(f"Prefetched expert {expert_id}")
        
    def get_expert(self, expert_id: int) -> nn.Module:
        """Get expert, using cache if available"""
        # Try cache first
        expert = self.cache.get(expert_id)
        if expert is not None:
            return expert
            
        # Cache miss - load directly
        logger.debug(f"Cache miss for expert {expert_id}")
        expert = self.experts[expert_id]
        
        # Add to cache for future use
        self.cache.put(expert_id, expert)
        
        # Record usage
        if self.config.track_usage_history:
            self.usage_history.append(expert_id)
            
        return expert
        
    def predict_and_prefetch(self, current_experts: List[int]) -> None:
        """Predict future expert usage and prefetch"""
        if not self.config.use_lstm_predictor:
            # Simple frequency-based prefetching
            self._frequency_based_prefetch()
            return
            
        # Convert to tensor
        recent_experts = torch.tensor(
            list(self.usage_history)[-50:],  # Last 50 experts
            dtype=torch.long
        ).unsqueeze(0)
        
        # Predict future usage
        with torch.no_grad():
            predictions, _ = self.predictor(recent_experts)
            
        # Get top experts to prefetch
        for t in range(self.config.prediction_horizon):
            probs = predictions[0, t]
            
            # Get experts above threshold
            expert_indices = torch.where(probs > self.config.prefetch_threshold)[0]
            
            # Add to prefetch queue
            for idx in expert_indices:
                self.prefetch_queue.add(idx.item(), probs[idx].item())
                
    def _frequency_based_prefetch(self):
        """Simple frequency-based prefetching"""
        # Count recent usage
        recent_usage = defaultdict(int)
        for expert_id in list(self.usage_history)[-100:]:
            recent_usage[expert_id] += 1
            
        # Prefetch most frequently used
        sorted_experts = sorted(
            recent_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for expert_id, count in sorted_experts[:self.config.prefetch_queue_size]:
            probability = count / 100.0
            if probability > self.config.prefetch_threshold:
                self.prefetch_queue.add(expert_id, probability)
                
    def update_predictor(self, actual_experts: List[int]):
        """Update predictor with actual usage for training"""
        if not self.usage_history:
            return
            
        # Prepare training data
        recent = list(self.usage_history)[-50:-1]
        if len(recent) < 10:
            return
            
        recent_tensor = torch.tensor(recent, dtype=torch.long).unsqueeze(0)
        actual_tensor = torch.tensor(actual_experts, dtype=torch.long)
        
        # Forward pass
        predictions, _ = self.predictor(recent_tensor)
        
        # Compute loss (cross entropy)
        loss = 0
        for t, expert_id in enumerate(actual_experts[:self.config.prediction_horizon]):
            if t < predictions.size(1):
                loss += F.cross_entropy(
                    predictions[0, t].unsqueeze(0),
                    actual_tensor[t:t+1]
                )
                
        # Backward pass (if training)
        if self.predictor.training:
            loss.backward()
            
        # Track accuracy
        with torch.no_grad():
            predicted_experts = predictions[0].argmax(dim=-1)
            accuracy = (predicted_experts == actual_tensor[:predictions.size(1)]).float().mean()
            self.prediction_accuracy.append(accuracy.item())
            
    def adapt_threshold(self):
        """Adapt prefetch threshold based on hit rate"""
        if not self.config.adaptive_threshold:
            return
            
        stats = self.cache.get_stats()
        hit_rate = stats["hit_rate"]
        
        if hit_rate < self.config.min_hit_rate:
            # Lower threshold to prefetch more
            self.config.prefetch_threshold *= 0.9
            logger.info(f"Lowered prefetch threshold to {self.config.prefetch_threshold:.3f}")
        elif hit_rate > self.config.min_hit_rate + 0.1:
            # Raise threshold to prefetch less
            self.config.prefetch_threshold *= 1.1
            logger.info(f"Raised prefetch threshold to {self.config.prefetch_threshold:.3f}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get prefetching statistics"""
        cache_stats = self.cache.get_stats()
        
        return {
            **cache_stats,
            "prediction_accuracy": np.mean(self.prediction_accuracy) if self.prediction_accuracy else 0,
            "prefetch_queue_size": len(self.prefetch_queue.queue),
            "prefetch_threshold": self.config.prefetch_threshold
        }
        
    def shutdown(self):
        """Shutdown prefetching system"""
        self.prefetch_queue.shutdown()


class AsyncExpertLoader:
    """
    Asynchronous expert loading with futures
    """
    
    def __init__(self, experts: List[nn.Module], num_workers: int = 4):
        self.experts = experts
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures = {}
        
    def load_async(self, expert_id: int, device: str) -> asyncio.Future:
        """Start loading expert asynchronously"""
        if expert_id in self.futures:
            return self.futures[expert_id]
            
        future = self.executor.submit(self._load_expert, expert_id, device)
        self.futures[expert_id] = future
        return future
        
    def _load_expert(self, expert_id: int, device: str) -> nn.Module:
        """Load expert to specified device"""
        expert = self.experts[expert_id]
        return expert.to(device)
        
    def wait_for_expert(self, expert_id: int) -> nn.Module:
        """Wait for expert to finish loading"""
        if expert_id in self.futures:
            expert = self.futures[expert_id].result()
            del self.futures[expert_id]
            return expert
        else:
            # Not loading, return directly
            return self.experts[expert_id]
            
    def shutdown(self):
        """Shutdown async loader"""
        self.executor.shutdown(wait=True)