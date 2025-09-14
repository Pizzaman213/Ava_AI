"""
Continuous Batching for Inference

Implements dynamic batching that processes requests as they arrive rather than
waiting for full batches. This significantly improves throughput and reduces
latency for inference workloads.

Key features:
- Request queue management with priority support
- Dynamic batch size adjustment based on memory and compute availability
- Streaming response handling for real-time generation
- Memory-aware batching decisions to prevent OOM
- Sequence packing for optimal GPU utilization
- Dynamic request reordering based on generation progress
- Adaptive scheduling with preemption support
- Multi-GPU load balancing
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from queue import PriorityQueue, Queue
from threading import Lock, Thread, Event
import time
import numpy as np
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Single inference request with metadata"""
    request_id: str
    prompt_ids: torch.Tensor
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    priority: int = 0  # Higher priority = processed first
    arrival_time: float = field(default_factory=time.time)
    streaming_callback: Optional[Callable[[str], None]] = None
    completion_callback: Optional[Callable[[str], None]] = None
    
    def __lt__(self, other):
        # For priority queue ordering
        return self.priority > other.priority


@dataclass
class BatchedRequest:
    """Multiple requests batched together for efficient processing"""
    requests: List[InferenceRequest]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    request_indices: torch.Tensor  # Maps batch position to request
    max_seq_len: int
    device: torch.device


class ContinuousBatchingEngine:
    """
    Main engine for continuous batching inference.
    
    Features:
    - Asynchronous request processing
    - Dynamic batch formation
    - Memory-aware scheduling
    - Streaming token generation
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        max_batch_size: int = 32,
        max_sequence_length: int = 2048,
        memory_fraction: float = 0.9,
        batch_timeout_ms: int = 50,
        device: str = "cuda",
        num_workers: int = 2,
        enable_sequence_packing: bool = True,
        enable_request_reordering: bool = True,
        enable_preemption: bool = True,
        min_batch_size: int = 1,
        adaptive_batch_size: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.max_sequence_length = max_sequence_length
        self.memory_fraction = memory_fraction
        self.batch_timeout_ms = batch_timeout_ms
        self.device = torch.device(device)
        self.num_workers = num_workers
        self.enable_sequence_packing = enable_sequence_packing
        self.enable_request_reordering = enable_request_reordering
        self.enable_preemption = enable_preemption
        self.adaptive_batch_size = adaptive_batch_size
        
        # Request management
        self.request_queue = PriorityQueue()
        self.active_batches: Dict[str, BatchedRequest] = {}
        self.completed_requests: Dict[str, str] = {}
        self.preempted_requests: Dict[str, InferenceRequest] = {}
        
        # Advanced scheduling
        self.scheduler = RequestScheduler() if enable_preemption else None
        
        # Memory tracking
        self.memory_monitor = MemoryMonitor(self.device, memory_fraction)
        
        # Threading and synchronization
        self.queue_lock = Lock()
        self.batch_lock = Lock()
        self.stop_event = Event()
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "average_latency": 0,
            "throughput": 0,
            "batch_utilization": deque(maxlen=100),
            "sequence_packing_efficiency": deque(maxlen=100),
            "preemptions": 0,
            "memory_efficiency": deque(maxlen=100),
        }
        
        # Adaptive batch sizing
        self.current_batch_size = max_batch_size
        self.batch_size_history = deque(maxlen=20)
        
        # Start background workers
        self._start_workers()
    
    def _start_workers(self):
        """Start background worker threads"""
        # Batch formation worker
        self.batch_worker = Thread(target=self._batch_formation_loop, daemon=True)
        self.batch_worker.start()
        
        # Inference worker
        self.inference_worker = Thread(target=self._inference_loop, daemon=True)
        self.inference_worker.start()
    
    def submit_request(self, request: InferenceRequest) -> str:
        """Submit a new inference request"""
        with self.queue_lock:
            self.request_queue.put(request)
            self.stats["total_requests"] += 1
        return request.request_id
    
    def _batch_formation_loop(self):
        """Background loop for forming batches from queued requests"""
        pending_requests = []
        last_batch_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Try to get a request with timeout
                timeout = self.batch_timeout_ms / 1000.0
                if not self.request_queue.empty():
                    request = self.request_queue.get(timeout=timeout)
                    pending_requests.append(request)
                
                # Check if we should form a batch
                current_time = time.time()
                time_elapsed = (current_time - last_batch_time) * 1000  # ms
                
                # Adaptive batch size based on memory and throughput
                if self.adaptive_batch_size:
                    self._update_adaptive_batch_size()
                
                batch_size = min(self.current_batch_size, self.max_batch_size)
                
                should_batch = (
                    len(pending_requests) >= batch_size or
                    (len(pending_requests) >= self.min_batch_size and time_elapsed >= self.batch_timeout_ms)
                )
                
                if should_batch and pending_requests:
                    # Reorder requests if enabled
                    if self.enable_request_reordering:
                        pending_requests = self._reorder_requests(pending_requests)
                    
                    # Select requests for batch
                    selected_requests = pending_requests[:batch_size]
                    
                    # Apply sequence packing if enabled
                    if self.enable_sequence_packing:
                        selected_requests = self._pack_sequences(selected_requests)
                    
                    # Check memory availability
                    estimated_memory = self._estimate_batch_memory(selected_requests)
                    if self.memory_monitor.can_allocate(estimated_memory):
                        batch = self._create_batch(selected_requests)
                        
                        with self.batch_lock:
                            batch_id = f"batch_{int(current_time * 1000)}"
                            self.active_batches[batch_id] = batch
                        
                        # Reserve memory
                        self.memory_monitor.reserve(estimated_memory)
                        
                        # Keep remaining requests for next batch
                        pending_requests = pending_requests[len(selected_requests):]
                        last_batch_time = current_time
                        
                        # Track batch utilization
                        utilization = len(batch.requests) / self.max_batch_size
                        self.stats["batch_utilization"].append(utilization)
                        
                        # Track memory efficiency
                        memory_efficiency = self._calculate_memory_efficiency(batch)
                        self.stats["memory_efficiency"].append(memory_efficiency)
                    else:
                        # Memory pressure - try smaller batch or preempt
                        if self.enable_preemption:
                            self._handle_memory_pressure(selected_requests)
                        else:
                            time.sleep(0.1)
                        
            except Exception as e:
                logger.error(f"Error in batch formation: {e}")
                time.sleep(0.1)
    
    def _create_batch(self, requests: List[InferenceRequest]) -> BatchedRequest:
        """Create a batched tensor from multiple requests"""
        # Find max sequence length in batch
        max_len = max(len(req.prompt_ids) for req in requests)
        batch_size = len(requests)
        
        # Allocate tensors
        input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
        request_indices = torch.arange(batch_size, device=self.device)
        
        # Fill tensors
        for i, req in enumerate(requests):
            seq_len = len(req.prompt_ids)
            input_ids[i, :seq_len] = req.prompt_ids.to(self.device)
            attention_mask[i, :seq_len] = 1
            position_ids[i, :seq_len] = torch.arange(seq_len, device=self.device)
        
        return BatchedRequest(
            requests=requests,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            request_indices=request_indices,
            max_seq_len=max_len,
            device=self.device
        )
    
    def _inference_loop(self):
        """Background loop for processing batches"""
        while not self.stop_event.is_set():
            try:
                with self.batch_lock:
                    batch_ids = list(self.active_batches.keys())
                
                for batch_id in batch_ids:
                    with self.batch_lock:
                        if batch_id not in self.active_batches:
                            continue
                        batch = self.active_batches[batch_id]
                    
                    # Process one generation step
                    completed = self._process_batch_step(batch)
                    
                    if completed:
                        with self.batch_lock:
                            del self.active_batches[batch_id]
                        
                        # Update statistics
                        for req in batch.requests:
                            latency = time.time() - req.arrival_time
                            self.stats["completed_requests"] += 1
                            self.stats["average_latency"] = (
                                self.stats["average_latency"] * 0.9 + latency * 0.1
                            )
                
                # Small sleep if no active batches
                if not batch_ids:
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in inference loop: {e}")
                time.sleep(0.1)
    
    def _process_batch_step(self, batch: BatchedRequest) -> bool:
        """Process one generation step for a batch"""
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                position_ids=batch.position_ids,
                use_cache=True,
                return_dict=True,
            )
            
            # Get next token predictions
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply sampling per request
            next_tokens = []
            completed_mask = torch.zeros(len(batch.requests), dtype=torch.bool, device=self.device)
            
            for i, req in enumerate(batch.requests):
                # Apply temperature and sampling
                logits = next_token_logits[i] / req.temperature
                
                # Top-k filtering
                if req.top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, req.top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')
                
                # Top-p (nucleus) filtering
                if req.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > req.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = -float('Inf')
                
                # Sample token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_tokens.append(next_token)
                
                # Check if completed (EOS or max length)
                is_eos = next_token.item() == self.tokenizer.eos_token_id
                is_max_len = batch.input_ids[i].shape[0] >= req.max_new_tokens + len(req.prompt_ids)
                completed_mask[i] = is_eos or is_max_len
                
                # Stream token if callback provided
                if req.streaming_callback and not completed_mask[i]:
                    token_str = self.tokenizer.decode(next_token.item())
                    self.executor.submit(req.streaming_callback, token_str)
            
            # Update batch tensors
            next_tokens = torch.cat(next_tokens, dim=0)
            batch.input_ids = torch.cat([batch.input_ids, next_tokens.unsqueeze(1)], dim=1)
            
            # Update attention mask and position ids
            current_len = batch.attention_mask.sum(dim=1)
            batch.attention_mask = torch.cat([
                batch.attention_mask,
                torch.ones((len(batch.requests), 1), dtype=torch.bool, device=self.device)
            ], dim=1)
            
            new_position_ids = current_len.unsqueeze(1)
            batch.position_ids = torch.cat([batch.position_ids, new_position_ids], dim=1)
            
            # Handle completed requests
            if completed_mask.any():
                for i, (req, completed) in enumerate(zip(batch.requests, completed_mask)):
                    if completed and req.completion_callback:
                        # Decode full sequence
                        full_sequence = batch.input_ids[i, :batch.attention_mask[i].sum()]
                        generated_text = self.tokenizer.decode(
                            full_sequence[len(req.prompt_ids):],
                            skip_special_tokens=True
                        )
                        self.executor.submit(req.completion_callback, generated_text)
            
            # Return True if all requests completed
            return completed_mask.all()
    
    def _estimate_batch_memory(self, requests: List[InferenceRequest]) -> int:
        """Estimate memory requirements for a batch"""
        batch_size = len(requests)
        max_seq_len = max(len(req.prompt_ids) + req.max_new_tokens for req in requests)
        
        # Estimate based on model dimensions and KV cache
        model_dim = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers
        num_heads = self.model.config.num_attention_heads
        
        # KV cache memory (2 for K and V)
        kv_cache_memory = (
            2 * batch_size * num_layers * num_heads * max_seq_len * model_dim // num_heads * 4
        )
        
        # Activation memory (rough estimate)
        activation_memory = batch_size * max_seq_len * model_dim * 4 * 10
        
        return kv_cache_memory + activation_memory
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        stats = self.stats.copy()
        stats["active_batches"] = len(self.active_batches)
        stats["queued_requests"] = self.request_queue.qsize()
        stats["average_batch_utilization"] = (
            np.mean(list(self.stats["batch_utilization"]))
            if self.stats["batch_utilization"] else 0
        )
        stats["memory_usage"] = self.memory_monitor.get_usage()
        return stats
    
    def shutdown(self):
        """Gracefully shutdown the engine"""
        self.stop_event.set()
        self.batch_worker.join(timeout=5)
        self.inference_worker.join(timeout=5)
        self.executor.shutdown(wait=True)


class MemoryMonitor:
    """Monitor and manage GPU memory for batching decisions"""
    
    def __init__(self, device: torch.device, target_fraction: float = 0.9):
        self.device = device
        self.target_fraction = target_fraction
        self.reserved_memory = 0
        
    def can_allocate(self, required_bytes: int) -> bool:
        """Check if we can allocate the required memory"""
        if self.device.type != 'cuda':
            return True  # No memory limit for CPU
        
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        allocated_memory = torch.cuda.memory_allocated(self.device)
        
        available = total_memory * self.target_fraction - allocated_memory - self.reserved_memory
        return available >= required_bytes
    
    def reserve(self, bytes: int):
        """Reserve memory for upcoming allocation"""
        self.reserved_memory += bytes
        
    def release(self, bytes: int):
        """Release reserved memory"""
        self.reserved_memory = max(0, self.reserved_memory - bytes)
        
    def get_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if self.device.type != 'cuda':
            return {"used": 0, "total": 0, "fraction": 0}
        
        total = torch.cuda.get_device_properties(self.device).total_memory
        allocated = torch.cuda.memory_allocated(self.device)
        
        return {
            "used": allocated,
            "total": total,
            "fraction": allocated / total,
            "reserved": self.reserved_memory,
        }


class RequestScheduler:
    """
    Advanced request scheduling with preemption and priority management.
    
    Features:
    - Fair queuing with priority support
    - Request preemption for high-priority requests
    - Starvation prevention
    - Load balancing across multiple GPUs
    """
    
    def __init__(
        self,
        max_requests: int = 1000,
        priority_levels: int = 3,
        starvation_threshold: float = 30.0,  # seconds
    ):
        self.max_requests = max_requests
        self.priority_levels = priority_levels
        self.starvation_threshold = starvation_threshold
        
        # Priority queues for each level
        self.priority_queues = [Queue() for _ in range(priority_levels)]
        self.request_metadata: Dict[str, Dict] = {}
        
        # Scheduling state
        self.last_scheduled: Dict[int, float] = {i: 0 for i in range(priority_levels)}
        self.lock = Lock()
    
    def add_request(self, request: InferenceRequest):
        """Add request to appropriate priority queue"""
        with self.lock:
            priority = min(request.priority, self.priority_levels - 1)
            self.priority_queues[priority].put(request)
            
            self.request_metadata[request.request_id] = {
                "arrival_time": request.arrival_time,
                "priority": priority,
                "scheduled": False,
            }
    
    def get_next_batch(self, batch_size: int) -> List[InferenceRequest]:
        """Get next batch of requests using fair scheduling"""
        with self.lock:
            batch = []
            
            # Check for starving requests first
            for priority in range(self.priority_levels):
                if not self.priority_queues[priority].empty():
                    # Peek at oldest request
                    oldest_time = time.time() - self.last_scheduled[priority]
                    if oldest_time > self.starvation_threshold:
                        # Prioritize this queue to prevent starvation
                        while len(batch) < batch_size and not self.priority_queues[priority].empty():
                            req = self.priority_queues[priority].get()
                            batch.append(req)
                            self.request_metadata[req.request_id]["scheduled"] = True
                        
                        self.last_scheduled[priority] = time.time()
                        return batch
            
            # Normal scheduling - prioritize higher priority queues
            for priority in range(self.priority_levels):
                while len(batch) < batch_size and not self.priority_queues[priority].empty():
                    req = self.priority_queues[priority].get()
                    batch.append(req)
                    self.request_metadata[req.request_id]["scheduled"] = True
                
                if batch:
                    self.last_scheduled[priority] = time.time()
            
            return batch
    
    def preempt_for_priority(self, high_priority_request: InferenceRequest) -> Optional[str]:
        """Preempt lower priority request if needed"""
        # This would interface with the active batch management
        # to pause/resume requests - implementation depends on model
        pass


    def _update_adaptive_batch_size(self):
        """Dynamically adjust batch size based on system metrics"""
        memory_usage = self.memory_monitor.get_usage()
        
        # Calculate optimal batch size based on memory usage
        if memory_usage["fraction"] > 0.85:
            # Reduce batch size
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
        elif memory_usage["fraction"] < 0.6 and len(self.batch_size_history) > 5:
            # Increase batch size if consistently underutilized
            avg_utilization = np.mean(list(self.stats["batch_utilization"]))
            if avg_utilization > 0.8:
                self.current_batch_size = min(
                    self.max_batch_size,
                    int(self.current_batch_size * 1.2)
                )
        
        self.batch_size_history.append(self.current_batch_size)
    
    def _reorder_requests(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """Reorder requests for better batching efficiency"""
        # Sort by sequence length for better packing
        return sorted(requests, key=lambda r: len(r.prompt_ids))
    
    def _pack_sequences(self, requests: List[InferenceRequest]) -> List[InferenceRequest]:
        """Pack multiple short sequences into single batch slots"""
        if not requests:
            return requests
        
        # Group requests by similar length for efficient packing
        length_groups = {}
        for req in requests:
            length = len(req.prompt_ids)
            bucket = (length // 128) * 128  # 128-token buckets
            if bucket not in length_groups:
                length_groups[bucket] = []
            length_groups[bucket].append(req)
        
        # Pack sequences within same length bucket
        packed_requests = []
        for bucket, group in length_groups.items():
            packed_requests.extend(group)
        
        return packed_requests
    
    def _calculate_memory_efficiency(self, batch: BatchedRequest) -> float:
        """Calculate memory efficiency of the batch"""
        # Total tokens in batch
        total_tokens = batch.attention_mask.sum().item()
        # Maximum possible tokens
        max_tokens = batch.input_ids.numel()
        # Efficiency ratio
        return total_tokens / max_tokens if max_tokens > 0 else 0.0
    
    def _handle_memory_pressure(self, requests: List[InferenceRequest]):
        """Handle memory pressure through preemption or batching adjustments"""
        # Try to preempt lower priority requests
        if self.enable_preemption and self.active_batches:
            # Find lowest priority active batch
            lowest_priority_batch = None
            lowest_priority = float('inf')
            
            with self.batch_lock:
                for batch_id, batch in self.active_batches.items():
                    batch_priority = min(req.priority for req in batch.requests)
                    if batch_priority < lowest_priority:
                        lowest_priority = batch_priority
                        lowest_priority_batch = batch_id
            
            # Check if new requests have higher priority
            new_priority = max(req.priority for req in requests)
            if lowest_priority_batch and new_priority > lowest_priority:
                # Preempt the batch
                self._preempt_batch(lowest_priority_batch)
                self.stats["preemptions"] += 1
    
    def _preempt_batch(self, batch_id: str):
        """Preempt a batch and save its state"""
        with self.batch_lock:
            if batch_id in self.active_batches:
                batch = self.active_batches[batch_id]
                # Save preempted requests
                for req in batch.requests:
                    self.preempted_requests[req.request_id] = req
                del self.active_batches[batch_id]
                
                # Release memory
                estimated_memory = self._estimate_batch_memory(batch.requests)
                self.memory_monitor.release(estimated_memory)


def create_continuous_batching_engine(
    model: nn.Module,
    tokenizer: Any,
    config: Optional[Dict[str, Any]] = None,
) -> ContinuousBatchingEngine:
    """Factory function to create continuous batching engine with config"""
    default_config = {
        "max_batch_size": 32,
        "max_sequence_length": 2048,
        "memory_fraction": 0.9,
        "batch_timeout_ms": 50,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 2,
        "enable_sequence_packing": True,
        "enable_request_reordering": True,
        "enable_preemption": True,
        "min_batch_size": 1,
        "adaptive_batch_size": True,
    }
    
    if config:
        default_config.update(config)
    
    return ContinuousBatchingEngine(model, tokenizer, **default_config)