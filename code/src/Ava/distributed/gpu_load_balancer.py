"""
GPU Load Balancer for MoE Expert Distribution

This module provides intelligent GPU load balancing for MoE models:
- Dynamic expert placement across GPUs
- Real-time GPU utilization monitoring
- Load-aware expert routing
- Automatic rebalancing based on workload
- Multi-GPU memory-aware scheduling

Features:
- Track per-GPU memory, compute, and expert usage
- Dynamically migrate experts between GPUs
- Balance workload across available GPUs
- Minimize cross-GPU communication
- Support for heterogeneous GPU configurations
"""

import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GPUStats:
    """Statistics for a single GPU."""
    gpu_id: int
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_utilization: float = 0.0
    compute_utilization: float = 0.0
    num_experts: int = 0
    expert_ids: Set[int] = field(default_factory=set)
    tokens_processed: int = 0
    last_update: float = field(default_factory=time.time)

    # NEW: Token-based load tracking
    tokens_per_second: float = 0.0
    peak_tokens_per_second: float = 0.0
    avg_tokens_per_second: float = 0.0

    # NEW: Temperature and power metrics (if available)
    temperature_c: float = 0.0
    power_usage_w: float = 0.0

    # NEW: Expert workload distribution
    expert_token_counts: Dict[int, int] = field(default_factory=dict)

    # NEW: Historical tracking for trend analysis
    memory_history: List[float] = field(default_factory=list)
    compute_history: List[float] = field(default_factory=list)
    _history_max_size: int = 100

    @property
    def memory_free_mb(self) -> float:
        """Available memory in MB."""
        return self.memory_total_mb - self.memory_used_mb

    @property
    def load_score(self) -> float:
        """
        Composite load score (0-1, higher = more loaded).

        Considers:
        - Memory utilization (50%)
        - Compute utilization (30%)
        - Token throughput (20%)
        """
        # Weighted combination of multiple metrics
        memory_weight = 0.5
        compute_weight = 0.3
        throughput_weight = 0.2

        # Normalize throughput to 0-1 range (assume 10k tokens/sec is very high)
        normalized_throughput = min(self.tokens_per_second / 10000.0, 1.0)

        return (
            memory_weight * self.memory_utilization +
            compute_weight * self.compute_utilization +
            throughput_weight * normalized_throughput
        )

    @property
    def is_overloaded(self) -> bool:
        """Check if GPU is overloaded (>90% on any metric)."""
        return (
            self.memory_utilization > 0.9 or
            self.compute_utilization > 0.9
        )

    @property
    def is_underutilized(self) -> bool:
        """Check if GPU is underutilized (<30% on all metrics)."""
        return (
            self.memory_utilization < 0.3 and
            self.compute_utilization < 0.3
        )

    def update_history(self):
        """Update historical metrics for trend analysis."""
        self.memory_history.append(self.memory_utilization)
        self.compute_history.append(self.compute_utilization)

        # Keep only recent history
        if len(self.memory_history) > self._history_max_size:
            self.memory_history.pop(0)
        if len(self.compute_history) > self._history_max_size:
            self.compute_history.pop(0)

    def get_load_trend(self) -> float:
        """
        Get load trend (-1 to 1, negative = decreasing, positive = increasing).

        Returns:
            Trend coefficient based on recent history
        """
        if len(self.memory_history) < 10:
            return 0.0

        # Simple linear regression on recent load scores
        recent_loads = [
            0.6 * mem + 0.4 * comp
            for mem, comp in zip(self.memory_history[-20:], self.compute_history[-20:])
        ]

        if len(recent_loads) < 5:
            return 0.0

        # Calculate slope using least squares
        x = np.arange(len(recent_loads))
        y = np.array(recent_loads)
        slope = np.polyfit(x, y, 1)[0]

        # Normalize to -1 to 1 range
        return np.clip(slope * 10, -1.0, 1.0)


@dataclass
class ExpertPlacement:
    """Expert placement information."""
    expert_id: int
    gpu_id: int
    memory_mb: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    is_pinned: bool = False  # Pinned experts won't be migrated

    # NEW: Token-based workload tracking
    tokens_processed: int = 0
    tokens_per_access: float = 0.0

    # NEW: Access pattern prediction
    access_frequency: float = 0.0  # Accesses per second
    predicted_next_access: float = 0.0  # Predicted time of next access

    # NEW: Co-access patterns (which experts are accessed together)
    co_accessed_with: Counter = field(default_factory=Counter)

    # NEW: Migration cost tracking
    migration_count: int = 0
    last_migration: float = 0.0
    migration_cost_history: List[float] = field(default_factory=list)

    def update_access(self, tokens: int = 1):
        """Update access statistics."""
        current_time = time.time()
        time_since_last = current_time - self.last_access

        # Update access count and tokens
        self.access_count += 1
        self.tokens_processed += tokens

        # Calculate average tokens per access
        self.tokens_per_access = self.tokens_processed / max(self.access_count, 1)

        # Update access frequency (exponential moving average)
        if time_since_last > 0:
            instant_freq = 1.0 / time_since_last
            alpha = 0.1  # Smoothing factor
            self.access_frequency = alpha * instant_freq + (1 - alpha) * self.access_frequency

        self.last_access = current_time

    def estimate_migration_cost(self) -> float:
        """
        Estimate cost of migrating this expert to another GPU.

        Returns:
            Estimated cost in milliseconds
        """
        # Base cost: memory transfer time
        # Assume ~10 GB/s PCIe bandwidth
        transfer_time_ms = (self.memory_mb / 1024) / 10 * 1000

        # Add overhead for frequent accesses (migration during high usage is expensive)
        frequency_penalty = self.access_frequency * 10  # ms

        # Add overhead for recent migrations (thrashing)
        if self.migration_count > 0:
            time_since_migration = time.time() - self.last_migration
            if time_since_migration < 60:  # Recent migration
                thrashing_penalty = 50 * (60 - time_since_migration) / 60
            else:
                thrashing_penalty = 0
        else:
            thrashing_penalty = 0

        total_cost = transfer_time_ms + frequency_penalty + thrashing_penalty
        return total_cost


class GPULoadBalancer:
    """
    Intelligent GPU load balancer for MoE models.

    Monitors GPU utilization and dynamically places experts to balance
    workload across available GPUs.

    Args:
        num_experts: Total number of experts
        num_gpus: Number of available GPUs
        balancing_strategy: Strategy for expert placement
            - 'round_robin': Simple round-robin distribution
            - 'memory_aware': Balance based on memory usage
            - 'compute_aware': Balance based on compute load
            - 'adaptive': Dynamically adapt based on workload
        rebalance_interval: Steps between rebalancing (0 = disabled)
        memory_headroom_mb: Reserve this much memory on each GPU
        enable_expert_migration: Allow moving experts between GPUs
        migration_threshold: Load imbalance threshold to trigger migration
        logger: Optional logger

    Example:
        >>> balancer = GPULoadBalancer(
        ...     num_experts=32,
        ...     num_gpus=4,
        ...     balancing_strategy='adaptive',
        ...     rebalance_interval=1000
        ... )
        >>> # Get initial expert placement
        >>> placement = balancer.get_expert_placement()
        >>> # Update with workload statistics
        >>> balancer.update_gpu_stats(gpu_id=0, memory_mb=8192, compute_util=0.75)
        >>> balancer.update_expert_access(expert_id=5, gpu_id=0)
    """

    def __init__(
        self,
        num_experts: int,
        num_gpus: int = 1,
        balancing_strategy: str = 'adaptive',
        rebalance_interval: int = 1000,
        memory_headroom_mb: float = 1024.0,
        enable_expert_migration: bool = True,
        migration_threshold: float = 0.2,  # 20% imbalance
        logger: Optional[logging.Logger] = None,
        # NEW: Advanced features
        enable_predictive_placement: bool = True,
        enable_token_aware_balancing: bool = True,
        enable_dynamic_thresholds: bool = True,
        enable_migration_cost_analysis: bool = True,
        enable_gpu_monitoring: bool = True,
        max_migrations_per_rebalance: int = 5,
    ):
        self.num_experts = num_experts
        self.num_gpus = num_gpus
        self.balancing_strategy = balancing_strategy
        self.rebalance_interval = rebalance_interval
        self.base_rebalance_interval = rebalance_interval  # PHASE 2 OPTIMIZATION: Store base interval
        self.adaptive_rebalance_interval = True  # PHASE 2 OPTIMIZATION: Enable adaptive interval
        self.min_rebalance_interval = max(500, rebalance_interval // 2)  # Min: half of base
        self.max_rebalance_interval = rebalance_interval * 2  # Max: 2x base
        self.load_variance_history = []  # Track load variance for adaptation
        self.load_variance_window = 10  # Look at last 10 rebalances
        self.memory_headroom_mb = memory_headroom_mb
        self.enable_expert_migration = enable_expert_migration
        self.migration_threshold = migration_threshold
        self.logger = logger or logging.getLogger(__name__)

        # NEW: Advanced feature flags
        self.enable_predictive_placement = enable_predictive_placement
        self.enable_token_aware_balancing = enable_token_aware_balancing
        self.enable_dynamic_thresholds = enable_dynamic_thresholds
        self.enable_migration_cost_analysis = enable_migration_cost_analysis
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.max_migrations_per_rebalance = max_migrations_per_rebalance

        # GPU statistics
        self.gpu_stats: Dict[int, GPUStats] = {}
        for gpu_id in range(num_gpus):
            self.gpu_stats[gpu_id] = GPUStats(gpu_id=gpu_id)

        # Expert placement mapping
        self.expert_placements: Dict[int, ExpertPlacement] = {}

        # Tracking
        self.step_count = 0
        self.last_rebalance_step = 0
        self.rebalance_history: List[Dict] = []
        self.expert_access_history: Dict[int, List[int]] = defaultdict(list)

        # NEW: Predictive tracking
        self.expert_cooccurrence: Dict[Tuple[int, int], int] = defaultdict(int)
        self.recent_expert_batch: List[int] = []

        # NEW: Dynamic threshold tracking
        self.dynamic_threshold = migration_threshold
        self.threshold_history: List[float] = []

        # NEW: Performance telemetry
        self.telemetry = {
            'total_migrations': 0,
            'successful_migrations': 0,
            'failed_migrations': 0,
            'avg_migration_cost_ms': 0.0,
            'total_rebalance_time_ms': 0.0,
            'load_balance_improvements': [],
        }

        # Initialize expert placement
        self._initialize_placement()

    def _initialize_placement(self):
        """Initialize expert placement across GPUs."""
        if self.balancing_strategy == 'round_robin':
            self._place_experts_round_robin()
        else:
            # For other strategies, start with round-robin then adapt
            self._place_experts_round_robin()

    def _place_experts_round_robin(self):
        """Place experts using simple round-robin distribution."""
        experts_per_gpu = self.num_experts // self.num_gpus
        remainder = self.num_experts % self.num_gpus

        expert_id = 0
        for gpu_id in range(self.num_gpus):
            # Distribute remainder across first N GPUs
            num_on_this_gpu = experts_per_gpu + (1 if gpu_id < remainder else 0)

            for _ in range(num_on_this_gpu):
                self.expert_placements[expert_id] = ExpertPlacement(
                    expert_id=expert_id,
                    gpu_id=gpu_id,
                    memory_mb=0.0  # Will be updated when experts are created
                )
                self.gpu_stats[gpu_id].expert_ids.add(expert_id)
                self.gpu_stats[gpu_id].num_experts += 1
                expert_id += 1

        self.logger.info(
            f"✓ Initial expert placement: {experts_per_gpu}-{experts_per_gpu+1} experts per GPU"
        )

    def update_gpu_stats(
        self,
        gpu_id: int,
        memory_mb: Optional[float] = None,
        memory_total_mb: Optional[float] = None,
        compute_util: Optional[float] = None,
        tokens_processed: Optional[int] = None,
    ):
        """
        Update statistics for a GPU.

        Args:
            gpu_id: GPU identifier
            memory_mb: Current memory usage in MB
            memory_total_mb: Total GPU memory in MB
            compute_util: Compute utilization (0-1)
            tokens_processed: Number of tokens processed
        """
        if gpu_id not in self.gpu_stats:
            return

        stats = self.gpu_stats[gpu_id]

        if memory_mb is not None:
            stats.memory_used_mb = memory_mb
        if memory_total_mb is not None:
            stats.memory_total_mb = memory_total_mb
        if stats.memory_total_mb > 0:
            stats.memory_utilization = stats.memory_used_mb / stats.memory_total_mb
        if compute_util is not None:
            stats.compute_utilization = compute_util
        if tokens_processed is not None:
            stats.tokens_processed = tokens_processed

        stats.last_update = time.time()

    def update_expert_access(self, expert_id: int, tokens: int = 1):
        """
        Record expert access for load tracking.

        Args:
            expert_id: Expert that was accessed
            tokens: Number of tokens processed by this expert
        """
        if expert_id in self.expert_placements:
            placement = self.expert_placements[expert_id]

            # Use the new update_access method
            placement.update_access(tokens)

            # Update GPU token count and per-expert tracking
            if placement.gpu_id in self.gpu_stats:
                gpu_stats = self.gpu_stats[placement.gpu_id]
                gpu_stats.tokens_processed += tokens

                # Track per-expert token counts on this GPU
                if expert_id not in gpu_stats.expert_token_counts:
                    gpu_stats.expert_token_counts[expert_id] = 0
                gpu_stats.expert_token_counts[expert_id] += tokens

            # NEW: Track co-occurrence for predictive placement
            if self.enable_predictive_placement:
                self._track_cooccurrence(expert_id)

            # Add to recent batch for pattern tracking
            self.recent_expert_batch.append(expert_id)

    def update_expert_memory(self, expert_id: int, memory_mb: float):
        """Update memory usage for an expert."""
        if expert_id in self.expert_placements:
            self.expert_placements[expert_id].memory_mb = memory_mb

    def get_expert_gpu(self, expert_id: int) -> int:
        """Get the GPU where an expert is placed."""
        if expert_id in self.expert_placements:
            return self.expert_placements[expert_id].gpu_id
        # Fallback to round-robin
        return expert_id % self.num_gpus

    def get_experts_on_gpu(self, gpu_id: int) -> List[int]:
        """Get list of expert IDs on a specific GPU."""
        if gpu_id in self.gpu_stats:
            return sorted(list(self.gpu_stats[gpu_id].expert_ids))
        return []

    def get_expert_placement(self) -> Dict[int, int]:
        """
        Get current expert-to-GPU mapping.

        Returns:
            Dictionary mapping expert_id -> gpu_id
        """
        return {
            expert_id: placement.gpu_id
            for expert_id, placement in self.expert_placements.items()
        }

    def check_rebalance_needed(self) -> bool:
        """Check if rebalancing is needed based on load imbalance."""
        if not self.enable_expert_migration:
            return False

        # PHASE 2 OPTIMIZATION: Adaptive rebalance interval adjustment
        if self.adaptive_rebalance_interval and len(self.load_variance_history) >= self.load_variance_window:
            # Calculate average load variance
            avg_variance = sum(self.load_variance_history[-self.load_variance_window:]) / self.load_variance_window

            # High variance (>0.15) = unstable load = rebalance more frequently
            # Low variance (<0.05) = stable load = rebalance less frequently
            if avg_variance > 0.15 and self.rebalance_interval > self.min_rebalance_interval:
                self.rebalance_interval = max(self.min_rebalance_interval, self.rebalance_interval - 100)
            elif avg_variance < 0.05 and self.rebalance_interval < self.max_rebalance_interval:
                self.rebalance_interval = min(self.max_rebalance_interval, self.rebalance_interval + 100)

        # Check if it's time for scheduled rebalance
        if self.rebalance_interval > 0:
            if self.step_count - self.last_rebalance_step >= self.rebalance_interval:
                # PHASE 2 OPTIMIZATION: Track load variance before rebalancing
                if self.adaptive_rebalance_interval and len(self.gpu_stats) >= 2:
                    load_scores = [stats.load_score for stats in self.gpu_stats.values()]
                    if len(load_scores) > 0:
                        max_load = max(load_scores)
                        min_load = min(load_scores)
                        variance = max_load - min_load
                        self.load_variance_history.append(variance)
                        # Keep only recent history
                        if len(self.load_variance_history) > self.load_variance_window * 2:
                            self.load_variance_history = self.load_variance_history[-self.load_variance_window:]

                return True

        # Check for severe load imbalance
        if len(self.gpu_stats) < 2:
            return False

        # Use dynamic threshold if enabled
        threshold = self.dynamic_threshold if self.enable_dynamic_thresholds else self.migration_threshold

        # Check load score imbalance
        load_scores = [stats.load_score for stats in self.gpu_stats.values()]
        max_load = max(load_scores)
        min_load = min(load_scores)

        # NEW: Also check token-based imbalance if enabled
        if self.enable_token_aware_balancing:
            token_balance = self.get_gpu_token_balance_score()
            if token_balance < 0.7:  # Less than 70% balanced
                self.logger.info(
                    f"⚠️  Token imbalance detected: balance score = {token_balance:.2f}"
                )
                return True

        # Trigger if imbalance exceeds threshold
        if max_load - min_load > threshold:
            self.logger.info(
                f"⚠️  Load imbalance detected: {max_load:.2f} vs {min_load:.2f} "
                f"(threshold: {threshold:.2f})"
            )
            return True

        # NEW: Check for overloaded/underutilized GPUs
        for gpu_id, stats in self.gpu_stats.items():
            if stats.is_overloaded:
                self.logger.info(f"⚠️  GPU {gpu_id} is overloaded")
                return True

        return False

    def rebalance(self) -> Dict:
        """
        Rebalance expert placement across GPUs.

        Returns:
            Dictionary with rebalancing results and migrations
        """
        if not self.enable_expert_migration:
            return {'status': 'disabled', 'migrations': []}

        self.last_rebalance_step = self.step_count
        start_time = time.time()

        # Choose rebalancing strategy
        if self.balancing_strategy == 'memory_aware':
            migrations = self._rebalance_memory_aware()
        elif self.balancing_strategy == 'compute_aware':
            migrations = self._rebalance_compute_aware()
        elif self.balancing_strategy == 'adaptive':
            migrations = self._rebalance_adaptive()
        else:
            migrations = []

        elapsed_ms = (time.time() - start_time) * 1000

        result = {
            'status': 'completed',
            'migrations': migrations,
            'num_migrations': len(migrations),
            'elapsed_ms': elapsed_ms,
            'step': self.step_count,
        }

        self.rebalance_history.append(result)

        if migrations:
            self.logger.info(
                f"✓ Rebalanced {len(migrations)} experts in {elapsed_ms:.1f}ms"
            )

        return result

    def _rebalance_memory_aware(self) -> List[Dict]:
        """Rebalance based on memory usage."""
        migrations = []

        # Sort GPUs by memory usage
        sorted_gpus = sorted(
            self.gpu_stats.items(),
            key=lambda x: x[1].memory_utilization
        )

        # Move experts from most loaded to least loaded
        for i in range(len(sorted_gpus) // 2):
            source_gpu_id, source_stats = sorted_gpus[-(i+1)]
            target_gpu_id, target_stats = sorted_gpus[i]

            # Find experts to migrate
            candidates = self._get_migration_candidates(source_gpu_id)

            for expert_id in candidates:
                placement = self.expert_placements[expert_id]

                # Check if migration would help
                if target_stats.memory_free_mb > placement.memory_mb + self.memory_headroom_mb:
                    # Perform migration
                    self._migrate_expert(expert_id, source_gpu_id, target_gpu_id)
                    migrations.append({
                        'expert_id': expert_id,
                        'from_gpu': source_gpu_id,
                        'to_gpu': target_gpu_id,
                        'reason': 'memory_balance'
                    })
                    break  # One migration per iteration

        return migrations

    def _rebalance_compute_aware(self) -> List[Dict]:
        """Rebalance based on compute utilization."""
        migrations = []

        # Sort GPUs by compute load
        sorted_gpus = sorted(
            self.gpu_stats.items(),
            key=lambda x: x[1].compute_utilization
        )

        # Move heavily-used experts from busy GPUs to idle GPUs
        for i in range(len(sorted_gpus) // 2):
            source_gpu_id, source_stats = sorted_gpus[-(i+1)]
            target_gpu_id, target_stats = sorted_gpus[i]

            # Find most accessed expert on source GPU
            experts_on_source = self.get_experts_on_gpu(source_gpu_id)
            if not experts_on_source:
                continue

            # Sort by access count
            experts_by_access = sorted(
                experts_on_source,
                key=lambda e: self.expert_placements[e].access_count,
                reverse=True
            )

            # Migrate hottest expert
            for expert_id in experts_by_access[:1]:
                if not self.expert_placements[expert_id].is_pinned:
                    self._migrate_expert(expert_id, source_gpu_id, target_gpu_id)
                    migrations.append({
                        'expert_id': expert_id,
                        'from_gpu': source_gpu_id,
                        'to_gpu': target_gpu_id,
                        'reason': 'compute_balance'
                    })
                    break

        return migrations

    def _rebalance_adaptive(self) -> List[Dict]:
        """
        Adaptive rebalancing based on combined metrics.

        Uses:
        - Token-aware balancing
        - Migration cost estimation
        - Predictive co-access patterns
        - Dynamic load trends
        """
        migrations = []

        # Sort GPUs by composite load score
        sorted_gpus = sorted(
            self.gpu_stats.items(),
            key=lambda x: x[1].load_score
        )

        # Use dynamic threshold if enabled
        threshold = self.dynamic_threshold if self.enable_dynamic_thresholds else self.migration_threshold

        # Identify overloaded and underloaded GPUs
        avg_load = sum(s.load_score for s in self.gpu_stats.values()) / len(self.gpu_stats)

        # NEW: Consider load trends when identifying problematic GPUs
        overloaded = []
        underloaded = []

        for gpu_id, stats in sorted_gpus:
            load_diff = stats.load_score - avg_load
            trend = stats.get_load_trend()

            # Overloaded: high load OR increasing trend
            if load_diff > threshold or (load_diff > threshold * 0.5 and trend > 0.3):
                overloaded.append((gpu_id, stats))

            # Underloaded: low load AND not increasing
            elif load_diff < -threshold and trend <= 0:
                underloaded.append((gpu_id, stats))

        # Migrate experts from overloaded to underloaded
        for source_gpu_id, source_stats in overloaded:
            if len(migrations) >= self.max_migrations_per_rebalance:
                break

            for target_gpu_id, target_stats in underloaded:
                if len(migrations) >= self.max_migrations_per_rebalance:
                    break

                # Find best expert to migrate
                candidates = self._get_migration_candidates(source_gpu_id)

                if not candidates:
                    continue

                # NEW: Select expert based on migration cost and benefit
                best_expert = None
                best_score = float('-inf')

                for expert_id in candidates[:10]:  # Consider top 10 candidates
                    placement = self.expert_placements[expert_id]

                    # Estimate migration cost
                    migration_cost = placement.estimate_migration_cost() if self.enable_migration_cost_analysis else 0

                    # Estimate benefit (load reduction on source GPU)
                    if self.enable_token_aware_balancing:
                        # Use token workload as benefit metric
                        benefit = placement.tokens_per_access * 100  # Normalize
                    else:
                        # Use access count as benefit metric
                        benefit = placement.access_count

                    # NEW: Bonus for co-accessed experts on target GPU
                    coaccessed_bonus = 0
                    if self.enable_predictive_placement:
                        coaccessed_experts = self._get_predicted_coaccessed_experts(expert_id)
                        experts_on_target = self.get_experts_on_gpu(target_gpu_id)
                        coaccessed_bonus = len(set(coaccessed_experts) & set(experts_on_target)) * 50

                    # Calculate score: benefit - cost + co-access bonus
                    score = benefit - migration_cost + coaccessed_bonus

                    if score > best_score:
                        best_score = score
                        best_expert = expert_id

                if best_expert is not None and best_score > 0:
                    self._migrate_expert(best_expert, source_gpu_id, target_gpu_id)
                    migrations.append({
                        'expert_id': best_expert,
                        'from_gpu': source_gpu_id,
                        'to_gpu': target_gpu_id,
                        'reason': 'adaptive_balance',
                        'source_load': source_stats.load_score,
                        'target_load': target_stats.load_score,
                        'migration_score': best_score,
                        'migration_cost_ms': self.expert_placements[best_expert].estimate_migration_cost(),
                    })
                    break

        return migrations

    def _get_migration_candidates(self, gpu_id: int) -> List[int]:
        """Get list of experts that can be migrated from a GPU."""
        experts = self.get_experts_on_gpu(gpu_id)

        # Filter out pinned experts
        candidates = [
            e for e in experts
            if not self.expert_placements[e].is_pinned
        ]

        # Sort by access count (prefer less-accessed experts)
        candidates.sort(
            key=lambda e: self.expert_placements[e].access_count
        )

        return candidates

    def _migrate_expert(self, expert_id: int, from_gpu: int, to_gpu: int):
        """Migrate an expert from one GPU to another."""
        if expert_id not in self.expert_placements:
            return

        placement = self.expert_placements[expert_id]

        # NEW: Track migration cost and history
        if self.enable_migration_cost_analysis:
            migration_cost = placement.estimate_migration_cost()
            placement.migration_cost_history.append(migration_cost)
            placement.migration_count += 1
            placement.last_migration = time.time()

            # Update telemetry
            self.telemetry['total_migrations'] += 1
            current_avg = self.telemetry['avg_migration_cost_ms']
            total = self.telemetry['total_migrations']
            self.telemetry['avg_migration_cost_ms'] = (
                (current_avg * (total - 1) + migration_cost) / total
            )

        # Update placement
        placement.gpu_id = to_gpu

        # Update GPU stats
        if from_gpu in self.gpu_stats:
            self.gpu_stats[from_gpu].expert_ids.discard(expert_id)
            self.gpu_stats[from_gpu].num_experts -= 1

            # Transfer token counts
            if expert_id in self.gpu_stats[from_gpu].expert_token_counts:
                token_count = self.gpu_stats[from_gpu].expert_token_counts.pop(expert_id)
                if to_gpu in self.gpu_stats:
                    self.gpu_stats[to_gpu].expert_token_counts[expert_id] = token_count

        if to_gpu in self.gpu_stats:
            self.gpu_stats[to_gpu].expert_ids.add(expert_id)
            self.gpu_stats[to_gpu].num_experts += 1

    def _track_cooccurrence(self, expert_id: int):
        """Track which experts are frequently accessed together."""
        # Track co-occurrence with recent experts
        for prev_expert in self.recent_expert_batch[-10:]:  # Last 10 experts
            if prev_expert != expert_id:
                key = tuple(sorted([prev_expert, expert_id]))
                self.expert_cooccurrence[key] += 1

                # Update placement co-access tracking
                if expert_id in self.expert_placements:
                    self.expert_placements[expert_id].co_accessed_with[prev_expert] += 1
                if prev_expert in self.expert_placements:
                    self.expert_placements[prev_expert].co_accessed_with[expert_id] += 1

    def _update_dynamic_threshold(self):
        """Dynamically adjust migration threshold based on performance."""
        if not self.enable_dynamic_thresholds:
            return

        # Analyze recent rebalancing effectiveness
        if len(self.rebalance_history) < 5:
            return

        recent_rebalances = self.rebalance_history[-5:]

        # Count how many migrations were made
        total_migrations = sum(r['num_migrations'] for r in recent_rebalances)

        # If too many migrations (thrashing), increase threshold
        if total_migrations > self.max_migrations_per_rebalance * 3:
            self.dynamic_threshold = min(self.dynamic_threshold * 1.1, 0.5)
            self.logger.info(f"Increasing migration threshold to {self.dynamic_threshold:.3f} (reducing thrashing)")

        # If no migrations but high imbalance, decrease threshold
        elif total_migrations == 0:
            metrics = self.get_load_balance_metrics()
            if metrics.get('load_imbalance', 0) > 0.3:
                self.dynamic_threshold = max(self.dynamic_threshold * 0.9, 0.05)
                self.logger.info(f"Decreasing migration threshold to {self.dynamic_threshold:.3f} (improving balance)")

        self.threshold_history.append(self.dynamic_threshold)

    def _get_predicted_coaccessed_experts(self, expert_id: int, top_k: int = 3) -> List[int]:
        """
        Get experts likely to be accessed together with the given expert.

        Args:
            expert_id: Expert to find co-access patterns for
            top_k: Number of top co-accessed experts to return

        Returns:
            List of expert IDs likely to be accessed together
        """
        if expert_id not in self.expert_placements:
            return []

        placement = self.expert_placements[expert_id]
        if not placement.co_accessed_with:
            return []

        # Get top-k most frequently co-accessed experts
        top_coaccessed = placement.co_accessed_with.most_common(top_k)
        return [expert for expert, count in top_coaccessed]

    def get_gpu_token_balance_score(self) -> float:
        """
        Calculate how well-balanced GPUs are by token load (0-1, 1 = perfect balance).

        Returns:
            Balance score (higher is better)
        """
        if not self.gpu_stats:
            return 1.0

        token_counts = [stats.tokens_processed for stats in self.gpu_stats.values()]

        if max(token_counts) == 0:
            return 1.0

        # Calculate coefficient of variation (lower is better balanced)
        mean_tokens = np.mean(token_counts)
        if mean_tokens == 0:
            return 1.0

        std_tokens = np.std(token_counts)
        cv = std_tokens / mean_tokens

        # Convert to 0-1 score (1 = perfect balance)
        balance_score = max(0.0, 1.0 - cv)
        return balance_score

    def pin_expert(self, expert_id: int):
        """Pin an expert to its current GPU (prevent migration)."""
        if expert_id in self.expert_placements:
            self.expert_placements[expert_id].is_pinned = True

    def unpin_expert(self, expert_id: int):
        """Unpin an expert (allow migration)."""
        if expert_id in self.expert_placements:
            self.expert_placements[expert_id].is_pinned = False

    def get_load_balance_metrics(self) -> Dict:
        """Get comprehensive load balancing metrics."""
        if not self.gpu_stats:
            return {}

        load_scores = [stats.load_score for stats in self.gpu_stats.values()]
        memory_utils = [stats.memory_utilization for stats in self.gpu_stats.values()]
        compute_utils = [stats.compute_utilization for stats in self.gpu_stats.values()]
        expert_counts = [stats.num_experts for stats in self.gpu_stats.values()]
        token_counts = [stats.tokens_processed for stats in self.gpu_stats.values()]

        # NEW: Token throughput metrics
        tokens_per_sec = [stats.tokens_per_second for stats in self.gpu_stats.values()]
        avg_tokens_per_sec = sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else 0

        return {
            'num_gpus': len(self.gpu_stats),
            'avg_load_score': sum(load_scores) / len(load_scores),
            'max_load_score': max(load_scores),
            'min_load_score': min(load_scores),
            'load_imbalance': max(load_scores) - min(load_scores),
            'avg_memory_util': sum(memory_utils) / len(memory_utils),
            'avg_compute_util': sum(compute_utils) / len(compute_utils),
            'avg_experts_per_gpu': sum(expert_counts) / len(expert_counts),
            'max_experts_per_gpu': max(expert_counts),
            'min_experts_per_gpu': min(expert_counts),
            'num_rebalances': len(self.rebalance_history),
            'balancing_strategy': self.balancing_strategy,

            # NEW: Token-based metrics
            'total_tokens_processed': sum(token_counts),
            'avg_tokens_per_gpu': sum(token_counts) / len(token_counts) if token_counts else 0,
            'token_balance_score': self.get_gpu_token_balance_score(),
            'avg_tokens_per_second': avg_tokens_per_sec,

            # NEW: Migration metrics
            'total_migrations': self.telemetry['total_migrations'],
            'avg_migration_cost_ms': self.telemetry['avg_migration_cost_ms'],

            # NEW: Dynamic threshold
            'current_threshold': self.dynamic_threshold if self.enable_dynamic_thresholds else self.migration_threshold,
        }

    def get_detailed_gpu_stats(self) -> Dict[int, Dict]:
        """
        Get detailed statistics for each GPU.

        Returns:
            Dictionary mapping GPU ID to detailed stats
        """
        detailed_stats = {}

        for gpu_id, stats in self.gpu_stats.items():
            detailed_stats[gpu_id] = {
                'gpu_id': gpu_id,
                'memory_used_mb': stats.memory_used_mb,
                'memory_total_mb': stats.memory_total_mb,
                'memory_utilization': stats.memory_utilization,
                'memory_free_mb': stats.memory_free_mb,
                'compute_utilization': stats.compute_utilization,
                'load_score': stats.load_score,
                'num_experts': stats.num_experts,
                'expert_ids': list(stats.expert_ids),
                'tokens_processed': stats.tokens_processed,
                'tokens_per_second': stats.tokens_per_second,
                'is_overloaded': stats.is_overloaded,
                'is_underutilized': stats.is_underutilized,
                'load_trend': stats.get_load_trend(),

                # Per-expert breakdown
                'expert_token_distribution': dict(stats.expert_token_counts),
            }

        return detailed_stats

    def get_expert_statistics(self) -> Dict[int, Dict]:
        """
        Get detailed statistics for each expert.

        Returns:
            Dictionary mapping expert ID to detailed stats
        """
        expert_stats = {}

        for expert_id, placement in self.expert_placements.items():
            expert_stats[expert_id] = {
                'expert_id': expert_id,
                'gpu_id': placement.gpu_id,
                'memory_mb': placement.memory_mb,
                'access_count': placement.access_count,
                'tokens_processed': placement.tokens_processed,
                'tokens_per_access': placement.tokens_per_access,
                'access_frequency': placement.access_frequency,
                'migration_count': placement.migration_count,
                'is_pinned': placement.is_pinned,

                # Co-access patterns
                'top_coaccessed_experts': self._get_predicted_coaccessed_experts(expert_id, top_k=5),
                'coaccessed_count': len(placement.co_accessed_with),

                # Migration cost
                'estimated_migration_cost_ms': placement.estimate_migration_cost(),
            }

        return expert_stats

    def print_load_balance_report(self):
        """Print a comprehensive load balancing report."""
        print("\n" + "=" * 80)
        print("GPU Load Balance Report")
        print("=" * 80)

        metrics = self.get_load_balance_metrics()

        print(f"\nStrategy: {metrics['balancing_strategy']}")
        print(f"Number of GPUs: {metrics['num_gpus']}")
        print(f"Total Experts: {self.num_experts}")

        print(f"\n--- Load Balance Metrics ---")
        print(f"Average Load Score: {metrics['avg_load_score']:.3f}")
        print(f"Load Imbalance: {metrics['load_imbalance']:.3f}")
        print(f"Token Balance Score: {metrics['token_balance_score']:.3f}")

        print(f"\n--- Resource Utilization ---")
        print(f"Avg Memory Utilization: {metrics['avg_memory_util']:.1%}")
        print(f"Avg Compute Utilization: {metrics['avg_compute_util']:.1%}")

        print(f"\n--- Expert Distribution ---")
        print(f"Experts per GPU: {metrics['min_experts_per_gpu']}-{metrics['max_experts_per_gpu']}")
        print(f"Average: {metrics['avg_experts_per_gpu']:.1f}")

        print(f"\n--- Token Throughput ---")
        print(f"Total Tokens Processed: {metrics['total_tokens_processed']:,}")
        print(f"Average Tokens/GPU: {metrics['avg_tokens_per_gpu']:,.0f}")
        print(f"Throughput: {metrics['avg_tokens_per_second']:.1f} tokens/sec")

        print(f"\n--- Migration Statistics ---")
        print(f"Total Rebalances: {metrics['num_rebalances']}")
        print(f"Total Migrations: {metrics['total_migrations']}")
        print(f"Avg Migration Cost: {metrics['avg_migration_cost_ms']:.2f} ms")
        print(f"Current Threshold: {metrics['current_threshold']:.3f}")

        print(f"\n--- Per-GPU Breakdown ---")
        detailed = self.get_detailed_gpu_stats()
        for gpu_id in sorted(detailed.keys()):
            stats = detailed[gpu_id]
            print(f"\nGPU {gpu_id}:")
            print(f"  Load Score: {stats['load_score']:.3f} {'[OVERLOADED]' if stats['is_overloaded'] else ''}")
            print(f"  Memory: {stats['memory_used_mb']:.0f}/{stats['memory_total_mb']:.0f} MB ({stats['memory_utilization']:.1%})")
            print(f"  Compute: {stats['compute_utilization']:.1%}")
            print(f"  Experts: {stats['num_experts']}")
            print(f"  Tokens: {stats['tokens_processed']:,} ({stats['tokens_per_second']:.1f}/sec)")
            print(f"  Trend: {stats['load_trend']:+.3f}")

        print("\n" + "=" * 80)

    def step(self):
        """Increment step counter and check for rebalancing."""
        self.step_count += 1

        # Update GPU stats history
        for stats in self.gpu_stats.values():
            stats.update_history()

        # Update dynamic thresholds
        if self.enable_dynamic_thresholds:
            self._update_dynamic_threshold()

        # Check if rebalancing is needed
        if self.check_rebalance_needed():
            return self.rebalance()
        return None

    def __repr__(self):
        metrics = self.get_load_balance_metrics()
        return (
            f"GPULoadBalancer(gpus={self.num_gpus}, experts={self.num_experts}, "
            f"strategy={self.balancing_strategy}, "
            f"load_imbalance={metrics.get('load_imbalance', 0):.3f})"
        )
