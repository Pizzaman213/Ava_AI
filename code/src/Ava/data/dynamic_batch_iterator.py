"""
Dynamic Batch Iterator for Memory-Aware Batch Size Adjustment

This module is part of the data pipeline and provides wrappers around
PyTorch DataLoader that dynamically adjust batch sizes based on GPU memory.

Location: code/src/Ava/data/dynamic_batch_iterator.py (DATA module)
Purpose: Pre-training batch preparation and iteration

Related module: code/src/Ava/training/optimizations/dynamic_batching.py
    - That module handles runtime memory monitoring during training
    - This module handles data loading and batch construction

Key classes:
    - DynamicBatchIterator: Wraps DataLoader with memory-aware batching
    - TokenBudgetBatchIterator: Targets token count instead of sample count
    - UnifiedBatchingStrategy: Combines both approaches
    - PassThroughBatchIterator: No-op wrapper for compatibility

The approach uses mini-batch concatenation:
- Base DataLoader yields batches of min_batch_size
- Iterator concatenates multiple mini-batches based on memory availability
- Yields batch sizes that are multiples of min_batch_size (64, 128, 192, 256)

This works with IterableDataset (unlike batch_sampler which requires map-style).

Enhanced features:
- Token-budget batching: Target token count instead of sample count
- Sequence-aware batching: Account for sequence length in decisions
- Scheduler integration: Pass batch info for token/sequence tracking

Configuration: Uses DynamicBatchingConfig from config/training_config.py
"""

import logging
import threading
from collections import deque
from typing import Any, Deque, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import DataLoader

# Import from the training optimizations module
# Note: This is a cross-package import, handled at runtime
try:
    from src.Ava.training.optimizations.dynamic_batching import (
        DynamicBatchConfig,
        DynamicBatchScheduler,
        create_dynamic_batch_scheduler,
        KalmanMemoryPredictor,
    )
except ImportError:
    # Fallback for different import contexts
    from Ava.training.optimizations.dynamic_batching import (
        DynamicBatchConfig,
        DynamicBatchScheduler,
        create_dynamic_batch_scheduler,
        KalmanMemoryPredictor,
    )

logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent duplicate logs


# =============================================================================
# Momentum Accumulator for Sustained High Throughput
# =============================================================================

class MomentumAccumulator:
    """
    Track success of batch sizes and build momentum for larger batches.

    Key insight: After successfully processing several large batches, it's
    likely we can continue at that size. Build momentum to resist premature
    yielding and maintain high throughput.

    Benefits:
    - Prevents conservative behavior after temporary memory spikes
    - Maintains higher average batch sizes
    - Smoother training with fewer batch size changes
    """

    def __init__(
        self,
        momentum_threshold: int = 5,
        max_momentum: float = 1.5,
        momentum_decay: float = 0.95,
    ):
        """
        Initialize momentum accumulator.

        Args:
            momentum_threshold: Consecutive successes before increasing momentum
            max_momentum: Maximum momentum multiplier (e.g., 1.5 = 50% larger target)
            momentum_decay: Rate of momentum decay on each step (0.95 = slow decay)
        """
        self.momentum_threshold = momentum_threshold
        self.max_momentum = max_momentum
        self.momentum_decay = momentum_decay

        # State tracking
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.current_momentum = 1.0  # Multiplier for batch size target

        # History for analysis
        self.max_successful_batch = 0
        self.recent_batch_sizes: List[int] = []
        self._max_history = 50

    def record_success(self, batch_size: int) -> float:
        """
        Record a successful batch (no OOM, completed forward/backward).

        Args:
            batch_size: Size of the successful batch

        Returns:
            Updated momentum value
        """
        self.consecutive_successes += 1
        self.consecutive_failures = 0
        self.max_successful_batch = max(self.max_successful_batch, batch_size)

        # Record history
        self.recent_batch_sizes.append(batch_size)
        if len(self.recent_batch_sizes) > self._max_history:
            self.recent_batch_sizes.pop(0)

        # Build momentum after threshold consecutive successes
        if self.consecutive_successes >= self.momentum_threshold:
            # Increase momentum based on how far past threshold we are
            bonus = 0.1 * (self.consecutive_successes - self.momentum_threshold + 1)
            self.current_momentum = min(self.max_momentum, self.current_momentum + bonus)

        return self.current_momentum

    def record_failure(self) -> float:
        """
        Record a batch failure (OOM or memory pressure).

        Returns:
            Updated (reduced) momentum value
        """
        self.consecutive_successes = 0
        self.consecutive_failures += 1

        # Decay momentum faster on failure
        decay_factor = 0.7 if self.consecutive_failures > 1 else 0.8
        self.current_momentum = max(1.0, self.current_momentum * decay_factor)

        return self.current_momentum

    def apply_momentum(self, target_samples: int) -> int:
        """
        Apply momentum to target sample count.

        Higher momentum means we try to accumulate more samples before yielding.

        Args:
            target_samples: Base target sample count

        Returns:
            Momentum-adjusted target sample count
        """
        adjusted = int(target_samples * self.current_momentum)

        # Never exceed historical max successful by much
        if self.max_successful_batch > 0:
            # Allow up to 20% above historical max
            adjusted = min(adjusted, int(self.max_successful_batch * 1.2))

        return adjusted

    def decay_momentum(self) -> float:
        """
        Apply decay to momentum (call periodically even without success/failure).

        Returns:
            Updated momentum value
        """
        # Very slow decay toward 1.0
        if self.current_momentum > 1.0:
            self.current_momentum = 1.0 + (self.current_momentum - 1.0) * self.momentum_decay

        return self.current_momentum

    def get_statistics(self) -> Dict[str, Any]:
        """Get momentum statistics."""
        return {
            'current_momentum': self.current_momentum,
            'consecutive_successes': self.consecutive_successes,
            'consecutive_failures': self.consecutive_failures,
            'max_successful_batch': self.max_successful_batch,
            'avg_recent_batch': (
                sum(self.recent_batch_sizes) / len(self.recent_batch_sizes)
                if self.recent_batch_sizes else 0
            ),
        }


# =============================================================================
# Unified Batching Strategy (Token-Budget + Sequence-Aware Combined)
# =============================================================================

class UnifiedBatchingConfig:
    """Configuration for unified token-budget + sequence-aware batching."""

    def __init__(
        self,
        enabled: bool = True,
        base_token_budget: int = 8192,
        min_token_budget: int = 512,
        max_token_budget: int = 32768,
        reference_seq_len: int = 512,
        sequence_adjustment_strength: float = 0.5,
        target_memory_utilization: float = 0.80,
        attention_memory_factor: float = 0.001,
    ):
        """
        Initialize unified batching configuration.

        Args:
            enabled: Whether unified batching is enabled
            base_token_budget: Token budget at reference_seq_len
            min_token_budget: Minimum token budget
            max_token_budget: Maximum token budget
            reference_seq_len: Sequence length where base_token_budget applies
            sequence_adjustment_strength: How much to adjust for seq len (0=ignore, 1=full)
            target_memory_utilization: Target GPU memory utilization
            attention_memory_factor: Factor for quadratic attention memory scaling
        """
        self.enabled = enabled
        self.base_token_budget = base_token_budget
        self.min_token_budget = min_token_budget
        self.max_token_budget = max_token_budget
        self.reference_seq_len = reference_seq_len
        self.sequence_adjustment_strength = sequence_adjustment_strength
        self.target_memory_utilization = target_memory_utilization
        self.attention_memory_factor = attention_memory_factor


def get_memory_efficient_batching_config() -> UnifiedBatchingConfig:
    """
    Memory-efficient preset for RAM-constrained systems (<32GB).

    Reduces peak memory by ~15% at cost of ~10% throughput.
    Use when:
    - System RAM < 32GB
    - Training alongside other GPU workloads
    - Experiencing OOM errors
    """
    return UnifiedBatchingConfig(
        enabled=True,
        base_token_budget=4096,  # Reduced from 8192
        min_token_budget=256,  # Reduced from 512
        max_token_budget=16384,  # Reduced from 32768
        target_memory_utilization=0.70,  # More conservative (was 0.80)
        attention_memory_factor=0.002,  # Higher = more conservative
    )


def get_high_throughput_batching_config() -> UnifiedBatchingConfig:
    """
    High-throughput preset for systems with abundant RAM (>64GB).

    Maximizes GPU utilization for faster training.
    Use when:
    - System RAM > 64GB
    - Dedicated training machine
    - No memory issues observed
    """
    return UnifiedBatchingConfig(
        enabled=True,
        base_token_budget=16384,  # Increased from 8192
        max_token_budget=65536,  # Increased from 32768
        target_memory_utilization=0.85,  # More aggressive
    )


class UnifiedBatchingStrategy:
    """
    Unified batching that considers BOTH token count AND sequence length.

    Key insight: GPU memory for transformers scales as:
        Memory = O(batch * seq) for embeddings/FFN
               + O(batch * seq^2) for attention

    So the "effective load" isn't just token count, but:
        effective_load = tokens * (1 + alpha * seq_len)

    where alpha is a model-dependent constant for attention overhead.

    This strategy:
    1. Targets a consistent "effective memory load" instead of raw tokens
    2. Automatically adjusts token budget based on sequence lengths
    3. Uses Kalman predictor for accurate memory estimation

    Benefits over separate token-budget / sequence-aware modes:
    - Single unified approach (no mode switching)
    - Accounts for quadratic attention memory properly
    - Adapts to variable-length sequences automatically
    """

    def __init__(
        self,
        config: UnifiedBatchingConfig,
        predictor: Optional[KalmanMemoryPredictor] = None,
    ):
        """
        Initialize unified batching strategy.

        Args:
            config: Unified batching configuration
            predictor: Optional Kalman predictor for memory estimation
        """
        self.config = config
        self.predictor = predictor

        # Track recent sequence lengths for planning (thread-safe)
        self.recent_seq_window = 50
        self._seq_lens_lock = threading.Lock()
        self._recent_seq_lens: Deque[int] = deque(maxlen=self.recent_seq_window)

        # Calibrated attention factor (updated online from predictor)
        self.attention_factor = config.attention_memory_factor

        # Current dynamic token budget
        self.current_token_budget = config.base_token_budget

        # Statistics
        self._decisions_made = 0
        self._budget_adjustments = 0

    @property
    def recent_seq_lens(self) -> List[int]:
        """Get a copy of recent sequence lengths (for backwards compatibility)."""
        with self._seq_lens_lock:
            return list(self._recent_seq_lens)

    def get_effective_load(self, tokens: int, avg_seq_len: int) -> float:
        """
        Calculate effective memory load considering attention overhead.

        This normalizes different (batch, seq) combinations to a common scale.
        Higher effective load = more GPU memory used.

        Args:
            tokens: Total token count (batch_size * seq_len)
            avg_seq_len: Average sequence length

        Returns:
            Effective load value (higher = more memory)
        """
        if avg_seq_len <= 0:
            avg_seq_len = self.config.reference_seq_len
        return tokens * (1.0 + self.attention_factor * avg_seq_len)

    def get_target_token_budget(
        self,
        avg_seq_len: Optional[int] = None,
        memory_utilization: Optional[float] = None,
    ) -> int:
        """
        Calculate target token budget adjusted for sequence length.

        When sequences are longer, we need fewer tokens to achieve
        the same memory usage (due to quadratic attention).

        Args:
            avg_seq_len: Average sequence length of current/upcoming batch
            memory_utilization: Current GPU memory utilization (0-1)

        Returns:
            Adjusted token budget
        """
        if avg_seq_len is None:
            avg_seq_len = self._get_expected_seq_len()

        base_budget = self.config.base_token_budget
        ref_seq = self.config.reference_seq_len
        strength = self.config.sequence_adjustment_strength

        if avg_seq_len <= 0:
            avg_seq_len = ref_seq

        # Calculate scaling factor based on sequence length
        # At reference_seq_len, factor = 1.0
        # At 2x reference_seq_len, effective load is ~4x due to attention
        # So we need ~4x fewer tokens

        # Linear term (embeddings/FFN) ratio
        linear_ratio = ref_seq / avg_seq_len

        # Quadratic term (attention) ratio
        quadratic_ratio = (ref_seq / avg_seq_len) ** 2

        # Blend based on attention factor and strength
        # When attention dominates, use more quadratic scaling
        # When FFN dominates, use more linear scaling
        attention_weight = min(1.0, self.attention_factor * avg_seq_len)

        scaling = (
            (1 - attention_weight) * linear_ratio +
            attention_weight * quadratic_ratio
        )

        # Apply strength parameter (0 = no adjustment, 1 = full adjustment)
        scaling = 1.0 + (scaling - 1.0) * strength

        adjusted_budget = int(base_budget * scaling)

        # Memory-aware adjustment if utilization provided
        if memory_utilization is not None:
            target_util = self.config.target_memory_utilization

            if memory_utilization > target_util + 0.05:
                # Over target - reduce budget
                reduction = (memory_utilization - target_util) / (1.0 - target_util)
                adjusted_budget = int(adjusted_budget * (1 - reduction * 0.3))
                self._budget_adjustments += 1

            elif memory_utilization < target_util - 0.1:
                # Under target - increase budget
                headroom = (target_util - memory_utilization) / target_util
                adjusted_budget = int(adjusted_budget * (1 + headroom * 0.2))
                self._budget_adjustments += 1

        # Clamp to valid range
        adjusted_budget = max(
            self.config.min_token_budget,
            min(self.config.max_token_budget, adjusted_budget)
        )

        self.current_token_budget = adjusted_budget
        return adjusted_budget

    def calculate_batch_size_for_budget(
        self,
        token_budget: int,
        avg_seq_len: int,
        min_batch: int,
        max_batch: int,
    ) -> int:
        """
        Calculate batch size to achieve token budget given sequence length.

        Also considers memory predictor for safety if available.

        Args:
            token_budget: Target token count
            avg_seq_len: Average sequence length
            min_batch: Minimum batch size
            max_batch: Maximum batch size

        Returns:
            Recommended batch size
        """
        if avg_seq_len <= 0:
            avg_seq_len = self.config.reference_seq_len

        # Simple calculation
        target_batch = token_budget // avg_seq_len

        # Check with predictor if available
        if self.predictor and self.predictor.calibrated:
            # Binary search for safe batch size
            safe_max = max_batch
            while safe_max > min_batch:
                if self.predictor.is_safe(safe_max, avg_seq_len):
                    break
                safe_max = int(safe_max * 0.9)

            target_batch = min(target_batch, safe_max)

        # Ensure multiple of min_batch (guard against zero)
        min_batch = max(1, min_batch)
        target_batch = (target_batch // min_batch) * min_batch
        target_batch = max(min_batch, min(max_batch, target_batch))

        return target_batch

    def should_yield_batch(
        self,
        accumulated_tokens: int,
        accumulated_samples: int,
        avg_seq_len: int,
        memory_utilization: float,
    ) -> bool:
        """
        Determine if current accumulated batch should be yielded.

        Uses effective load comparison, not raw token count.
        Also includes memory safety checks to prevent OOM.

        IMPROVED: Now more aggressive when memory headroom is available.
        Will keep accumulating larger batches when GPU has room.

        Args:
            accumulated_tokens: Tokens accumulated so far
            accumulated_samples: Samples accumulated so far
            avg_seq_len: Average sequence length of accumulated batch
            memory_utilization: Current GPU memory utilization

        Returns:
            True if batch should be yielded now
        """
        if accumulated_tokens == 0:
            return False

        self._decisions_made += 1

        # SAFETY CHECK 1: Yield immediately if memory is too high
        # This prevents OOM by not accumulating more when GPU is stressed
        if memory_utilization > self.config.target_memory_utilization + 0.05:
            # Memory is above target + 5% buffer - yield what we have
            if accumulated_samples > 0:
                logger.debug(
                    f"[UnifiedBatching] Early yield due to high memory: "
                    f"{memory_utilization:.1%} > {self.config.target_memory_utilization:.1%}"
                )
                return True

        # SAFETY CHECK 2: Use Kalman predictor if available
        if self.predictor and self.predictor.calibrated and accumulated_samples > 0:
            # Check if the current accumulated batch is safe
            if not self.predictor.is_safe(
                accumulated_samples,
                avg_seq_len,
                threshold=self.config.target_memory_utilization
            ):
                logger.debug(
                    f"[UnifiedBatching] Predictor says unsafe: BS={accumulated_samples}, seq={avg_seq_len}"
                )
                return True

        # Calculate memory headroom for potential boost (conservative)
        headroom = self.config.target_memory_utilization - memory_utilization

        # Only apply small boost when we have significant headroom (>20%)
        # Much more conservative than before to avoid OOM
        if headroom > 0.20:
            # With 20%+ headroom, small 10% boost max
            headroom_boost = 1.0 + min(0.10, (headroom - 0.20) * 0.5)
        else:
            # No boost - stay conservative
            headroom_boost = 1.0

        # Get target budget adjusted for sequence length and memory
        target_budget = self.get_target_token_budget(avg_seq_len, memory_utilization)

        # Apply conservative headroom boost
        boosted_budget = int(target_budget * headroom_boost)
        boosted_budget = min(boosted_budget, self.config.max_token_budget)

        # Calculate effective loads
        current_load = self.get_effective_load(accumulated_tokens, avg_seq_len)
        target_load = self.get_effective_load(boosted_budget, self.config.reference_seq_len)

        # Yield if current load >= target load
        return current_load >= target_load

    def calibrate_attention_factor(self) -> None:
        """
        Calibrate attention_factor from Kalman predictor if available.

        Uses the predictor's quadratic coefficient to estimate
        the attention memory contribution.
        """
        if self.predictor and self.predictor.calibrated:
            # Extract from Kalman state
            # quadratic_coef represents memory per (token * seq)
            # attention_factor is normalized version
            linear_coef = self.predictor.state[1]
            quad_coef = self.predictor.state[2]

            if linear_coef > 0 and quad_coef > 0:
                new_factor = quad_coef / linear_coef
                # Sanity check - should be small positive number
                if 0 < new_factor < 0.1:
                    old_factor = self.attention_factor
                    self.attention_factor = new_factor
                    logger.debug(
                        f"Calibrated attention_factor: {old_factor:.6f} -> {new_factor:.6f}"
                    )

    def record_seq_len(self, seq_len: int) -> None:
        """Record sequence length for planning (thread-safe)."""
        with self._seq_lens_lock:
            self._recent_seq_lens.append(seq_len)
            # deque with maxlen handles eviction automatically

    def _get_expected_seq_len(self) -> int:
        """Get expected sequence length from history (thread-safe)."""
        with self._seq_lens_lock:
            if not self._recent_seq_lens:
                return self.config.reference_seq_len
            return int(sum(self._recent_seq_lens) / len(self._recent_seq_lens))

    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        return {
            'current_token_budget': self.current_token_budget,
            'attention_factor': self.attention_factor,
            'avg_recent_seq_len': self._get_expected_seq_len(),
            'decisions_made': self._decisions_made,
            'budget_adjustments': self._budget_adjustments,
            'effective_load_ratio': self.get_effective_load(
                self.current_token_budget,
                self._get_expected_seq_len()
            ) / max(1, self.get_effective_load(
                self.config.base_token_budget,
                self.config.reference_seq_len
            )),
        }


class DynamicBatchIterator:
    """
    Wraps a DataLoader and yields dynamically-sized batches by concatenating
    multiple mini-batches based on GPU memory utilization.

    This enables dynamic batch sizing with IterableDataset, which doesn't
    support PyTorch's native batch_sampler.

    The base DataLoader should be created with batch_size=min_batch_size.
    This iterator will concatenate 1-N mini-batches to create larger batches
    when GPU memory allows.

    Enhanced with:
    - Token-budget batching: Yield when target token count is reached
    - Sequence info tracking: Pass sequence lengths to scheduler
    - Gradient accumulation awareness: Coordinate with training loop

    Example:
        ```python
        # Create base DataLoader with min_batch_size
        base_loader = DataLoader(dataset, batch_size=64, ...)

        # Create scheduler from config
        scheduler = create_dynamic_batch_scheduler(config_dict)

        # Wrap with dynamic iterator
        dynamic_loader = DynamicBatchIterator(
            base_dataloader=base_loader,
            scheduler=scheduler,
            min_batch_size=64,
            max_batch_size=256,
        )

        # Use in training loop
        for batch in dynamic_loader:
            # batch size varies: 64, 128, 192, or 256
            loss = model(batch)
        ```

    Args:
        base_dataloader: DataLoader with batch_size=min_batch_size
        scheduler: DynamicBatchScheduler for memory-aware decisions
        min_batch_size: Minimum batch size (should match base_dataloader.batch_size)
        max_batch_size: Maximum batch size (must be multiple of min_batch_size)
        token_budget_enabled: Enable token-budget batching mode
        target_tokens_per_batch: Target token count when token_budget_enabled
        max_tokens_per_batch: Maximum tokens per batch
    """

    def __init__(
        self,
        base_dataloader: DataLoader,
        scheduler: DynamicBatchScheduler,
        min_batch_size: int = 64,
        max_batch_size: int = 256,
        token_budget_enabled: bool = False,
        target_tokens_per_batch: int = 4096,
        max_tokens_per_batch: int = 8192,
        unified_batching_config: Optional[UnifiedBatchingConfig] = None,
    ):
        self.base_dataloader = base_dataloader
        self.scheduler = scheduler
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        # Token budget settings
        self.token_budget_enabled = token_budget_enabled
        self.target_tokens_per_batch = target_tokens_per_batch
        self.max_tokens_per_batch = max_tokens_per_batch

        # Unified batching strategy (combines token-budget + sequence-aware)
        self.unified_strategy: Optional[UnifiedBatchingStrategy] = None
        if unified_batching_config is not None and unified_batching_config.enabled:
            # Get Kalman predictor from scheduler if available
            predictor = None
            if hasattr(scheduler, 'kalman_predictor'):
                predictor = scheduler.kalman_predictor
            elif hasattr(scheduler, 'memory_predictor') and isinstance(
                getattr(scheduler, 'memory_predictor', None), KalmanMemoryPredictor
            ):
                predictor = scheduler.memory_predictor

            self.unified_strategy = UnifiedBatchingStrategy(
                config=unified_batching_config,
                predictor=predictor,
            )
            # Unified mode supersedes token_budget mode
            self.token_budget_enabled = False
            logger.info(
                f"Unified batching enabled: base_budget={unified_batching_config.base_token_budget}, "
                f"ref_seq={unified_batching_config.reference_seq_len}, "
                f"strength={unified_batching_config.sequence_adjustment_strength}"
            )

        # Validate min_batch_size > 0 to prevent division by zero
        if min_batch_size <= 0:
            raise ValueError(f"min_batch_size must be > 0, got {min_batch_size}")

        # Validate max is multiple of min
        if max_batch_size % min_batch_size != 0:
            logger.warning(
                f"max_batch_size ({max_batch_size}) is not a multiple of "
                f"min_batch_size ({min_batch_size}). Adjusting to nearest multiple."
            )
            self.max_batch_size = (max_batch_size // min_batch_size) * min_batch_size

        self.max_multiplier = self.max_batch_size // self.min_batch_size

        # Statistics
        self._step_count = 0
        self._total_batches_yielded = 0
        self._batch_size_history: List[int] = []
        self._tokens_per_batch_history: List[int] = []
        self._total_tokens_processed = 0

        # Batch size controller integration (for OOM recovery)
        self._batch_controller = None  # Set via set_batch_controller()
        self._target_batch_size = max_batch_size  # Can be updated by controller

        mode = "unified" if self.unified_strategy else ("token-budget" if token_budget_enabled else "sample-count")
        logger.info(
            f"DynamicBatchIterator initialized: "
            f"min={self.min_batch_size}, max={self.max_batch_size}, "
            f"max_multiplier={self.max_multiplier}, mode={mode}"
        )
        if token_budget_enabled and not self.unified_strategy:
            logger.info(
                f"Token budget: target={target_tokens_per_batch}, "
                f"max={max_tokens_per_batch}"
            )

    def set_batch_size(self, new_size: int) -> None:
        """
        Update target batch size (called by BatchSizeController on OOM recovery).

        This allows the training loop to dynamically reduce batch size when
        OOM occurs, without restarting the iterator.

        Args:
            new_size: New maximum batch size to use
        """
        old_size = self._target_batch_size
        # Clamp to valid range
        new_size = max(self.min_batch_size, min(self.max_batch_size, new_size))
        self._target_batch_size = new_size
        self.max_batch_size = new_size  # Also update max for safety checks
        self.max_multiplier = new_size // self.min_batch_size

        if new_size != old_size:
            logger.info(f"DynamicBatchIterator: batch size updated {old_size} -> {new_size}")

    def set_batch_controller(self, controller) -> None:
        """
        Set the BatchSizeController for coordinated batch sizing.

        When set, the iterator will record successes with the controller.

        Args:
            controller: BatchSizeController instance
        """
        self._batch_controller = controller
        if controller is not None:
            logger.info("DynamicBatchIterator: BatchSizeController attached")

    def _count_tokens(self, batch: Dict[str, torch.Tensor]) -> int:
        """
        Count actual tokens in a batch using attention mask.

        GPU SYNC FIX: Uses shape-based estimation to avoid .sum().item() sync.
        For dynamic batching decisions, we use the maximum possible token count
        (batch_size * seq_len) which is a safe upper bound.
        """
        if 'input_ids' in batch:
            # Use shape-based count (no GPU sync!) - this is the max possible tokens
            # For packed sequences this slightly overestimates, but avoids GPU stalls
            return batch['input_ids'].numel()
        elif 'attention_mask' in batch:
            return batch['attention_mask'].numel()
        return 0

    def _get_avg_sequence_length(self, batch: Dict[str, torch.Tensor]) -> int:
        """
        Get average sequence length in a batch.

        GPU SYNC FIX: Uses shape-based calculation to avoid .sum().item() sync.
        """
        if 'input_ids' in batch:
            # Use shape directly (no GPU sync!)
            return batch['input_ids'].size(1)
        elif 'attention_mask' in batch:
            return batch['attention_mask'].size(1)
        return 0

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over base DataLoader, yielding dynamically-sized batches.

        Accumulates mini-batches in a buffer and yields when:
        - Unified mode: Target effective load is reached (considering seq²)
        - Token budget mode: Target token count is reached
        - Sample count mode: Target sample count is reached (original behavior)
        """
        buffer: List[Dict[str, torch.Tensor]] = []
        accumulated_tokens = 0
        accumulated_samples = 0

        for mini_batch in self.base_dataloader:
            # Count tokens and samples in this mini-batch BEFORE accumulating
            batch_tokens = self._count_tokens(mini_batch)
            batch_samples = mini_batch['input_ids'].size(0) if 'input_ids' in mini_batch else 1

            # SAFETY CHECK: Yield existing buffer if adding this batch would exceed target
            # Uses _target_batch_size which can be updated by BatchSizeController on OOM
            # This prevents unbounded growth by checking BEFORE accumulating
            if buffer and (accumulated_samples + batch_samples > self._target_batch_size):
                # Yield what we have before adding more
                concatenated = self._concatenate_batches(buffer)
                actual_batch_size = concatenated['input_ids'].size(0)
                actual_tokens = self._count_tokens(concatenated)

                # Track statistics
                self._batch_size_history.append(actual_batch_size)
                self._tokens_per_batch_history.append(actual_tokens)
                self._total_tokens_processed += actual_tokens
                self._total_batches_yielded += 1

                yield concatenated

                # Reset buffer
                buffer = []
                accumulated_tokens = 0
                accumulated_samples = 0

                # Update step (GPU SYNC FIX: is_log_step=False - no GPU memory query)
                self._step_count += 1
                self.scheduler.step(self._step_count, concatenated, is_log_step=False)

            # Now safe to add mini-batch to buffer
            buffer.append(mini_batch)
            accumulated_tokens += batch_tokens
            accumulated_samples += batch_samples

            # Get average sequence length
            avg_seq_len = self._get_avg_sequence_length(mini_batch)

            # Record sequence length for unified strategy
            if self.unified_strategy:
                self.unified_strategy.record_seq_len(avg_seq_len)

            # Determine if we should yield based on mode
            should_yield = False

            if self.unified_strategy:
                # Unified mode: use effective load comparison
                # GPU SYNC FIX: Don't query memory every mini-batch - use cached value
                # Memory queries cause implicit cudaStreamSynchronize
                mem_util = getattr(self, '_cached_mem_util', 0.0)

                should_yield = self.unified_strategy.should_yield_batch(
                    accumulated_tokens=accumulated_tokens,
                    accumulated_samples=accumulated_samples,
                    avg_seq_len=avg_seq_len,
                    memory_utilization=mem_util,
                )

                # SAFETY: Also yield if we hit max tokens (safety limit)
                if accumulated_tokens >= self.unified_strategy.config.max_token_budget:
                    should_yield = True

                # SAFETY: Also yield if we hit target_batch_size to prevent unbounded growth
                if accumulated_samples >= self._target_batch_size:
                    should_yield = True

            elif self.token_budget_enabled:
                # Token budget mode: yield when we reach target tokens
                # GPU SYNC FIX: Use cached token budget to avoid calling get_memory_stats()
                # on every mini-batch. Budget is updated after each yielded batch.
                current_target = getattr(self, '_cached_token_budget', self.target_tokens_per_batch)

                if accumulated_tokens >= current_target:
                    should_yield = True
                # Also yield if we would exceed max tokens with next batch
                elif accumulated_tokens >= self.max_tokens_per_batch:
                    should_yield = True
            else:
                # Sample count mode: use scheduler's batch size
                target_size = self.scheduler.current_batch_size
                target_multiplier = max(1, round(target_size / self.min_batch_size))
                target_multiplier = min(target_multiplier, self.max_multiplier)

                if len(buffer) >= target_multiplier:
                    should_yield = True

            if should_yield:
                concatenated = self._concatenate_batches(buffer)
                actual_batch_size = concatenated['input_ids'].size(0)
                actual_tokens = self._count_tokens(concatenated)
                avg_seq_len = self._get_avg_sequence_length(concatenated)

                # Track statistics
                self._batch_size_history.append(actual_batch_size)
                self._tokens_per_batch_history.append(actual_tokens)
                self._total_tokens_processed += actual_tokens
                self._total_batches_yielded += 1

                # Log batch info periodically
                if self._total_batches_yielded % 100 == 0 or (
                    len(self._batch_size_history) >= 2 and
                    self._batch_size_history[-1] != self._batch_size_history[-2]
                ):
                    mem_info = ""
                    if hasattr(self.scheduler, 'get_memory_stats'):
                        mem = self.scheduler.get_memory_stats()
                        mem_info = f" | GPU mem: {mem.get('utilization', 0):.1%}"

                    if self.unified_strategy:
                        budget = self.unified_strategy.current_token_budget
                        logger.debug(
                            f"[DynamicBatch] Step {self._step_count}: "
                            f"BS={actual_batch_size}, tokens={actual_tokens}, "
                            f"avg_seq={avg_seq_len}, budget={budget}{mem_info}"
                        )
                    elif self.token_budget_enabled:
                        logger.debug(
                            f"[DynamicBatch] Step {self._step_count}: "
                            f"BS={actual_batch_size}, tokens={actual_tokens}, "
                            f"avg_seq={avg_seq_len}{mem_info}"
                        )
                    else:
                        target_size = self.scheduler.current_batch_size
                        target_multiplier = max(1, round(target_size / self.min_batch_size))
                        logger.debug(
                            f"[DynamicBatch] Step {self._step_count}: BS={actual_batch_size} "
                            f"(target={target_size}, multiplier={target_multiplier}x){mem_info}"
                        )

                yield concatenated

                # Reset buffer and counts
                buffer = []
                accumulated_tokens = 0
                accumulated_samples = 0

                # Calibrate unified strategy periodically
                if self.unified_strategy and self._total_batches_yielded % 50 == 0:
                    self.unified_strategy.calibrate_attention_factor()

                # Update step and let scheduler adjust (pass batch for token/seq tracking)
                # GPU SYNC FIX: is_log_step=False by default - no GPU memory query
                # Memory is only queried at log intervals in the training loop
                self._step_count += 1
                new_size = self.scheduler.step(self._step_count, concatenated, is_log_step=False)

                # GPU SYNC FIX: Use cached memory stats from scheduler (no GPU sync)
                # The scheduler's cache is populated at log intervals by the training loop
                if hasattr(self.scheduler, '_memory_stats_cache') and self.scheduler._memory_stats_cache:
                    self._cached_mem_util = self.scheduler._memory_stats_cache.get('utilization', 0.0)
                if hasattr(self.scheduler, 'get_dynamic_token_budget'):
                    self._cached_token_budget = self.scheduler.get_dynamic_token_budget()

                if new_size is not None:
                    logger.debug(
                        f"Step {self._step_count}: Batch size adjusted to {new_size} "
                        f"(multiplier: {new_size // self.min_batch_size}x)"
                    )

        # Yield remaining mini-batches at end of epoch
        if buffer:
            concatenated = self._concatenate_batches(buffer)
            actual_tokens = self._count_tokens(concatenated)
            self._batch_size_history.append(concatenated['input_ids'].size(0))
            self._tokens_per_batch_history.append(actual_tokens)
            self._total_tokens_processed += actual_tokens
            self._total_batches_yielded += 1
            yield concatenated

    def _concatenate_batches(
        self, batches: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Concatenate multiple mini-batches into one large batch.

        Args:
            batches: List of mini-batches (dicts with input_ids, attention_mask, labels)

        Returns:
            Single concatenated batch
        """
        if len(batches) == 1:
            return batches[0]

        # Concatenate along batch dimension (dim=0)
        result = {}

        # Handle standard keys
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in batches[0]:
                result[key] = torch.cat([b[key] for b in batches], dim=0)

        # Handle any additional keys (e.g., position_ids, token_type_ids)
        for key in batches[0].keys():
            if key not in result:
                try:
                    result[key] = torch.cat([b[key] for b in batches], dim=0)
                except (TypeError, RuntimeError):
                    # Non-concatenatable value, take from first batch
                    result[key] = batches[0][key]

        return result

    def __len__(self) -> int:
        """
        Return estimated number of batches.

        For token-budget mode: estimate based on actual token throughput
        For sample-count mode: estimate based on samples / batch size
        """
        # Get base dataloader length - this might be batches OR samples depending on dataset type
        try:
            base_len = len(self.base_dataloader)
        except TypeError:
            # IterableDataset may not have __len__ on the DataLoader, but the dataset might
            # Try to get length from the underlying dataset
            try:
                dataset = self.base_dataloader.dataset
                # Handle InfiniteUltraFastDataset wrapper
                if hasattr(dataset, 'base_dataset'):
                    dataset = dataset.base_dataset
                base_len = len(dataset)
            except (TypeError, AttributeError):
                # Last resort: return current + estimate
                return self._total_batches_yielded + 1000

        # Get base batch size
        try:
            base_batch_size = self.base_dataloader.batch_size or self.min_batch_size
        except AttributeError:
            base_batch_size = self.min_batch_size

        # IMPORTANT: Check if base_len looks like number of samples (too high) vs batches
        # If base_len > 100000 and base_batch_size is small, it's probably samples not batches
        if base_len > 100000 and base_batch_size <= 64:
            # Assume base_len is number of samples, convert to batches
            num_mini_batches = base_len // base_batch_size
        else:
            num_mini_batches = base_len

        if self.token_budget_enabled:
            # Token-budget mode: use actual average tokens per yielded batch if available
            if self._tokens_per_batch_history and len(self._tokens_per_batch_history) >= 10:
                # We have enough history - use actual throughput
                avg_tokens_per_batch = sum(self._tokens_per_batch_history) / len(self._tokens_per_batch_history)
                avg_samples_per_batch = sum(self._batch_size_history) / len(self._batch_size_history)

                # Estimate total tokens in dataset
                total_samples = num_mini_batches * base_batch_size
                tokens_per_sample = avg_tokens_per_batch / max(1, avg_samples_per_batch)
                total_tokens = total_samples * tokens_per_sample

                # Batches = total tokens / target tokens per batch
                estimated_batches = max(1, int(total_tokens / self.target_tokens_per_batch))
                return estimated_batches
            else:
                # Not enough history - use simple estimate
                # Each yielded batch has ~target_tokens_per_batch tokens
                # Estimate: (num_samples * avg_seq_len) / target_tokens
                total_samples = num_mini_batches * base_batch_size
                avg_seq_len = 64  # Conservative default
                total_tokens = total_samples * avg_seq_len
                estimated_batches = max(1, int(total_tokens / self.target_tokens_per_batch))
                return estimated_batches
        else:
            # Sample-count mode: original calculation
            # Guard against division by zero
            safe_min_batch = max(1, self.min_batch_size)
            current_multiplier = max(1, round(self.scheduler.current_batch_size / safe_min_batch))
            current_multiplier = min(current_multiplier, self.max_multiplier)
            return max(1, num_mini_batches // current_multiplier)

    def get_dynamic_total(self) -> int:
        """
        Get the expected total number of batches based on current batch size.

        Use this to update tqdm progress bar total dynamically.
        """
        return len(self)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about dynamic batching performance."""
        # Determine mode
        if self.unified_strategy:
            mode = 'unified'
        elif self.token_budget_enabled:
            mode = 'token-budget'
        else:
            mode = 'sample-count'

        if not self._batch_size_history:
            return {
                'total_batches': 0,
                'avg_batch_size': 0,
                'min_batch_size_used': 0,
                'max_batch_size_used': 0,
                'batch_size_variance': 0,
                'mode': mode,
            }

        import statistics

        stats = {
            'total_batches': self._total_batches_yielded,
            'avg_batch_size': statistics.mean(self._batch_size_history),
            'min_batch_size_used': min(self._batch_size_history),
            'max_batch_size_used': max(self._batch_size_history),
            'batch_size_variance': statistics.variance(self._batch_size_history)
                if len(self._batch_size_history) > 1 else 0,
            'scheduler_stats': self.scheduler.get_statistics(),
            'mode': mode,
        }

        # Add token statistics if available
        if self._tokens_per_batch_history:
            stats['total_tokens_processed'] = self._total_tokens_processed
            stats['avg_tokens_per_batch'] = statistics.mean(self._tokens_per_batch_history)
            stats['min_tokens_per_batch'] = min(self._tokens_per_batch_history)
            stats['max_tokens_per_batch'] = max(self._tokens_per_batch_history)

        # Add unified strategy statistics if available
        if self.unified_strategy:
            stats['unified_strategy_stats'] = self.unified_strategy.get_statistics()

        return stats

    def log_summary(self) -> None:
        """Log a summary of dynamic batching performance."""
        stats = self.get_statistics()

        logger.info("=" * 70)
        logger.info("DYNAMIC BATCH ITERATOR SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Mode: {stats.get('mode', 'unknown')}")
        logger.info(f"Total batches yielded: {stats['total_batches']}")
        logger.info(f"Average batch size: {stats['avg_batch_size']:.1f}")
        logger.info(f"Batch size range: [{stats['min_batch_size_used']}, {stats['max_batch_size_used']}]")
        logger.info(f"Batch size variance: {stats['batch_size_variance']:.2f}")

        # Log token stats if available
        if 'total_tokens_processed' in stats:
            logger.info(f"Total tokens processed: {stats['total_tokens_processed']:,}")
            logger.info(f"Avg tokens per batch: {stats['avg_tokens_per_batch']:.0f}")
            logger.info(f"Token range: [{stats['min_tokens_per_batch']}, {stats['max_tokens_per_batch']}]")

        logger.info("=" * 70)

        # Also log scheduler summary
        self.scheduler.log_summary()

    def reset_statistics(self) -> None:
        """Reset statistics for new epoch."""
        self._batch_size_history = []
        self._tokens_per_batch_history = []
        self._total_batches_yielded = 0
        self._total_tokens_processed = 0
        # Don't reset step_count - scheduler needs continuous steps

    def refresh_memory_stats(self) -> None:
        """
        Refresh GPU memory stats - call this at log intervals.

        GPU SYNC FIX: This method triggers the actual GPU memory query.
        Call this from the training loop at log intervals (e.g., every 100 steps)
        to update the cached memory values used for batch size decisions.

        This design ensures memory queries (which cause cudaStreamSynchronize)
        only happen at controlled intervals, not on every batch.
        """
        if hasattr(self.scheduler, 'get_memory_stats'):
            # Force refresh to query GPU
            self.scheduler.get_memory_stats(force_refresh=True)
            # Update our cached value
            if hasattr(self.scheduler, '_memory_stats_cache') and self.scheduler._memory_stats_cache:
                self._cached_mem_util = self.scheduler._memory_stats_cache.get('utilization', 0.0)

    @property
    def current_batch_size(self) -> int:
        """Get current batch size from scheduler."""
        return self.scheduler.current_batch_size

    def get_recommended_grad_accum_steps(self) -> int:
        """Get recommended gradient accumulation steps from scheduler."""
        if hasattr(self.scheduler, 'get_recommended_grad_accum_steps'):
            return self.scheduler.get_recommended_grad_accum_steps()
        return 1


class TokenBudgetBatchIterator(DynamicBatchIterator):
    """
    Specialized iterator that yields batches based on token budget.

    This is a convenience class that defaults to token-budget mode.
    Use this when you want to maximize GPU utilization by targeting
    a specific token count per batch rather than sample count.

    Example:
        ```python
        iterator = TokenBudgetBatchIterator(
            base_dataloader=loader,
            scheduler=scheduler,
            target_tokens=4096,
            max_tokens=8192,
        )

        for batch in iterator:
            # Each batch will have approximately 4096 tokens
            pass
        ```
    """

    def __init__(
        self,
        base_dataloader: DataLoader,
        scheduler: DynamicBatchScheduler,
        min_batch_size: int = 64,
        max_batch_size: int = 256,
        target_tokens: int = 4096,
        max_tokens: int = 8192,
    ):
        super().__init__(
            base_dataloader=base_dataloader,
            scheduler=scheduler,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            token_budget_enabled=True,
            target_tokens_per_batch=target_tokens,
            max_tokens_per_batch=max_tokens,
        )


def create_dynamic_batch_iterator(
    base_dataloader: DataLoader,
    config_dict: Dict[str, Any],
) -> Union[DynamicBatchIterator, 'PassThroughBatchIterator']:
    """
    Factory function to create a DynamicBatchIterator from config.

    Args:
        base_dataloader: DataLoader with batch_size=min_batch_size
        config_dict: Configuration dictionary containing dynamic_batching section

    Returns:
        DynamicBatchIterator wrapping the base DataLoader, or
        PassThroughBatchIterator if dynamic batching is disabled
    """
    # Extract dynamic batching config (support multiple nesting levels)
    db_config = config_dict.get('dynamic_batching', {})

    if not db_config:
        training = config_dict.get('training', {})
        db_config = training.get('dynamic_batching', {})
        if not db_config:
            batching = training.get('batching', {})
            db_config = batching.get('dynamic_batching', {})

    if not db_config.get('enabled', False):
        logger.info(
            "Dynamic batching disabled. Using pass-through iterator."
        )
        return PassThroughBatchIterator(base_dataloader)

    # Get batch size parameters
    min_batch_size = db_config.get('min_batch_size', 64)
    max_batch_size = db_config.get('max_batch_size', 256)

    # Get token budget parameters
    token_budget = db_config.get('token_budget', {})
    token_budget_enabled = token_budget.get('enabled', db_config.get('token_budget_enabled', False))
    target_tokens = token_budget.get('target_tokens_per_batch', db_config.get('target_tokens_per_batch', 4096))
    max_tokens = token_budget.get('max_tokens_per_batch', db_config.get('max_tokens_per_batch', 8192))

    # Get unified batching parameters (NEW)
    unified_config_dict = db_config.get('unified_batching', {})
    unified_batching_config = None
    if unified_config_dict.get('enabled', False):
        unified_batching_config = UnifiedBatchingConfig(
            enabled=True,
            base_token_budget=unified_config_dict.get('base_token_budget', 8192),
            min_token_budget=unified_config_dict.get('min_token_budget', 512),
            max_token_budget=unified_config_dict.get('max_token_budget', 32768),
            reference_seq_len=unified_config_dict.get('reference_seq_len', 512),
            sequence_adjustment_strength=unified_config_dict.get('sequence_adjustment_strength', 0.5),
            target_memory_utilization=unified_config_dict.get('target_memory_utilization', 0.80),
            attention_memory_factor=unified_config_dict.get('attention_memory_factor', 0.001),
        )
        logger.info(f"Unified batching config loaded: base_budget={unified_batching_config.base_token_budget}")

    # Create scheduler
    scheduler = create_dynamic_batch_scheduler(config_dict)

    return DynamicBatchIterator(
        base_dataloader=base_dataloader,
        scheduler=scheduler,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        token_budget_enabled=token_budget_enabled,
        target_tokens_per_batch=target_tokens,
        max_tokens_per_batch=max_tokens,
        unified_batching_config=unified_batching_config,
    )


class PassThroughBatchIterator:
    """
    Pass-through iterator that doesn't modify batches.

    Used when dynamic batching is disabled but code expects a DynamicBatchIterator.
    """

    def __init__(self, base_dataloader: DataLoader):
        self.base_dataloader = base_dataloader
        self._total_batches_yielded = 0
        self._total_tokens_processed = 0
        self.current_batch_size = getattr(base_dataloader, 'batch_size', 1) or 1

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for batch in self.base_dataloader:
            self._total_batches_yielded += 1
            # Count tokens using shape (no GPU sync!)
            if isinstance(batch, dict) and 'input_ids' in batch:
                self._total_tokens_processed += batch['input_ids'].numel()
            elif isinstance(batch, dict) and 'attention_mask' in batch:
                self._total_tokens_processed += batch['attention_mask'].numel()
            yield batch

    def __len__(self) -> int:
        return len(self.base_dataloader)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_batches': self._total_batches_yielded,
            'total_tokens_processed': self._total_tokens_processed,
            'mode': 'pass-through',
        }

    def log_summary(self) -> None:
        logger.info(
            f"PassThroughBatchIterator: {self._total_batches_yielded} batches, "
            f"{self._total_tokens_processed:,} tokens"
        )

    def reset_statistics(self) -> None:
        self._total_batches_yielded = 0
        self._total_tokens_processed = 0

    def get_recommended_grad_accum_steps(self) -> int:
        """Return 1 since pass-through doesn't coordinate with grad accum."""
        return 1
