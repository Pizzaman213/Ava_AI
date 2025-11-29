"""
Dynamic Batch Iterator for Memory-Aware Batch Size Adjustment

This module provides a wrapper around PyTorch DataLoader that dynamically
adjusts batch sizes based on GPU memory utilization, implementing the
dynamic_batching configuration from YAML.

The approach uses mini-batch concatenation:
- Base DataLoader yields batches of min_batch_size
- Iterator concatenates multiple mini-batches based on memory availability
- Yields batch sizes that are multiples of min_batch_size (64, 128, 192, 256)

This works with IterableDataset (unlike batch_sampler which requires map-style).

Enhanced features:
- Token-budget batching: Target token count instead of sample count
- Sequence-aware batching: Account for sequence length in decisions
- Scheduler integration: Pass batch info for token/sequence tracking
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Union

import torch
from torch.utils.data import DataLoader

# Import from the training optimizations module
# Note: This is a cross-package import, handled at runtime
try:
    from src.Ava.training.optimizations.dynamic_batching import (
        DynamicBatchConfig,
        DynamicBatchScheduler,
        create_dynamic_batch_scheduler,
    )
except ImportError:
    # Fallback for different import contexts
    from Ava.training.optimizations.dynamic_batching import (
        DynamicBatchConfig,
        DynamicBatchScheduler,
        create_dynamic_batch_scheduler,
    )

logger = logging.getLogger(__name__)
logger.propagate = False  # Prevent duplicate logs


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
    ):
        self.base_dataloader = base_dataloader
        self.scheduler = scheduler
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        # Token budget settings
        self.token_budget_enabled = token_budget_enabled
        self.target_tokens_per_batch = target_tokens_per_batch
        self.max_tokens_per_batch = max_tokens_per_batch

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

        mode = "token-budget" if token_budget_enabled else "sample-count"
        logger.info(
            f"DynamicBatchIterator initialized: "
            f"min={self.min_batch_size}, max={self.max_batch_size}, "
            f"max_multiplier={self.max_multiplier}, mode={mode}"
        )
        if token_budget_enabled:
            logger.info(
                f"Token budget: target={target_tokens_per_batch}, "
                f"max={max_tokens_per_batch}"
            )

    def _count_tokens(self, batch: Dict[str, torch.Tensor]) -> int:
        """Count actual tokens in a batch using attention mask."""
        if 'attention_mask' in batch:
            return int(batch['attention_mask'].sum().item())
        elif 'input_ids' in batch:
            # Fallback: assume all tokens are valid
            return batch['input_ids'].numel()
        return 0

    def _get_avg_sequence_length(self, batch: Dict[str, torch.Tensor]) -> int:
        """Get average sequence length in a batch."""
        if 'attention_mask' in batch:
            batch_size = batch['attention_mask'].size(0)
            if batch_size > 0:
                total_tokens = int(batch['attention_mask'].sum().item())
                return total_tokens // batch_size
        elif 'input_ids' in batch:
            return batch['input_ids'].size(1)
        return 0

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over base DataLoader, yielding dynamically-sized batches.

        Accumulates mini-batches in a buffer and yields when:
        - Token budget mode: Target token count is reached
        - Sample count mode: Target sample count is reached (original behavior)
        """
        buffer: List[Dict[str, torch.Tensor]] = []
        accumulated_tokens = 0

        for mini_batch in self.base_dataloader:
            buffer.append(mini_batch)

            # Count tokens in this mini-batch
            batch_tokens = self._count_tokens(mini_batch)
            accumulated_tokens += batch_tokens

            # Determine if we should yield based on mode
            should_yield = False

            if self.token_budget_enabled:
                # Token budget mode: yield when we reach target tokens
                # Use dynamic token budget from scheduler if available
                if hasattr(self.scheduler, 'get_dynamic_token_budget'):
                    current_target = self.scheduler.get_dynamic_token_budget()
                else:
                    current_target = self.target_tokens_per_batch

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

                    if self.token_budget_enabled:
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

                # Reset buffer and token count
                buffer = []
                accumulated_tokens = 0

                # Update step and let scheduler adjust (pass batch for token/seq tracking)
                self._step_count += 1
                new_size = self.scheduler.step(self._step_count, concatenated)

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
            # IterableDataset may not have __len__
            return self._total_batches_yielded + 1000  # Return current + reasonable estimate

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
            current_multiplier = max(1, round(self.scheduler.current_batch_size / self.min_batch_size))
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
        if not self._batch_size_history:
            return {
                'total_batches': 0,
                'avg_batch_size': 0,
                'min_batch_size_used': 0,
                'max_batch_size_used': 0,
                'batch_size_variance': 0,
                'mode': 'token-budget' if self.token_budget_enabled else 'sample-count',
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
            'mode': 'token-budget' if self.token_budget_enabled else 'sample-count',
        }

        # Add token statistics if available
        if self._tokens_per_batch_history:
            stats['total_tokens_processed'] = self._total_tokens_processed
            stats['avg_tokens_per_batch'] = statistics.mean(self._tokens_per_batch_history)
            stats['min_tokens_per_batch'] = min(self._tokens_per_batch_history)
            stats['max_tokens_per_batch'] = max(self._tokens_per_batch_history)

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
            # Count tokens if possible
            if isinstance(batch, dict) and 'attention_mask' in batch:
                self._total_tokens_processed += int(batch['attention_mask'].sum().item())
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
