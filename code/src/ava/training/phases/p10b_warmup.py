"""
Phase 10b: Model Warmup.

Warms up the model with dummy batches to prime CUDA kernels and
trigger any JIT compilation before training begins. This improves
initial throughput and provides more consistent timing metrics.

Benefits:
- Primes CUDA kernels for consistent performance
- Triggers torch.compile graph compilation
- Allows JIT compilation to complete before timing
- Ensures memory is properly initialized
"""

import logging
import time
from typing import List

import torch

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class WarmupPhase(TrainingPhase):
    """
    Phase 10b: Warm up the model.

    This optional phase runs a few dummy forward/backward passes
    to prime CUDA kernels and any JIT compilation.

    Config options (under compute.warmup):
        enabled: bool (default: False) - Must be explicitly enabled
        num_warmup_batches: int (default: 3)
        use_real_data: bool (default: False)
        log_memory: bool (default: True)
    """

    name = "warmup"
    description = "Model warmup"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Run warmup iterations.

        Args:
            ctx: Phase context

        Returns:
            Context unchanged (warmup only affects CUDA state)
        """
        # Check if warmup is enabled (default: False to avoid unexpected delays)
        warmup_config = ctx.config.get('compute', {}).get('warmup', {})
        if not warmup_config.get('enabled', False):
            return ctx

        # CRITICAL: Skip warmup when using DeepSpeed
        # DeepSpeed requires model.backward(loss) instead of loss.backward().
        # Using loss.backward() during warmup corrupts DeepSpeed's gradient
        # reduction state, causing "parameter already reduced" errors.
        distributed_cfg = ctx.config.get('distributed', {})
        is_deepspeed = (
            distributed_cfg.get('deepspeed', {}).get('enabled', False) or
            ctx.config.get('deepspeed', {}).get('enabled', False)
        )
        if is_deepspeed:
            if ctx.is_main_process:
                self.log(ctx, "Skipping warmup (incompatible with DeepSpeed)")
            return ctx

        num_batches = warmup_config.get('num_warmup_batches', 3)
        use_real_data = warmup_config.get('use_real_data', False)
        log_memory = warmup_config.get('log_memory', True)

        if ctx.is_main_process:
            self.log(ctx, f"Running {num_batches} warmup iterations...")

        # Get model and required components
        model = ctx.model
        optimizer = ctx.optimizer
        device = ctx.device

        if model is None or optimizer is None:
            self.log(ctx, "Model or optimizer not available, skipping warmup", "warning")
            return ctx

        # Get data configuration
        data_config = ctx.config.get('data', {})
        model_config = ctx.config.get('model', {})
        training_config = ctx.config.get('training', {})

        seq_len = data_config.get('max_length', 512)
        vocab_size = model_config.get('vocab_size', 32000)
        batch_size = ctx.batch_size or training_config.get('batching', {}).get('batch_size', 8)

        # Get AMP settings
        use_amp = ctx.context.use_amp if ctx.context else True
        amp_dtype = ctx.context.amp_dtype if ctx.context else torch.bfloat16

        # Track initial memory
        initial_memory = 0
        if torch.cuda.is_available() and log_memory:
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated() / 1e9

        # Create synthetic batches or use real data
        model.train()
        warmup_times = []

        try:
            for i in range(num_batches):
                start_time = time.perf_counter()

                # Create batch
                if use_real_data and ctx.train_loader is not None:
                    batch = self._get_real_batch(ctx)
                else:
                    batch = self._create_synthetic_batch(
                        batch_size, seq_len, vocab_size, device
                    )

                # Forward pass
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch.get('labels', batch['input_ids']),
                    )

                    if isinstance(outputs, dict):
                        loss = outputs.get('loss', outputs.get('ce_loss'))
                    else:
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs

                # Backward pass
                if loss is not None:
                    loss.backward()

                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # Synchronize for accurate timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time
                warmup_times.append(elapsed * 1000)

                if ctx.is_main_process:
                    self.log(ctx, f"  Warmup batch {i + 1}/{num_batches}: {elapsed * 1000:.1f}ms")

            # Report warmup summary
            if ctx.is_main_process:
                avg_time = sum(warmup_times) / len(warmup_times) if warmup_times else 0
                self.log(ctx, f"Warmup complete. Avg time: {avg_time:.1f}ms")

                if torch.cuda.is_available() and log_memory:
                    torch.cuda.synchronize()
                    final_memory = torch.cuda.memory_allocated() / 1e9
                    peak_memory = torch.cuda.max_memory_allocated() / 1e9
                    self.log(
                        ctx,
                        f"Memory after warmup: {final_memory:.2f} GB "
                        f"(peak: {peak_memory:.2f} GB, delta: +{final_memory - initial_memory:.2f} GB)"
                    )

        except Exception as e:
            if ctx.is_main_process:
                self.log(ctx, f"Warmup failed: {e}", "warning")
                self.log(ctx, "Continuing without warmup", "warning")

        # Clear gradients and memory
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ctx

    def _create_synthetic_batch(
        self,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        device: torch.device,
    ) -> dict:
        """Create a synthetic batch for warmup."""
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
        }

    def _get_real_batch(self, ctx: PhaseContext) -> dict:
        """Get a real batch from the dataloader."""
        if ctx.train_loader is None:
            raise RuntimeError("train_loader is None")

        # Get one batch
        data_iter = iter(ctx.train_loader)
        batch = next(data_iter)

        # Move to device
        device = ctx.device
        if isinstance(batch, dict):
            return {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        else:
            return {'input_ids': batch.to(device)}

    def validate(self, ctx: PhaseContext) -> List[str]:
        """Validate preconditions."""
        errors = []
        # Warmup requires model and optimizer
        if ctx.model is None:
            errors.append("model must be built before warmup")
        if ctx.optimizer is None:
            errors.append("optimizer must be created before warmup")
        return errors
