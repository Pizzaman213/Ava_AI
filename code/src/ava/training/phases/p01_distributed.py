"""
Phase 1: Distributed training setup.

Initializes DDP for multi-GPU or single GPU training.
Sets up rank, world_size, and device selection.
"""

import logging
from typing import List

import torch

from .base import PhaseContext, TrainingPhase

logger = logging.getLogger(__name__)


class DistributedPhase(TrainingPhase):
    """
    Phase 1: Setup distributed training environment.

    This phase:
    1. Initializes DDP process group (if multi-GPU)
    2. Sets rank and world_size
    3. Selects best GPU for single-GPU training
    4. Sets TF32 precision for Ampere+ GPUs
    5. Configures CUDA device

    Features:
        - Automatic GPU selection based on free memory
        - TF32 precision for 3x faster matmuls
        - Proper device context for multi-GPU
    """

    name = "distributed"
    description = "Setup distributed training"

    def execute(self, ctx: PhaseContext) -> PhaseContext:
        """
        Setup distributed training.

        Args:
            ctx: Phase context

        Returns:
            Context with rank, world_size, device set
        """
        from ava.training import setup_distributed

        # Initialize distributed
        rank, world_size = setup_distributed()
        ctx.rank = rank
        ctx.world_size = world_size

        # Set TF32 precision early (PyTorch 2.9+ API)
        if torch.cuda.is_available():
            # Use new API to avoid deprecation warnings
            if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
                torch.backends.cuda.matmul.fp32_precision = 'tf32'
            else:
                torch.backends.cuda.matmul.allow_tf32 = True

            if hasattr(torch.backends.cudnn.conv, 'fp32_precision'):
                torch.backends.cudnn.conv.fp32_precision = 'tf32'
            else:
                torch.backends.cudnn.allow_tf32 = True

        # Select device
        if world_size == 1 and torch.cuda.is_available():
            # Single GPU: select best available
            best_gpu = self._select_best_gpu(ctx)
            torch.cuda.set_device(best_gpu)
            ctx.device = torch.device(f'cuda:{best_gpu}')
        else:
            # Multi-GPU: use rank's assigned GPU
            if torch.cuda.is_available():
                torch.cuda.set_device(rank)
                ctx.device = torch.device(f'cuda:{rank}')
            else:
                ctx.device = torch.device('cpu')

        self.log(ctx, f"Device: {ctx.device}, World size: {world_size}")

        return ctx

    def _select_best_gpu(self, ctx: PhaseContext) -> int:
        """
        Select the best available GPU for single-GPU training.

        Selection criteria (in order):
        1. Most free memory
        2. Highest compute capability
        3. Lowest GPU index as tiebreaker

        Args:
            ctx: Phase context

        Returns:
            GPU index to use
        """
        if not torch.cuda.is_available():
            return 0

        num_gpus = torch.cuda.device_count()
        if num_gpus <= 1:
            return 0

        best_gpu = 0
        best_score = -1

        if ctx.is_main_process:
            self.log(ctx, f"GPU Selection - Found {num_gpus} GPUs")

        for i in range(num_gpus):
            try:
                props = torch.cuda.get_device_properties(i)
                torch.cuda.set_device(i)
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                free_gb = free_mem / (1024**3)

                # Compute capability as tie-breaker
                compute_cap = props.major * 10 + props.minor

                # Score: prioritize free memory, then compute capability
                score = free_gb * 100 + compute_cap

                if ctx.is_main_process:
                    self.log(
                        ctx,
                        f"  GPU {i}: {props.name}, {free_gb:.1f}GB free, "
                        f"compute {props.major}.{props.minor}, score {score:.1f}",
                        "debug"
                    )

                if score > best_score:
                    best_score = score
                    best_gpu = i

            except Exception as e:
                self.log(ctx, f"GPU {i}: Error querying - {e}", "warning")
                continue

        if ctx.is_main_process:
            self.log(ctx, f"Selected GPU {best_gpu}")

        return best_gpu

    def validate(self, ctx: PhaseContext) -> List[str]:
        """Validate preconditions."""
        return []

    def cleanup(self, ctx: PhaseContext) -> None:
        """Cleanup distributed resources."""
        from ava.training import cleanup_distributed
        if ctx.world_size > 1:
            cleanup_distributed(ctx.rank, ctx.world_size)
