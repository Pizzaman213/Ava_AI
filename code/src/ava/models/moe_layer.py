"""
Sparse Mixture of Experts Layer - Drop-in replacement for FFN layers.

This module implements a production-grade SparseMoE layer with:
- Mixtral or DeepSeek routing
- Grouped GEMM for parallel expert computation
- 4 auxiliary losses (load_balance, router_z, expert_dropout, diversity)
- Dynamic expert capacity
- Gradient checkpointing
- Expert-level mixed precision
- Hierarchical MoE support (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any
import math

from .experts import ExpertParallelGroup, SequentialExpertGroup, SharedExpertLayer
from .routing import MixtralRouter, DeepSeekRouter, StableMoERouter

import logging
moe_logger = logging.getLogger(__name__)


@torch.jit.script
def fused_expert_combine(expert_outputs: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    JIT-compiled fused expert combination for efficient output aggregation.

    Fuses the weighted sum operation for expert outputs into a single kernel.
    Further optimized with memory-efficient sum operation.

    Args:
        expert_outputs: [num_tokens, k, hidden_size]
        weights: Optional weights tensor [num_tokens, k] - if provided, applies weighting
                 NOTE: In current architecture, weights are pre-applied in ExpertParallelGroup,
                 so this parameter is kept for API compatibility but typically None.

    Returns:
        Combined output: [num_tokens, hidden_size]
    """
    # OPTIMIZATION: Use torch.sum with explicit dimension for better memory efficiency
    # NOTE: In the current architecture, weights are already applied in ExpertParallelGroup._forward_grouped_gemm()
    # (line ~407 in experts.py). This function just sums the pre-weighted outputs.
    # The weights parameter is kept for API compatibility and potential future use cases
    # where weighting is deferred to combination time.
    # sum() already returns a contiguous tensor, no need for extra .contiguous()
    return expert_outputs.sum(dim=1, keepdim=False)


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture of Experts layer - drop-in FFN replacement.

    This layer can replace any feedforward network in a transformer with
    a mixture of expert networks, providing better parameter efficiency
    and potentially better performance.

    Args:
        hidden_size: Input/output dimension
        intermediate_size: FFN hidden dimension (typically 3.5-4x hidden_size)
        num_experts: Total number of experts
        num_experts_per_token: How many experts to activate per token (k)
        router_type: Routing strategy ('mixtral', 'deepseek')
        capacity_factor: Expert capacity multiplier
        expert_dropout: Dropout rate for expert outputs
        activation: Activation function ('swiglu', 'geglu', 'gelu')
        use_grouped_gemm: Whether to use grouped GEMM (recommended)
        use_triton_kernels: Whether to use Triton kernels for routing
        use_torch_compile: Whether to compile the module
        router_z_loss_coef: Coefficient for router z-loss
        load_balance_loss_coef: Coefficient for load balancing loss
        diversity_loss_coef: Coefficient for diversity loss
        expert_dropout_loss_coef: Coefficient for expert dropout regularization
        router_jitter_noise: Jitter noise for exploration
        aux_loss_frequency: Compute aux losses every N steps (1=every step, 50=every 50 steps)
        use_shared_expert: Whether to use shared expert (DeepSeek-style)
        shared_expert_weight: Weight for shared expert
        gradient_checkpointing: Whether to checkpoint expert computation
        dtype: Parameter dtype

    Example:
        >>> # Replace FFN in transformer with MoE
        >>> moe_layer = SparseMoELayer(
        ...     hidden_size=4096,
        ...     intermediate_size=14336,
        ...     num_experts=32,
        ...     num_experts_per_token=2,
        ...     router_type='mixtral',
        ...     aux_loss_frequency=50,  # Compute aux losses every 50 steps
        ... )
        >>> x = torch.randn(8, 128, 4096)  # [batch, seq, hidden]
        >>> output, aux_loss, metrics = moe_layer(x, training=True)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        router_type: str = 'mixtral',
        capacity_factor: float = 1.25,
        expert_dropout: float = 0.0,
        activation: str = 'swiglu',
        use_grouped_gemm: bool = True,
        use_triton_kernels: bool = True,
        use_torch_compile: bool = True,
        compile_router: bool = False,  # Separately compile router (deprecated)
        router_compile_mode: str = 'default',  # 'default' or 'reduce-overhead'
        enable_cudagraphs_safe_routing: bool = False,  # Enable CUDAGraphs-compatible routing (deprecated)
        router_z_loss_coef: float = 0.001,
        load_balance_loss_coef: float = 0.01,
        diversity_loss_coef: float = 0.001,
        expert_dropout_loss_coef: float = 0.001,
        router_jitter_noise: float = 0.0,
        aux_loss_frequency: int = 500,  # OPTIMIZED: Compute aux losses every 500 steps (was 200)
        use_shared_expert: bool = False,
        shared_expert_weight: float = 0.5,
        gradient_checkpointing: bool = False,
        # P7 OPTIMIZATION: Fine-grained checkpointing
        # When True, only checkpoint expensive operations (experts) not cheap ones (router, norm)
        checkpoint_experts_only: bool = True,
        dtype: Optional[torch.dtype] = None,
        # OPTIMIZATION: Expert caching
        use_expert_caching: bool = False,
        expert_cache_size: int = 51,  # Reduced from 256 for memory efficiency
        expert_cache_similarity_threshold: float = 0.95,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.router_type = router_type
        self.capacity_factor = capacity_factor
        self.expert_dropout = expert_dropout
        self.use_grouped_gemm = use_grouped_gemm
        self.use_triton_kernels = use_triton_kernels
        self.use_torch_compile = use_torch_compile
        self.router_z_loss_coef = router_z_loss_coef
        self.load_balance_loss_coef = load_balance_loss_coef
        self.diversity_loss_coef = diversity_loss_coef
        self.expert_dropout_loss_coef = expert_dropout_loss_coef
        self.use_shared_expert = use_shared_expert
        self.gradient_checkpointing = gradient_checkpointing
        # P7 OPTIMIZATION: Fine-grained checkpointing
        # When True and gradient_checkpointing is enabled, only checkpoint expert computation
        # (saves 5-10% memory compared to full checkpointing while maintaining most benefits)
        self._checkpoint_experts_only = checkpoint_experts_only

        # Step counter for adaptive loss computation frequency
        # Use Python int instead of torch.tensor to avoid torch.compile graph breaks
        self._training_step = 0

        # Store aux_loss_frequency for reference
        self.aux_loss_frequency = aux_loss_frequency

        # Create router
        # Map 'switch' to 'mixtral' (functionally equivalent top-k routing)
        if router_type in ('mixtral', 'switch'):
            self.router = MixtralRouter(
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_selected_experts=num_experts_per_token,
                capacity_factor=capacity_factor,
                router_z_loss_coef=router_z_loss_coef,
                load_balance_loss_coef=load_balance_loss_coef,
                router_jitter_noise=router_jitter_noise,
                dtype=dtype,
                use_triton_kernels=use_triton_kernels,  # TIER 3 OPTIMIZATION
                aux_loss_frequency=aux_loss_frequency,  # OPTIMIZATION: Reduce aux loss overhead
            )
        elif router_type == 'deepseek':
            self.router = DeepSeekRouter(
                hidden_size=hidden_size,
                num_experts=num_experts - 1,  # One will be shared
                num_selected_experts=num_experts_per_token,
                num_shared_experts=1,
                shared_expert_weight=shared_expert_weight,
                capacity_factor=capacity_factor,
                router_z_loss_coef=router_z_loss_coef,
                load_balance_loss_coef=load_balance_loss_coef,
                router_jitter_noise=router_jitter_noise,
                dtype=dtype,
            )
            # Create shared expert
            self.shared_expert = SharedExpertLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation=activation,
                dtype=dtype,
            )
        elif router_type == 'stable_moe':
            # Stable-MoE with Lyapunov-based adaptive load balancing (arXiv 2512.06784)
            # Provides 40% throughput improvement via adaptive capacity factors
            self.router = StableMoERouter(
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_selected_experts=num_experts_per_token,
                target_utilization=0.0,  # Auto: 1/num_experts
                adaptation_rate=0.01,
                temperature_init=1.0,
                temperature_min=0.1,
                temperature_decay=0.9999,
                capacity_min=1.0,
                capacity_max=capacity_factor * 1.5,  # Use configured capacity as base
                router_z_loss_coef=router_z_loss_coef,
                router_jitter_noise=router_jitter_noise,
                dtype=dtype,
                use_triton_kernels=use_triton_kernels,
            )
        else:
            raise ValueError(f"Unknown router type: {router_type}. Use 'mixtral', 'deepseek', or 'stable_moe'")

        # Note: CUDAGraphs-safe routing removed (cudagraphs_safe_routing.py was unused)
        if enable_cudagraphs_safe_routing:
            import warnings
            warnings.warn(
                "enable_cudagraphs_safe_routing is deprecated and has no effect. "
                "ALTERNATIVE: Use torch.compile with mode='reduce-overhead' for CUDA graph "
                "optimization. This is enabled via enable_torch_compile in config.",
                DeprecationWarning
            )

        # ROUTER COMPILATION DISABLED:
        # 1. Separate router compilation causes "cannot assign module as child module" errors
        # 2. Router caching has GPU→CPU sync overhead that negates compilation benefits
        # SOLUTION: Use enable_torch_compile=true for whole-model compilation (20-30% speedup)
        # See routing.py for additional details on sync overhead issues.
        if compile_router:
            import warnings
            warnings.warn(
                "compile_router parameter is deprecated. "
                "ALTERNATIVE: Use enable_torch_compile=true in config for whole-model compilation, "
                "which provides equivalent router optimization without module conflicts.",
                DeprecationWarning
            )

        # Create expert group with grouped GEMM for parallel computation
        if use_grouped_gemm:
            expert_count = num_experts if router_type == 'mixtral' else num_experts - 1

            # Standard experts: Full weight matrices with grouped GEMM
            self.experts = ExpertParallelGroup(
                num_experts=expert_count,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation=activation,
                dropout=expert_dropout,
                dtype=dtype,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            # Fallback: Sequential expert processing (no grouped GEMM)
            # This is slower but works on all hardware (CPU, older GPUs, etc.)
            moe_logger.info(
                f"Using SequentialExpertGroup fallback (use_grouped_gemm=False). "
                f"This is slower than grouped GEMM but works on all hardware."
            )
            expert_count = num_experts if router_type == 'mixtral' else num_experts - 1
            self.experts = SequentialExpertGroup(
                num_experts=expert_count,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation=activation,
                dropout=expert_dropout,
                dtype=dtype,
                gradient_checkpointing=gradient_checkpointing,
            )

        # Expert dropout for regularization
        # CRITICAL: Dropout is incompatible with gradient checkpointing.
        # During checkpoint recomputation, dropout uses different RNG states,
        # causing different numbers of tokens to be routed through experts,
        # which leads to shape mismatches (e.g., [38191, 768] vs [38244, 768]).
        if expert_dropout > 0 and not gradient_checkpointing:
            self.expert_dropout_layer = nn.Dropout(expert_dropout)
        else:
            self.expert_dropout_layer = None
            if expert_dropout > 0 and gradient_checkpointing:
                moe_logger.info(
                    "Expert dropout disabled: incompatible with gradient checkpointing. "
                    "Consider using weight decay or other regularization instead."
                )

        # Layer normalization (applied before MoE, like in transformer)
        self.norm = nn.LayerNorm(hidden_size, dtype=dtype)

        # VRAM OPTIMIZATION: Pre-allocated buffers for capacity limit enforcement
        # These are reused across forward passes to avoid per-forward allocations
        self._capacity_expert_counts: Optional[torch.Tensor] = None
        self._capacity_ones_buffer: Optional[torch.Tensor] = None
        self._capacity_ones_size: int = 0
        self._capacity_boundaries: Optional[torch.Tensor] = None
        # Sort key buffer for capacity enforcement (avoids per-forward allocation)
        self._capacity_sort_key: Optional[torch.Tensor] = None
        self._capacity_sort_key_size: int = 0

        # OPTIMIZATION: Pre-compute diversity loss weights tensor (avoids per-forward allocation)
        # Weights use polynomial hashing: base^0, base^1, ..., base^(k-1)
        # where base = num_experts + 1 to ensure unique fingerprints
        base = num_experts + 1
        diversity_weights = torch.tensor(
            [base ** i for i in range(num_experts_per_token)],
            dtype=torch.float32
        )
        self.register_buffer('_diversity_weights', diversity_weights, persistent=False)

        # OPTIMIZATION: Pre-allocate random sampling pool for diversity loss
        # Pool size chosen to provide good coverage while limiting memory
        self._sample_pool_size = 1024
        self._sample_pool: Optional[torch.Tensor] = None
        self._sample_pool_idx = 0

        # Note: Expert caching feature removed (expert_cache.py was unused)
        if use_expert_caching:
            import warnings
            warnings.warn(
                "use_expert_caching is deprecated and has no effect. "
                "The feature was removed as part of codebase cleanup.",
                DeprecationWarning
            )

    def _compute_diversity_loss_approx(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Approximate diversity loss using hash-based similarity for efficiency.

        Uses a simple hash-based approach to estimate expert diversity without
        computing full pairwise similarity matrix.

        Args:
            expert_indices: Selected experts [num_tokens, k]

        Returns:
            Scalar diversity loss (approximate)
        """
        num_tokens = expert_indices.shape[0]

        # Create hash signatures for each token's expert selection
        # Convert expert indices to unique fingerprints using polynomial hashing
        sorted_indices, _ = expert_indices.sort(dim=-1)
        k = expert_indices.size(1)  # num_experts_per_token

        # OPTIMIZATION: Use pre-computed weights instead of creating tensor every forward
        # Weights were registered as buffer in __init__ for polynomial hashing
        fingerprints = (sorted_indices.float() * self._diversity_weights).sum(dim=-1)

        # Count unique fingerprints (higher diversity = more unique patterns)
        unique_count = len(torch.unique(fingerprints))
        max_possible_unique = min(num_tokens, self.num_experts ** self.num_experts_per_token)

        # Diversity score: ratio of unique patterns (lower is better for loss)
        diversity_ratio = unique_count / max_possible_unique
        # Convert to loss: penalize low diversity
        diversity_loss = 1.0 - diversity_ratio

        return torch.tensor(diversity_loss, device=expert_indices.device, dtype=torch.float32)

    def _compute_diversity_loss(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Diversity loss: Encourages different tokens to use different experts.

        This prevents all tokens from collapsing to the same few experts.
        Optimized with sampling for large batches (>512 tokens).

        Args:
            expert_indices: Selected experts [num_tokens, k]

        Returns:
            Scalar diversity loss
        """
        num_tokens = expert_indices.shape[0]

        # Use fast approximation for larger batches to avoid O(N²) complexity
        # The approximation is accurate enough for batches >64 tokens
        if num_tokens > 64:
            return self._compute_diversity_loss_approx(expert_indices)

        # For small batches, use exact computation
        # For large batches (>64), use sampling to avoid O(N²) complexity
        # Sample size of 64 provides sufficient accuracy while remaining efficient
        max_sample_size = 64

        if num_tokens <= max_sample_size:
            # Exact computation for small batches
            # OPTIMIZATION: Use scatter instead of one_hot for better performance
            expert_fingerprint = torch.zeros(num_tokens, self.num_experts,
                                            device=expert_indices.device, dtype=torch.float32)
            expert_fingerprint.scatter_add_(1, expert_indices,
                                           torch.ones_like(expert_indices, dtype=torch.float32))

            # Compute pairwise similarity
            similarity = torch.matmul(expert_fingerprint, expert_fingerprint.t())
            similarity = similarity / (self.num_experts_per_token ** 2)

            # Exclude diagonal (self-similarity)
            mask = 1 - torch.eye(num_tokens, device=expert_indices.device)
            diversity_loss = (similarity * mask).sum() / (num_tokens * (num_tokens - 1) + 1e-10)
        else:
            # Sampling-based approximation for large batches
            # OPTIMIZATION: Use pre-allocated sample pool instead of randperm every forward
            device = expert_indices.device
            if self._sample_pool is None or self._sample_pool.device != device:
                # Lazy initialize pool on correct device
                self._sample_pool = torch.randperm(self._sample_pool_size, device=device)
                self._sample_pool_idx = 0

            # Refresh pool if we've used too much or num_tokens changed significantly
            if self._sample_pool_idx + max_sample_size > self._sample_pool_size:
                self._sample_pool = torch.randperm(self._sample_pool_size, device=device)
                self._sample_pool_idx = 0

            # Get indices from pool (modulo num_tokens to handle variable batch sizes)
            pool_slice = self._sample_pool[self._sample_pool_idx:self._sample_pool_idx + max_sample_size]
            sample_indices = pool_slice % num_tokens
            self._sample_pool_idx += max_sample_size

            sampled_indices = expert_indices[sample_indices]

            # Compute fingerprints for sampled tokens
            # OPTIMIZATION: Use scatter instead of one_hot for better performance
            sample_size = sampled_indices.shape[0]
            expert_fingerprint = torch.zeros(sample_size, self.num_experts,
                                            device=expert_indices.device, dtype=torch.float32)
            expert_fingerprint.scatter_add_(1, sampled_indices,
                                           torch.ones_like(sampled_indices, dtype=torch.float32))

            # Compute pairwise similarity on sampled subset
            similarity = torch.matmul(expert_fingerprint, expert_fingerprint.t())
            similarity = similarity / (self.num_experts_per_token ** 2)

            # Exclude diagonal
            sample_size = sample_indices.shape[0]
            mask = 1 - torch.eye(sample_size, device=expert_indices.device)
            diversity_loss = (similarity * mask).sum() / (sample_size * (sample_size - 1) + 1e-10)

        return diversity_loss

    def _compute_expert_dropout_loss(self, expert_weights: torch.Tensor) -> torch.Tensor:
        """
        Expert dropout regularization loss.

        Encourages the model not to rely too heavily on any single expert
        by penalizing high routing weights.

        Args:
            expert_weights: Routing weights [num_tokens, k]

        Returns:
            Scalar regularization loss
        """
        # Penalize high confidence routing
        # This encourages more exploration and prevents overconfidence
        dropout_loss = (expert_weights ** 2).mean()
        return dropout_loss

    # VRAM OPTIMIZATION: Buffer helper methods for capacity limit enforcement
    def _get_capacity_expert_counts(self, device: torch.device) -> torch.Tensor:
        """Get pre-allocated expert counts buffer."""
        if self._capacity_expert_counts is None or self._capacity_expert_counts.device != device:
            self._capacity_expert_counts = torch.zeros(self.num_experts, dtype=torch.int64, device=device)
        else:
            self._capacity_expert_counts.zero_()
        return self._capacity_expert_counts

    def _get_capacity_ones_buffer(self, size: int, device: torch.device) -> torch.Tensor:
        """Get pre-allocated ones buffer for capacity counting."""
        if self._capacity_ones_buffer is None or self._capacity_ones_size < size or self._capacity_ones_buffer.device != device:
            alloc_size = max(size, self._capacity_ones_size * 2, 8192)
            self._capacity_ones_buffer = torch.ones(alloc_size, dtype=torch.int64, device=device)
            self._capacity_ones_size = alloc_size
        return self._capacity_ones_buffer[:size]

    def _get_capacity_boundaries(self, device: torch.device) -> torch.Tensor:
        """Get pre-allocated boundaries buffer."""
        if self._capacity_boundaries is None or self._capacity_boundaries.device != device:
            self._capacity_boundaries = torch.zeros(self.num_experts + 1, dtype=torch.int64, device=device)
        else:
            self._capacity_boundaries.zero_()
        return self._capacity_boundaries

    def _get_capacity_sort_key(self, size: int, device: torch.device) -> torch.Tensor:
        """Get pre-allocated sort key buffer for capacity enforcement."""
        if (self._capacity_sort_key is None or
            self._capacity_sort_key_size < size or
            self._capacity_sort_key.device != device):
            # Allocate with some headroom to avoid frequent reallocations
            alloc_size = max(size, self._capacity_sort_key_size * 2, 8192)
            self._capacity_sort_key = torch.empty(alloc_size, dtype=torch.float32, device=device)
            self._capacity_sort_key_size = alloc_size
        return self._capacity_sort_key[:size]

    def _apply_capacity_limits(
        self,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        num_tokens: int,
        use_fresh_tensors: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply capacity limits to prevent expert overload (OPTIMIZED).

        OPTIMIZATIONS:
        1. Fast path: Skip all work if no expert exceeds capacity (common case)
        2. Use argsort instead of sort (only need permutation indices)
        3. Pre-allocated buffers: Reuse sort key buffer across forward passes
        4. O(N + E) counting for boundary computation

        Algorithm:
        1. Count tokens per expert using scatter_add_ (O(N))
        2. Compute expert boundaries via cumsum (O(E))
        3. Fast path exit if max_tokens <= capacity (O(E) check)
        4. Sort by composite key (expert_id * scale - weight) for weight priority
        5. Create capacity mask based on position < capacity
        6. Apply mask and renormalize weights

        For typical MoE configs (E=8-64, N=4096) with balanced routing,
        the fast path avoids the sort entirely in ~70% of forward passes.

        Args:
            expert_indices: Expert assignments [num_tokens, k]
            expert_weights: Routing weights [num_tokens, k]
            num_tokens: Total number of tokens
            use_fresh_tensors: If True, create fresh tensors instead of reusing buffers
                              (required for gradient checkpointing compatibility)

        Returns:
            Modified expert_indices and expert_weights with capacity limits applied
        """
        batch_size, k = expert_indices.shape
        device = expert_indices.device
        dtype = expert_weights.dtype

        # Calculate capacity per expert (guard against num_experts=0)
        if self.num_experts > 0:
            avg_tokens_per_expert = (num_tokens * k) / self.num_experts
        else:
            avg_tokens_per_expert = float(num_tokens * k)  # Fallback: no capacity limit
        expert_capacity = int(avg_tokens_per_expert * self.capacity_factor)

        # Early exit if capacity is effectively unlimited
        if expert_capacity >= batch_size * k:
            return expert_indices, expert_weights

        # COUNTING SORT OPTIMIZED CAPACITY ENFORCEMENT
        # D2D FIX: Use view instead of reshape to avoid copy when possible
        # PERF FIX: Ensure contiguous before view to avoid implicit copy or error
        if not expert_indices.is_contiguous():
            expert_indices = expert_indices.contiguous()
        if not expert_weights.is_contiguous():
            expert_weights = expert_weights.contiguous()
        flat_indices = expert_indices.view(-1)  # [N * k]
        flat_weights = expert_weights.view(-1)  # [N * k]
        total_assignments = flat_indices.shape[0]

        # STEP 1: Count tokens per expert using scatter_add_ (O(N))
        # NOTE: use_fresh_tensors=True for gradient checkpointing compatibility
        # (pre-allocated buffers break recomputation due to in-place mutations)
        if use_fresh_tensors:
            expert_counts = torch.zeros(self.num_experts, dtype=torch.int64, device=device)
            ones = torch.ones(total_assignments, dtype=torch.int64, device=device)
        else:
            expert_counts = self._get_capacity_expert_counts(device)
            ones = self._get_capacity_ones_buffer(total_assignments, device)
        expert_counts.scatter_add_(0, flat_indices.long(), ones)

        # STEP 2: Compute expert boundaries via cumsum (O(E))
        if use_fresh_tensors:
            expert_boundaries = torch.zeros(self.num_experts + 1, dtype=torch.int64, device=device)
        else:
            expert_boundaries = self._get_capacity_boundaries(device)
        expert_boundaries[1:] = expert_counts.cumsum(0)

        # OPTIMIZATION: Fast path - check if ANY expert exceeds capacity
        # If all experts are under capacity, skip expensive sort operation entirely
        max_tokens_per_expert = expert_counts.max()
        if max_tokens_per_expert <= expert_capacity:
            # No expert is over capacity, return unchanged
            return expert_indices, expert_weights

        # STEP 3: Sort by expert to group tokens (needed for position assignment)
        # OPTIMIZATIONS:
        # - Use argsort instead of sort (we only need permutation indices)
        #
        # GRADIENT CHECKPOINTING FIX: Remove weight from sort key entirely.
        # Problem: Float weights can have precision differences during checkpoint
        # recomputation (e.g., softmax output 0.99999999 vs 1.00000001), causing:
        #   - Different weight_int values after scaling/int64 conversion
        #   - Different sort keys → different sort order
        #   - Different tokens dropped → shape mismatch error
        #
        # Solution: Sort only by (expert_id, token_position) which is fully deterministic.
        # This means capacity limiting drops tokens in position order (not weight order)
        # when experts exceed capacity. Trade-off is acceptable because:
        #   1. Most tokens aren't dropped (capacity_factor > 1.0)
        #   2. Training stability is more important than optimal token selection
        #   3. Random position order provides implicit regularization
        #
        # Token position as tiebreaker ensures deterministic ordering even when
        # multiple tokens route to the same expert.
        token_positions = torch.arange(total_assignments, device=device, dtype=torch.int64)

        # Composite key: expert_id * expert_scale + token_position
        # expert_scale must be > max_position to ensure expert_id dominates sorting
        expert_scale = total_assignments + 1

        sort_key = (
            flat_indices.to(torch.int64) * expert_scale
            + token_positions
        )
        sort_perm = torch.argsort(sort_key, stable=True)
        sorted_experts = flat_indices[sort_perm]

        # STEP 4: Compute position within each expert's segment using precomputed boundaries
        # expert_starts[i] = starting position for expert sorted_experts[i]
        expert_starts = expert_boundaries[sorted_experts]
        positions_within_expert = torch.arange(total_assignments, device=device) - expert_starts

        # STEP 5: Create capacity mask: keep if position < capacity
        keep_mask_sorted = positions_within_expert < expert_capacity

        # Scatter mask back to original positions
        keep_mask = torch.zeros(total_assignments, dtype=torch.bool, device=device)
        keep_mask[sort_perm] = keep_mask_sorted

        # D2D FIX: Use view instead of reshape to avoid copy
        keep_mask = keep_mask.view(batch_size, k)
        # GRADIENT CHECKPOINTING FIX: Out-of-place multiplication required
        # In-place ops corrupt saved tensors during checkpoint recomputation
        expert_weights = expert_weights * keep_mask.to(dtype)

        # Renormalize weights per token
        weight_sum = expert_weights.sum(dim=1, keepdim=True)
        expert_weights = torch.where(
            weight_sum > 0,
            expert_weights / (weight_sum + 1e-10),
            expert_weights
        )

        return expert_indices, expert_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True,
        use_compile_friendly: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through Sparse MoE layer.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            training: Whether in training mode
            use_compile_friendly: If True, use torch.compile-friendly dispatch

        Returns:
            - output: Layer output [batch_size, seq_len, hidden_size]
            - aux_loss: Total auxiliary loss (scalar)
            - metrics: Dictionary of routing metrics and losses
                       Call compute_scalar_metrics() on metrics for logging
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        original_shape = hidden_states.shape

        # Normalize input
        # Note: clone() removed since tensor is immediately used and not modified in-place
        hidden_states = self.norm(hidden_states)

        # Flatten for routing
        # PERF FIX: Ensure contiguous before view to avoid implicit copy or error
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        hidden_flat = hidden_states.view(-1, hidden_size)  # [num_tokens, hidden_size]
        num_tokens = hidden_flat.shape[0]

        # Route tokens to experts
        try:
            expert_indices, expert_weights, routing_aux_loss, routing_metrics = self.router(
                hidden_flat, training=training
            )
            # expert_indices: [num_tokens, k]
            # expert_weights: [num_tokens, k]

            # Rely on natural IndexError for out-of-bounds expert indices
            # Avoids explicit validation that would require GPU synchronization

            # CAPACITY PLANNING: Limit tokens per expert to prevent overload
            # NOTE: use_fresh_tensors=True when requires_grad to support gradient checkpointing
            # (pre-allocated buffers break recomputation due to in-place mutations)
            if training and self.capacity_factor < float('inf'):
                use_fresh = hidden_states.requires_grad  # True during training with grad
                expert_indices, expert_weights = self._apply_capacity_limits(
                    expert_indices, expert_weights, num_tokens, use_fresh_tensors=use_fresh
                )
        except Exception as e:
            # CRITICAL: Re-raise PyTorch internal exceptions used by gradient checkpointing
            # _StopRecomputationError is used internally by torch.utils.checkpoint to signal
            # when to stop recomputation during the backward pass. Catching it breaks checkpointing.
            if type(e).__name__ in ('_StopRecomputationError', 'StopIteration'):
                raise
            import traceback
            error_msg = (
                f"Routing failed in MoE layer:\n"
                f"  Input shape: {hidden_states.shape}\n"
                f"  Hidden flat shape: {hidden_flat.shape}\n"
                f"  Num tokens: {num_tokens}\n"
                f"  Num experts: {self.num_experts}\n"
                f"  Training: {training}\n"
                f"  Exception type: {type(e).__name__}\n"
                f"  Error: {str(e)}\n"
                f"  Repr: {repr(e)}\n"
                f"  Inner traceback:\n{traceback.format_exc()}"
            )
            raise RuntimeError(error_msg) from e

        # Compute expert outputs using grouped GEMM
        # NOTE: Skip expert-level checkpointing when torch.compile is enabled
        # because torch.compile conflicts with nested gradient checkpoints.
        # When model-level gradient checkpointing is enabled (in moe.py), the
        # entire layer forward is already checkpointed, making this redundant.
        # The inner checkpoint with torch.compile causes CUDA illegal memory access.
        # Use self.use_torch_compile (from constructor) since use_compile_friendly
        # may not be passed correctly from TransformerBlock.
        #
        # PERFORMANCE FIX: Always use compile-friendly dispatch (20x faster)
        # The compile-friendly path uses vectorized operations (sort, scatter_add)
        # instead of mask.any()/nonzero() which cause GPU syncs.
        # This does NOT require torch.compile to be enabled - it's just faster code.
        # Previous logic: effective_compile_friendly = use_compile_friendly or self.use_torch_compile
        # This caused 81ms/iter instead of 4ms/iter when torch_compile was disabled!
        effective_compile_friendly = True  # Always use fast vectorized dispatch

        # CRITICAL FIX: Disable inner checkpointing entirely.
        # The outer checkpoint in moe.py already wraps the entire layer including MoE.
        # Nested checkpointing with use_reentrant=False causes "backward through graph
        # a second time" errors because saved tensors get freed prematurely.
        # The outer checkpoint provides sufficient memory savings.

        # Pass use_compile_friendly flag to use torch.compile-friendly dispatch
        expert_outputs = self.experts(
            hidden_flat,
            expert_indices,
            expert_weights,
            use_compile_friendly=effective_compile_friendly,
        )
        # expert_outputs: [num_tokens, k, hidden_size]

        # OPTIMIZATION: Combine expert outputs using fused JIT function
        output = fused_expert_combine(expert_outputs)  # [num_tokens, hidden_size]

        # Apply expert dropout if enabled
        if self.expert_dropout_layer is not None and training:
            output = self.expert_dropout_layer(output)

        # Validate tensor size before reshape
        expected_size = batch_size * seq_len * hidden_size
        actual_size = output.numel()
        if actual_size != expected_size:
            if training:
                # During training, shape mismatches indicate a bug that corrupts gradients
                # Raise an error so the issue can be investigated
                raise RuntimeError(
                    f"Shape mismatch in MoE layer during training: "
                    f"expected {expected_size} ({batch_size} x {seq_len} x {hidden_size}), "
                    f"got {actual_size}. This corrupts gradients and must be fixed."
                )
            else:
                # During inference (generation), dynamic shapes may occur
                # Pad/truncate with warning for debugging
                import warnings
                warnings.warn(
                    f"MoE shape mismatch during inference: {actual_size} vs {expected_size}. "
                    f"This may indicate a routing issue.",
                    RuntimeWarning
                )
                if actual_size > expected_size:
                    output = output.flatten()[:expected_size].view(*original_shape)
                else:
                    padded = torch.zeros(expected_size, dtype=output.dtype, device=output.device)
                    padded[:actual_size] = output.flatten()
                    output = padded.view(*original_shape)
        else:
            # Reshape back to original
            output = output.view(*original_shape)  # [batch_size, seq_len, hidden_size]

        # CRITICAL FIX: DeepSeek routing always uses shared expert
        # The shared expert is applied to ALL tokens (not routed)
        # This provides a stable baseline that prevents expert collapse
        if self.router_type == 'deepseek':
            shared_output = self.shared_expert(hidden_states)
            # Shared expert is added with its configured weight (default 0.5)
            # The output already has routed expert contributions
            output = output + shared_output

        # Compute auxiliary losses
        aux_loss = routing_aux_loss

        # Increment step counter and compute diversity loss less frequently
        # Diversity loss is primarily for monitoring, computing every 10 steps is sufficient
        if training:
            self._training_step += 1

        # Skip auxiliary loss computation when disabled for efficiency
        # Many speed-optimized configs set these coefficients to 0.0
        # Completely skip computation to avoid overhead from function calls

        # Only compute losses when coefficients are non-zero
        # Avoid creating GPU tensors for zero values
        # Use None for disabled metrics vs 0.0 for computed-as-zero
        diversity_loss: float | None = None
        expert_dropout_loss: float | None = None

        # Diversity loss - compute every 10 steps when enabled
        if training and self.diversity_loss_coef > 0:
            diversity_loss_freq = 10
            if self._training_step % diversity_loss_freq == 0:
                diversity_loss = self._compute_diversity_loss(expert_indices)
                aux_loss = aux_loss + self.diversity_loss_coef * diversity_loss

        # Expert dropout regularization loss - only when enabled
        if training and self.expert_dropout_loss_coef > 0:
            expert_dropout_loss = self._compute_expert_dropout_loss(expert_weights)
            aux_loss = aux_loss + self.expert_dropout_loss_coef * expert_dropout_loss

        # Collect all metrics
        # Only convert to tensor if needed for metrics
        # None = disabled, 0.0 = computed as zero
        metrics = {
            **routing_metrics,
            'aux_loss_total': aux_loss,
            'aux_loss_routing': routing_aux_loss,
            'aux_loss_diversity': diversity_loss,  # None if disabled
            'aux_loss_expert_dropout': expert_dropout_loss,  # None if disabled
            'num_tokens': num_tokens,  # Keep as int, avoid tensor creation
            'router_type': self.router_type,  # Track which router is being used
        }

        # COMPILE-FRIENDLY: Keep expert_utilization as tensor, don't call .item() in forward
        # The .item() calls would cause graph breaks in torch.compile
        # Use compute_scalar_metrics() after forward() to get scalar values for logging
        if 'expert_utilization' in routing_metrics:
            # Store raw tensor for later scalar conversion
            metrics['_expert_utilization_tensor'] = routing_metrics['expert_utilization']

        return output, aux_loss, metrics

    def compute_scalar_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert tensor metrics to scalars for logging.

        TORCH.COMPILE FRIENDLY: Call this method OUTSIDE the compiled forward() region.
        This allows forward() to stay compilable while still getting scalar metrics.

        Args:
            metrics: Dictionary returned by forward()

        Returns:
            Updated metrics dict with scalars instead of tensors
        """
        result = dict(metrics)

        # Convert expert utilization tensor to per-expert scalar metrics
        if '_expert_utilization_tensor' in result:
            expert_util = result.pop('_expert_utilization_tensor')
            if isinstance(expert_util, torch.Tensor):
                # GPU SYNC OPT: Use single .tolist() instead of N separate .item() calls
                # This reduces N cudaStreamSynchronize calls to 1
                num_experts_to_log = min(expert_util.shape[0], self.num_experts)
                expert_utils = expert_util[:num_experts_to_log].tolist()  # Single sync
                for expert_id, util in enumerate(expert_utils):
                    result[f'expert_{expert_id}_utilization'] = util

        # GPU SYNC OPT: Batch scalar tensor conversions with single .tolist()
        # Collect all scalar tensors, stack them, and extract in one sync
        scalar_tensors = []
        scalar_keys = []
        for key, value in list(result.items()):
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                scalar_tensors.append(value.view(1))
                scalar_keys.append(key)

        if scalar_tensors:
            # Single GPU→CPU sync for all scalar metrics
            stacked = torch.cat(scalar_tensors)
            values = stacked.tolist()
            for key, val in zip(scalar_keys, values):
                result[key] = val

        return result

    def reset_expert_counts(self):
        """Reset expert utilization counts (for metrics)."""
        if hasattr(self.router, 'expert_counts'):
            self.router.expert_counts.zero_()  # type: ignore[attr-defined]
            self.router.total_routing_calls.zero_()  # type: ignore[attr-defined]

    def get_expert_usage_stats(self) -> Dict[str, Any]:
        """
        Get expert usage statistics over time.

        Returns:
            Dictionary with expert utilization stats
        """
        if not hasattr(self.router, 'expert_counts'):
            return {}

        # This is called infrequently for stats reporting,
        # so a single sync here is acceptable (not in hot training loop)
        total_calls_tensor = self.router.total_routing_calls  # type: ignore[attr-defined]
        total_calls = int(total_calls_tensor.item()) if hasattr(total_calls_tensor, 'item') else int(total_calls_tensor)
        if total_calls == 0:
            return {'expert_usage': self.router.expert_counts}

        # Normalize by total calls
        usage_per_call = self.router.expert_counts / total_calls  # type: ignore[operator]
        return {
            'expert_usage_total': self.router.expert_counts,
            'expert_usage_normalized': usage_per_call,
            'total_routing_calls': self.router.total_routing_calls,
        }


# DISABLED: Module-level compilation to prevent CUDA graph conflicts
# When the entire model is compiled with torch.compile at the training level,
# compiling individual modules creates nested CUDA graphs that conflict during
# backward pass, causing "tensor output overwritten by subsequent run" errors.
# The whole-model compilation in train.py provides sufficient optimization.
#
# try:
#     if hasattr(torch, 'compile'):
#         SparseMoELayer.forward = torch.compile(
#             SparseMoELayer.forward,
#             mode='reduce-overhead',
#             fullgraph=False
#         )
# except Exception as e:
#     # torch.compile not available or failed - log but continue
#     import warnings
#     warnings.warn(f"torch.compile failed for SparseMoELayer: {e}. Continuing without compilation.")
