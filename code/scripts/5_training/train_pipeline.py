#!/usr/bin/env python3
"""
Ava Pipeline Training Script - Main entry point for MoE model training.

This is the new pipeline-based training script that uses the modular
Ava component architecture. It replaces the monolithic train_100m_full.py
with a cleaner, more maintainable structure.

Training Pipeline Phases:
    Phase 0:  Dependency checking (optional, quick package verification)
    Phase 1:  Distributed setup (DDP/single GPU, rank/world_size)
    Phase 2:  Configuration loading (YAML → dict, path resolution)
    Phase 3:  RunManager setup (output directories, logging)
    Phase 4:  TrainingContext creation (shared state hub)
    Phase 5:  TrainingPipeline registration (component wiring)
    Phase 6:  Model building (create → optimize → move → wrap DDP)
    Phase 6.1: Batch size calibration (optional, auto-tune for GPU memory)
    Phase 7:  Optimizer & scheduler creation
    Phase 7.1: Checkpoint loading (if resuming from previous run)
    Phase 8:  DataLoader creation (streaming, pre-tokenized, etc.)
    Phase 8.1: Scheduler creation (needs total_steps from DataLoader)
    Phase 9:  Metrics setup (WandB, logging, validation manager)
    Phase 10: Additional components (diagnostics, generation)
    Phase 11: Training loop (epochs, validation, checkpointing)
    Phase 12: Finalization (metrics summary, cleanup)

Key Design Decisions:
    - Phases are ordered for dependency satisfaction (model before optimizer)
    - Cleanup is in reverse order to prevent resource deadlocks
    - Batch calibration is optional and runs before DataLoader creation
    - Distributed barriers synchronize ranks at critical points

Usage:
    python train_pipeline.py --config code/configs/moe/large.yaml

    # Multi-GPU
    torchrun --nproc_per_node=4 train_pipeline.py --config code/configs/moe/4x_a6000_max_speed.yaml
"""

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Default timeout for distributed barriers (30 minutes)
_BARRIER_TIMEOUT = timedelta(minutes=30)

import torch

# Add src to path before importing Ava modules
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[2]  # scripts/5_training -> code -> project root
_src_dir = _project_root / "code" / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Now we can import Ava modules
from ava.core.paths import get_tokenizer_path
from ava.core.logging import (
    Colors, Icons, ColoredFormatter,
    print_header, print_subheader, print_success, print_warning, print_error,
    print_info, print_config, configure_root_logger, print_phase,
    print_epoch_summary, print_checkpoint, print_calibration,
)
from ava.config.training_config import get_mixed_precision

# Ava pipeline imports
from ava.training import (
    TrainingContext,
    DataLoaderManager,
    ModelBuilder,
    OptimizerManager,
    TrainingLoopManager,
    TrainingLoopConfig,
    ValidationManager,
    GenerationManager,
    MetricsManager,
    RunManager,
    TrainingPipeline,
    setup_distributed,
    cleanup_distributed,
)
from ava.training.diagnostics import DiagnosticsManager
from ava.training.episodic_memory import EpisodicMemoryManager
from ava.training.distributed import DistributedStateManager
from ava.config.yaml_loader import load_yaml_with_path_resolution
from ava.config.training_config import DynamicConfig, ModelSelectionConfig, DiagnosticsConfig
# Issue #10 fix: Guard Triton import in case ava.kernels fails to load
try:
    from ava.kernels import KernelConfig, set_kernel_config, TRITON_AVAILABLE
except ImportError as e:
    # Fallback if kernels module fails to load (e.g., Triton not installed)
    logger = logging.getLogger(__name__)
    TRITON_AVAILABLE = False
    KernelConfig = None
    set_kernel_config = lambda x: None

from ava.core.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


def _status_indicator(enabled: bool, impact: Optional[str] = None) -> str:
    """Return a colored YES/NO indicator with optional impact hint."""
    if enabled:
        status = f"{Colors.GREEN}{Colors.BOLD}YES{Colors.RESET}"
        if impact:
            status += f" {Colors.GRAY}({impact}){Colors.RESET}"
        return status
    return f"{Colors.GRAY}NO{Colors.RESET}"


def _format_value(value: Any, unit: str = "", color: str = Colors.CYAN) -> str:
    """Format a configuration value with color."""
    if value is None:
        return f"{Colors.GRAY}default{Colors.RESET}"
    return f"{color}{value}{unit}{Colors.RESET}"


def _format_memory(bytes_val: int) -> str:
    """Format bytes as human-readable memory size."""
    if bytes_val >= 1024**3:
        return f"{bytes_val / 1024**3:.1f}GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / 1024**2:.1f}MB"
    else:
        return f"{bytes_val / 1024:.1f}KB"


def log_optimization_status(config: dict, rank: int = 0) -> None:
    """
    Log detailed status of all optimization features at training start.

    This function displays a comprehensive report of:
    - Model architecture and parameter count
    - Memory optimizations (gradient checkpointing, flash attention, etc.)
    - Training configuration (batch size, learning rate, precision)
    - Data loading optimizations (pre-tokenization, sequence packing)
    - Advanced memory techniques (hybrid caching, overlapped computation)
    - Performance settings (TF32, CuDNN benchmarking)
    - Overall optimization score and recommendations

    Only logs on rank 0 to avoid duplicate output in distributed training.
    """
    if rank != 0:
        return

    model_config = config.get('model', {})
    training_config = config.get('training', {})
    batching_config = training_config.get('batching', {})
    data_config = config.get('data', {})
    perf_config = config.get('performance', {})
    hybrid_config = config.get('hybrid_caching', {})
    overlapped_config = config.get('overlapped_checkpointing', {})
    double_config = config.get('double_checkpointing', {})
    fp8_config = config.get('fp8', {})

    print_header("OPTIMIZATIONS", icon=Icons.GEAR)

    # ═══════════════════════════════════════════════════════════════════
    # Model Architecture Summary
    # Shows: hidden dimensions, layer count, attention heads, MoE experts,
    # vocabulary size, and estimated parameter count. These metrics help
    # understand model capacity and memory requirements.
    # ═══════════════════════════════════════════════════════════════════
    print_subheader(f"{Icons.BRAIN} Model Architecture")
    hidden_size = model_config.get('hidden_size', 768)
    num_layers = model_config.get('num_layers', 12)
    num_heads = model_config.get('num_attention_heads', 12)
    num_experts = model_config.get('num_experts', 8)
    experts_per_tok = model_config.get('num_experts_per_token', 2)
    vocab_size = model_config.get('vocab_size', 50257)
    intermediate_size = model_config.get('intermediate_size', hidden_size * 4)

    print(f"  {Colors.WHITE}hidden_size:{Colors.RESET}           {_format_value(hidden_size)}")
    print(f"  {Colors.WHITE}num_layers:{Colors.RESET}            {_format_value(num_layers)}")
    print(f"  {Colors.WHITE}attention_heads:{Colors.RESET}       {_format_value(num_heads)} {Colors.GRAY}(head_dim={hidden_size // num_heads}){Colors.RESET}")
    print(f"  {Colors.WHITE}intermediate_size:{Colors.RESET}     {_format_value(intermediate_size)}")
    print(f"  {Colors.WHITE}vocab_size:{Colors.RESET}            {_format_value(vocab_size)}")
    print(f"  {Colors.WHITE}MoE experts:{Colors.RESET}           {_format_value(num_experts)} {Colors.GRAY}(top-{experts_per_tok} routing){Colors.RESET}")

    router_type = model_config.get('router_type', 'mixtral')
    activation = model_config.get('activation', 'swiglu')
    print(f"  {Colors.WHITE}router_type:{Colors.RESET}           {_format_value(router_type)}")
    print(f"  {Colors.WHITE}activation:{Colors.RESET}            {_format_value(activation)} {Colors.GRAY}(gated activation){Colors.RESET}")

    # Estimate parameter count
    # Rough estimate: embeddings + layers * (attention + ffn/experts)
    embed_params = vocab_size * hidden_size * 2  # input + output embeddings
    attn_params = num_layers * (4 * hidden_size * hidden_size)  # Q, K, V, O projections
    # FIX: SwiGLU experts have 3 weight matrices (gate_proj + up_proj + down_proj), not 2
    expert_params = num_layers * num_experts * (3 * hidden_size * intermediate_size)  # gate + up + down projections
    total_params = embed_params + attn_params + expert_params
    print(f"  {Colors.WHITE}estimated_params:{Colors.RESET}      {Colors.ORANGE}~{total_params / 1e6:.0f}M{Colors.RESET}")

    # ═══════════════════════════════════════════════════════════════════
    # Model Optimizations (Memory & Speed)
    # These are techniques that reduce memory usage or improve training speed
    # at the model level: flash attention, gradient checkpointing, compiled
    # kernels, and other architectural optimizations.
    # ═══════════════════════════════════════════════════════════════════
    print_subheader(f"{Icons.LIGHTNING} Model Optimizations")

    flash_attn = model_config.get('use_flash_attention', False)
    print(f"  {Colors.WHITE}use_flash_attention:{Colors.RESET}   {_status_indicator(flash_attn, '40% mem savings, 2-4x faster')}")

    grad_ckpt = model_config.get('gradient_checkpointing', False)
    print(f"  {Colors.WHITE}gradient_checkpointing:{Colors.RESET} {_status_indicator(grad_ckpt, '70-80% mem savings')}")

    grouped_gemm = model_config.get('use_grouped_gemm', False)
    print(f"  {Colors.WHITE}use_grouped_gemm:{Colors.RESET}      {_status_indicator(grouped_gemm, '5-10x expert speedup')}")

    triton = model_config.get('use_triton_kernels', False)
    print(f"  {Colors.WHITE}use_triton_kernels:{Colors.RESET}    {_status_indicator(triton, 'fused ops, 10-20% faster')}")

    torch_compile = model_config.get('use_torch_compile', False)
    compile_mode = model_config.get('torch_compile_mode', 'reduce-overhead')
    if torch_compile:
        print(f"  {Colors.WHITE}use_torch_compile:{Colors.RESET}     {_status_indicator(torch_compile, '15-25% after warmup')}")
        print(f"    {Colors.GRAY}└─ mode:{Colors.RESET} {_format_value(compile_mode)}")
    else:
        print(f"  {Colors.WHITE}use_torch_compile:{Colors.RESET}     {_status_indicator(torch_compile)}")

    opt_moe = model_config.get('use_optimized_moe', False)
    print(f"  {Colors.WHITE}use_optimized_moe:{Colors.RESET}     {_status_indicator(opt_moe, 'fused routing, 10-20% faster')}")

    # ═══════════════════════════════════════════════════════════════════
    # Training Configuration
    # Core training hyperparameters: batch size, learning rate, optimizer
    # choice, mixed precision settings (FP16/BF16/FP32), weight decay,
    # and maximum sequence length. These directly impact learning dynamics.
    # ═══════════════════════════════════════════════════════════════════
    print_subheader(f"{Icons.TARGET} Training Configuration")

    batch_size = training_config.get('batch_size', 32)
    grad_accum = training_config.get('gradient_accumulation_steps', 1)
    effective_batch = batch_size * grad_accum
    print(f"  {Colors.WHITE}batch_size:{Colors.RESET}            {_format_value(batch_size)}")
    print(f"  {Colors.WHITE}grad_accumulation:{Colors.RESET}     {_format_value(grad_accum)} {Colors.GRAY}(effective: {effective_batch}){Colors.RESET}")

    lr = training_config.get('learning_rate', 1e-4)
    print(f"  {Colors.WHITE}learning_rate:{Colors.RESET}         {Colors.CYAN}{lr:.2e}{Colors.RESET}")

    optimizer = training_config.get('optimizer', 'adamw')
    weight_decay = training_config.get('weight_decay', 0.01)
    print(f"  {Colors.WHITE}optimizer:{Colors.RESET}             {_format_value(optimizer)} {Colors.GRAY}(wd={weight_decay}){Colors.RESET}")

    mixed_precision = get_mixed_precision(config)
    print(f"  {Colors.WHITE}mixed_precision:{Colors.RESET}       {_format_value(mixed_precision, color=Colors.GREEN if mixed_precision in ['bf16', 'fp16'] else Colors.GRAY)}")

    max_length = data_config.get('max_length', 512)
    print(f"  {Colors.WHITE}max_seq_length:{Colors.RESET}        {_format_value(max_length)}")

    # ═══════════════════════════════════════════════════════════════════
    # Data Loading
    # Shows efficiency improvements in data pipeline: pre-tokenized (Arrow
    # files), sequence packing (reduces padding), lazy file discovery
    # (streaming mode), and number of data loading workers.
    # ═══════════════════════════════════════════════════════════════════
    print_subheader(f"{Icons.DATA} Data Loading")

    pretokenized = data_config.get('use_pretokenized', False)
    print(f"  {Colors.WHITE}use_pretokenized:{Colors.RESET}      {_status_indicator(pretokenized, '60x faster loading')}")

    seq_packing = data_config.get('use_sequence_packing', False)
    packing_strategy = data_config.get('packing_strategy', 'greedy')
    if seq_packing:
        print(f"  {Colors.WHITE}sequence_packing:{Colors.RESET}      {_status_indicator(seq_packing, '20-35% speedup')}")
        print(f"    {Colors.GRAY}└─ strategy:{Colors.RESET} {_format_value(packing_strategy)}")
    else:
        print(f"  {Colors.WHITE}sequence_packing:{Colors.RESET}      {_status_indicator(seq_packing)}")

    lazy_discovery = data_config.get('lazy_file_discovery', False)
    print(f"  {Colors.WHITE}lazy_file_discovery:{Colors.RESET}   {_status_indicator(lazy_discovery, 'memory-efficient for large datasets')}")

    num_workers = data_config.get('num_workers', 4)
    print(f"  {Colors.WHITE}num_workers:{Colors.RESET}           {_format_value(num_workers)}")

    # ═══════════════════════════════════════════════════════════════════
    # Advanced Memory Optimizations
    # Sophisticated techniques for handling very large models and sequences:
    # - Hybrid caching: Mixes KV cache and activation caching
    # - Overlapped checkpointing: Runs backward pass during forward
    # - Double checkpointing: Reduces memory to O(sqrt(N)) for long sequences
    # - FP8 training: Ultra-low precision for H100+ GPUs only
    # ═══════════════════════════════════════════════════════════════════
    print_subheader(f"{Icons.FIRE} Advanced Memory Optimizations")

    hybrid_enabled = hybrid_config.get('enabled', False)
    if hybrid_enabled:
        cache_size = hybrid_config.get('cache_size', '2GB')
        print(f"  {Colors.WHITE}hybrid_caching:{Colors.RESET}        {_status_indicator(hybrid_enabled, '2.19x throughput')}")
        print(f"    {Colors.GRAY}└─ cache_size:{Colors.RESET} {_format_value(cache_size)}")
    else:
        print(f"  {Colors.WHITE}hybrid_caching:{Colors.RESET}        {_status_indicator(hybrid_enabled)}")

    overlapped_enabled = overlapped_config.get('enabled', False)
    print(f"  {Colors.WHITE}overlapped_ckpt:{Colors.RESET}       {_status_indicator(overlapped_enabled, '10-20% speedup, parallel backward')}")

    double_enabled = double_config.get('enabled', False)
    print(f"  {Colors.WHITE}double_checkpointing:{Colors.RESET}  {_status_indicator(double_enabled, 'O(sqrt(n)) memory, 10x longer seqs')}")

    fp8_enabled = fp8_config.get('enabled', False)
    print(f"  {Colors.WHITE}fp8_training:{Colors.RESET}          {_status_indicator(fp8_enabled, '2x memory savings, requires H100+')}")

    # ═══════════════════════════════════════════════════════════════════
    # Performance Backend Settings
    # Low-level GPU/CUDA optimizations that can boost performance:
    # - Torch compile: JIT compiles model graph (15-25% speedup after warmup)
    # - TF32: Fast reduced precision on Ampere+ GPUs
    # - CuDNN benchmark: Auto-tunes convolution algorithms
    # ═══════════════════════════════════════════════════════════════════
    print_subheader(f"{Icons.ROCKET} Performance Backend")

    perf_compile = perf_config.get('enable_torch_compile', False)
    print(f"  {Colors.WHITE}torch_compile:{Colors.RESET}         {_status_indicator(perf_compile, '15-25% speedup')}")

    tf32 = perf_config.get('enable_tf32', False)
    print(f"  {Colors.WHITE}enable_tf32:{Colors.RESET}           {_status_indicator(tf32, '3x faster matmuls on Ampere+')}")

    cudnn_bench = perf_config.get('enable_cudnn_benchmark', False)
    print(f"  {Colors.WHITE}cudnn_benchmark:{Colors.RESET}       {_status_indicator(cudnn_bench, 'auto-tune convolutions')}")

    # ═══════════════════════════════════════════════════════════════════
    # Summary with Recommendations
    # ═══════════════════════════════════════════════════════════════════
    print_subheader(f"{Icons.STAR} Optimization Summary")

    # Count enabled optimizations
    enabled_count = sum([
        flash_attn, grad_ckpt, grouped_gemm, triton, torch_compile, opt_moe,
        pretokenized, seq_packing, hybrid_enabled, overlapped_enabled,
        double_enabled, fp8_enabled, tf32, cudnn_bench
    ])
    total_opts = 14

    if enabled_count >= 10:
        score_color = Colors.GREEN
        score_text = "Highly optimized"
    elif enabled_count >= 5:
        score_color = Colors.YELLOW
        score_text = "Moderately optimized"
    else:
        score_color = Colors.RED
        score_text = "Minimal optimization"

    print(f"  {Colors.WHITE}Enabled optimizations:{Colors.RESET} {score_color}{enabled_count}/{total_opts}{Colors.RESET} {Colors.GRAY}({score_text}){Colors.RESET}")

    # Memory impact estimate
    mem_savings = []
    if flash_attn:
        mem_savings.append("40%")
    if grad_ckpt:
        mem_savings.append("70-80%")
    if mixed_precision in ['fp16', 'bf16']:
        mem_savings.append("50%")
    if mem_savings:
        print(f"  {Colors.WHITE}Est. memory savings:{Colors.RESET}  {Colors.GREEN}{' + '.join(mem_savings)}{Colors.RESET}")

    # Recommendations
    recommendations = []
    if not flash_attn:
        recommendations.append("Enable flash_attention for 40% memory savings")
    if not grad_ckpt and total_params > 100e6:
        recommendations.append("Enable gradient_checkpointing for large models")
    if not tf32 and torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 8:
                recommendations.append("Enable TF32 for 3x faster matmuls on Ampere+")
        except Exception:
            pass
    if not pretokenized:
        recommendations.append("Use pretokenized data for 60x faster loading")

    if recommendations:
        print(f"  {Colors.YELLOW}{Icons.WARNING} Recommendations:{Colors.RESET}")
        for rec in recommendations[:3]:  # Show top 3 recommendations
            print(f"    {Colors.GRAY}*{Colors.RESET} {rec}")

    print()  # Extra newline at end


def setup_logging(log_dir: Path, rank: int = 0) -> logging.Logger:
    """Setup logging for training with colorful output."""
    logger = logging.getLogger('train_pipeline')
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers.clear()  # Clear existing handlers

    if rank == 0:
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ColoredFormatter(show_level=False, show_icons=True))
        logger.addHandler(console_handler)

        # File handler (plain format)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'training.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s - %(message)s')
        )
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def select_best_gpu() -> int:
    """
    Select the best available GPU for single-GPU training.

    Selection criteria (in order):
    1. Most free memory
    2. Highest compute capability (newer GPU)
    3. Lowest GPU index as tiebreaker

    Returns:
        GPU index to use (0 if no GPU available or on error)
    """
    if not torch.cuda.is_available():
        return 0

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return 0
    if num_gpus == 1:
        return 0

    best_gpu = 0
    best_score = -1

    print_subheader(f"{Icons.GPU} GPU Selection - Found {num_gpus} GPUs")

    for i in range(num_gpus):
        try:
            props = torch.cuda.get_device_properties(i)
            # Get free memory
            torch.cuda.set_device(i)
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            free_gb = free_mem / (1024**3)
            total_gb = total_mem / (1024**3)

            # Compute capability as a tie-breaker
            compute_cap = props.major * 10 + props.minor

            # Score: prioritize free memory (in GB), then compute capability
            score = free_gb * 100 + compute_cap

            print(f"  {Colors.CYAN}GPU {i}:{Colors.RESET} {Colors.WHITE}{props.name}{Colors.RESET}")
            print(f"         {Colors.GRAY}Memory:{Colors.RESET} {Colors.ORANGE}{free_gb:.1f}GB{Colors.RESET} free / {total_gb:.1f}GB total")
            print(f"         {Colors.GRAY}Compute:{Colors.RESET} {props.major}.{props.minor}, {Colors.GRAY}Score:{Colors.RESET} {Colors.LIME}{score:.1f}{Colors.RESET}")

            if score > best_score:
                best_score = score
                best_gpu = i

        except Exception as e:
            print_error(f"GPU {i}: Error querying - {e}")
            continue

    print_success(f"Selected GPU {best_gpu}")
    return best_gpu


def create_calibration_dataloader(
    config: dict,
    tokenizer: Any,
    batch_size: int,
    device: torch.device,
    rank: int = 0,
    logger_instance: Optional[logging.Logger] = None
) -> Optional[Any]:
    """
    Create a minimal DataLoader for batch size calibration.

    This loader includes real data loading overhead (workers, pinned memory,
    prefetch) but uses minimal resources for fast calibration.

    Args:
        config: Full training config dict
        tokenizer: Tokenizer (may be None for pretokenized data)
        batch_size: Batch size for calibration
        device: Target device
        rank: Process rank
        logger_instance: Logger to use (optional)

    Returns:
        DataLoader or None if creation fails
    """
    log = logger_instance or logger

    try:
        from ava.data.pretokenized import create_ultra_fast_dataloaders

        data_config = config.get('data', {})

        # Extract data configuration parameters
        data_dir = data_config.get('data_dir')
        max_length = data_config.get('max_length', 512)

        if not data_dir:
            if rank == 0:
                log.warning("No data_dir specified, cannot create calibration DataLoader")
            return None

        # Get model config for token IDs
        model_config = config.get('model', {})

        # Create minimal DataLoader with calibration settings
        train_loader, _ = create_ultra_fast_dataloaders(
            batch_size=batch_size,
            max_length=max_length,
            data_dir=data_dir,
            num_workers=2,  # Minimal workers (vs 4 in training) for faster calibration
            prefetch_factor=2,  # Standard prefetch to capture overhead
            persistent_workers=False,  # Don't keep alive (faster cleanup)
            lazy_file_discovery=data_config.get('lazy_file_discovery', True),
            cache_size=data_config.get('cache_size', 20),
            max_files_to_load=5,  # Just load a few files for calibration
            verbose=False,  # Quiet output for calibration
            pad_token_id=model_config.get('pad_token_id', 0),
            eos_token_id=model_config.get('eos_token_id', 2),
        )

        if rank == 0:
            try:
                loader_len = len(train_loader)
                log.info(f"Created calibration DataLoader: {loader_len} batches")
            except TypeError:
                log.info("Created calibration DataLoader (streaming, no fixed length)")

        return train_loader

    except Exception as e:
        if rank == 0:
            log.warning(f"Failed to create calibration DataLoader: {e}")
            log.warning("Falling back to synthetic batches")
        return None


def main(args: argparse.Namespace) -> None:
    """
    Main training function using the Ava pipeline architecture.

    This function orchestrates the complete training pipeline in phases:
    0. Dependency checking (optional)
    1. Distributed setup (DDP/single GPU)
    2. Configuration loading and validation
    3. RunManager setup (output directory structure)
    4. TrainingContext creation (shared state hub)
    5. TrainingPipeline registration (component wiring)
    6. Model building (creation + optimizations + device movement)
    7. Batch size calibration (optional auto-tuning)
    8. DataLoader creation and scheduler setup
    9. Metrics and validation manager setup
    10. Training loop execution
    11. Training complete (finalization)
    """

    # =========================================================================
    # Phase 0: Dependency Check
    # Verifies all required packages are available (transformers, torch, etc.)
    # =========================================================================
    if not getattr(args, 'skip_dep_check', False):
        # Import here to avoid circular imports and keep startup fast when skipped
        try:
            from check_dependencies import quick_check
            quick_check(warn_only=True)
        except ImportError:
            # check_dependencies.py not in path, try relative import
            import importlib.util
            dep_check_path = Path(__file__).parent.parent / "check_dependencies.py"
            if dep_check_path.exists():
                spec = importlib.util.spec_from_file_location("check_dependencies", dep_check_path)
                dep_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(dep_module)
                dep_module.quick_check(warn_only=True)

    # Configure root logger early to prevent duplicate log messages
    # This sets up proper formatting and prevents module loggers from propagating duplicates
    configure_root_logger(level=logging.WARNING)

    # =========================================================================
    # Phase 1: Distributed Setup
    # Initialize DDP for multi-GPU or single GPU training.
    # Sets up rank (process ID) and world_size (total processes).
    # =========================================================================
    rank, world_size = setup_distributed()

    # For single-GPU training, select the best available GPU
    if world_size == 1 and torch.cuda.is_available():
        best_gpu = select_best_gpu()
        torch.cuda.set_device(best_gpu)
        device = torch.device(f'cuda:{best_gpu}')
    else:
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # =========================================================================
    # Phase 2: Load Configuration
    # Load YAML config and extract key sections (model, training, data).
    # YAML paths are resolved relative to config file location.
    # =========================================================================
    try:
        config = load_yaml_with_path_resolution(args.config)
    except FileNotFoundError:
        print_error(f"Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        sys.exit(1)

    # Extract key config sections
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    data_config = config.get('data', {})
    kernel_opt_config = config.get('kernel_optimization', {})

    # =========================================================================
    # Phase 2.1: Configure Kernel Optimizations
    # =========================================================================
    if kernel_opt_config and TRITON_AVAILABLE:
        kernel_config = KernelConfig(
            use_bitonic_topk=kernel_opt_config.get('use_fused_softmax_topk', True),
            use_heap_topk=kernel_opt_config.get('use_fused_softmax_topk', True),
            router_block_size=kernel_opt_config.get('router_block_size', 4),
            use_fused_softmax_topk=kernel_opt_config.get('use_fused_softmax_topk', True),
        )
        set_kernel_config(kernel_config)
        if rank == 0:
            print_subheader(f"{Icons.LIGHTNING} Triton Kernel Optimization")
            print_config("Fused softmax+topk", str(kernel_config.use_fused_softmax_topk))
            print_config("Router block size", str(kernel_config.router_block_size))
            print_config("Fused activations", str(kernel_opt_config.get('use_fused_activations', True)))
            print_config("Vectorized capacity", str(kernel_opt_config.get('use_vectorized_capacity', True)))
    elif rank == 0:
        if not TRITON_AVAILABLE:
            print_warning("Triton not available, using PyTorch fallbacks")
        else:
            print_info("No kernel_optimization config found, using defaults")

    # =========================================================================
    # Phase 2.2: Log Optimization Status (visibility into what's enabled)
    # =========================================================================
    log_optimization_status(config, rank)

    # Override with CLI args
    num_epochs = args.epochs or training_config.get('num_epochs', 3)
    # batch_size can be at training.batch_size OR training.batching.batch_size
    batching_config = training_config.get('batching', {})

    # Check if calibration is enabled BEFORE setting batch_size
    # CLI batch_size should only take precedence if calibration is disabled
    batch_size_calibration_config = config.get('batch_size_calibration', {})
    calibration_will_run = batch_size_calibration_config.get('enabled', False)

    if args.batch_size is not None and not calibration_will_run:
        # CLI override only when calibration is disabled
        batch_size = args.batch_size
        if rank == 0:
            print_warning(f"Using CLI batch_size={batch_size} (calibration disabled)")
    else:
        # Use config value; calibration will override later if enabled
        batch_size = batching_config.get('batch_size') or training_config.get('batch_size', 8)

    # Read learning rate from training.optimizer.learning_rate (preferred) or training.learning_rate
    optimizer_config = training_config.get('optimizer', {})
    # Handle optimizer config as dict or string
    if isinstance(optimizer_config, dict):
        opt_lr = optimizer_config.get('learning_rate')
    else:
        opt_lr = None

    learning_rate = (
        args.learning_rate or
        opt_lr or
        training_config.get('learning_rate') or
        5e-5
    )

    # Validate learning rate type and value
    if not isinstance(learning_rate, (int, float)):
        raise ValueError(
            f"learning_rate must be numeric, got {type(learning_rate).__name__}: {learning_rate}"
        )
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

    # Read log_interval from multiple sources (CLI > top-level logging > training.logging > defaults)
    # Priority: CLI arg > logging.log_interval > training.logging.logging_steps > training.log_interval > default
    top_level_logging = config.get('logging', {})
    training_logging = training_config.get('logging', {})
    log_interval = (
        args.log_interval or
        top_level_logging.get('log_interval') or
        training_logging.get('logging_steps') or
        training_config.get('log_interval', 100)
    )
    val_interval = args.val_interval

    # Get output directory from config or CLI args
    output_config = config.get('output', {})
    output_dir = output_config.get('output_dir', args.save_dir)

    # =========================================================================
    # Phase 3: Create RunManager (Ava's output organizer)
    # RunManager handles output directory structure:
    # - Run folder: {output_dir}/pretraining/run_YYYYMMDD_HHMMSS_ID/
    # - Checkpoints: run_folder/checkpoints/
    # - Logs: run_folder/logs/
    # - Config copies: run_folder/*.yaml
    # =========================================================================
    run_manager = RunManager(
        base_output_dir=str(output_dir),
        run_name=config.get('experiment_name', 'ava_training'),
        description=f"Training with config: {args.config}"
    )

    # Save the actual configuration files to the run directory
    if rank == 0:
        # Save full config
        run_manager.save_config(config, 'full')
        # Save individual sections for easy reference
        if model_config:
            run_manager.save_config(model_config, 'model')
        if training_config:
            run_manager.save_config(training_config, 'training')
        if data_config:
            run_manager.save_config(data_config, 'data')
        # Save command line args
        run_manager.save_args(args)

    # Setup logging
    log_dir = run_manager.run_dir / 'logs' if hasattr(run_manager, 'run_dir') else Path(args.log_dir)
    train_logger = setup_logging(log_dir, rank)

    # =========================================================================
    # Phase 3.1: Create CheckpointManager
    # =========================================================================
    checkpoint_dir = run_manager.run_dir / 'checkpoints' if hasattr(run_manager, 'run_dir') else Path(args.save_dir) / 'checkpoints'
    checkpoint_manager = CheckpointManager(
        save_dir=checkpoint_dir,
        max_keep=training_config.get('max_checkpoints', 3),
        config=config,
        async_save=True
    )

    # Resume state tracking
    resume_epoch = 0
    resume_step = 0

    # =========================================================================
    # Phase 3.2: Load Checkpoint if Resuming
    # =========================================================================
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            if rank == 0:
                train_logger.info(f"Will resume from checkpoint: {resume_path}")
            # We'll load the checkpoint after model/optimizer are created
        else:
            # Issue #16 fix: Add fail option for missing resume checkpoint
            require_resume = getattr(args, 'require_resume', False) or \
                            config.get('training', {}).get('require_resume', False)

            if require_resume:
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {resume_path} "
                    f"(--require-resume set, not starting fresh)"
                )

            if rank == 0:
                train_logger.warning(f"Resume checkpoint not found: {resume_path}, starting fresh")
            args.resume = None

    max_steps = getattr(args, 'max_steps', None)

    # Check if batch size calibration is enabled
    batch_size_calibration_config = config.get('batch_size_calibration', {})
    calibration_enabled = batch_size_calibration_config.get('enabled', False)

    if rank == 0:
        print_header("AVA PIPELINE TRAINING", icon=Icons.ROCKET)
        print_config("Config", str(args.config))
        print_config("Device", str(device))
        print_config("World size", str(world_size))
        print_config("Epochs", str(num_epochs))
        # Only show batch size if calibration is disabled (enabled shows it after calibration)
        if not calibration_enabled:
            print_config("Batch size", f"{batch_size} (fixed)")
        print_config("Learning rate", f"{learning_rate:.2e}")
        if max_steps:
            print_config("Max steps", str(max_steps))

    # =========================================================================
    # Phase 4: Create TrainingContext (Ava's shared state hub)
    # TrainingContext is a central object that all managers access for:
    # - Model and optimizer references
    # - Device, rank, world_size for distributed training
    # - Config and RunManager for output/logging
    # - Metadata dict for storing runtime state (resume_step, etc.)
    # =========================================================================
    context = TrainingContext(
        model=None,  # Will be set by ModelBuilder
        device=device,
        config=config,
        run_manager=run_manager,
        rank=rank,
        world_size=world_size,
        is_main_process=(rank == 0),
    )
    context.update_from_config(config)

    # =========================================================================
    # Phase 5: Create TrainingPipeline and Register Components
    # The TrainingPipeline acts as a central component manager that:
    # - Registers all manager components (model, optimizer, data, etc.)
    # - Calls lifecycle hooks (on_epoch_start, on_epoch_end, etc.)
    # - Handles component errors and cleanup
    # Components are retrieved with: pipeline.get('model'), etc.
    # =========================================================================
    pipeline = TrainingPipeline(context)

    # Register all components
    pipeline.register('model', ModelBuilder(context))
    pipeline.register('optimizer', OptimizerManager(context))
    pipeline.register('data', DataLoaderManager(context))
    pipeline.register('training', TrainingLoopManager(context))
    pipeline.register('validation', ValidationManager(context))
    pipeline.register('generation', GenerationManager(context))
    pipeline.register('metrics', MetricsManager(context))

    try:
        # =====================================================================
        # Phase 6: Build Model
        # Builds MoE model with full pipeline:
        # 1. Create model (EnhancedMoEModel)
        # 2. Apply quantization if enabled (8-bit, 4-bit, etc.)
        # 3. Apply pre-device optimizations (torch.compile on CPU)
        # 4. Move to GPU
        # 5. Apply post-device optimizations (FP8, gradient checkpointing)
        # 6. Wrap in DDP for multi-GPU
        # =====================================================================
        model_builder = pipeline.get('model')
        model_builder.initialize()

        # Configure fail-fast behavior from config
        fail_fast = config.get('model_building', {}).get('fail_on_optimization_error', True)
        model_builder.set_fail_on_optimization_error(fail_fast)

        model = model_builder.build_model(config, device)

        # Apply quantization if enabled (before device move)
        quant_config = config.get('quantization', {})
        if quant_config.get('enabled', False):
            model = model_builder.apply_quantization(model, quant_config)

        # Apply pre-device optimizations (torch.compile - works better on CPU)
        model = model_builder.apply_pre_device_optimizations(model, config)

        # Move to device
        model = model_builder.move_to_device(model, device)

        # Apply post-device optimizations (FP8, hybrid caching, checkpointing)
        model = model_builder.apply_post_device_optimizations(model, config)

        # Wrap in DDP for distributed
        model = model_builder.wrap_distributed(model, rank, world_size)

        context.model = model

        # =====================================================================
        # Phase 6.1: Batch Size Calibration (Independent from Dynamic Batching)
        # Auto-tunes optimal batch size to maximize GPU utilization while
        # staying within memory limits. Features:
        # - Real DataLoader or synthetic batches (full-length sequences)
        # - Binary search between min and max batch size
        # - Target memory % (default 70%)
        # - Synchronizes across all GPUs (uses MIN in multi-GPU)
        # =====================================================================
        calibration_config = config.get('batch_size_calibration', {})
        calibration_enabled = calibration_config.get('enabled', False)

        # Auto-disable calibration when DeepSpeed is enabled (incompatible with DeepSpeed optimizer)
        deepspeed_config = config.get('deepspeed', {})
        if calibration_enabled and deepspeed_config.get('enabled', False):
            if rank == 0:
                print_warning("Batch size calibration disabled (incompatible with DeepSpeed)")
                print_info("Using batch size from config: deepspeed.micro_batch_size")
            calibration_enabled = False

        # Issue #9 fix: Initialize optimal_batch_size before calibration block
        # to avoid fragile 'in locals()' check later
        optimal_batch_size = None

        if calibration_enabled:  # All ranks participate in calibration for DDP compatibility
            # Synchronize all ranks before starting calibration
            if world_size > 1:
                import torch.distributed as dist
                dist.barrier()

            try:
                from ava.optimizations.batch_controller import BatchSizeController
                # Configure batch_controller logger to show detailed calibration progress (ALL RANKS)
                batch_controller_logger = logging.getLogger('ava.optimizations.batch_controller')
                batch_controller_logger.setLevel(logging.INFO)

                # Add handler from train_logger if available
                if train_logger.handlers:
                    batch_controller_logger.addHandler(train_logger.handlers[0])

                if rank == 0:
                    print_calibration("Running batch size calibration with REAL OPTIMIZER...")

                # ═══════════════════════════════════════════════════════════════
                # AUTO-FETCH: Pull calibration settings from main config sections
                # ═══════════════════════════════════════════════════════════════

                # Fetch from training.batching or use calibration overrides
                batching_cfg = config.get('training', {}).get('batching', {})

                # min_batch_size: calibration override or default 1
                min_bs = calibration_config.get('min_batch_size', 1)

                # Issue #15 fix: Make multiplier configurable instead of hardcoded 4
                # max_batch_size: calibration override or training.batch_size * multiplier
                max_batch_multiplier = calibration_config.get('max_batch_multiplier', 4)
                default_max = batching_cfg.get('batch_size', 32) * max_batch_multiplier
                max_bs = calibration_config.get('max_batch_size', default_max)

                # target_memory: calibration override or 0.70
                target_mem = calibration_config.get('target_memory', 0.70)

                # Validate calibration config values
                if min_bs >= max_bs:
                    raise ValueError(
                        f"Calibration min_batch_size ({min_bs}) must be < max_batch_size ({max_bs}). "
                        f"Check training.batch_size_calibration config."
                    )
                if not 0.0 < target_mem <= 1.0:
                    raise ValueError(
                        f"Calibration target_memory must be in (0.0, 1.0], got {target_mem}"
                    )

                if rank == 0:
                    train_logger.info(f"  Calibration settings (auto-fetched from config):")
                    train_logger.info(f"    min_batch_size: {min_bs}, max_batch_size: {max_bs}, target_memory: {target_mem}")

                # Create controller for calibration
                calib_headroom = calibration_config.get('calibration_headroom', 0.90)
                controller = BatchSizeController(
                    min_batch_size=min_bs,
                    max_batch_size=max_bs,
                    target_memory=target_mem,
                    calibration_headroom=calib_headroom,
                )

                # Check if using real DataLoader for calibration
                use_real_dataloader = calibration_config.get('use_real_dataloader', False)
                calib_loader = None
                calib_iter = None  # Initialize early for safe cleanup
                tokenizer_calib = None

                if use_real_dataloader:
                    if rank == 0:
                        train_logger.info("Attempting to create REAL DataLoader for calibration...")

                    # Try to load tokenizer early for DataLoader
                    try:
                        from transformers import AutoTokenizer, PreTrainedTokenizerFast
                        tokenizer_path = (
                            data_config.get('tokenizer_path') or
                            data_config.get('tokenizer_name') or
                            str(get_tokenizer_path())
                        )
                        # Handle local tokenizer.json files
                        tokenizer_path_obj = Path(tokenizer_path)
                        if tokenizer_path_obj.exists() and tokenizer_path_obj.suffix == '.json':
                            tokenizer_calib = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path_obj))
                            if tokenizer_calib.pad_token is None:
                                tokenizer_calib.pad_token = tokenizer_calib.eos_token or '[PAD]'
                        elif tokenizer_path_obj.is_dir() and (tokenizer_path_obj / 'tokenizer.json').exists():
                            tokenizer_calib = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path_obj / 'tokenizer.json'))
                            if tokenizer_calib.pad_token is None:
                                tokenizer_calib.pad_token = tokenizer_calib.eos_token or '[PAD]'
                        else:
                            tokenizer_calib = AutoTokenizer.from_pretrained(tokenizer_path)
                        if rank == 0:
                            train_logger.info(f"Loaded tokenizer for calibration: {tokenizer_path}")
                    except Exception as e:
                        if rank == 0:
                            train_logger.warning(f"Failed to load tokenizer for calibration: {e}")
                            train_logger.warning("Falling back to synthetic batches")
                        use_real_dataloader = False

                    # Create calibration DataLoader if tokenizer loaded successfully
                    if use_real_dataloader:
                        calib_loader = create_calibration_dataloader(
                            config=config,
                            tokenizer=tokenizer_calib,
                            batch_size=calibration_config.get('min_batch_size', 8),
                            device=device,
                            rank=rank,
                            logger_instance=train_logger if rank == 0 else None
                        )

                        if calib_loader is None:
                            if rank == 0:
                                train_logger.warning("Failed to create calibration DataLoader, using synthetic batches")
                            use_real_dataloader = False

                # ═══════════════════════════════════════════════════════════════
                # BATCH SAMPLING STRATEGY
                # Calibration needs a function to produce batches of variable sizes.
                # Two strategies are available:
                #
                # 1. REAL DataLoader (use_real_dataloader=True):
                #    - Uses actual training data from disk/network
                #    - More accurate: includes I/O overhead and realistic tensor shapes
                #    - Slower: requires tokenizer and dataset setup
                #    - May fail if data is unavailable
                #
                # 2. SYNTHETIC Batches (fallback):
                #    - Generates random tensors matching max_length
                #    - Faster: no I/O, no tokenizer needed
                #    - Less accurate: doesn't reflect real data distribution
                #    - Always works: no external dependencies
                #
                # The sample_batch_fn() is called by BatchSizeController during
                # binary search to test if a given batch size fits in GPU memory.
                # ═══════════════════════════════════════════════════════════════
                if use_real_dataloader and calib_loader is not None:
                    if rank == 0:
                        train_logger.info("Using REAL DataLoader for calibration (includes I/O overhead)")

                    # Create iterator from DataLoader
                    calib_iter = iter(calib_loader)

                    # Retry counter to prevent infinite recursion on empty/broken DataLoader
                    # If DataLoader fails 3 times, we fall back to synthetic batches
                    _sample_retry_count = [0]  # Use list to allow mutation in closure
                    _MAX_SAMPLE_RETRIES = 3

                    def sample_batch_fn(bs: int) -> Dict[str, torch.Tensor]:
                        """Get real batch from DataLoader by collecting and concatenating batches."""
                        nonlocal calib_iter, calib_loader

                        try:
                            # Get first batch
                            batch = next(calib_iter)
                            _sample_retry_count[0] = 0  # Reset on successful batch retrieval

                            # Move to device if not already there
                            if isinstance(batch, dict):
                                batch = {
                                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                                    for k, v in batch.items()
                                }
                            else:
                                batch = batch.to(device) if isinstance(batch, torch.Tensor) else batch

                            # Handle batch size mismatch by collecting multiple batches
                            actual_bs = batch['input_ids'].shape[0] if isinstance(batch, dict) else batch.shape[0]

                            if actual_bs < bs:
                                # Need to collect more batches to reach target size
                                batches_needed = (bs + actual_bs - 1) // actual_bs
                                batches = [batch]

                                # Collect additional batches
                                for _ in range(batches_needed - 1):
                                    try:
                                        next_batch = next(calib_iter)
                                        if isinstance(next_batch, dict):
                                            next_batch = {
                                                k: v.to(device) if isinstance(v, torch.Tensor) else v
                                                for k, v in next_batch.items()
                                            }
                                        else:
                                            next_batch = next_batch.to(device) if isinstance(next_batch, torch.Tensor) else next_batch
                                        batches.append(next_batch)
                                    except StopIteration:
                                        # Restart iterator if we run out
                                        calib_iter = iter(calib_loader)
                                        next_batch = next(calib_iter)
                                        if isinstance(next_batch, dict):
                                            next_batch = {
                                                k: v.to(device) if isinstance(v, torch.Tensor) else v
                                                for k, v in next_batch.items()
                                            }
                                        batches.append(next_batch)

                                # Concatenate batches
                                if isinstance(batch, dict):
                                    batch = {
                                        k: torch.cat([b[k] for b in batches], dim=0)[:bs]
                                        if isinstance(batches[0][k], torch.Tensor) else batches[0][k]
                                        for k in batch.keys()
                                    }
                                else:
                                    batch = torch.cat(batches, dim=0)[:bs]

                                # Explicit cleanup to prevent memory accumulation during calibration
                                del batches

                            elif actual_bs > bs:
                                # Truncate batch
                                if isinstance(batch, dict):
                                    batch = {
                                        k: v[:bs] if isinstance(v, torch.Tensor) else v
                                        for k, v in batch.items()
                                    }
                                else:
                                    batch = batch[:bs]

                            return batch

                        except StopIteration:
                            # Restart iterator with retry limit to prevent infinite recursion
                            _sample_retry_count[0] += 1
                            if _sample_retry_count[0] > _MAX_SAMPLE_RETRIES:
                                raise RuntimeError(
                                    f"Calibration DataLoader exhausted after {_MAX_SAMPLE_RETRIES} retries. "
                                    f"DataLoader may be empty or have insufficient samples for batch size {bs}."
                                )
                            calib_iter = iter(calib_loader)
                            return sample_batch_fn(bs)

                else:
                    # Fallback: Synthetic batches (current behavior)
                    if rank == 0:
                        train_logger.info("Using synthetic batches for calibration")

                    def sample_batch_fn(bs: int):
                        """Create a sample batch for calibration using WORST-CASE memory.

                        Uses full-length sequences (100% max_length) to ensure calibration
                        finds a batch size that works for ALL real training data, not just
                        shorter sequences.

                        Token distribution is still realistic (skewed toward common tokens).
                        """
                        # Use data.max_length as default if base_sequence_length not explicitly set
                        data_max_length = config.get('data', {}).get('max_length', 512)
                        seq_len = calibration_config.get('base_sequence_length', data_max_length)
                        vocab_size = config.get('model', {}).get('vocab_size', 50680)

                        # WORST-CASE: Use full-length sequences (all attention = 1)
                        # This ensures calibrated BS works for longest sequences in real data
                        attention_mask = torch.ones(bs, seq_len, device=device)

                        # Generate input_ids with realistic token distribution
                        # Use squared distribution (common tokens more frequent, like real tokenizers)
                        uniform_rand = torch.rand(bs, seq_len, device=device)
                        skewed_rand = torch.pow(uniform_rand, 2.0)  # Square for right-skew
                        input_ids = (skewed_rand * vocab_size).long().clamp(0, vocab_size - 1)

                        return {
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                        }

                # Create optimizer factory for calibration
                # AUTO-FETCH: Uses same optimizer settings as training.optimizer
                temp_opt_mgr = OptimizerManager(context)
                temp_opt_mgr.initialize()

                training_config = config.get('training', {})
                optimizer_config = training_config.get('optimizer', {})

                # Fetch all settings from training.optimizer (automatic - no hardcoding)
                optimizer_type = optimizer_config.get('type', 'adamw')
                learning_rate = optimizer_config.get('learning_rate', 1e-4)
                weight_decay = optimizer_config.get('weight_decay', 0.01)

                if rank == 0:
                    train_logger.info(f"  Calibration using training optimizer: {optimizer_type}")

                # Check if using fresh models for each calibration test
                use_fresh_models = calibration_config.get('use_fresh_models', True)

                optimizer_factory = temp_opt_mgr.create_optimizer_factory(
                    model=model,
                    config=config,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    support_model_arg=use_fresh_models,  # Enable model arg when using fresh models
                )
                if rank == 0:
                    train_logger.info("Created optimizer factory for calibration")

                # Create model_factory for fresh models per calibration test
                model_factory = None
                if use_fresh_models:
                    if rank == 0:
                        train_logger.info("Using FRESH MODEL per calibration test (accurate memory measurement)")

                    def model_factory():
                        """Create a fresh model instance for calibration."""
                        fresh_model = model_builder.build_model(config, device)

                        # Apply quantization if enabled
                        quant_config_local = config.get('quantization', {})
                        if quant_config_local.get('enabled', False):
                            fresh_model = model_builder.apply_quantization(fresh_model, quant_config_local)

                        # Apply optimizations (but NOT torch.compile for calibration - too slow)
                        # Apply only critical optimizations
                        optim_config = config.copy()
                        if 'performance' in optim_config:
                            optim_config['performance'] = optim_config['performance'].copy()
                            optim_config['performance']['enable_torch_compile'] = False

                        # Apply pre-device optimizations (gradient checkpointing, etc.)
                        fresh_model = model_builder.apply_pre_device_optimizations(fresh_model, optim_config)

                        # Move to device
                        fresh_model = model_builder.move_to_device(fresh_model, device)

                        # Apply post-device optimizations (torch.compile, etc.)
                        fresh_model = model_builder.apply_post_device_optimizations(fresh_model, optim_config)

                        # NOTE: Don't wrap in DDP for calibration - each test is independent
                        return fresh_model

                # Run calibration with optimizer (all ranks participate)
                optimal_batch_size = controller.startup_calibration(
                    model=model,
                    sample_batch_fn=sample_batch_fn,
                    optimizer_factory=optimizer_factory,
                    perform_optimizer_step=calibration_config.get('perform_optimizer_step', True),
                    max_time_seconds=calibration_config.get('calibration_timeout_sec', 90.0),  # More time for fresh models
                    rank=rank,
                    world_size=world_size,
                    model_factory=model_factory,  # Fresh model per test if enabled
                )

                # Only rank 0 logs calibration result (don't update config until after sync)
                if rank == 0:
                    print_success(f"Calibration complete: optimal batch size = {optimal_batch_size}")

                # Synchronize batch size across all ranks using DistributedStateManager
                # This implements a proper two-phase commit pattern to prevent race conditions
                dist_manager = DistributedStateManager(rank=rank, world_size=world_size, timeout_seconds=1800)

                if world_size > 1:
                    import torch.distributed as dist

                    # Issue #17 fix: Synchronize CUDA before memory measurement
                    torch.cuda.synchronize()

                    # Gather VRAM info from all ranks to detect mismatches
                    total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                    vram_sizes = dist_manager.all_gather_values(total_memory_gb, "vram_gb")

                    # Check for VRAM mismatch and warn user (only on rank 0)
                    if rank == 0:
                        vram_sizes_rounded = [round(v, 1) for v in vram_sizes]
                        unique_vram = set(vram_sizes_rounded)
                        if len(unique_vram) > 1:
                            train_logger.warning(
                                f"Mixed VRAM detected across {world_size} GPUs: {vram_sizes_rounded} GB\n"
                                f"   Using MINIMUM batch size to prevent OOM on smaller GPU(s)"
                            )
                        else:
                            train_logger.info(f"Synchronizing optimal batch size across {world_size} identical GPUs ({vram_sizes_rounded[0]} GB each)...")

                # Synchronize batch size using two-phase commit (works for single or multi-GPU)
                # DistributedStateManager handles single GPU case automatically
                if world_size > 1:
                    synced_batch_size = dist_manager.synchronized_update(
                        value=optimal_batch_size,
                        reduction_op=dist.ReduceOp.MIN,
                        validate_fn=lambda x: x > 0,
                        value_name="optimal_batch_size"
                    )
                else:
                    # Single GPU - no synchronization needed, but still validate
                    if optimal_batch_size <= 0:
                        raise RuntimeError(f"Invalid batch size: {optimal_batch_size}")
                    synced_batch_size = optimal_batch_size

                # Now update config with synchronized value using context manager
                # This ensures all ranks update config atomically
                with dist_manager.config_update_context("batch_size"):
                    if 'training' not in config:
                        config['training'] = {}
                    if 'batching' not in config['training']:
                        config['training']['batching'] = {}
                    config['training']['batching']['batch_size'] = synced_batch_size
                    config['training']['batch_size'] = synced_batch_size
                    batch_size = synced_batch_size

                if rank == 0:
                    print_config("Batch size", f"{optimal_batch_size} (auto-calibrated)")

                # CRITICAL: Store controller in context for OOM handling during training
                context.batch_controller = controller
                if rank == 0:
                    train_logger.info(f"[OK] BatchSizeController stored in context for OOM recovery")

                # Cleanup calibration DataLoader if it was created
                if use_real_dataloader and calib_loader is not None:
                    try:
                        # Shutdown workers to free memory
                        # Issue #4 fix: Check method existence before calling (PyTorch version compat)
                        if hasattr(calib_loader, '_iterator') and calib_loader._iterator is not None:
                            try:
                                if hasattr(calib_loader._iterator, '_shutdown_workers'):
                                    calib_loader._iterator._shutdown_workers()
                                    if rank == 0:
                                        train_logger.info("Cleaned up calibration DataLoader workers")
                                else:
                                    # Fallback: delete iterator to trigger cleanup
                                    del calib_loader._iterator
                                    if rank == 0:
                                        train_logger.debug("DataLoader iterator deleted (no _shutdown_workers)")
                            except Exception:
                                pass
                        # Delete references to free memory
                        del calib_loader
                        if calib_iter is not None:
                            del calib_iter
                        if tokenizer_calib is not None:
                            del tokenizer_calib
                        # Force garbage collection
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if rank == 0:
                            train_logger.info("Freed calibration DataLoader memory")
                    except Exception as e:
                        if rank == 0:
                            train_logger.warning(f"Failed to cleanup calibration DataLoader: {e}")

            except ImportError as e:
                train_logger.warning(f"Could not import BatchSizeController: {e}")
                # Issue #6 fix: Reset optimal_batch_size on import failure
                optimal_batch_size = None
            except Exception as e:
                # Issue #6 fix: Proper error handling with fail-fast option
                fail_on_calibration_error = calibration_config.get('fail_on_error', False)

                train_logger.error(f"Batch size calibration failed: {e}")
                import traceback
                train_logger.error(f"Full traceback:\n{traceback.format_exc()}")

                if fail_on_calibration_error:
                    raise RuntimeError(f"Batch size calibration failed (fail_on_error=True): {e}") from e

                # Reset optimal_batch_size so verification check passes
                optimal_batch_size = None

                if rank == 0:
                    train_logger.warning("Training will continue with configured batch_size")

        # =====================================================================
        # Phase 7: Create Optimizer (scheduler created after dataloader)
        # Creates optimizer (AdamW, SGD, etc.) with settings from config.
        # Learning rate comes from CLI > config.training.optimizer.learning_rate.
        # Scheduler is created later after we know total_steps from dataloader.
        # =====================================================================
        optimizer_mgr = pipeline.get('optimizer')
        optimizer_mgr.initialize()

        optimizer = optimizer_mgr.create_optimizer(
            model, config, learning_rate,
            weight_decay=training_config.get('weight_decay', 0.01)
        )

        context.optimizer = optimizer

        # =====================================================================
        # Phase 7.1: Load Checkpoint if Resuming
        # Restores model and optimizer state from checkpoint.
        # Returns resume_epoch and resume_step to continue from there.
        # =====================================================================
        if args.resume:
            resume_path = Path(args.resume)
            if rank == 0:
                print_checkpoint("Loading checkpoint", path=str(resume_path))

            resume_epoch, resume_step = checkpoint_manager.load(
                model=model,
                optimizer=optimizer,
                checkpoint_path=resume_path
            )

            if rank == 0:
                print_success(f"Resumed from epoch {resume_epoch}, step {resume_step}")

        # =====================================================================
        # Phase 8: Create DataLoaders
        # Creates train and validation dataloaders with:
        # - Tokenizer loading (multiple fallback paths)
        # - Pre-tokenized or streaming datasets
        # - Sequence packing (optional)
        # - Multiple workers for parallel loading
        # =====================================================================
        data_mgr = pipeline.get('data')
        data_mgr.initialize()

        # Load tokenizer
        tokenizer = None
        try:
            from transformers import AutoTokenizer, PreTrainedTokenizerFast
            # Try multiple possible config locations for tokenizer path
            tokenizer_path = (
                data_config.get('tokenizer_path') or
                data_config.get('tokenizer_name') or
                config.get('data', {}).get('tokenizer_name') or
                str(get_tokenizer_path())  # Use utility function for default path
            )
            # Handle local tokenizer.json files
            tokenizer_path_obj = Path(tokenizer_path)
            if tokenizer_path_obj.exists() and tokenizer_path_obj.suffix == '.json':
                # Load from local tokenizer.json file
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path_obj))
                # Set special tokens if not defined
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
            elif tokenizer_path_obj.is_dir() and (tokenizer_path_obj / 'tokenizer.json').exists():
                # Load from directory containing tokenizer.json
                tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path_obj / 'tokenizer.json'))
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token or '[PAD]'
            else:
                # Try as HuggingFace model ID
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            context.tokenizer = tokenizer
            if rank == 0:
                train_logger.info(f"Loaded tokenizer: {tokenizer_path} (vocab_size={len(tokenizer)})")
        except Exception as e:
            if rank == 0:
                train_logger.warning(f"Could not load tokenizer: {e}")

        # Disable generation if no tokenizer available
        # Extract generation config from training.generation (nested) or fallback to raw config
        generation_config = training_config.get('generation', {})
        if not generation_config:
            generation_config = config.get('training', {}).get('generation', {})
        if generation_config and tokenizer is None:
            if rank == 0:
                train_logger.warning(
                    f"Generation disabled: no tokenizer available. "
                    f"Ignoring settings: {list(generation_config.keys())}"
                )
            generation_config = {}

        # Convert config dict to DynamicConfig for DataLoaderManager
        config_obj = DynamicConfig(config) if isinstance(config, dict) else config

        train_loader, val_loader = data_mgr.create_dataloaders(
            training_config=config_obj,
            tokenizer=tokenizer,
            config_dict=config,
            batch_size=batch_size
        )

        # Issue #11 fix: Synchronize all ranks after dataloader creation
        # Ensures all ranks have their data ready before proceeding to training
        if world_size > 1:
            import torch.distributed as dist
            dist.barrier()
            if rank == 0:
                train_logger.debug("All ranks synchronized after dataloader creation")

        # Validate calibration was applied correctly
        if rank == 0:
            train_logger.info(f"Dataloader created with batch_size={batch_size}")

            # Verify calibration was applied if it was enabled
            # Issue #9 fix: Use 'is not None' instead of fragile 'in locals()' check
            calibration_enabled = config.get('batch_size_calibration', {}).get('enabled', False)
            if calibration_enabled and optimal_batch_size is not None:
                if batch_size != optimal_batch_size:
                    train_logger.warning(
                        f"WARNING: Calibrated batch size ({optimal_batch_size}) "
                        f"does not match dataloader batch size ({batch_size})!"
                    )
                else:
                    train_logger.info(
                        f"[OK] Calibrated batch size successfully applied to dataloaders"
                    )

        # Update total steps now that we have the data loader
        # Note: IterableDatasets (like InfiniteUltraFastDataset) don't have __len__
        # Try to get length safely, falling back to config value or default
        try:
            loader_len = len(train_loader)
        except TypeError:
            # Infinite/streaming datasets don't have length
            loader_len = config.get('training', {}).get('steps_per_epoch', 1000)

        # Validate gradient_accumulation_steps to prevent division by zero
        if context.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"Invalid gradient_accumulation_steps: {context.gradient_accumulation_steps}. "
                f"Must be >= 1. Check your config at 'training.gradient_accumulation_steps' or "
                f"'training.batching.gradient_accumulation_steps'."
            )
        # FIX: Use integer division ceiling to ensure scheduler gets correct total_steps
        # This prevents off-by-one errors with cosine schedulers and ensures we don't truncate
        total_steps = (num_epochs * loader_len + context.gradient_accumulation_steps - 1) // context.gradient_accumulation_steps

        if rank == 0:
            train_logger.info(f"Total training steps: {total_steps}")

        # =====================================================================
        # Phase 8.1: Create Scheduler (now that we know total_steps)
        # Creates learning rate scheduler with:
        # - Scheduler type (cosine, linear, etc.)
        # - Warmup steps (linear ramp from 0 to LR)
        # - Total steps (calculated from epochs * loader_len / grad_accum)
        # - Minimum LR floor (for cosine annealing)
        # =====================================================================
        # Schedule config can be at training.schedule or training level
        schedule_config = training_config.get('schedule', {})
        warmup_steps = schedule_config.get('warmup_steps', training_config.get('warmup_steps', 1000))
        min_lr = schedule_config.get('min_lr', training_config.get('min_lr', 0.0))
        scheduler_type = schedule_config.get('scheduler_type', 'cosine')
        num_cycles = schedule_config.get('num_cycles', 1)

        scheduler = optimizer_mgr.create_scheduler(
            optimizer, warmup_steps, total_steps,
            min_lr=min_lr,
            scheduler_type=scheduler_type,
            num_cycles=num_cycles,
        )
        context.scheduler = scheduler

        if rank == 0:
            train_logger.info(f"Scheduler: type={scheduler_type}, warmup={warmup_steps}, total={total_steps}, min_lr={min_lr:.2e}")

        # =====================================================================
        # Phase 9: Setup Metrics
        # Initializes metrics tracking:
        # - WandB integration (if enabled)
        # - Local logging (CSV, JSON)
        # - WandB directory inside run folder
        # =====================================================================
        metrics_mgr = pipeline.get('metrics')
        metrics_mgr.initialize()

        # WandB config is under logging.wandb in the config schema
        logging_config = config.get('logging', {})
        wandb_config = logging_config.get('wandb', {})
        # Set wandb directory inside the run folder
        wandb_dir = run_manager.run_dir / 'wandb'
        wandb_enabled = wandb_config.get('enabled', False)
        train_logger.info(f"WandB config: enabled={wandb_enabled}, project={wandb_config.get('project', 'N/A')}")

        # Check if all logging is disabled
        if logging_config.get('disabled', False):
            train_logger.info("All logging DISABLED via config for maximum training speed")
        metrics_mgr.setup(
            log_dir=log_dir,
            wandb_config=wandb_config if wandb_enabled else None,
            use_wandb=wandb_enabled,
            wandb_dir=wandb_dir,
            logging_config=logging_config,
        )

        # =====================================================================
        # Phase 10: Initialize Remaining Components
        # Sets up validation manager, generation manager, and diagnostics:
        # - Quality scoring (multi-metric model selection)
        # - Generation config (sampling parameters)
        # - Diagnostics (per-layer gradients, routing analysis, etc.)
        # =====================================================================
        validation_mgr = pipeline.get('validation')
        validation_mgr.initialize()

        # Setup multi-metric quality scoring for model selection
        model_selection_config = config.get('model_selection', {})
        if model_selection_config.get('enabled', True):
            ms_config = ModelSelectionConfig(
                enabled=model_selection_config.get('enabled', True),
                val_loss_weight=model_selection_config.get('val_loss_weight', 0.5),
                coherence_score_weight=model_selection_config.get('coherence_score_weight', 0.3),
                perplexity_weight=model_selection_config.get('perplexity_weight', 0.2),
                perplexity_cap=model_selection_config.get('perplexity_cap', 100.0),
                val_loss_cap=model_selection_config.get('val_loss_cap', 10.0),
            )
            validation_mgr.setup_quality_scoring(ms_config)
            train_logger.info(
                f"Quality scoring enabled: val_loss={ms_config.val_loss_weight:.0%}, "
                f"coherence={ms_config.coherence_score_weight:.0%}, "
                f"perplexity={ms_config.perplexity_weight:.0%}"
            )

        generation_mgr = pipeline.get('generation')
        generation_mgr.initialize()
        generation_mgr.set_log_dir(log_dir)
        if generation_config:
            generation_mgr.set_generation_config(generation_config)

        # Setup diagnostics manager for detailed training insights
        diagnostics_mgr = None
        diagnostics_config = config.get('diagnostics', {})
        if diagnostics_config.get('enabled', False):
            diag_config = DiagnosticsConfig(
                enabled=True,
                enable_per_layer_gradients=diagnostics_config.get('enable_per_layer_gradients', False),
                per_layer_log_freq=diagnostics_config.get('per_layer_log_freq', 500),
                enable_routing_diagnostics=diagnostics_config.get('enable_routing_diagnostics', False),
                routing_log_freq=diagnostics_config.get('routing_log_freq', 100),
                enable_memory_breakdown=diagnostics_config.get('enable_memory_breakdown', False),
                memory_log_freq=diagnostics_config.get('memory_log_freq', 500),
                enable_timing_profiling=diagnostics_config.get('enable_timing_profiling', False),
                timing_log_freq=diagnostics_config.get('timing_log_freq', 100),
            )
            diagnostics_mgr = DiagnosticsManager(context)
            diagnostics_mgr.initialize()
            diagnostics_mgr.configure(diag_config)
            logger.info("Diagnostics manager initialized for detailed training insights")

        # Setup episodic memory manager for experience replay
        episodic_memory_mgr = None
        episodic_config = config.get('experimental', {}).get('episodic_memory', {})
        # Also check top-level episodic_memory for convenience
        if not episodic_config:
            episodic_config = config.get('episodic_memory', {})

        if episodic_config.get('use_episodic_memory', False) or episodic_config.get('enabled', False):
            # Create a config object that EpisodicMemoryManager expects
            class EpisodicMemConfig:
                use_episodic_memory = True
                memory_capacity = episodic_config.get('memory_capacity', 10000)
                memory_replay_ratio = episodic_config.get('memory_replay_ratio', 0.2)
                buffer_warmup_steps = episodic_config.get('buffer_warmup_steps', 100)
                memory_selection_strategy = episodic_config.get('memory_selection_strategy', 'importance')
                priority_exponent = episodic_config.get('priority_exponent', 0.6)
                importance_weight_exponent = episodic_config.get('importance_weight_exponent', 0.4)
                store_aux_info = episodic_config.get('store_aux_info', False)
                silent_mode = episodic_config.get('silent_mode', False)

            episodic_memory_mgr = EpisodicMemoryManager(EpisodicMemConfig(), device=device)
            if rank == 0:
                train_logger.info(
                    f"Episodic memory enabled: capacity={EpisodicMemConfig.memory_capacity}, "
                    f"replay_ratio={EpisodicMemConfig.memory_replay_ratio}, "
                    f"warmup_steps={EpisodicMemConfig.buffer_warmup_steps}"
                )

        training_mgr = pipeline.get('training')
        training_mgr.initialize()
        training_mgr.set_components(
            metrics_manager=metrics_mgr,
            generation_manager=generation_mgr,
            checkpoint_manager=checkpoint_manager,
            diagnostics_manager=diagnostics_mgr,
            episodic_memory_manager=episodic_memory_mgr,
        )

        # Set up overlapped gradient sync for multi-GPU DDP (10-30% speedup)
        is_deepspeed = config.get('deepspeed', {}).get('enabled', False)
        training_mgr.setup_gradient_sync(model, is_deepspeed=is_deepspeed)

        # Issue #2 fix: Set resume info in context.metadata ONLY (single source of truth)
        # TrainingLoopManager reads from context.metadata['resume_step'] in train_epoch()
        # Do NOT set training_mgr._global_step directly to avoid dual sources of truth
        if resume_step > 0:
            context.metadata['resume_step'] = resume_step
            context.metadata['resume_epoch'] = resume_epoch
            if rank == 0:
                train_logger.info(f"Resume info stored in context: step={resume_step}, epoch={resume_epoch}")

        # Determine profile directory - use run folder if not explicitly specified
        profile_dir = getattr(args, 'profile_dir', None)
        if profile_dir is None or profile_dir == './profiles':
            # Default to run folder/profiles
            profile_dir = str(run_manager.run_dir / 'profiles') if hasattr(run_manager, 'run_dir') else './profiles'

        # Create training loop config
        # Logging options: 'tqdm' = clean progress bar only, 'verbose' = both tqdm + INFO logs
        # Read from top-level logging config first, then fall back to training config
        log_mode = top_level_logging.get('log_mode') or training_config.get('log_mode', 'tqdm')
        verbose_log_interval = (
            top_level_logging.get('verbose_log_interval') or
            training_config.get('verbose_log_interval', 500)
        )
        tqdm_update_interval = top_level_logging.get('tqdm_update_interval', 10)

        # CUDA Graphs config - 15-25% throughput improvement
        cuda_graphs_config = config.get('cuda_graphs', {})
        use_cuda_graphs = cuda_graphs_config.get('enabled', False)
        cuda_graph_warmup_steps = cuda_graphs_config.get('warmup_steps', 10)

        if use_cuda_graphs and rank == 0:
            logger.info(f"CUDA Graphs ENABLED - will capture after {cuda_graph_warmup_steps} warmup steps")
            logger.info("  Note: Requires fixed batch sizes. Disable if you see shape mismatch errors.")

        loop_config = TrainingLoopConfig(
            gradient_accumulation_steps=context.gradient_accumulation_steps,
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            use_amp=context.use_amp,
            amp_dtype=context.amp_dtype,
            log_interval=log_interval,
            generate_every_n_steps=generation_config.get('generate_every_n_steps', 500),
            save_steps=training_config.get('logging', {}).get('save_steps', 500),
            max_steps=getattr(args, 'max_steps', None),
            # Profiling options (disabled by default)
            enable_profiling=getattr(args, 'enable_profiling', False),
            profile_start_step=getattr(args, 'profile_start_step', 0),
            profile_end_step=getattr(args, 'profile_end_step', 999999),
            profile_dir=profile_dir,
            # Logging options
            log_mode=log_mode,
            verbose_log_interval=verbose_log_interval,
            # CUDA Graphs - 15-25% speedup with fixed batch sizes
            use_cuda_graphs=use_cuda_graphs,
            cuda_graph_warmup_steps=cuda_graph_warmup_steps,
        )

        # Setup profiler if enabled
        if loop_config.enable_profiling and rank == 0:
            training_mgr.setup_profiler(loop_config)

        # =====================================================================
        # Phase 11: Training Loop
        # Main training loop that:
        # - Iterates through epochs (respecting resume_epoch)
        # - Calls train_epoch() for each epoch
        # - Optionally validates (every val_interval epochs)
        # - Saves checkpoints and best models
        # - Checks max_steps termination condition
        # - Notifies components via on_epoch_start/end hooks
        # =====================================================================
        if rank == 0:
            print_phase(11, "Training Loop")

        best_val_loss = float('inf')

        # Issue #1 fix: Define coherence_config at training loop scope
        # Previously only defined inline in train_epoch call, but referenced in validation block
        coherence_config = training_config.get('coherence', config.get('coherence', {}))

        # Issue #13 fix: Log warning if validation dataloader is missing
        if val_loader is None and rank == 0:
            train_logger.warning(
                "Validation dataloader is None - validation will be skipped for all epochs. "
                "Check data.val_split_ratio in config if this is unexpected."
            )

        for epoch in range(resume_epoch, num_epochs):
            # Notify components of epoch start
            pipeline.on_epoch_start(epoch)

            # Train one epoch
            train_loss = training_mgr.train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=loop_config,
                vocab_size=model_config.get('vocab_size', 50680),
                tokenizer=tokenizer,
                generation_config=generation_config,
                coherence_config=coherence_config,  # Use pre-defined variable
            )

            if rank == 0:
                print_epoch_summary(epoch, num_epochs, train_loss)

            # Check if max_steps reached
            if training_mgr.reached_max_steps(loop_config):
                if rank == 0:
                    train_logger.info(f"Reached max_steps ({loop_config.max_steps}), stopping training")
                break

            # ═══════════════════════════════════════════════════════════════════
            # VALIDATION & MULTI-METRIC MODEL SELECTION
            # Runs every val_interval epochs to evaluate model quality.
            #
            # Quality Score Formula (when model_selection.enabled=true):
            #   quality = (val_loss_weight * normalized_val_loss) +
            #             (coherence_weight * coherence_score) +
            #             (perplexity_weight * normalized_perplexity)
            #
            # Where normalized values are capped to prevent outliers from
            # dominating: val_loss_cap=10.0, perplexity_cap=100.0
            #
            # Best model selection: Lower quality score = better model
            # (not just lowest val_loss, but balanced across all metrics)
            # ═══════════════════════════════════════════════════════════════════
            if val_loader is not None and (epoch + 1) % val_interval == 0:
                val_loss = validation_mgr.validate(
                    model=model,
                    val_loader=val_loader,
                    use_amp=context.use_amp,
                    amp_dtype=context.amp_dtype,
                )

                metrics_mgr.log_validation(
                    step=training_mgr.get_global_step(),
                    epoch=epoch,
                    val_loss=val_loss
                )

                # Get coherence metrics for multi-metric model selection
                # coherence_metrics is None when: (a) coherence disabled, (b) method missing, (c) error
                coherence_metrics = None
                if coherence_config and coherence_config.get('enabled', False):
                    # Issue #20 fix: Verify measure_coherence method exists
                    if hasattr(generation_mgr, 'measure_coherence'):
                        try:
                            coherence_metrics = generation_mgr.measure_coherence(
                                model, training_mgr.get_global_step(), coherence_config
                            )
                        except Exception as e:
                            train_logger.warning(f"Coherence measurement failed: {e}")
                    else:
                        train_logger.debug("GenerationManager does not have measure_coherence method")

                current_step = training_mgr.get_global_step()
                if validation_mgr.is_best(
                    val_loss,
                    coherence_metrics=coherence_metrics,
                    step=current_step,
                    epoch=epoch,
                ):
                    best_val_loss = val_loss
                    quality_score = validation_mgr.get_best_quality_score()

                    if rank == 0:
                        if quality_score:
                            train_logger.info(
                                f"New best model! Quality={quality_score.quality_score:.4f} "
                                f"(val_loss={val_loss:.4f})"
                            )
                            # Log quality score to metrics
                            metrics_mgr.log_quality_score(current_step, quality_score)
                        else:
                            train_logger.info(f"New best validation loss: {val_loss:.4f}")

                        # Save best checkpoint with quality score (with error handling)
                        try:
                            run_manager.save_checkpoint(
                                model_state=model.state_dict(),
                                optimizer_state=optimizer.state_dict(),
                                epoch=epoch,
                                step=current_step,
                                loss=val_loss,
                                is_best=True,
                                additional_data={
                                    'train_loss': train_loss,
                                    'quality_score': quality_score.to_dict() if quality_score else None,
                                }
                            )
                        except Exception as e:
                            train_logger.error(f"Failed to save best checkpoint: {e}")
                            train_logger.warning("Training will continue, but best model may not be saved")

                # Synchronize all ranks after validation to prevent desync
                if world_size > 1:
                    import torch.distributed as dist
                    dist.barrier()

            # Notify components of epoch end
            pipeline.on_epoch_end(epoch)

        # =====================================================================
        # Phase 12: Finalize
        # Final cleanup after training:
        # - Log best validation loss
        # - Save metrics summary (JSON)
        # - Log generation history table
        # - Stop profiler and report location
        # - Mark run as complete
        # =====================================================================
        if rank == 0:
            print_phase(12, "Training Complete")
            print_success(f"Best validation loss: {best_val_loss:.4f}")

            # Save final metrics
            metrics_mgr.save_summary(log_dir / 'metrics_summary.json')

            # Log final generations table
            generations = generation_mgr.get_generation_history()
            if generations:
                metrics_mgr.log_generation_table(generations)

            # Stop profiler and log output location
            profile_dir = training_mgr.stop_profiler()
            if profile_dir:
                train_logger.info(f"Profile data saved to: {profile_dir}")

        # Finish run
        run_manager.finish_run(status='completed')

    except Exception as e:
        train_logger.error(f"Training failed: {e}", exc_info=True)
        pipeline.on_error(e)
        # Issue #14 fix: Include more error context in final metrics
        run_manager.finish_run(
            status='failed',
            final_metrics={
                'error': str(e),
                'error_type': type(e).__name__,
                'error_module': type(e).__module__,
            }
        )
        raise  # Re-raises with full traceback preserved

    finally:
        # CRITICAL: Cleanup order matters to prevent deadlocks!
        # 1. DataLoaders FIRST (workers may be using distributed primitives)
        # 2. Checkpoints SECOND (flush pending async saves)
        # 3. Other CUDA resources THIRD
        # 4. Distributed process group LAST

        train_logger.info("Starting cleanup sequence...")

        # Step 1: Cleanup DataLoaders BEFORE distributed cleanup
        # This prevents workers from being blocked on dist.barrier() when group is destroyed
        try:
            train_logger.info("Cleaning up pipeline components (including dataloaders)...")
            pipeline.cleanup_all()
        except Exception as e:
            train_logger.error(f"Error during pipeline cleanup: {e}")

        # Step 2: Flush and shutdown checkpoint manager
        # Wait for all pending async checkpoint saves to complete
        try:
            train_logger.info("Shutting down checkpoint manager...")
            checkpoint_manager.shutdown()
        except Exception as e:
            train_logger.error(f"Error during checkpoint shutdown: {e}")

        # Step 3: Cleanup optional global CUDA resources
        try:
            from ava.cuda.metrics import shutdown_async_logger
            shutdown_async_logger()
        except Exception:
            pass

        try:
            from ava.cuda.streams import clear_buffer_pool
            clear_buffer_pool()
        except Exception:
            pass

        # Step 4: Sync CUDA before distributed cleanup (prevents NCCL errors)
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        # Step 5: Distributed cleanup LAST (after all workers and CUDA ops complete)
        try:
            train_logger.info("Cleaning up distributed process group...")
            cleanup_distributed(rank, world_size)
        except Exception as e:
            train_logger.error(f"Error during distributed cleanup: {e}")

        train_logger.info("Cleanup complete.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Ava Pipeline Training Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config file'
    )

    # Training hyperparameters (override config)
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--max-steps', type=int, default=None, help='Maximum training steps (overrides epochs)')

    # Directories - default to project outputs directory
    default_output_dir = str(Path(__file__).resolve().parent.parent.parent / 'outputs')
    parser.add_argument('--save-dir', type=str, default=default_output_dir, help='Base output directory (runs saved to {save-dir}/pretraining/)')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')

    # Logging
    parser.add_argument('--log-interval', type=int, default=None, help='Steps between log messages')
    parser.add_argument('--val-interval', type=int, default=1, help='Epochs between validation')

    # Resume
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    # Issue #16 fix: Add --require-resume flag
    parser.add_argument('--require-resume', action='store_true',
                       help='Fail if resume checkpoint is missing (instead of starting fresh)')

    # Profiling options for Nsight Systems/Compute
    parser.add_argument('--enable-profiling', action='store_true',
                       help='Enable Nsight-compatible GPU profiling')
    parser.add_argument('--profile-dir', type=str, default='./profiles',
                       help='Directory to save profiling outputs')
    parser.add_argument('--profile-start-step', type=int, default=0,
                       help='Step to start profiling (default: 0, profiles all)')
    parser.add_argument('--profile-end-step', type=int, default=999999,
                       help='Step to end profiling (default: 999999, profiles all)')

    # Dependency checking
    parser.add_argument('--skip-dep-check', action='store_true',
                       help='Skip dependency check at startup')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
