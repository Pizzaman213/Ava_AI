#!/usr/bin/env python3
"""
Ava Pipeline Training Script

This is the new pipeline-based training script that uses the modular
Ava component architecture. It replaces the monolithic train_100m_full.py
with a cleaner, more maintainable structure.

Usage:
    python train_pipeline.py --config code/configs/moe/large.yaml

    # Multi-GPU
    torchrun --nproc_per_node=4 train_pipeline.py --config code/configs/moe/4x_a6000_max_speed.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add src to path before importing Ava modules
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parents[2]  # scripts/5_training -> code -> project root
_src_dir = _project_root / "code" / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Now we can import Ava modules
from ava.core.paths import get_project_root, get_tokenizer_path
from ava.core.logging import (
    Colors, Icons, ColoredFormatter,
    print_header, print_subheader, print_success, print_warning, print_error,
    print_info, print_config, configure_root_logger,
)
project_root = get_project_root()

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
from ava.config.yaml_loader import load_yaml_with_path_resolution
from ava.config.training_config import DynamicConfig
from ava.kernels import KernelConfig, set_kernel_config, TRITON_AVAILABLE
from ava.core.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


def _status_indicator(enabled: bool, impact: str = None) -> str:
    """Return a colored YES/NO indicator with optional impact hint."""
    if enabled:
        status = f"{Colors.GREEN}{Colors.BOLD}YES{Colors.RESET}"
        if impact:
            status += f" {Colors.GRAY}({impact}){Colors.RESET}"
        return status
    return f"{Colors.GRAY}NO{Colors.RESET}"


def _format_value(value, unit: str = "", color: str = Colors.CYAN) -> str:
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

    Provides visibility into which optimizations are enabled/disabled
    with performance impact estimates and configuration details.
    """
    if rank != 0:
        return

    model_config = config.get('model', {})
    training_config = config.get('training', {})
    batching_config = training_config.get('batching', {})
    dynamic_batching = batching_config.get('dynamic_batching', {})
    data_config = config.get('data', {})
    perf_config = config.get('performance', {})
    hybrid_config = config.get('hybrid_caching', {})
    overlapped_config = config.get('overlapped_checkpointing', {})
    double_config = config.get('double_checkpointing', {})
    fp8_config = config.get('fp8', {})

    print_header("ACTIVE OPTIMIZATIONS STATUS", icon=Icons.GEAR)

    # ═══════════════════════════════════════════════════════════════════
    # Model Architecture Summary
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
    expert_params = num_layers * num_experts * (2 * hidden_size * intermediate_size)  # up + down projections
    total_params = embed_params + attn_params + expert_params
    print(f"  {Colors.WHITE}estimated_params:{Colors.RESET}      {Colors.ORANGE}~{total_params / 1e6:.0f}M{Colors.RESET}")

    # ═══════════════════════════════════════════════════════════════════
    # Model Optimizations (Memory & Speed)
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

    mixed_precision = training_config.get('mixed_precision', 'fp16')
    print(f"  {Colors.WHITE}mixed_precision:{Colors.RESET}       {_format_value(mixed_precision, color=Colors.GREEN if mixed_precision in ['bf16', 'fp16'] else Colors.GRAY)}")

    max_length = data_config.get('max_length', 512)
    print(f"  {Colors.WHITE}max_seq_length:{Colors.RESET}        {_format_value(max_length)}")

    # ═══════════════════════════════════════════════════════════════════
    # Dynamic Batching
    # ═══════════════════════════════════════════════════════════════════
    print_subheader(f"{Icons.CHART} Dynamic Batching")
    db_enabled = dynamic_batching.get('enabled', False)
    print(f"  {Colors.WHITE}enabled:{Colors.RESET}               {_status_indicator(db_enabled, '15-25% throughput gain')}")
    if db_enabled:
        min_bs = dynamic_batching.get('min_batch_size', 16)
        max_bs = dynamic_batching.get('max_batch_size', 256)
        target_mem = dynamic_batching.get('target_memory_threshold', 0.7)
        print(f"    {Colors.GRAY}├─{Colors.RESET} {Colors.WHITE}batch_range:{Colors.RESET}       {_format_value(f'{min_bs}-{max_bs}')}")
        print(f"    {Colors.GRAY}├─{Colors.RESET} {Colors.WHITE}target_memory:{Colors.RESET}     {_format_value(f'{target_mem:.0%}')}")

        token_budget = dynamic_batching.get('token_budget', {})
        tb_enabled = token_budget.get('enabled', False)
        if tb_enabled:
            target_tokens = token_budget.get('target_tokens_per_batch', 4096)
            print(f"    {Colors.GRAY}├─{Colors.RESET} {Colors.WHITE}token_budget:{Colors.RESET}      {_status_indicator(tb_enabled)} {Colors.GRAY}(target: {target_tokens:,} tokens/batch){Colors.RESET}")
        else:
            print(f"    {Colors.GRAY}├─{Colors.RESET} {Colors.WHITE}token_budget:{Colors.RESET}      {_status_indicator(tb_enabled)}")

        warmup = dynamic_batching.get('warmup_steps', 100)
        print(f"    {Colors.GRAY}└─{Colors.RESET} {Colors.WHITE}warmup_steps:{Colors.RESET}      {_format_value(warmup)}")

    # ═══════════════════════════════════════════════════════════════════
    # Data Loading
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
        db_enabled, pretokenized, seq_packing, hybrid_enabled, overlapped_enabled,
        double_enabled, fp8_enabled, tf32, cudnn_bench
    ])
    total_opts = 15

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
            print(f"    {Colors.GRAY}•{Colors.RESET} {rec}")

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


def main(args):
    """Main training function using the Ava pipeline architecture."""

    # Configure root logger early to prevent duplicate log messages
    # This sets up proper formatting and prevents module loggers from propagating duplicates
    configure_root_logger(level=logging.WARNING)

    # =========================================================================
    # Phase 1: Distributed Setup
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
    # =========================================================================
    config = load_yaml_with_path_resolution(args.config)

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
    batch_size = args.batch_size or training_config.get('batch_size', 8)
    learning_rate = args.learning_rate or training_config.get('learning_rate', 5e-5)
    log_interval = args.log_interval or training_config.get('log_interval', 10)
    val_interval = args.val_interval

    # Get output directory from config or CLI args
    output_config = config.get('output', {})
    output_dir = output_config.get('output_dir', args.save_dir)

    # =========================================================================
    # Phase 3: Create RunManager (Ava's output organizer)
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
            if rank == 0:
                train_logger.warning(f"Resume checkpoint not found: {resume_path}, starting fresh")
            args.resume = None

    max_steps = getattr(args, 'max_steps', None)

    if rank == 0:
        train_logger.info("=" * 60)
        train_logger.info("Ava Pipeline Training")
        train_logger.info("=" * 60)
        train_logger.info(f"Config: {args.config}")
        train_logger.info(f"Device: {device}")
        train_logger.info(f"World size: {world_size}")
        train_logger.info(f"Epochs: {num_epochs}")
        train_logger.info(f"Batch size: {batch_size}")
        train_logger.info(f"Learning rate: {learning_rate:.2e}")
        if max_steps:
            train_logger.info(f"Max steps: {max_steps}")

    # =========================================================================
    # Phase 4: Create TrainingContext (Ava's shared state hub)
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
        # =====================================================================
        model_builder = pipeline.get('model')
        model_builder.initialize()

        model = model_builder.build_model(config, device)

        # Apply quantization if enabled
        quant_config = config.get('quantization', {})
        if quant_config.get('enabled', False):
            model = model_builder.apply_quantization(model, quant_config)

        # Move to device
        model = model_builder.move_to_device(model, device)

        # Apply optimizations (FP8, hybrid caching, torch.compile, etc.)
        model = model_builder.apply_optimizations(model, config)

        # Wrap in DDP for distributed
        model = model_builder.wrap_distributed(model, rank, world_size)

        context.model = model

        # =====================================================================
        # Phase 6.1: Initialize BatchSizeController for Dynamic Batching
        # =====================================================================
        dynamic_batching_config = config.get('training', {}).get('batching', {}).get('dynamic_batching', {})
        if not dynamic_batching_config:
            dynamic_batching_config = config.get('dynamic_batching', {})

        batch_controller = None
        if dynamic_batching_config.get('enabled', False):
            try:
                from ava.optimizations.batch_controller import (
                    BatchSizeController,
                    create_batch_size_controller,
                )

                batch_controller = create_batch_size_controller(dynamic_batching_config)

                if batch_controller is not None:
                    # Run startup calibration if enabled
                    run_calibration = dynamic_batching_config.get('run_startup_calibration', True)

                    if run_calibration and rank == 0:
                        train_logger.info("Running batch size calibration...")

                        # Create sample batch function for calibration
                        def sample_batch_fn(bs: int):
                            """Create a sample batch for calibration."""
                            seq_len = dynamic_batching_config.get('base_sequence_length', 512)
                            vocab_size = config.get('model', {}).get('vocab_size', 50000)
                            return {
                                'input_ids': torch.randint(0, vocab_size, (bs, seq_len), device=device),
                                'attention_mask': torch.ones(bs, seq_len, device=device),
                            }

                        optimal_bs = batch_controller.startup_calibration(
                            model=model,
                            sample_batch_fn=sample_batch_fn,
                            max_time_seconds=dynamic_batching_config.get('calibration_timeout_sec', 30.0),
                        )
                        train_logger.info(f"Calibration complete: optimal batch size = {optimal_bs}")

                    context.batch_controller = batch_controller
                    if rank == 0:
                        train_logger.info(f"BatchSizeController initialized: {batch_controller.get_state()}")

            except ImportError as e:
                if rank == 0:
                    train_logger.warning(f"Could not import BatchSizeController: {e}")
            except Exception as e:
                if rank == 0:
                    train_logger.warning(f"BatchSizeController initialization failed: {e}")

        # =====================================================================
        # Phase 7: Create Optimizer (scheduler created after dataloader)
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
        # =====================================================================
        if args.resume:
            resume_path = Path(args.resume)
            if rank == 0:
                train_logger.info(f"Loading checkpoint from: {resume_path}")

            resume_epoch, resume_step = checkpoint_manager.load(
                model=model,
                optimizer=optimizer,
                checkpoint_path=resume_path
            )

            if rank == 0:
                train_logger.info(f"Resumed from epoch {resume_epoch}, step {resume_step}")

        # =====================================================================
        # Phase 8: Create DataLoaders
        # =====================================================================
        data_mgr = pipeline.get('data')
        data_mgr.initialize()

        # Load tokenizer
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            # Try multiple possible config locations for tokenizer path
            tokenizer_path = (
                data_config.get('tokenizer_path') or
                data_config.get('tokenizer_name') or
                config.get('data', {}).get('tokenizer_name') or
                str(get_tokenizer_path())  # Use utility function for default path
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            context.tokenizer = tokenizer
            if rank == 0:
                train_logger.info(f"Loaded tokenizer: {tokenizer_path}")
        except Exception as e:
            if rank == 0:
                train_logger.warning(f"Could not load tokenizer: {e}")

        # Disable generation if no tokenizer available
        generation_config = config.get('generation', {})
        if generation_config and tokenizer is None:
            if rank == 0:
                train_logger.warning("Generation disabled: no tokenizer available")
            generation_config = {}

        # Convert config dict to DynamicConfig for DataLoaderManager
        config_obj = DynamicConfig(config) if isinstance(config, dict) else config

        train_loader, val_loader = data_mgr.create_dataloaders(
            training_config=config_obj,
            tokenizer=tokenizer,
            config_dict=config,
            batch_size=batch_size
        )

        # Update total steps now that we have the data loader
        # Note: IterableDatasets (like InfiniteUltraFastDataset) don't have __len__
        # Try to get length safely, falling back to config value or default
        try:
            loader_len = len(train_loader)
        except TypeError:
            # Infinite/streaming datasets don't have length
            loader_len = config.get('training', {}).get('steps_per_epoch', 1000)
        total_steps = num_epochs * loader_len // context.gradient_accumulation_steps

        if rank == 0:
            train_logger.info(f"Total training steps: {total_steps}")

        # =====================================================================
        # Phase 8.1: Create Scheduler (now that we know total_steps)
        # =====================================================================
        warmup_steps = training_config.get('warmup_steps', 1000)
        scheduler = optimizer_mgr.create_scheduler(
            optimizer, warmup_steps, total_steps,
            min_lr=training_config.get('min_lr', 0.0)
        )
        context.scheduler = scheduler

        if rank == 0:
            train_logger.info(f"Scheduler: warmup={warmup_steps}, total={total_steps}")

        # =====================================================================
        # Phase 9: Setup Metrics
        # =====================================================================
        metrics_mgr = pipeline.get('metrics')
        metrics_mgr.initialize()

        wandb_config = config.get('wandb', {})
        # Set wandb directory inside the run folder
        wandb_dir = run_manager.run_dir / 'wandb'
        metrics_mgr.setup(
            log_dir=log_dir,
            wandb_config=wandb_config if wandb_config.get('enabled', False) else None,
            use_wandb=wandb_config.get('enabled', False),
            wandb_dir=wandb_dir,
        )

        # =====================================================================
        # Phase 10: Initialize Remaining Components
        # =====================================================================
        validation_mgr = pipeline.get('validation')
        validation_mgr.initialize()

        generation_mgr = pipeline.get('generation')
        generation_mgr.initialize()
        generation_mgr.set_log_dir(log_dir)

        training_mgr = pipeline.get('training')
        training_mgr.initialize()
        training_mgr.set_components(
            metrics_manager=metrics_mgr,
            generation_manager=generation_mgr,
            checkpoint_manager=checkpoint_manager
        )

        # Set global step if resuming (so TrainingLoopManager tracks correctly)
        if resume_step > 0:
            training_mgr._global_step = resume_step

        # Determine profile directory - use run folder if not explicitly specified
        profile_dir = getattr(args, 'profile_dir', None)
        if profile_dir is None or profile_dir == './profiles':
            # Default to run folder/profiles
            profile_dir = str(run_manager.run_dir / 'profiles') if hasattr(run_manager, 'run_dir') else './profiles'

        # Create training loop config
        # Logging options: 'tqdm' = clean progress bar only, 'verbose' = both tqdm + INFO logs
        log_mode = training_config.get('log_mode', 'tqdm')
        verbose_log_interval = training_config.get('verbose_log_interval', 500)

        loop_config = TrainingLoopConfig(
            gradient_accumulation_steps=context.gradient_accumulation_steps,
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            use_amp=context.use_amp,
            amp_dtype=context.amp_dtype,
            log_interval=log_interval,
            generate_every_n_steps=config.get('generation', {}).get('generate_every_n_steps', 500),
            save_steps=training_config.get('save_steps', 0),
            max_steps=getattr(args, 'max_steps', None),
            # Profiling options (disabled by default)
            enable_profiling=getattr(args, 'enable_profiling', False),
            profile_start_step=getattr(args, 'profile_start_step', 0),
            profile_end_step=getattr(args, 'profile_end_step', 999999),
            profile_dir=profile_dir,
            # Logging options
            log_mode=log_mode,
            verbose_log_interval=verbose_log_interval,
        )

        # Setup profiler if enabled
        if loop_config.enable_profiling and rank == 0:
            training_mgr.setup_profiler(loop_config)

        # =====================================================================
        # Phase 11: Training Loop
        # =====================================================================
        if rank == 0:
            train_logger.info("\n" + "=" * 60)
            train_logger.info("Starting Training")
            train_logger.info("=" * 60)

        best_val_loss = float('inf')

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
                coherence_config=config.get('coherence', {}),
            )

            if rank == 0:
                train_logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")

            # Check if max_steps reached
            if training_mgr.reached_max_steps(loop_config):
                if rank == 0:
                    train_logger.info(f"Reached max_steps ({loop_config.max_steps}), stopping training")
                break

            # Validation
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

                if validation_mgr.is_best(val_loss):
                    best_val_loss = val_loss
                    if rank == 0:
                        train_logger.info(f"New best validation loss: {val_loss:.4f}")
                        # Save best checkpoint
                        run_manager.save_checkpoint(
                            model_state=model.state_dict(),
                            optimizer_state=optimizer.state_dict(),
                            epoch=epoch,
                            step=training_mgr.get_global_step(),
                            loss=val_loss,
                            is_best=True,
                            additional_data={'train_loss': train_loss}
                        )

            # Notify components of epoch end
            pipeline.on_epoch_end(epoch)

        # =====================================================================
        # Phase 12: Finalize
        # =====================================================================
        if rank == 0:
            train_logger.info("\n" + "=" * 60)
            train_logger.info("Training Complete")
            train_logger.info("=" * 60)
            train_logger.info(f"Best validation loss: {best_val_loss:.4f}")

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

        # Cleanup all components
        pipeline.cleanup_all()

        # Shutdown checkpoint manager (waits for pending async saves)
        checkpoint_manager.shutdown()

    except Exception as e:
        train_logger.error(f"Training failed: {e}", exc_info=True)
        pipeline.on_error(e)
        run_manager.finish_run(status='failed', final_metrics={'error': str(e)})
        # Still shutdown checkpoint manager on error
        checkpoint_manager.shutdown()
        raise

    finally:
        cleanup_distributed(rank, world_size)


def parse_args():
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

    # Profiling options for Nsight Systems/Compute
    parser.add_argument('--enable-profiling', action='store_true',
                       help='Enable Nsight-compatible GPU profiling')
    parser.add_argument('--profile-dir', type=str, default='./profiles',
                       help='Directory to save profiling outputs')
    parser.add_argument('--profile-start-step', type=int, default=0,
                       help='Step to start profiling (default: 0, profiles all)')
    parser.add_argument('--profile-end-step', type=int, default=999999,
                       help='Step to end profiling (default: 999999, profiles all)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
