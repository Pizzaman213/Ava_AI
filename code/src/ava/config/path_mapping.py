"""
Configuration Path Mapping for Backward Compatibility.

This module provides mappings from old configuration paths to new canonical paths,
enabling a seamless migration while maintaining backward compatibility.

The configuration system has been reorganized from 38 scattered top-level sections
into 8 logical categories:

    1. model         - Architecture (core, tokens, moe, position, regularization)
    2. training      - Training loop (batch, optimizer, schedule, validation, generation)
    3. data          - Data pipeline (source, splits, loading, packing, indexed)
    4. compute       - Hardware/performance (device, precision, cuda, kernels, memory)
    5. distributed   - Multi-GPU (strategy, deepspeed)
    6. logging       - Observability (console, wandb, tensorboard, diagnostics)
    7. checkpoints   - Save/load (output_dir, selection)
    8. experimental  - Advanced features disabled by default

Usage:
    from ava.config.path_mapping import get_new_path, is_deprecated_path

    # Check if a path is deprecated
    if is_deprecated_path('hardware.device'):
        new_path = get_new_path('hardware.device')  # Returns 'compute.device.type'
"""

from typing import Dict, Optional, Set
import os
import warnings
import logging

logger = logging.getLogger(__name__)

# Environment variable to suppress deprecation warnings
SUPPRESS_DEPRECATION_WARNINGS = os.environ.get('SUPPRESS_CONFIG_DEPRECATION', '').lower() in ('1', 'true', 'yes')

# Track which warnings have been shown to avoid spam
_warned_paths: Set[str] = set()


# =============================================================================
# PATH MAPPINGS: old_path -> new_path
# =============================================================================

CONFIG_PATH_MAPPINGS: Dict[str, str] = {
    # =========================================================================
    # Hardware -> compute.device
    # =========================================================================
    'hardware': 'compute.device',
    'hardware.device': 'compute.device.type',
    'hardware.mixed_precision': 'compute.precision.mixed_precision',
    'hardware.compile': 'compute.kernels.torch_compile.enabled',
    'hardware.num_gpus': 'compute.device.num_gpus',
    'hardware.use_gpu_load_balancing': 'distributed.load_balancing.enabled',
    'hardware.balancing_strategy': 'distributed.load_balancing.strategy',
    'hardware.rebalance_interval': 'distributed.load_balancing.rebalance_interval',
    'hardware.enable_expert_migration': 'distributed.load_balancing.enable_expert_migration',
    'hardware.migration_threshold': 'distributed.load_balancing.migration_threshold',
    'hardware.log_gpu_metrics': 'distributed.load_balancing.log_gpu_metrics',

    # =========================================================================
    # Performance -> compute.performance
    # =========================================================================
    'performance': 'compute.performance',
    'performance.ultra_fast_mode': 'compute.performance.ultra_fast_mode',
    'performance.fast_progress': 'compute.performance.fast_progress',
    'performance.minimal_progress': 'compute.performance.minimal_progress',
    'performance.no_sync': 'compute.performance.no_sync',
    'performance.express_mode': 'compute.performance.express_mode',
    'performance.enable_tf32': 'compute.performance.enable_tf32',
    'performance.float32_matmul_precision': 'compute.performance.float32_matmul_precision',
    'performance.enable_cudnn_benchmark': 'compute.performance.enable_cudnn_benchmark',
    'performance.cudagraph_skip_dynamic_shapes': 'compute.cuda.graphs.skip_dynamic_shapes',
    'performance.cudagraph_dynamic_shape_warn_limit': 'compute.cuda.graphs.dynamic_shape_warn_limit',
    'performance.torchinductor_max_autotune': 'compute.kernels.torchinductor_max_autotune',
    'performance.enable_torch_compile': 'compute.kernels.torch_compile.enabled',
    'performance.torch_compile_mode': 'compute.kernels.torch_compile.mode',
    'performance.torch_compile_fullgraph': 'compute.kernels.torch_compile.fullgraph',
    'performance.torch_compile_dynamic': 'compute.kernels.torch_compile.dynamic',

    # =========================================================================
    # kernel_optimization -> compute.kernels
    # =========================================================================
    'kernel_optimization': 'compute.kernels',
    'kernel_optimization.router_kernel_mode': 'compute.kernels.router.mode',
    'kernel_optimization.use_fused_softmax_topk': 'compute.kernels.router.use_fused_softmax_topk',
    'kernel_optimization.router_block_size': 'compute.kernels.router.block_size',
    'kernel_optimization.use_sparse_expert_dispatch': 'compute.kernels.expert.use_sparse_dispatch',
    'kernel_optimization.use_fused_activations': 'compute.kernels.expert.use_fused_activations',
    'kernel_optimization.use_selective_expert_loading': 'compute.kernels.expert.use_selective_loading',
    'kernel_optimization.use_vectorized_capacity': 'compute.kernels.expert.use_vectorized_capacity',
    'kernel_optimization.use_fused_moe_kernel': 'compute.kernels.expert.use_fused_moe_kernel',
    'kernel_optimization.enable_kernel_profiling': 'compute.kernels.enable_profiling',

    # =========================================================================
    # cuda_streams -> compute.cuda.streams
    # =========================================================================
    'cuda_streams': 'compute.cuda.streams',
    'cuda_streams.enabled': 'compute.cuda.streams.enabled',
    'cuda_streams.num_streams': 'compute.cuda.streams.num_streams',
    'cuda_streams.use_event_timing': 'compute.cuda.streams.use_event_timing',
    'cuda_streams.use_stream_pool': 'compute.cuda.streams.use_stream_pool',
    'cuda_streams.high_priority_transfers': 'compute.cuda.streams.high_priority_transfers',

    # =========================================================================
    # cuda_graphs -> compute.cuda.graphs
    # =========================================================================
    'cuda_graphs': 'compute.cuda.graphs',
    'cuda_graphs.enabled': 'compute.cuda.graphs.enabled',
    'cuda_graphs.warmup_steps': 'compute.cuda.graphs.warmup_steps',
    'cuda_graphs.capture_mode': 'compute.cuda.graphs.capture_mode',

    # =========================================================================
    # optimizations -> compute.optimizations
    # =========================================================================
    'optimizations': 'compute.optimizations',
    'optimizations.use_overlapped_checkpointing': 'experimental.checkpointing.overlapped.enabled',
    'optimizations.torchinductor_autotune': 'compute.kernels.torchinductor_autotune',
    'optimizations.memory_headroom_gb': 'compute.memory.headroom_gb',
    'optimizations.memory_cleanup_thresholds': 'compute.memory.cleanup_thresholds',
    'optimizations.expert_prefetch': 'compute.memory.expert_prefetch',
    'optimizations.expert_cache': 'compute.memory.expert_cache',
    'optimizations.checkpoint': 'checkpoints.options',
    'optimizations.memory_cleanup': 'compute.memory.cleanup',
    'optimizations.dataloader': 'data.loading.optimizations',
    'optimizations.gradient_checkpointing': 'compute.memory.gradient_checkpointing',
    'optimizations.router': 'compute.kernels.router.optimizations',

    # =========================================================================
    # batch_size_calibration -> compute.calibration.batch_size
    # =========================================================================
    'batch_size_calibration': 'compute.calibration.batch_size',
    'batch_size_calibration.enabled': 'compute.calibration.batch_size.enabled',
    'batch_size_calibration.target_memory': 'compute.calibration.batch_size.target_memory',
    'batch_size_calibration.max_batch_size': 'compute.calibration.batch_size.max_batch_size',
    'batch_size_calibration.min_batch_size': 'compute.calibration.batch_size.min_batch_size',

    # =========================================================================
    # calibration -> compute.calibration.system
    # =========================================================================
    'calibration': 'compute.calibration.system',
    'calibration.enabled': 'compute.calibration.system.enabled',
    'calibration.run_memory_profiling': 'compute.calibration.system.run_memory_profiling',
    'calibration.run_backward_profiling': 'compute.calibration.system.run_backward_profiling',
    'calibration.run_throughput_profiling': 'compute.calibration.system.run_throughput_profiling',

    # =========================================================================
    # Logging sections -> logging.*
    # =========================================================================
    'wandb': 'logging.wandb',
    'wandb.enabled': 'logging.wandb.enabled',
    'wandb.project': 'logging.wandb.project',
    'wandb.entity': 'logging.wandb.entity',
    'wandb.name': 'logging.wandb.name',
    'wandb.tags': 'logging.wandb.tags',
    'wandb.notes': 'logging.wandb.notes',
    'wandb.group': 'logging.wandb.group',
    'wandb.log_freq': 'logging.wandb.log_freq',

    'logging.tensorboard': 'logging.tensorboard',
    'logging.tensorboard.enabled': 'logging.tensorboard.enabled',
    'logging.tensorboard.log_dir': 'logging.tensorboard.log_dir',

    'logging_config': 'logging.config',
    'logging_config.verbosity': 'logging.console.verbosity',
    'logging_config.console_level': 'logging.console.level',
    'logging_config.file_level': 'logging.file.level',
    'logging_config.metrics_log_freq': 'logging.frequencies.metrics',
    'logging_config.memory_check_freq': 'logging.frequencies.memory_check',
    'logging_config.health_summary_freq': 'logging.frequencies.health_summary',
    'logging_config.moe_metrics_freq': 'logging.frequencies.moe_metrics',
    'logging_config.enable_timing_breakdown': 'logging.features.timing_breakdown',
    'logging_config.enable_memory_profiling': 'logging.features.memory_profiling',
    'logging_config.enable_health_summaries': 'logging.features.health_summaries',
    'logging_config.log_tensor_shapes': 'logging.features.log_tensor_shapes',
    'logging_config.log_checkpoint_validation': 'logging.features.log_checkpoint_validation',
    'logging_config.save_sample_generations': 'logging.features.save_sample_generations',
    'logging_config.use_structured_logging': 'logging.features.use_structured_logging',
    'logging_config.log_format': 'logging.features.log_format',
    'logging_config.log_gradients_to_wandb': 'logging.wandb.log_gradients',
    'logging_config.log_model_topology': 'logging.wandb.log_model_topology',

    'diagnostics': 'logging.diagnostics',
    'diagnostics.enabled': 'logging.diagnostics.enabled',
    'diagnostics.enable_per_layer_gradients': 'logging.diagnostics.per_layer_gradients.enabled',
    'diagnostics.per_layer_log_freq': 'logging.diagnostics.per_layer_gradients.log_freq',
    'diagnostics.layer_name_patterns': 'logging.diagnostics.per_layer_gradients.layer_name_patterns',
    'diagnostics.enable_routing_diagnostics': 'logging.diagnostics.routing.enabled',
    'diagnostics.routing_log_freq': 'logging.diagnostics.routing.log_freq',
    'diagnostics.track_per_expert_load': 'logging.diagnostics.routing.track_per_expert_load',
    'diagnostics.track_routing_entropy': 'logging.diagnostics.routing.track_routing_entropy',
    'diagnostics.track_expert_capacity_usage': 'logging.diagnostics.routing.track_expert_capacity_usage',
    'diagnostics.enable_memory_breakdown': 'logging.diagnostics.memory.enabled',
    'diagnostics.memory_log_freq': 'logging.diagnostics.memory.log_freq',
    'diagnostics.track_activation_memory': 'logging.diagnostics.memory.track_activation_memory',
    'diagnostics.track_gradient_memory': 'logging.diagnostics.memory.track_gradient_memory',
    'diagnostics.track_optimizer_state_memory': 'logging.diagnostics.memory.track_optimizer_state_memory',
    'diagnostics.track_parameter_memory': 'logging.diagnostics.memory.track_parameter_memory',
    'diagnostics.enable_timing_profiling': 'logging.diagnostics.timing.enabled',
    'diagnostics.timing_log_freq': 'logging.diagnostics.timing.log_freq',
    'diagnostics.profile_forward': 'logging.diagnostics.timing.profile_forward',
    'diagnostics.profile_backward': 'logging.diagnostics.timing.profile_backward',
    'diagnostics.profile_optimizer_step': 'logging.diagnostics.timing.profile_optimizer_step',
    'diagnostics.profile_data_loading': 'logging.diagnostics.timing.profile_data_loading',

    'dev_log': 'logging.dev',
    'dev_log.enabled': 'logging.dev.enabled',
    'dev_log.show_file_timings': 'logging.dev.show_file_timings',
    'dev_log.show_batch_timings': 'logging.dev.show_batch_timings',
    'dev_log.show_step_breakdown': 'logging.dev.show_step_breakdown',
    'dev_log.report_interval': 'logging.dev.report_interval',

    'moe_metrics': 'logging.moe_metrics',
    'moe_metrics.track_expert_utilization': 'logging.moe_metrics.track_expert_utilization',
    'moe_metrics.log_frequency': 'logging.moe_metrics.log_frequency',
    'moe_metrics.track_routing_decisions': 'logging.moe_metrics.track_routing_decisions',
    'moe_metrics.track_load_balance': 'logging.moe_metrics.track_load_balance',

    'run_management': 'logging.run_management',
    'run_management.run_name': 'logging.run_management.run_name',
    'run_management.run_tags': 'logging.run_management.run_tags',
    'run_management.run_description': 'logging.run_management.run_description',
    'run_management.disable_run_manager': 'logging.run_management.disable_run_manager',

    # =========================================================================
    # Output -> checkpoints
    # =========================================================================
    'output': 'checkpoints',
    'output.output_dir': 'checkpoints.output_dir',
    'output.save_every': 'checkpoints.save_every',
    'output.resume': 'checkpoints.resume',
    'output.fresh_start': 'checkpoints.fresh_start',

    'model_selection': 'checkpoints.selection',
    'model_selection.enabled': 'checkpoints.selection.enabled',
    'model_selection.val_loss_weight': 'checkpoints.selection.val_loss_weight',
    'model_selection.coherence_score_weight': 'checkpoints.selection.coherence_score_weight',
    'model_selection.perplexity_weight': 'checkpoints.selection.perplexity_weight',
    'model_selection.perplexity_cap': 'checkpoints.selection.perplexity_cap',
    'model_selection.val_loss_cap': 'checkpoints.selection.val_loss_cap',
    'model_selection.higher_is_better': 'checkpoints.selection.higher_is_better',
    'model_selection.require_all_metrics': 'checkpoints.selection.require_all_metrics',
    'model_selection.fallback_to_val_loss': 'checkpoints.selection.fallback_to_val_loss',

    # =========================================================================
    # DeepSpeed -> distributed.deepspeed
    # =========================================================================
    'deepspeed': 'distributed.deepspeed',
    'deepspeed.enabled': 'distributed.deepspeed.enabled',
    'deepspeed.config_file': 'distributed.deepspeed.config_file',
    'deepspeed.zero_stage': 'distributed.deepspeed.zero_stage',
    'deepspeed.cpu_offload': 'distributed.deepspeed.cpu_offload',
    'deepspeed.nvme_offload': 'distributed.deepspeed.nvme_offload',
    'deepspeed.nvme_path': 'distributed.deepspeed.nvme_path',
    'deepspeed.precision_type': 'distributed.deepspeed.precision_type',
    'deepspeed.enable_mixed_precision': 'distributed.deepspeed.enable_mixed_precision',
    'deepspeed.gradient_accumulation_steps': 'distributed.deepspeed.gradient_accumulation_steps',
    'deepspeed.train_batch_size': 'distributed.deepspeed.train_batch_size',
    'deepspeed.micro_batch_size': 'distributed.deepspeed.micro_batch_size',
    'deepspeed.activation_checkpointing': 'distributed.deepspeed.activation_checkpointing',
    'deepspeed.gradient_clipping': 'distributed.deepspeed.gradient_clipping',

    # =========================================================================
    # Experimental features -> experimental.*
    # =========================================================================
    'progressive': 'experimental.progressive',
    'progressive.enable_progressive_training': 'experimental.progressive.enabled',
    'progressive.enable_sequence_scaling': 'experimental.progressive.sequence_scaling.enabled',
    'progressive.initial_seq_length': 'experimental.progressive.sequence_scaling.initial_seq_length',
    'progressive.final_seq_length': 'experimental.progressive.sequence_scaling.final_seq_length',
    'progressive.length_schedule': 'experimental.progressive.sequence_scaling.length_schedule',
    'progressive.length_growth_epochs': 'experimental.progressive.sequence_scaling.length_growth_epochs',
    'progressive.enable_length_bucketing': 'experimental.progressive.sequence_scaling.enable_length_bucketing',
    'progressive.enable_curriculum': 'experimental.progressive.curriculum.enabled',
    'progressive.curriculum_metric': 'experimental.progressive.curriculum.metric',
    'progressive.enable_score_caching': 'experimental.progressive.curriculum.enable_score_caching',
    'progressive.cache_dir': 'experimental.progressive.curriculum.cache_dir',
    'progressive.cache_version': 'experimental.progressive.curriculum.cache_version',

    'hybrid_caching': 'experimental.caching',
    'hybrid_caching.enabled': 'experimental.caching.enabled',
    'hybrid_caching.mode': 'experimental.caching.mode',
    'hybrid_caching.activation_cache': 'experimental.caching.activation_cache',
    'hybrid_caching.kv_cache': 'experimental.caching.kv_cache',

    'overlapped_checkpointing': 'experimental.checkpointing.overlapped',
    'overlapped_checkpointing.enabled': 'experimental.checkpointing.overlapped.enabled',
    'overlapped_checkpointing.stream_overlap': 'experimental.checkpointing.overlapped.stream_overlap',
    'overlapped_checkpointing.target_layers': 'experimental.checkpointing.overlapped.target_layers',

    'double_checkpointing': 'experimental.checkpointing.double',
    'double_checkpointing.enabled': 'experimental.checkpointing.double.enabled',
    'double_checkpointing.coarse_checkpoint_interval': 'experimental.checkpointing.double.coarse_checkpoint_interval',
    'double_checkpointing.fine_checkpoint_interval': 'experimental.checkpointing.double.fine_checkpoint_interval',
    'double_checkpointing.use_cuda_streams': 'experimental.checkpointing.double.use_cuda_streams',

    'fp8': 'experimental.fp8',
    'fp8.enabled': 'experimental.fp8.enabled',
    'fp8.use_transformer_engine': 'experimental.fp8.use_transformer_engine',
    'fp8.format': 'experimental.fp8.format',
    'fp8.margin': 'experimental.fp8.margin',

    'lr_finder': 'experimental.lr_finder',
    'lr_finder.run_lr_finder': 'experimental.lr_finder.enabled',
    'lr_finder.start_lr': 'experimental.lr_finder.start_lr',
    'lr_finder.end_lr': 'experimental.lr_finder.end_lr',
    'lr_finder.num_iterations': 'experimental.lr_finder.num_iterations',
    'lr_finder.suggestion_method': 'experimental.lr_finder.suggestion_method',
    'lr_finder.use_suggested_lr': 'experimental.lr_finder.use_suggested_lr',
    'lr_finder.plot_path': 'experimental.lr_finder.plot_path',
    'lr_finder.smooth_beta': 'experimental.lr_finder.smooth_beta',
    'lr_finder.stop_div_threshold': 'experimental.lr_finder.stop_div_threshold',

    'episodic_memory': 'experimental.episodic_memory',
    'episodic_memory.use_episodic_memory': 'experimental.episodic_memory.enabled',
    'episodic_memory.memory_capacity': 'experimental.episodic_memory.memory_capacity',
    'episodic_memory.memory_selection_strategy': 'experimental.episodic_memory.memory_selection_strategy',
    'episodic_memory.memory_importance_threshold': 'experimental.episodic_memory.memory_importance_threshold',
    'episodic_memory.memory_retrieval_method': 'experimental.episodic_memory.memory_retrieval_method',
    'episodic_memory.memory_replay_ratio': 'experimental.episodic_memory.memory_replay_ratio',
    'episodic_memory.memory_replay_strategy': 'experimental.episodic_memory.memory_replay_strategy',
    'episodic_memory.memory_adaptation_rate': 'experimental.episodic_memory.memory_adaptation_rate',
    'episodic_memory.memory_performance_window': 'experimental.episodic_memory.memory_performance_window',
    'episodic_memory.task_id': 'experimental.episodic_memory.task_id',
    'episodic_memory.silent_mode': 'experimental.episodic_memory.silent_mode',
    'episodic_memory.enable_auto_grad_accumulation': 'experimental.episodic_memory.enable_auto_grad_accumulation',

    'architecture': 'experimental.architecture',
    'architecture.use_moh': 'experimental.architecture.use_moh',
    'architecture.use_moa': 'experimental.architecture.use_moa',
    'architecture.use_cross_attention': 'experimental.architecture.use_cross_attention',
    'architecture.use_alibi': 'experimental.architecture.use_alibi',
    'architecture.expert_routing_type': 'experimental.architecture.expert_routing_type',

    'rag': 'experimental.rag',
    'rag.use_rag': 'experimental.rag.enabled',
    'rag.knowledge_base_path': 'experimental.rag.knowledge_base_path',
    'rag.max_retrieved_docs': 'experimental.rag.max_retrieved_docs',
    'rag.rag_fusion_type': 'experimental.rag.fusion_type',

    'losses': 'experimental.losses',
    'losses.use_focal_loss': 'experimental.losses.focal_loss.enabled',
    'losses.use_contrastive_loss': 'experimental.losses.contrastive_loss.enabled',
    'losses.use_diversity_loss': 'experimental.losses.diversity_loss.enabled',
    'losses.adaptive_loss_scaling': 'experimental.losses.adaptive_scaling',
    'losses.use_multi_token_prediction': 'experimental.losses.multi_token_prediction.enabled',
    'losses.num_future_tokens': 'experimental.losses.multi_token_prediction.num_future_tokens',
    'losses.mtp_weight': 'experimental.losses.multi_token_prediction.weight',
    'losses.initial_temperature': 'experimental.losses.temperature.initial',
    'losses.adaptive_temperature': 'experimental.losses.temperature.adaptive',
    'losses.label_smoothing': 'experimental.losses.label_smoothing',
    'losses.use_moe_balancing': 'experimental.losses.moe_balancing.enabled',
    'losses.gradient_balance_weight': 'experimental.losses.moe_balancing.gradient_balance_weight',
    'losses.use_auxiliary_loss': 'experimental.losses.moe_balancing.use_auxiliary_loss',
    'losses.use_ngram_penalty': 'experimental.losses.ngram_penalty.enabled',
    'losses.ngram_size': 'experimental.losses.ngram_penalty.ngram_size',
    'losses.ngram_penalty_weight': 'experimental.losses.ngram_penalty.weight',
    'losses.use_immediate_repetition_detector': 'experimental.losses.ngram_penalty.use_immediate_repetition_detector',
    'losses.immediate_repetition_weight': 'experimental.losses.ngram_penalty.immediate_repetition_weight',

    'adaptive_mtp': 'experimental.adaptive_mtp',
    'adaptive_mtp.use_adaptive_mtp': 'experimental.adaptive_mtp.enabled',
    'adaptive_mtp.num_prediction_heads': 'experimental.adaptive_mtp.num_prediction_heads',
    'adaptive_mtp.confidence_threshold_train': 'experimental.adaptive_mtp.confidence_threshold_train',
    'adaptive_mtp.confidence_threshold_inference': 'experimental.adaptive_mtp.confidence_threshold_inference',
    'adaptive_mtp.gate_hidden_dims': 'experimental.adaptive_mtp.gate.hidden_dims',
    'adaptive_mtp.gate_dropout': 'experimental.adaptive_mtp.gate.dropout',
    'adaptive_mtp.gate_activation': 'experimental.adaptive_mtp.gate.activation',
    'adaptive_mtp.use_attention_pooling': 'experimental.adaptive_mtp.gate.use_attention_pooling',
    'adaptive_mtp.head_type': 'experimental.adaptive_mtp.head.type',
    'adaptive_mtp.head_intermediate_size': 'experimental.adaptive_mtp.head.intermediate_size',
    'adaptive_mtp.head_dropout': 'experimental.adaptive_mtp.head.dropout',
    'adaptive_mtp.share_projections': 'experimental.adaptive_mtp.head.share_projections',
    'adaptive_mtp.mtp_warmup_epochs': 'experimental.adaptive_mtp.training.warmup_epochs',
    'adaptive_mtp.confidence_reg_strength': 'experimental.adaptive_mtp.training.confidence_reg_strength',
    'adaptive_mtp.use_confidence_weighting': 'experimental.adaptive_mtp.loss.use_confidence_weighting',
    'adaptive_mtp.primary_loss_weight': 'experimental.adaptive_mtp.loss.primary_loss_weight',
    'adaptive_mtp.additional_loss_base_weight': 'experimental.adaptive_mtp.loss.additional_loss_base_weight',
    'adaptive_mtp.enable_dynamic_prediction': 'experimental.adaptive_mtp.efficiency.enable_dynamic_prediction',
    'adaptive_mtp.min_confidence_for_computation': 'experimental.adaptive_mtp.efficiency.min_confidence_for_computation',

    'gradient': 'experimental.gradient',
    'gradient.gradient_surgery': 'experimental.gradient.surgery.enabled',
    'gradient.adaptive_gradient_surgery': 'experimental.gradient.surgery.adaptive',
    'gradient.gradient_surgery_method': 'experimental.gradient.surgery.method',

    'evaluation': 'training.evaluation',
    'evaluation.eval_during_training': 'training.evaluation.enabled',
    'evaluation.eval_metrics': 'training.evaluation.metrics',
    'evaluation.eval_frequency': 'training.evaluation.frequency',

    'quantization': 'experimental.quantization',
    'quantization.quantization_aware': 'experimental.quantization.aware_training',
    'quantization.bit_width': 'experimental.quantization.bit_width',
    'quantization.use_nvfp4': 'experimental.quantization.use_nvfp4',
    'quantization.nvfp4_block_size': 'experimental.quantization.nvfp4_block_size',
    'quantization.stochastic_rounding': 'experimental.quantization.stochastic_rounding',
    'quantization.use_hadamard_transform': 'experimental.quantization.use_hadamard_transform',
    'quantization.use_torchao_nvfp4': 'experimental.quantization.use_torchao_nvfp4',

    'generation': 'training.generation',

    'rlhf': 'experimental.rlhf',
    'rlhf.policy_model_path': 'experimental.rlhf.policy_model_path',
    'rlhf.judge_model_path': 'experimental.rlhf.judge_model_path',
    'rlhf.reward_model_path': 'experimental.rlhf.reward_model_path',
    'rlhf.use_model_to_model_reward': 'experimental.rlhf.use_model_to_model_reward',
    'rlhf.freeze_reward_model': 'experimental.rlhf.freeze_reward_model',
    'rlhf.ppo': 'experimental.rlhf.ppo',

    # =========================================================================
    # Data sections -> data.*
    # =========================================================================
    'multi_column_data': 'data.multi_column',
    'multi_column_data.use_multi_column': 'data.multi_column.enabled',
    'multi_column_data.dataset_config': 'data.multi_column.dataset_config',
    'multi_column_data.hf_dataset': 'data.multi_column.hf_dataset',
    'multi_column_data.hf_dataset_config': 'data.multi_column.hf_dataset_config',
    'multi_column_data.column_names': 'data.multi_column.column_names',
    'multi_column_data.column_types': 'data.multi_column.column_types',
    'multi_column_data.column_roles': 'data.multi_column.column_roles',
    'multi_column_data.combine_strategy': 'data.multi_column.combine_strategy',
    'multi_column_data.column_template': 'data.multi_column.column_template',
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_deprecated_path(path: str) -> bool:
    """
    Check if a configuration path is deprecated.

    Args:
        path: Dot-separated config path (e.g., 'hardware.device')

    Returns:
        True if the path is deprecated and has a new equivalent
    """
    return path in CONFIG_PATH_MAPPINGS


def get_new_path(old_path: str) -> Optional[str]:
    """
    Get the new canonical path for a deprecated path.

    Args:
        old_path: The deprecated config path

    Returns:
        The new canonical path, or None if not deprecated
    """
    return CONFIG_PATH_MAPPINGS.get(old_path)


def warn_deprecated_path(old_path: str, new_path: str = None) -> None:
    """
    Emit a deprecation warning for a config path (once per path).

    Args:
        old_path: The deprecated config path being accessed
        new_path: The new canonical path (auto-resolved if not provided)
    """
    if SUPPRESS_DEPRECATION_WARNINGS:
        return

    if old_path in _warned_paths:
        return

    _warned_paths.add(old_path)

    if new_path is None:
        new_path = get_new_path(old_path)

    if new_path:
        msg = (
            f"Config path '{old_path}' is deprecated. "
            f"Use '{new_path}' instead. "
            f"Set SUPPRESS_CONFIG_DEPRECATION=1 to silence this warning."
        )
        warnings.warn(msg, DeprecationWarning, stacklevel=3)
        logger.warning(msg)


def get_deprecated_top_level_sections() -> Set[str]:
    """
    Get all deprecated top-level config sections.

    Returns:
        Set of deprecated top-level section names
    """
    return {
        path.split('.')[0]
        for path in CONFIG_PATH_MAPPINGS.keys()
        if '.' not in path or path.count('.') == 0
    }


def clear_warning_cache() -> None:
    """Clear the cache of warned paths (useful for testing)."""
    _warned_paths.clear()
