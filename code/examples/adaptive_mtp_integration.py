"""
Example integration of Adaptive MTP with train.py

This script shows how to integrate the Adaptive Multi-Token Prediction system
with the existing train.py training pipeline.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# Import Adaptive MTP components
from .adaptive_mtp_model import AdaptiveMTPModel, AdaptiveMTPConfig
from ..losses import AdaptiveMTPLoss


def parse_gate_hidden_dims(dims_str: str) -> tuple:
    """Parse comma-separated hidden dimensions string."""
    return tuple(int(d) for d in dims_str.split(','))


def create_adaptive_mtp_config_from_training_config(training_config) -> AdaptiveMTPConfig:
    """
    Create AdaptiveMTPConfig from EnhancedTrainingConfig.

    Args:
        training_config: EnhancedTrainingConfig object

    Returns:
        AdaptiveMTPConfig object
    """
    mtp_cfg = training_config.adaptive_mtp

    # Parse gate hidden dims
    gate_hidden_dims = parse_gate_hidden_dims(mtp_cfg.gate_hidden_dims)

    return AdaptiveMTPConfig(
        num_prediction_heads=mtp_cfg.num_prediction_heads,
        confidence_threshold_train=mtp_cfg.confidence_threshold_train,
        confidence_threshold_inference=mtp_cfg.confidence_threshold_inference,
        gate_hidden_dims=gate_hidden_dims,
        gate_dropout=mtp_cfg.gate_dropout,
        gate_activation=mtp_cfg.gate_activation,
        use_attention_pooling=mtp_cfg.use_attention_pooling,
        head_type=mtp_cfg.head_type,
        head_intermediate_size=mtp_cfg.head_intermediate_size,
        head_dropout=mtp_cfg.head_dropout,
        share_projections=mtp_cfg.share_projections,
        mtp_warmup_epochs=mtp_cfg.mtp_warmup_epochs,
        confidence_reg_strength=mtp_cfg.confidence_reg_strength,
        use_confidence_weighting=mtp_cfg.use_confidence_weighting,
        primary_loss_weight=mtp_cfg.primary_loss_weight,
        additional_loss_base_weight=mtp_cfg.additional_loss_base_weight,
        enable_dynamic_prediction=mtp_cfg.enable_dynamic_prediction,
        min_confidence_for_computation=mtp_cfg.min_confidence_for_computation,
    )


def wrap_model_with_adaptive_mtp(
    base_model: nn.Module,
    training_config: Any,
    vocab_size: int,
    hidden_size: int,
) -> nn.Module:
    """
    Wrap a base model with Adaptive MTP if enabled in config.

    Args:
        base_model: The base transformer model
        training_config: EnhancedTrainingConfig with MTP settings
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension of the model

    Returns:
        Wrapped model or original model if MTP not enabled
    """
    # Check if adaptive MTP is enabled
    if not training_config.adaptive_mtp.use_adaptive_mtp:
        print("Adaptive MTP is disabled in config")
        return base_model

    print("\n" + "="*80)
    print("🚀 Initializing Adaptive Multi-Token Prediction System")
    print("="*80)

    # Create MTP configuration
    mtp_config = create_adaptive_mtp_config_from_training_config(training_config)

    print(f"Configuration:")
    print(f"  Number of prediction heads: {mtp_config.num_prediction_heads}")
    print(f"  Confidence threshold (train): {mtp_config.confidence_threshold_train}")
    print(f"  Confidence threshold (inference): {mtp_config.confidence_threshold_inference}")
    print(f"  Warmup epochs: {mtp_config.mtp_warmup_epochs}")
    print(f"  Head type: {mtp_config.head_type}")
    print(f"  Gate hidden dims: {mtp_config.gate_hidden_dims}")

    # Create wrapped model
    adaptive_model = AdaptiveMTPModel(
        base_model=base_model,
        config=mtp_config,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )

    # Count parameters
    gate_params = sum(p.numel() for p in adaptive_model.confidence_gate.parameters())
    head_params_info = adaptive_model.prediction_heads.get_parameters_count()

    print(f"\nParameter counts:")
    print(f"  Confidence gate: {gate_params:,} parameters")
    print(f"  Prediction heads: {head_params_info['total_params']:,} parameters")
    print(f"    Per head: {head_params_info['per_head_params']:,} parameters")

    total_base_params = sum(p.numel() for p in base_model.parameters())
    total_mtp_params = gate_params + head_params_info['total_params']
    overhead_pct = (total_mtp_params / total_base_params) * 100

    print(f"\nOverhead: {total_mtp_params:,} parameters ({overhead_pct:.2f}% of base model)")
    print("="*80 + "\n")

    return adaptive_model


def create_adaptive_mtp_loss(
    training_config: Any,
    vocab_size: int,
) -> AdaptiveMTPLoss:
    """
    Create AdaptiveMTPLoss from training configuration.

    Args:
        training_config: EnhancedTrainingConfig with MTP settings
        vocab_size: Vocabulary size

    Returns:
        AdaptiveMTPLoss instance
    """
    mtp_cfg = training_config.adaptive_mtp

    return AdaptiveMTPLoss(
        vocab_size=vocab_size,
        primary_loss_weight=mtp_cfg.primary_loss_weight,
        additional_loss_base_weight=mtp_cfg.additional_loss_base_weight,
        confidence_reg_strength=mtp_cfg.confidence_reg_strength,
        use_confidence_weighting=mtp_cfg.use_confidence_weighting,
        label_smoothing=training_config.losses.label_smoothing,
        ignore_index=-100,
    )


def compute_loss_with_adaptive_mtp(
    model_outputs: Dict[str, Any],
    labels: torch.Tensor,
    loss_fn: AdaptiveMTPLoss,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute loss using Adaptive MTP loss function.

    Args:
        model_outputs: Dictionary from AdaptiveMTPModel forward pass
        labels: Target labels [batch_size, seq_len]
        loss_fn: AdaptiveMTPLoss instance
        attention_mask: Attention mask [batch_size, seq_len]

    Returns:
        Dictionary with loss and metadata
    """
    return loss_fn(
        primary_logits=model_outputs['primary_logits'],
        targets=labels,
        additional_logits=model_outputs.get('additional_logits'),
        confidence_scores=model_outputs.get('confidence_scores'),
        attention_mask=attention_mask,
        mtp_active=model_outputs.get('mtp_active', False),
    )


def log_adaptive_mtp_metrics(
    model: AdaptiveMTPModel,
    loss_fn: AdaptiveMTPLoss,
    step: int,
    logger = None,
) -> Dict[str, float]:
    """
    Log Adaptive MTP metrics.

    Args:
        model: AdaptiveMTPModel instance
        loss_fn: AdaptiveMTPLoss instance
        step: Current training step
        logger: Optional logger (wandb, tensorboard, etc.)

    Returns:
        Dictionary with metrics
    """
    # Get model statistics
    model_stats = model.get_mtp_statistics()

    # Get loss statistics
    loss_stats = loss_fn.get_statistics()

    # Combine metrics
    metrics = {
        'mtp/usage_ratio': model_stats['mtp_usage_ratio'],
        'mtp/avg_confidence': model_stats['avg_confidence'],
        'mtp/high_confidence_ratio': model_stats['high_confidence_ratio'],
        'mtp/total_predictions': model_stats['total_predictions'],
        'mtp/in_warmup': float(model_stats['in_warmup']),
        'loss/primary': loss_stats['avg_primary_loss'],
        'loss/additional': loss_stats['avg_additional_loss'],
        'loss/confidence_reg': loss_stats['avg_confidence_reg'],
    }

    # Log to wandb if available
    if logger is not None and hasattr(logger, 'log'):
        logger.log(metrics, step=step)

    return metrics


# Example training loop integration
def example_training_loop_with_mtp():
    """
    Example showing how to integrate Adaptive MTP into training loop.
    This is pseudocode showing the key integration points.

    NOTE: This is pseudocode for documentation purposes.
    Variables like base_model, training_config, etc. are placeholders.
    """
    # type: ignore - This is pseudocode
    # ... (initialization code)
    # Placeholder variables (would be defined in actual code):
    base_model: Any = None  # type: ignore
    training_config: Any = None  # type: ignore
    vocab_size: int = 50000  # type: ignore
    hidden_size: int = 768  # type: ignore
    num_epochs: int = 3  # type: ignore
    train_loader: Any = None  # type: ignore
    optimizer: Any = None  # type: ignore
    wandb_logger: Any = None  # type: ignore

    # 1. Wrap model with Adaptive MTP
    model = wrap_model_with_adaptive_mtp(
        base_model=base_model,
        training_config=training_config,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )

    # 2. Create MTP loss function
    if training_config.adaptive_mtp.use_adaptive_mtp:  # type: ignore
        mtp_loss_fn = create_adaptive_mtp_loss(training_config, vocab_size)
    else:
        mtp_loss_fn = None

    # 3. Training loop
    for epoch in range(num_epochs):
        # Set current epoch for warmup tracking
        if isinstance(model, AdaptiveMTPModel):
            model.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )

            # Compute loss
            if isinstance(model, AdaptiveMTPModel) and mtp_loss_fn is not None:
                loss_outputs = compute_loss_with_adaptive_mtp(
                    model_outputs=outputs,
                    labels=batch['labels'],
                    loss_fn=mtp_loss_fn,
                    attention_mask=batch['attention_mask'],
                )
                loss = loss_outputs['loss']
            else:
                # Standard loss computation (pseudocode)
                def compute_standard_loss(outputs, labels):  # type: ignore
                    """Placeholder for standard loss function."""
                    return nn.CrossEntropyLoss()(outputs, labels)
                loss = compute_standard_loss(outputs, batch['labels'])

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log metrics periodically
            if step % 100 == 0 and isinstance(model, AdaptiveMTPModel) and mtp_loss_fn is not None:
                metrics = log_adaptive_mtp_metrics(
                    model=model,
                    loss_fn=mtp_loss_fn,
                    step=step,
                    logger=wandb_logger,
                )
                print(f"Step {step}: MTP Usage={metrics['mtp/usage_ratio']:.2%}, "
                      f"Confidence={metrics['mtp/avg_confidence']:.3f}")

        # Reset statistics at end of epoch
        if isinstance(model, AdaptiveMTPModel):
            model.reset_statistics()
            if mtp_loss_fn is not None:
                mtp_loss_fn.reset_statistics()
