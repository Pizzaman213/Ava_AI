#!/usr/bin/env python3
"""
Test minimal_working_fixed.yaml with DeepSpeed ZeRO-2.
Quick validation that the config works correctly.
"""

import os
import sys
import torch
import logging

# Set environment for single-GPU distributed
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29503'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['PYTHONPATH'] = 'code/src'

sys.path.insert(0, 'code/src')

from ava.config.yaml_loader import load_yaml_with_path_resolution
from ava.training.model_builder import ModelBuilder
from ava.training.context import TrainingContext
from ava.training.deepspeed_utils import is_deepspeed_engine, get_zero_stage

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_minimal_config():
    """Test minimal_working_fixed.yaml with DeepSpeed ZeRO-2"""

    logger.info("="*70)
    logger.info("Testing minimal_working_fixed.yaml with DeepSpeed ZeRO-2")
    logger.info("="*70)

    # Load config
    config = load_yaml_with_path_resolution('code/configs/moe/minimal_working_fixed.yaml')

    # Verify DeepSpeed is enabled
    assert config['deepspeed']['enabled'] == True, "DeepSpeed should be enabled in config"
    assert config['deepspeed']['zero_stage'] == 2, "Should use ZeRO-2"
    logger.info("✓ DeepSpeed enabled in config (ZeRO-2)")

    # Create training context
    context = TrainingContext(config)
    context.config = config
    context.rank = 0
    context.world_size = 1
    context.device = torch.device('cuda:0')

    # Initialize distributed
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
    logger.info("✓ Distributed initialized")

    # Build model
    logger.info("Building model...")
    builder = ModelBuilder(context)
    builder.initialize()
    model = builder.build_model(config=config, device=context.device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Model built: {num_params:,} parameters")

    # Wrap with DeepSpeed
    logger.info("Initializing DeepSpeed engine...")
    model = builder.wrap_distributed(model, rank=0, world_size=1)

    # Verify it's a DeepSpeed engine
    assert is_deepspeed_engine(model), "Model should be a DeepSpeed engine!"
    logger.info("✓ DeepSpeed engine initialized")

    # Get ZeRO stage
    zero_stage = get_zero_stage(model)
    assert zero_stage == 2, f"Expected ZeRO-2, got ZeRO-{zero_stage}"
    logger.info(f"✓ ZeRO stage verified: {zero_stage}")

    # Test forward pass with synthetic data
    logger.info("Testing forward pass...")
    batch_size = 4
    seq_len = 128
    vocab_size = config['model']['vocab_size']

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda:0')
    attention_mask = torch.ones(batch_size, seq_len, device='cuda:0')

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Handle output format
    if isinstance(outputs, dict):
        loss = outputs.get('loss') or outputs.get('logits').mean()
    elif hasattr(outputs, 'loss'):
        loss = outputs.loss
    else:
        loss = outputs.mean()

    logger.info(f"✓ Forward pass successful, loss: {loss.item():.6f}")

    # Test backward + step
    logger.info("Testing backward + optimizer step...")
    model.backward(loss)
    model.step()
    logger.info("✓ Training step successful")

    # Verify config consistency
    ds_config = model.config
    logger.info("\nDeepSpeed Configuration:")
    logger.info(f"  - Micro batch size: {ds_config.get('train_micro_batch_size_per_gpu')}")
    logger.info(f"  - Gradient accumulation: {ds_config.get('gradient_accumulation_steps')}")
    logger.info(f"  - Train batch size: {ds_config.get('train_batch_size')}")
    logger.info(f"  - Precision: {'BF16' if ds_config.get('bf16', {}).get('enabled') else 'FP32'}")
    logger.info(f"  - ZeRO stage: {zero_stage}")

    logger.info("="*70)
    logger.info("✓ ALL TESTS PASSED - minimal_working_fixed.yaml works with ZeRO-2!")
    logger.info("="*70)

    # Cleanup
    dist.destroy_process_group()
    return True

if __name__ == '__main__':
    try:
        success = test_minimal_config()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
