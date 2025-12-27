#!/usr/bin/env python3
"""
Test DeepSpeed integration with synthetic data.
Verifies that the DeepSpeed integration works correctly.
"""

import os
import sys
import torch
import logging

# Set environment for single-GPU distributed
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29502'
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['PYTHONPATH'] = 'code/src'

sys.path.insert(0, 'code/src')

from ava.config.yaml_loader import load_yaml_with_path_resolution
from ava.training.model_builder import ModelBuilder
from ava.training.context import TrainingContext
from ava.training.deepspeed_utils import is_deepspeed_engine, get_zero_stage, log_deepspeed_info

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_deepspeed_integration():
    """Test DeepSpeed integration with synthetic data"""

    logger.info("="*60)
    logger.info("Testing DeepSpeed Integration")
    logger.info("="*60)

    # Load config
    config = load_yaml_with_path_resolution('code/configs/moe/deepspeed_test.yaml')

    # Enable DeepSpeed
    if 'deepspeed' not in config:
        config['deepspeed'] = {}
    config['deepspeed']['enabled'] = True
    config['deepspeed']['zero_stage'] = 2

    # Create training context with the modified config
    context = TrainingContext(config)
    context.config = config  # Ensure context has the config
    context.rank = 0
    context.world_size = 1
    context.device = torch.device('cuda:0')

    # Initialize distributed
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')

    # Build model
    logger.info("Building model...")
    builder = ModelBuilder(context)
    builder.initialize()
    model = builder.build_model(config=config, device=context.device)

    # Wrap with DeepSpeed
    logger.info("Initializing DeepSpeed engine...")
    logger.info(f"DeepSpeed config enabled: {config['deepspeed']['enabled']}")
    logger.info(f"ZeRO stage: {config['deepspeed']['zero_stage']}")
    model = builder.wrap_distributed(model, rank=0, world_size=1)
    logger.info(f"Returned model type: {type(model).__name__}")

    # Verify it's a DeepSpeed engine
    assert is_deepspeed_engine(model), "Model is not a DeepSpeed engine!"
    logger.info("✓ Model is a DeepSpeed engine")

    # Get ZeRO stage
    zero_stage = get_zero_stage(model)
    logger.info(f"✓ ZeRO stage: {zero_stage}")
    assert zero_stage == 2, f"Expected ZeRO-2, got ZeRO-{zero_stage}"

    # Log DeepSpeed info
    log_deepspeed_info(model, rank=0)

    # Test forward pass
    logger.info("Testing forward pass...")
    batch_size = 4
    seq_len = 128
    vocab_size = config['model']['vocab_size']

    # Create synthetic input (valid token IDs)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda:0')
    attention_mask = torch.ones(batch_size, seq_len, device='cuda:0')

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # Handle different output formats
    if isinstance(outputs, dict):
        loss = outputs.get('loss') or outputs.get('logits').mean()
    elif hasattr(outputs, 'loss'):
        loss = outputs.loss
    else:
        loss = outputs.mean()
    logger.info(f"✓ Forward pass successful, loss: {loss.item():.4f}")

    # Test backward pass
    logger.info("Testing backward pass...")
    model.backward(loss)
    logger.info("✓ Backward pass successful")

    # Test optimizer step
    logger.info("Testing optimizer step...")
    model.step()
    logger.info("✓ Optimizer step successful")

    # Second iteration to verify it works consistently
    logger.info("Testing second iteration...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda:0')
    attention_mask = torch.ones(batch_size, seq_len, device='cuda:0')
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # Handle different output formats
    if isinstance(outputs, dict):
        loss = outputs.get('loss') or outputs.get('logits').mean()
    elif hasattr(outputs, 'loss'):
        loss = outputs.loss
    else:
        loss = outputs.mean()
    model.backward(loss)
    model.step()
    logger.info(f"✓ Second iteration successful, loss: {loss.item():.4f}")

    logger.info("="*60)
    logger.info("✓ ALL TESTS PASSED - DeepSpeed integration working!")
    logger.info("="*60)

    # Cleanup
    dist.destroy_process_group()

    return True

if __name__ == '__main__':
    try:
        success = test_deepspeed_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
