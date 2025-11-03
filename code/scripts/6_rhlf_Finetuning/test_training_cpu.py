#!/usr/bin/env python3
"""
Test RLHF training on CPU with minimal configuration.

This creates tiny test models and runs a minimal training loop.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import torch
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_tiny_test_model(vocab_size=65536):
    """Create a tiny GPT-2 model for testing."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=128,
        n_embd=128,
        n_layer=2,
        n_head=2,
        n_inner=256,
        activation_function="gelu",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )

    model = GPT2LMHeadModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created tiny test model: {num_params/1e6:.2f}M parameters")
    return model


def main():
    logger.info("=" * 70)
    logger.info("RLHF Training Test on CPU")
    logger.info("=" * 70)
    logger.info("")

    try:
        # Import RLHF components
        from src.rlhf.rlhf_trainer import RLHFTrainer, RLHFConfig
        from src.rlhf.ppo_trainer import PPOConfig
        from transformers import PreTrainedTokenizerFast

        # Load custom tokenizer
        tokenizer_path = "/project/code/models/tokenizer/enhanced-65536"
        logger.info(f"Loading custom tokenizer from {tokenizer_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        logger.info(f"✓ Loaded tokenizer with vocab size: {len(tokenizer)}")

        # Create tiny test models
        logger.info("\nCreating test models...")
        policy_model = create_tiny_test_model(vocab_size=len(tokenizer))
        judge_model = create_tiny_test_model(vocab_size=len(tokenizer))

        # Verify prompts exist
        prompts_path = "/project/code/data/rlhf/prompts.json"
        if not Path(prompts_path).exists():
            logger.error(f"Prompts file not found: {prompts_path}")
            logger.info("Creating test prompts...")
            with open(prompts_path, 'w') as f:
                json.dump({
                    "prompts": [
                        "What is AI?",
                        "Explain machine learning:",
                        "How do neural networks work?",
                        "What is deep learning?",
                    ]
                }, f)
            logger.info(f"✓ Created test prompts at {prompts_path}")

        # Create PPO config
        logger.info("\nConfiguring PPO trainer...")
        ppo_config = PPOConfig(
            learning_rate=1e-5,
            batch_size=2,
            mini_batch_size=1,
            gradient_accumulation_steps=1,
            ppo_epochs=2,
            clip_range=0.2,
            max_gen_length=32,
            logging_steps=1,
            save_steps=100,
            eval_steps=50,
            gradient_checkpointing=False,
            mixed_precision='no',
            warmup_steps=0
        )

        # Create RLHF config
        logger.info("Configuring RLHF trainer...")
        rlhf_config = RLHFConfig(
            ppo=ppo_config,
            rollout_batch_size=2,
            num_rollouts_per_epoch=2,  # Just 2 rollouts for quick test
            max_prompt_length=64,
            use_model_to_model_reward=True,
            num_epochs=1,
            save_dir="/tmp/rlhf_test_output",
            log_dir="/tmp/rlhf_test_logs",
            save_every=100,
            eval_every=50,
            num_eval_prompts=2,
            prompt_dataset_path=prompts_path,
            device='cpu',
            use_wandb=False
        )

        # Create output directories
        Path(rlhf_config.save_dir).mkdir(parents=True, exist_ok=True)
        Path(rlhf_config.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize RLHF trainer
        logger.info("\n" + "=" * 70)
        logger.info("Initializing RLHF Trainer")
        logger.info("=" * 70)
        trainer = RLHFTrainer(
            config=rlhf_config,
            model=policy_model,
            tokenizer=tokenizer,
            judge_model=judge_model
        )
        logger.info("✓ RLHF trainer initialized successfully!")

        # Test one training epoch
        logger.info("\n" + "=" * 70)
        logger.info("Starting Training Test (1 epoch, 2 rollouts)")
        logger.info("=" * 70)
        logger.info("")

        # Reduce the number of training steps for testing
        trainer.config.num_rollouts_per_epoch = 2

        # Run training
        epoch_stats = trainer.train_epoch(epoch=0)

        logger.info("\n" + "=" * 70)
        logger.info("✓ Training Test Completed Successfully!")
        logger.info("=" * 70)
        logger.info("\nEpoch Statistics:")
        for key, value in epoch_stats.items():
            logger.info(f"  {key}: {value:.4f}")

        # Verify outputs
        logger.info("\nVerifying outputs...")
        save_dir = Path(rlhf_config.save_dir)
        if save_dir.exists():
            files = list(save_dir.glob("*"))
            logger.info(f"✓ Found {len(files)} files in output directory")
            for f in files[:5]:  # Show first 5 files
                logger.info(f"  - {f.name}")

        logger.info("\n" + "=" * 70)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("=" * 70)
        logger.info("\nRLHF training pipeline is working correctly!")
        logger.info("\nNext steps:")
        logger.info("1. Train a base model (or use existing checkpoint)")
        logger.info("2. Prepare quality prompts for your domain")
        logger.info("3. Run full RLHF training on GPU:")
        logger.info("   python scripts/6_rhlf_Finetuning/train_rlhf.py \\")
        logger.info("     --config configs/gpu/small.yaml")
        logger.info("")

        return 0

    except Exception as e:
        logger.error("\n" + "=" * 70)
        logger.error(f"✗ Training test failed: {e}")
        logger.error("=" * 70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
