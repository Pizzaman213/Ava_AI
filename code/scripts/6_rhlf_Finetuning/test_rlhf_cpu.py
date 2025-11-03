#!/usr/bin/env python3
"""
Test RLHF Training on CPU

This script tests the RLHF pipeline with minimal settings on CPU.
"""

import sys
import os
from pathlib import Path
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import torch
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_tiny_test_model():
    """Create a tiny model for testing."""
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=128,
        n_layer=2,
        n_head=2,
        n_inner=256,
        activation_function="gelu",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )

    model = GPT2LMHeadModel(config)
    logger.info(f"Created tiny test model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    return model


def create_test_prompts():
    """Create simple test prompts."""
    prompts = [
        "What is AI?",
        "Explain machine learning:",
        "How does deep learning work?",
        "What are neural networks?",
        "Define artificial intelligence:",
    ]
    return prompts


def test_reward_model():
    """Test the reward model."""
    logger.info("\n=== Testing Reward Model ===")

    from src.rlhf.reward_model import ModelToModelReward
    from transformers import AutoTokenizer, GPT2Tokenizer

    # Create test model and tokenizer with matching vocab
    model = create_tiny_test_model()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Resize model embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # Create reward model
    reward = ModelToModelReward(
        judge_model=model,
        tokenizer=tokenizer,
        device='cpu'
    )

    # Test rating
    prompts = ["What is AI?"]
    responses = ["AI is artificial intelligence."]

    logger.info("Rating response...")
    rewards = reward.rate_responses(prompts, responses)
    logger.info(f"✓ Reward model works! Reward: {rewards[0]:.4f}")

    return True


def test_ppo_trainer():
    """Test the PPO trainer."""
    logger.info("\n=== Testing PPO Trainer ===")

    from src.rlhf.ppo_trainer import PPOTrainer, PPOConfig
    from src.rlhf.reward_model import ModelToModelReward
    from transformers import GPT2Tokenizer

    # Create models
    policy_model = create_tiny_test_model()
    ref_model = create_tiny_test_model()
    judge_model = create_tiny_test_model()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Resize all models to match tokenizer
    policy_model.resize_token_embeddings(len(tokenizer))
    ref_model.resize_token_embeddings(len(tokenizer))
    judge_model.resize_token_embeddings(len(tokenizer))

    # Create reward model
    reward_model = ModelToModelReward(
        judge_model=judge_model,
        tokenizer=tokenizer,
        device='cpu'
    )

    # Create PPO config
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
        ppo_epochs=1,
        max_gen_length=20,
        logging_steps=1,
        gradient_checkpointing=False,
        mixed_precision='no'
    )

    # Create PPO trainer
    trainer = PPOTrainer(
        config=ppo_config,
        model=policy_model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device='cpu'
    )

    logger.info("✓ PPO trainer initialized successfully!")

    # Test generation
    logger.info("Testing response generation...")
    prompts = ["What is AI?"]
    gen_ids, attention_mask, responses = trainer.generate_responses(prompts, max_length=20)
    logger.info(f"✓ Generated response: {responses[0][:100]}...")

    return True


def test_rlhf_trainer():
    """Test the full RLHF trainer."""
    logger.info("\n=== Testing RLHF Trainer ===")

    from src.rlhf.rlhf_trainer import RLHFTrainer, RLHFConfig
    from src.rlhf.ppo_trainer import PPOConfig
    from transformers import GPT2Tokenizer

    # Create test prompts file
    test_prompts_path = "/tmp/test_rlhf_prompts.json"
    prompts_data = {
        "prompts": create_test_prompts()
    }
    with open(test_prompts_path, 'w') as f:
        json.dump(prompts_data, f)
    logger.info(f"Created test prompts at {test_prompts_path}")

    # Create models
    policy_model = create_tiny_test_model()
    judge_model = create_tiny_test_model()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Resize models to match tokenizer
    policy_model.resize_token_embeddings(len(tokenizer))
    judge_model.resize_token_embeddings(len(tokenizer))

    # Create configs
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=1,
        ppo_epochs=1,
        max_gen_length=20,
        logging_steps=1,
        eval_steps=10,
        save_steps=10,
        gradient_checkpointing=False,
        mixed_precision='no'
    )

    rlhf_config = RLHFConfig(
        ppo=ppo_config,
        rollout_batch_size=2,
        num_rollouts_per_epoch=2,
        max_prompt_length=128,
        use_model_to_model_reward=True,
        num_epochs=1,
        save_dir="/tmp/test_rlhf_output",
        log_dir="/tmp/test_rlhf_logs",
        save_every=10,
        eval_every=5,
        num_eval_prompts=2,
        prompt_dataset_path=test_prompts_path,
        device='cpu',
        use_wandb=False
    )

    # Create RLHF trainer
    logger.info("Initializing RLHF trainer...")
    trainer = RLHFTrainer(
        config=rlhf_config,
        model=policy_model,
        tokenizer=tokenizer,
        judge_model=judge_model
    )

    logger.info("✓ RLHF trainer initialized successfully!")

    # Test experience collection
    logger.info("Testing experience collection...")
    test_prompts = ["What is AI?", "Explain ML:"]
    experience = trainer.collect_experience(test_prompts)
    logger.info(f"✓ Collected experience for {len(test_prompts)} prompts")
    rewards = experience['rewards']
    if isinstance(rewards, torch.Tensor):
        logger.info(f"  - Rewards: {rewards.tolist()}")
    else:
        logger.info(f"  - Rewards: {rewards}")

    # Test one training step
    logger.info("Testing one training step...")
    # Extract only tensor fields for train_step
    batch: Dict[str, torch.Tensor] = {k: v for k, v in experience.items() if isinstance(v, torch.Tensor) and k not in ['prompts', 'responses']}
    stats = trainer.ppo_trainer.train_step(batch)
    logger.info(f"✓ Training step completed!")
    logger.info(f"  - Policy loss: {stats.get('policy_loss', 0):.4f}")
    logger.info(f"  - KL divergence: {stats.get('kl_div', 0):.4f}")

    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("RLHF CPU Test Suite")
    logger.info("=" * 60)

    try:
        # Test 1: Reward Model
        test_reward_model()

        # Test 2: PPO Trainer
        test_ppo_trainer()

        # Test 3: Full RLHF Trainer
        test_rlhf_trainer()

        logger.info("\n" + "=" * 60)
        logger.info("✓ All tests passed successfully!")
        logger.info("=" * 60)
        logger.info("\nRLHF pipeline is working correctly on CPU!")
        logger.info("\nNext steps:")
        logger.info("1. Prepare your prompts dataset")
        logger.info("2. Train a base model (or use existing checkpoint)")
        logger.info("3. Run RLHF training:")
        logger.info("   python scripts/6_rhlf_Finetuning/train_rlhf.py --config configs/gpu/small.yaml")

        return 0

    except Exception as e:
        logger.error(f"\n✗ Test failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
