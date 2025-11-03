#!/usr/bin/env python3
"""
RLHF Fine-tuning Script

This script performs RLHF (Reinforcement Learning from Human Feedback) fine-tuning
using PPO (Proximal Policy Optimization) with model-to-model rating.

Usage:
    python train_rlhf.py --config configs/rlhf/rlhf_config.yaml
    python train_rlhf.py --config configs/rlhf/rlhf_config.yaml --resume checkpoints/checkpoint.pt
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import torch
import torch.nn as nn
import argparse
import logging
from transformers import AutoTokenizer
import yaml

from src.rlhf import RLHFTrainer
from src.rlhf.ppo_trainer import PPOConfig
from src.rlhf.rlhf_trainer import RLHFConfig
from Ava.config.training_config import TrainingConfigManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: str, config: dict, device: str) -> nn.Module:
    """
    Load a model from checkpoint or create new.

    Args:
        model_path: Path to model checkpoint
        config: Model configuration
        device: Device to load model on

    Returns:
        Loaded model
    """
    from Ava.models import EnhancedMoEModel  # type: ignore[attr-defined]
    from typing import cast, Type

    # Type narrowing: assert EnhancedMoEModel is available
    if EnhancedMoEModel is None:  # type: ignore[has-type]
        raise ImportError("EnhancedMoEModel is not available. Please check your installation.")

    # Type assertion for Pylance - ensures type checker knows this is a valid class
    ModelClass: Type[nn.Module] = cast(Type[nn.Module], EnhancedMoEModel)  # type: ignore[redundant-cast]

    logger.info(f"Loading model from {model_path}")

    # Load model
    if Path(model_path).exists():
        model: nn.Module = ModelClass.from_pretrained(model_path)  # type: ignore[attr-defined]
        logger.info(f"Loaded existing model from {model_path}")
    else:
        # Create new model from config
        model = ModelClass(config)  # type: ignore[call-arg]
        logger.info("Created new model from config")

    return model.to(device)


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_manager = TrainingConfigManager()
    config = config_manager.load_yaml_config(config_path)
    return config.to_dict()


def create_rlhf_config(config_dict: dict) -> RLHFConfig:
    """
    Create RLHF configuration from config dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        RLHFConfig object
    """
    rlhf_config = config_dict.get('rlhf', {})
    ppo_config_dict = rlhf_config.get('ppo', {})

    # Create PPO config
    ppo_config = PPOConfig(
        learning_rate=ppo_config_dict.get('learning_rate', 1e-5),
        batch_size=ppo_config_dict.get('batch_size', 128),
        mini_batch_size=ppo_config_dict.get('mini_batch_size', 32),
        gradient_accumulation_steps=ppo_config_dict.get('gradient_accumulation_steps', 1),
        ppo_epochs=ppo_config_dict.get('ppo_epochs', 4),
        clip_range=ppo_config_dict.get('clip_range', 0.2),
        clip_range_value=ppo_config_dict.get('clip_range_value', 0.2),
        value_loss_coef=ppo_config_dict.get('value_loss_coef', 0.5),
        entropy_coef=ppo_config_dict.get('entropy_coef', 0.01),
        max_grad_norm=ppo_config_dict.get('max_grad_norm', 1.0),
        gamma=ppo_config_dict.get('gamma', 1.0),
        lambda_=ppo_config_dict.get('lambda', 0.95),
        kl_penalty=ppo_config_dict.get('kl_penalty', 'kl'),
        target_kl=ppo_config_dict.get('target_kl', 0.01),
        init_kl_coef=ppo_config_dict.get('init_kl_coef', 0.2),
        adaptive_kl=ppo_config_dict.get('adaptive_kl', True),
        max_gen_length=ppo_config_dict.get('max_gen_length', 128),
        temperature=ppo_config_dict.get('temperature', 1.0),
        top_k=ppo_config_dict.get('top_k', 50),
        top_p=ppo_config_dict.get('top_p', 0.95),
        num_train_epochs=ppo_config_dict.get('num_train_epochs', 1),
        max_steps=ppo_config_dict.get('max_steps', None),
        warmup_steps=ppo_config_dict.get('warmup_steps', 100),
        logging_steps=ppo_config_dict.get('logging_steps', 10),
        save_steps=ppo_config_dict.get('save_steps', 500),
        eval_steps=ppo_config_dict.get('eval_steps', 500),
        use_score_scaling=ppo_config_dict.get('use_score_scaling', True),
        use_score_norm=ppo_config_dict.get('use_score_norm', True),
        whiten_rewards=ppo_config_dict.get('whiten_rewards', True),
        clip_reward=ppo_config_dict.get('clip_reward', 10.0),
        gradient_checkpointing=ppo_config_dict.get('gradient_checkpointing', True),
        mixed_precision=ppo_config_dict.get('mixed_precision', 'bf16')
    )

    # Create RLHF config
    config = RLHFConfig(
        ppo=ppo_config,
        rollout_batch_size=rlhf_config.get('rollout_batch_size', 128),
        num_rollouts_per_epoch=rlhf_config.get('num_rollouts_per_epoch', 1000),
        max_prompt_length=rlhf_config.get('max_prompt_length', 512),
        use_model_to_model_reward=rlhf_config.get('use_model_to_model_reward', True),
        reward_model_path=rlhf_config.get('reward_model_path', None),
        judge_model_path=rlhf_config.get('judge_model_path', None),
        freeze_reward_model=rlhf_config.get('freeze_reward_model', True),
        num_epochs=rlhf_config.get('num_epochs', 3),
        save_dir=rlhf_config.get('save_dir', 'outputs/rlhf'),
        log_dir=rlhf_config.get('log_dir', 'logs/rlhf'),
        save_every=rlhf_config.get('save_every', 1000),
        eval_every=rlhf_config.get('eval_every', 500),
        num_eval_prompts=rlhf_config.get('num_eval_prompts', 50),
        prompt_dataset_path=rlhf_config.get('prompt_dataset_path'),
        eval_prompt_dataset_path=rlhf_config.get('eval_prompt_dataset_path', None),
        device=rlhf_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        use_wandb=rlhf_config.get('use_wandb', True),
        wandb_project=rlhf_config.get('wandb_project', 'ava-rlhf'),
        wandb_name=rlhf_config.get('wandb_name', None)
    )

    return config


def main():
    parser = argparse.ArgumentParser(description='RLHF Fine-tuning')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to RLHF configuration file'
    )
    parser.add_argument(
        '--policy-model',
        type=str,
        default=None,
        help='Path to policy model checkpoint (overrides config)'
    )
    parser.add_argument(
        '--judge-model',
        type=str,
        default=None,
        help='Path to judge model checkpoint (overrides config)'
    )
    parser.add_argument(
        '--reward-model',
        type=str,
        default=None,
        help='Path to reward model checkpoint (overrides config)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, overrides config)'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config_dict = load_config(args.config)

    # Create RLHF config
    rlhf_config = create_rlhf_config(config_dict)

    # Apply command-line overrides
    if args.device is not None:
        rlhf_config.device = args.device
    if args.no_wandb:
        rlhf_config.use_wandb = False

    logger.info(f"Using device: {rlhf_config.device}")

    # Load tokenizer (custom tokenizer support)
    tokenizer_path = config_dict.get('data', {}).get('tokenizer_name', '/project/code/models/tokenizer/enhanced-65536')
    logger.info(f"Loading tokenizer from {tokenizer_path}")

    try:
        # Try loading as custom tokenizer first
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        logger.info(f"Loaded custom tokenizer with vocab size: {len(tokenizer)}")
    except:
        # Fallback to AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Loaded tokenizer with vocab size: {len(tokenizer)}")

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        else:
            # Add pad token if neither exists
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Added [PAD] token to tokenizer")

    # Load policy model
    policy_model_path = args.policy_model or config_dict.get('rlhf', {}).get('policy_model_path')
    if policy_model_path is None:
        raise ValueError("Policy model path not specified in config or command line")

    policy_model = load_model(policy_model_path, config_dict.get('model', {}), rlhf_config.device)

    # Load judge/reward model
    judge_model = None
    reward_model = None

    if rlhf_config.use_model_to_model_reward:
        # Load judge model
        judge_model_path = args.judge_model or rlhf_config.judge_model_path
        if judge_model_path is None:
            logger.warning("No judge model specified, using policy model as judge")
            judge_model = policy_model
        else:
            logger.info(f"Loading judge model from {judge_model_path}")
            judge_model = load_model(judge_model_path, config_dict.get('model', {}), rlhf_config.device)
    else:
        # Load reward model
        reward_model_path = args.reward_model or rlhf_config.reward_model_path
        if reward_model_path is not None:
            logger.info(f"Loading reward model from {reward_model_path}")
            reward_model = load_model(reward_model_path, config_dict.get('model', {}), rlhf_config.device)

    # Create RLHF trainer
    logger.info("Initializing RLHF trainer")
    trainer = RLHFTrainer(
        config=rlhf_config,
        model=policy_model,
        tokenizer=tokenizer,
        reward_model=reward_model,
        judge_model=judge_model
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    logger.info("Starting RLHF training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(
            epoch=trainer.global_step // len(trainer.train_prompts),
            step=trainer.global_step,
            is_epoch_end=False
        )
        logger.info("Saved checkpoint before exiting")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
