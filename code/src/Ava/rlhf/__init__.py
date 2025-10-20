"""
RLHF (Reinforcement Learning from Human Feedback) Module

This module provides components for RLHF finetuning including:
- Reward models for evaluating model responses
- PPO (Proximal Policy Optimization) trainer
- Model-to-model rating systems
"""

from .reward_model import RewardModel, ModelToModelReward
from .ppo_trainer import PPOTrainer, PPOConfig
from .rlhf_trainer import RLHFTrainer

__all__ = [
    'RewardModel',
    'ModelToModelReward',
    'PPOTrainer',
    'PPOConfig',
    'RLHFTrainer'
]