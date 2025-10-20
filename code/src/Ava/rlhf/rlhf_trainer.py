"""
Main RLHF Trainer

Orchestrates the full RLHF training pipeline including:
- Experience collection
- Reward computation
- PPO updates
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass
import logging
from pathlib import Path
from tqdm import tqdm
import json

from .ppo_trainer import PPOTrainer, PPOConfig
from .reward_model import RewardModel, ModelToModelReward

logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """Configuration for RLHF training."""
    # PPO configuration
    ppo: PPOConfig = None

    # Experience collection
    rollout_batch_size: int = 128
    num_rollouts_per_epoch: int = 1000
    max_prompt_length: int = 512

    # Reward model settings
    use_model_to_model_reward: bool = True
    reward_model_path: Optional[str] = None
    judge_model_path: Optional[str] = None
    freeze_reward_model: bool = True

    # Training
    num_epochs: int = 3
    save_dir: str = "outputs/rlhf"
    log_dir: str = "logs/rlhf"
    save_every: int = 1000
    eval_every: int = 500
    num_eval_prompts: int = 50

    # Data
    prompt_dataset_path: str = None
    eval_prompt_dataset_path: Optional[str] = None

    # Device
    device: str = "cuda"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "ava-rlhf"
    wandb_name: Optional[str] = None


class RLHFTrainer:
    """
    Main RLHF Trainer that orchestrates the full training pipeline.
    """

    def __init__(
        self,
        config: RLHFConfig,
        model: nn.Module,
        tokenizer: Any,
        reward_model: Optional[Any] = None,
        judge_model: Optional[nn.Module] = None
    ):
        """
        Initialize RLHF trainer.

        Args:
            config: RLHF configuration
            model: Policy model to train
            tokenizer: Tokenizer
            reward_model: Optional pre-trained reward model
            judge_model: Optional judge model for model-to-model rating
        """
        self.config = config
        self.tokenizer = tokenizer
        self.device = config.device

        # Setup models
        self.model = model.to(self.device)
        self.ref_model = self._create_reference_model(model)

        # Setup reward model
        if config.use_model_to_model_reward:
            assert judge_model is not None, "Judge model required for model-to-model reward"
            self.reward_model = ModelToModelReward(
                judge_model=judge_model,
                tokenizer=tokenizer,
                device=self.device
            )
            logger.info("Using model-to-model reward system")
        else:
            if reward_model is None:
                # Create reward model from policy model
                from .reward_model import RewardModelConfig
                reward_config = RewardModelConfig(
                    hidden_size=model.config.hidden_size if hasattr(model, 'config') else 512
                )
                self.reward_model = RewardModel(
                    base_model=model,
                    config=reward_config,
                    freeze_base=config.freeze_reward_model
                )
                logger.info("Created reward model from policy model")
            else:
                self.reward_model = reward_model
                logger.info("Using provided reward model")

        # Only call .to() if reward model is a nn.Module
        if hasattr(self.reward_model, 'to'):
            self.reward_model = self.reward_model.to(self.device)

        # Setup PPO trainer
        if config.ppo is None:
            config.ppo = PPOConfig()

        self.ppo_trainer = PPOTrainer(
            config=config.ppo,
            model=self.model,
            ref_model=self.ref_model,
            reward_model=self.reward_model,
            tokenizer=tokenizer,
            device=self.device
        )

        # Load datasets
        self.train_prompts = self._load_prompts(config.prompt_dataset_path)
        if config.eval_prompt_dataset_path:
            self.eval_prompts = self._load_prompts(config.eval_prompt_dataset_path)
        else:
            # Use a subset of training prompts for evaluation
            self.eval_prompts = self.train_prompts[:config.num_eval_prompts]

        logger.info(f"Loaded {len(self.train_prompts)} training prompts")
        logger.info(f"Loaded {len(self.eval_prompts)} evaluation prompts")

        # Setup directories
        self.save_dir = Path(config.save_dir)
        self.log_dir = Path(config.log_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb if enabled
        self.use_wandb = config.use_wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_name,
                    config=vars(config)
                )
                logger.info("Initialized Weights & Biases logging")
            except ImportError:
                logger.warning("wandb not installed, disabling wandb logging")
                self.use_wandb = False

        self.global_step = 0

    def _create_reference_model(self, model: nn.Module) -> nn.Module:
        """
        Create a frozen reference model copy.

        Args:
            model: Model to copy

        Returns:
            Frozen reference model
        """
        import copy
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model.to(self.device)

    def _load_prompts(self, path: str) -> List[str]:
        """
        Load prompts from a file.

        Args:
            path: Path to prompts file (text or JSON)

        Returns:
            List of prompts
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Prompts file not found: {path}")

        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    prompts = data
                elif isinstance(data, dict) and 'prompts' in data:
                    prompts = data['prompts']
                else:
                    raise ValueError("JSON file must contain a list or dict with 'prompts' key")
        else:
            # Assume text file with one prompt per line
            with open(path, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]

        return prompts

    def collect_experience(
        self,
        prompts: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Collect experience by generating responses and computing rewards.

        Args:
            prompts: List of prompts to generate responses for

        Returns:
            Dictionary containing experience data
        """
        # Generate responses
        gen_ids, attention_mask, responses = self.ppo_trainer.generate_responses(
            prompts,
            max_length=self.config.ppo.max_gen_length
        )

        # Compute rewards
        rewards = self.ppo_trainer.compute_rewards(
            gen_ids,
            attention_mask,
            prompts,
            responses
        )

        # Get log probabilities from policy
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=gen_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            action_logprobs = torch.gather(
                logprobs,
                2,
                gen_ids.unsqueeze(-1)
            ).squeeze(-1)

        # Get log probabilities from reference model
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=gen_ids, attention_mask=attention_mask)
            ref_logits = ref_outputs.logits if hasattr(ref_outputs, 'logits') else ref_outputs[0]
            ref_logprobs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
            ref_action_logprobs = torch.gather(
                ref_logprobs,
                2,
                gen_ids.unsqueeze(-1)
            ).squeeze(-1)

        # Placeholder for value estimates (could add a value head)
        # Ensure values match the shape of logprobs
        values = torch.zeros_like(action_logprobs)

        # Also ensure all tensors have consistent seq_len
        seq_len = gen_ids.size(1)
        if attention_mask.size(1) != seq_len:
            # Pad attention_mask if needed
            if attention_mask.size(1) < seq_len:
                padding = torch.zeros(
                    attention_mask.size(0),
                    seq_len - attention_mask.size(1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                attention_mask = torch.cat([attention_mask, padding], dim=1)
            else:
                attention_mask = attention_mask[:, :seq_len]

        return {
            'input_ids': gen_ids,
            'attention_mask': attention_mask,
            'logprobs': action_logprobs,
            'ref_logprobs': ref_action_logprobs,
            'values': values,
            'rewards': rewards,
            'prompts': prompts,
            'responses': responses
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

        epoch_stats = {
            'rewards': [],
            'policy_loss': [],
            'kl_div': [],
            'entropy': []
        }

        # Create progress bar
        pbar = tqdm(
            range(0, len(self.train_prompts), self.config.rollout_batch_size),
            desc=f"Epoch {epoch + 1}"
        )

        for i in pbar:
            # Get batch of prompts
            batch_prompts = self.train_prompts[i:i + self.config.rollout_batch_size]

            # Collect experience
            experience = self.collect_experience(batch_prompts)

            # Train with PPO
            train_stats = self.ppo_trainer.train_step(experience)

            # Update statistics
            epoch_stats['rewards'].append(experience['rewards'].mean().item())
            epoch_stats['policy_loss'].append(train_stats.get('policy_loss', 0))
            epoch_stats['kl_div'].append(train_stats.get('kl_div', 0))
            epoch_stats['entropy'].append(train_stats.get('entropy', 0))

            # Update progress bar
            pbar.set_postfix({
                'reward': f"{epoch_stats['rewards'][-1]:.4f}",
                'kl': f"{epoch_stats['kl_div'][-1]:.4f}"
            })

            # Logging
            if self.global_step % self.config.ppo.logging_steps == 0:
                log_dict = {
                    'train/reward': epoch_stats['rewards'][-1],
                    'train/policy_loss': epoch_stats['policy_loss'][-1],
                    'train/kl_div': epoch_stats['kl_div'][-1],
                    'train/entropy': epoch_stats['entropy'][-1],
                    'train/kl_coef': train_stats.get('kl_coef', 0),
                    'global_step': self.global_step
                }
                if self.use_wandb:
                    import wandb
                    wandb.log(log_dict)

                logger.info(f"Step {self.global_step}: " +
                          " | ".join([f"{k}: {v:.4f}" for k, v in log_dict.items() if k != 'global_step']))

            # Evaluation
            if self.global_step % self.config.eval_every == 0:
                eval_stats = self.evaluate()
                if self.use_wandb:
                    import wandb
                    wandb.log({f'eval/{k}': v for k, v in eval_stats.items()})
                logger.info(f"Evaluation: " +
                          " | ".join([f"{k}: {v:.4f}" for k, v in eval_stats.items()]))

            # Save checkpoint
            if self.global_step % self.config.save_every == 0:
                self.save_checkpoint(epoch, self.global_step)

            self.global_step += 1

        # Compute epoch averages
        epoch_avg = {k: sum(v) / len(v) for k, v in epoch_stats.items()}
        return epoch_avg

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Running evaluation...")
        eval_stats = self.ppo_trainer.evaluate(self.eval_prompts)
        return eval_stats

    def train(self):
        """
        Run the full RLHF training loop.
        """
        logger.info("Starting RLHF training")
        logger.info(f"Training for {self.config.num_epochs} epochs")
        logger.info(f"Rollout batch size: {self.config.rollout_batch_size}")
        logger.info(f"PPO config: {self.config.ppo}")

        for epoch in range(self.config.num_epochs):
            epoch_stats = self.train_epoch(epoch)

            logger.info(f"Epoch {epoch + 1} completed:")
            for k, v in epoch_stats.items():
                logger.info(f"  {k}: {v:.4f}")

            # Save end-of-epoch checkpoint
            self.save_checkpoint(epoch, self.global_step, is_epoch_end=True)

        logger.info("RLHF training completed!")

        # Final evaluation
        final_eval = self.evaluate()
        logger.info("Final evaluation:")
        for k, v in final_eval.items():
            logger.info(f"  {k}: {v:.4f}")

        if self.use_wandb:
            import wandb
            wandb.finish()

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        is_epoch_end: bool = False
    ):
        """
        Save training checkpoint.

        Args:
            epoch: Current epoch
            step: Current global step
            is_epoch_end: Whether this is an end-of-epoch checkpoint
        """
        if is_epoch_end:
            checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pt"
        else:
            checkpoint_name = f"checkpoint_step_{step}.pt"

        checkpoint_path = self.save_dir / checkpoint_name

        checkpoint = {
            'epoch': epoch,
            'global_step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.ppo_trainer.optimizer.state_dict(),
            'config': vars(self.config),
            'kl_coef': self.ppo_trainer.kl_coef
        }

        if self.ppo_trainer.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.ppo_trainer.lr_scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Also save the model separately for easy loading
        model_path = self.save_dir / f"model_step_{step}.pt"
        torch.save(self.model.state_dict(), model_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ppo_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.ppo_trainer.kl_coef = checkpoint.get('kl_coef', self.config.ppo.init_kl_coef)
        self.global_step = checkpoint['global_step']

        if 'scheduler_state_dict' in checkpoint and self.ppo_trainer.lr_scheduler is not None:
            self.ppo_trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
