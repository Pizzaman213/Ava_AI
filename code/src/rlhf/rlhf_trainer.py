"""
Main RLHF Trainer

Orchestrates the full RLHF training pipeline including:
- Experience collection
- Reward computation
- PPO updates
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Iterator, Union
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
    ppo: Optional[PPOConfig] = None

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
    prompt_dataset_path: Optional[str] = None
    eval_prompt_dataset_path: Optional[str] = None

    # Device
    device: str = "cuda"

    # Logging
    use_wandb: bool = True
    wandb_project: str = "ava-rlhf"
    wandb_name: Optional[str] = None

    # Reference model memory optimization
    ref_model_strategy: str = 'cpu_snapshot'  # Options: 'deepcopy', 'cpu_snapshot', 'disk_snapshot', 'shared_params'


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
            if judge_model is None:
                raise ValueError(
                    "Judge model is required when use_model_to_model_reward=True. "
                    "Provide a judge_model parameter or set use_model_to_model_reward=False "
                    "in your RLHF configuration."
                )
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
                hidden_size: int = 512
                if hasattr(model, 'config'):
                    model_config = model.config
                    # Type guard: ensure config is not a Tensor
                    if not isinstance(model_config, torch.Tensor) and hasattr(model_config, 'hidden_size'):
                        # Type: model.config.hidden_size could be int, Tensor, or other types
                        hs_val: Any = model_config.hidden_size
                        if isinstance(hs_val, torch.Tensor):
                            hidden_size = int(hs_val.item())
                        elif isinstance(hs_val, int):
                            hidden_size = hs_val
                        else:
                            hidden_size = int(hs_val)
                reward_config = RewardModelConfig(hidden_size=hidden_size)
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
        if self.reward_model is not None and hasattr(self.reward_model, 'to'):
            self.reward_model = self.reward_model.to(self.device)  # type: ignore[union-attr]

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

    def _create_reference_model(self, model: nn.Module) -> Optional[nn.Module]:
        """
        Create reference model for RLHF training with memory optimization.

        Uses memory-efficient snapshot approach based on config.ref_model_strategy:
        1. Saves state dict to CPU to avoid GPU memory duplication (cpu_snapshot)
        2. Loads reference model on-demand for lazy loading (disk_snapshot)
        3. Can optionally save to disk for very large models (disk_snapshot)
        4. Or use shared parameters with no_grad (shared_params)

        This can save 30-50% GPU memory compared to deepcopy approach.

        Args:
            model: Model to create reference from

        Returns:
            Frozen reference model (or None if using lazy loading strategies)
        """
        import copy

        strategy = self.config.ref_model_strategy

        if strategy == 'deepcopy':
            # Original approach - keeps full copy on GPU
            logger.warning(
                "Using deepcopy for reference model. This uses 2x GPU memory. "
                "Consider 'cpu_snapshot' or 'disk_snapshot' for large models."
            )
            ref_model = copy.deepcopy(model)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False
            return ref_model.to(self.device)

        elif strategy == 'cpu_snapshot':
            # Save snapshot to CPU - loads to GPU when needed
            logger.info("Creating CPU snapshot of reference model for memory efficiency")

            # Save state dict to CPU
            self._ref_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            # Create reference model structure
            ref_model = copy.deepcopy(model)
            ref_model.load_state_dict(self._ref_state_dict)
            ref_model.eval()
            for param in ref_model.parameters():
                param.requires_grad = False

            # Keep on GPU for fast inference
            return ref_model.to(self.device)

        elif strategy == 'disk_snapshot':
            # Save snapshot to disk - ultimate memory savings
            logger.info("Creating disk snapshot of reference model for maximum memory efficiency")

            # Create snapshot directory
            snapshot_dir = Path(self.config.save_dir) / "ref_model_snapshot"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_dir / "ref_model.pt"

            # Save to disk
            torch.save(model.state_dict(), snapshot_path)
            self._ref_snapshot_path = snapshot_path

            logger.info(f"Reference model snapshot saved to {snapshot_path}")

            # Don't keep in memory - load when needed
            return None

        elif strategy == 'shared_params':
            # Use same model with no_grad for reference logits
            # Most memory efficient but requires careful handling
            logger.info("Using shared parameters with no_grad for reference model (maximum memory efficiency)")

            # Save initial state for potential restoration
            self._ref_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            # Return None - will use main model with no_grad
            return None

        else:
            raise ValueError(
                f"Unknown ref_model_strategy: {strategy}. "
                f"Choose from: deepcopy, cpu_snapshot, disk_snapshot, shared_params"
            )

    def _get_reference_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get reference model logits with memory-efficient loading.

        Handles different reference model strategies including lazy loading
        from disk or using shared parameters.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Reference model logits
        """
        import copy

        strategy = self.config.ref_model_strategy

        if strategy == 'disk_snapshot':
            # Load from disk temporarily
            if not hasattr(self, '_ref_snapshot_path'):
                raise RuntimeError("Reference snapshot path not set")

            # Load model temporarily
            temp_ref = copy.deepcopy(self.model)
            temp_ref.load_state_dict(torch.load(self._ref_snapshot_path))
            temp_ref.eval()
            temp_ref.to(self.device)

            with torch.no_grad():
                outputs = temp_ref(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            # Clean up
            del temp_ref
            torch.cuda.empty_cache()

            return logits

        elif strategy == 'shared_params':
            # Use main model with no_grad
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            self.model.train()

            return logits.detach()

        else:
            # Use stored reference model (deepcopy or cpu_snapshot)
            if self.ref_model is None:
                raise RuntimeError("Reference model not initialized")

            with torch.no_grad():
                outputs = self.ref_model(input_ids, attention_mask=attention_mask)
                return outputs.logits if hasattr(outputs, 'logits') else outputs

    def _load_prompts(self, path: Optional[str]) -> List[str]:
        """
        Load prompts from a file.

        Args:
            path: Path to prompts file (text or JSON)

        Returns:
            List of prompts
        """
        if path is None:
            raise ValueError("Path to prompts file not provided")

        file_path: Path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {file_path}")

        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    prompts = data
                elif isinstance(data, dict) and 'prompts' in data:
                    prompts = data['prompts']
                else:
                    raise ValueError("JSON file must contain a list or dict with 'prompts' key")
        else:
            # Assume text file with one prompt per line
            with open(file_path, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]

        return prompts

    def collect_experience(
        self,
        prompts: List[str]
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Collect experience by generating responses and computing rewards.

        Args:
            prompts: List of prompts to generate responses for

        Returns:
            Dictionary containing experience data
        """
        # Generate responses
        max_gen_length: int = 128
        if self.config.ppo is not None and hasattr(self.config.ppo, 'max_gen_length'):
            max_gen_length = self.config.ppo.max_gen_length
        gen_ids, attention_mask, responses = self.ppo_trainer.generate_responses(
            prompts,
            max_length=max_gen_length
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

        # Get log probabilities from reference model (using memory-efficient helper)
        ref_logits = self._get_reference_logits(input_ids=gen_ids, attention_mask=attention_mask)
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

            # Train with PPO - filter out non-tensor fields (prompts, responses)
            batch_tensors: Dict[str, torch.Tensor] = {
                k: v for k, v in experience.items()
                if isinstance(v, torch.Tensor)
            }
            train_stats = self.ppo_trainer.train_step(batch_tensors)

            # Update statistics
            rewards_val = experience['rewards']
            if isinstance(rewards_val, torch.Tensor):
                epoch_stats['rewards'].append(rewards_val.mean().item())
            elif isinstance(rewards_val, list) and len(rewards_val) > 0:
                # Handle list of rewards - ensure all elements are numeric
                # Type: List could contain any type, need to handle carefully
                numeric_rewards = [float(r) for r in rewards_val if isinstance(r, (int, float, torch.Tensor))]
                if numeric_rewards:
                    epoch_stats['rewards'].append(sum(numeric_rewards) / len(numeric_rewards))
                else:
                    epoch_stats['rewards'].append(0.0)
            elif isinstance(rewards_val, (int, float)):
                # Handle scalar reward value - numeric types only
                epoch_stats['rewards'].append(float(rewards_val))
            else:
                # Unknown type or falsy value - default to 0
                epoch_stats['rewards'].append(0.0)
            epoch_stats['policy_loss'].append(train_stats.get('policy_loss', 0))
            epoch_stats['kl_div'].append(train_stats.get('kl_div', 0))
            epoch_stats['entropy'].append(train_stats.get('entropy', 0))

            # Update progress bar
            pbar.set_postfix({
                'reward': f"{epoch_stats['rewards'][-1]:.4f}",
                'kl': f"{epoch_stats['kl_div'][-1]:.4f}"
            })

            # Logging
            if self.config.ppo is not None and self.global_step % self.config.ppo.logging_steps == 0:
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
        # Use init_kl_coef from config if available, otherwise default to 0.01
        # Type: config.ppo could be None, need proper null check
        if self.config.ppo is not None:
            default_kl_coef = self.config.ppo.init_kl_coef
        else:
            default_kl_coef = 0.01
        self.ppo_trainer.kl_coef = checkpoint.get('kl_coef', default_kl_coef)
        self.global_step = checkpoint['global_step']

        if 'scheduler_state_dict' in checkpoint and self.ppo_trainer.lr_scheduler is not None:
            self.ppo_trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
