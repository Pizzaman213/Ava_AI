#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Training Example
Train MoE++ model using RLHF with reward models
"""
import sys
sys.path.append('../..')  # Add project root to path

import argparse
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import json
from tqdm import tqdm

from src.model.moe_transformer import MoEForCausalLM, MoEConfig
from src.training.ppo_trainer import PPOTrainer, PPOConfig
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def load_prompts(dataset_name_or_path, tokenizer, max_prompts=None):
    """Load prompts for PPO training"""
    
    if Path(dataset_name_or_path).exists():
        # Local file
        with open(dataset_name_or_path, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # HuggingFace dataset
        dataset = load_dataset(dataset_name_or_path, split='train')
        
        # Extract prompts based on dataset format
        if 'prompt' in dataset.column_names:
            prompts = dataset['prompt']
        elif 'question' in dataset.column_names:
            prompts = dataset['question']
        elif 'text' in dataset.column_names:
            prompts = dataset['text']
        else:
            raise ValueError(f"Cannot find prompt column in dataset: {dataset.column_names}")
    
    # Limit number of prompts if specified
    if max_prompts:
        prompts = prompts[:max_prompts]
    
    return prompts


def create_reward_function(reward_model_path, tokenizer, device='cuda'):
    """Create reward function from a trained reward model"""
    
    if reward_model_path == 'sentiment':
        # Use sentiment classifier as reward model
        from transformers import pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=0 if device == 'cuda' else -1
        )
        
        def reward_fn(prompts, responses):
            """Sentiment-based reward function"""
            rewards = []
            for response in responses:
                result = sentiment_pipeline(response)[0]
                # Convert sentiment to reward (1-5 stars to -2 to 2)
                stars = int(result['label'].split()[0])
                reward = (stars - 3) / 2.0
                rewards.append(reward)
            return torch.tensor(rewards, device=device)
        
        return reward_fn
    
    else:
        # Load custom reward model
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_path)
        reward_model.to(device)
        reward_model.eval()
        
        def reward_fn(prompts, responses):
            """Custom reward model function"""
            rewards = []
            
            for prompt, response in zip(prompts, responses):
                # Combine prompt and response
                text = f"{prompt}\n\n{response}"
                
                # Tokenize
                inputs = tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(device)
                
                # Get reward
                with torch.no_grad():
                    outputs = reward_model(**inputs)
                    reward = outputs.logits[0, 0].item()  # Assuming single score
                
                rewards.append(reward)
            
            return torch.tensor(rewards, device=device)
        
        return reward_fn


def main():
    parser = argparse.ArgumentParser(description="PPO Training for MoE++")
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--ref-model-path', type=str, default=None,
                        help='Path to reference model (uses model-path if not specified)')
    parser.add_argument('--reward-model-path', type=str, required=True,
                        help='Path to reward model or "sentiment" for sentiment-based rewards')
    
    # Data arguments
    parser.add_argument('--prompts', type=str, required=True,
                        help='Path to prompts file or dataset name')
    parser.add_argument('--max-prompts', type=int, default=10000,
                        help='Maximum number of prompts to use')
    
    # PPO arguments
    parser.add_argument('--learning-rate', type=float, default=1.4e-5,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Total batch size')
    parser.add_argument('--mini-batch-size', type=int, default=4,
                        help='Mini-batch size for PPO updates')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--ppo-epochs', type=int, default=4,
                        help='Number of PPO epochs per batch')
    
    # PPO hyperparameters
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Discount factor')
    parser.add_argument('--lam', type=float, default=0.95,
                        help='GAE lambda parameter')
    parser.add_argument('--clip-ratio', type=float, default=0.2,
                        help='PPO clipping parameter')
    parser.add_argument('--value-clip', type=float, default=0.2,
                        help='Value function clipping parameter')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='Entropy coefficient')
    parser.add_argument('--value-loss-coef', type=float, default=0.1,
                        help='Value loss coefficient')
    
    # KL penalty
    parser.add_argument('--init-kl-coef', type=float, default=0.2,
                        help='Initial KL penalty coefficient')
    parser.add_argument('--target-kl', type=float, default=6.0,
                        help='Target KL divergence')
    
    # Generation settings
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Generation temperature')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p (nucleus) sampling')
    
    # Training settings
    parser.add_argument('--max-steps', type=int, default=10000,
                        help='Maximum training steps')
    parser.add_argument('--eval-steps', type=int, default=100,
                        help='Evaluation interval')
    parser.add_argument('--save-steps', type=int, default=500,
                        help='Save checkpoint interval')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='outputs/ppo_model',
                        help='Output directory for checkpoints')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this training run')
    
    # Other arguments
    parser.add_argument('--mixed-precision', type=str, default='bf16',
                        choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision training')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing')
    parser.add_argument('--use-value-model', action='store_true',
                        help='Use separate value model')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    logger.info("Starting PPO Training")
    logger.info(f"Arguments: {args}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Helper function to load model with different checkpoint formats
    def load_model_from_checkpoint(checkpoint_path):
        config_json_path = os.path.join(checkpoint_path, "config.json")
        config_yaml_path = os.path.join(checkpoint_path, "config.yaml")
        
        if os.path.exists(config_json_path):
            with open(config_json_path, "r") as f:
                config_dict = json.load(f)
        elif os.path.exists(config_yaml_path):
            import yaml
            with open(config_yaml_path, "r") as f:
                config_dict = yaml.safe_load(f)
                if 'model' in config_dict:
                    config_dict = config_dict['model']
        else:
            raise FileNotFoundError(f"No config file found in {checkpoint_path}")
        
        # Remove any fields that MoEConfig doesn't expect
        if 'size' in config_dict:
            config_dict.pop('size')
        
        config = MoEConfig(**config_dict)
        model = MoEForCausalLM(config)
        
        # Load weights
        model_bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        model_pt_path = os.path.join(checkpoint_path, "model.pt")
        
        if os.path.exists(model_bin_path):
            state_dict = torch.load(model_bin_path, map_location="cpu")
        elif os.path.exists(model_pt_path):
            state_dict = torch.load(model_pt_path, map_location="cpu")
        else:
            raise FileNotFoundError(f"No model weights found in {checkpoint_path}")
        
        model.load_state_dict(state_dict)
        return model
    
    # Load models
    logger.info(f"Loading model from {args.model_path}...")
    model = load_model_from_checkpoint(args.model_path)
    
    logger.info(f"Loading reference model...")
    ref_model = load_model_from_checkpoint(args.ref_model_path or args.model_path)
    ref_model.eval()  # Reference model in eval mode
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Create reward function
    logger.info(f"Setting up reward model: {args.reward_model_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reward_fn = create_reward_function(args.reward_model_path, tokenizer, device)
    
    # Load prompts
    logger.info(f"Loading prompts from {args.prompts}")
    prompts = load_prompts(args.prompts, tokenizer, args.max_prompts)
    logger.info(f"Loaded {len(prompts)} prompts")
    
    # Configure PPO
    ppo_config = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        gamma=args.gamma,
        lam=args.lam,
        clip_ratio=args.clip_ratio,
        value_clip=args.value_clip,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        init_kl_coef=args.init_kl_coef,
        target_kl=args.target_kl,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        output_dir=args.output_dir,
        seed=42,
    )
    
    # Create trainer
    logger.info("Initializing PPO trainer...")
    trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_fn=reward_fn,
        tokenizer=tokenizer,
        config=ppo_config,
    )
    
    # Training loop
    logger.info("Starting PPO training...")
    global_step = 0
    
    try:
        progress_bar = tqdm(total=args.max_steps, desc="PPO Training")
        
        while global_step < args.max_steps:
            # Sample batch of prompts
            batch_size = min(args.batch_size, len(prompts))
            batch_indices = torch.randint(0, len(prompts), (batch_size,))
            batch_prompts = [prompts[i] for i in batch_indices]
            
            # Train on batch
            stats = trainer.step(batch_prompts)
            
            # Update progress
            global_step += 1
            progress_bar.update(1)
            
            # Log statistics
            if global_step % 10 == 0:
                progress_bar.set_postfix({
                    'reward': f"{stats['reward/mean']:.3f}",
                    'kl': f"{stats['kl/mean']:.3f}",
                    'loss': f"{stats['loss/total']:.3f}"
                })
            
            # Evaluate
            if global_step % args.eval_steps == 0:
                logger.info(f"Step {global_step} - Evaluation")
                eval_prompts = ["What is machine learning?", 
                               "Explain quantum computing",
                               "Write a Python function to sort a list"]
                
                for prompt in eval_prompts:
                    response = trainer.generate(prompt, max_length=200)
                    logger.info(f"Prompt: {prompt}")
                    logger.info(f"Response: {response}\n")
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                trainer.save_model(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        progress_bar.close()
        logger.info("PPO training completed successfully!")
        
        # Save final model
        final_path = Path(args.output_dir) / "final_model"
        trainer.save_model(final_path)
        logger.info(f"Model saved to {final_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        save_path = Path(args.output_dir) / "interrupted_checkpoint"
        trainer.save_model(save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()