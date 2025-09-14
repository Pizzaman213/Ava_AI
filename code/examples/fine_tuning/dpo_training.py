#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) Training Example
Train MoE++ model using preference data without reward models
"""
import sys
sys.path.append('../..')  # Add project root to path

import argparse
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os
import yaml
import json

from src.model.moe_transformer import MoEForCausalLM, MoEConfig
from src.training.dpo_trainer import DPOTrainer, DPOConfig
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def prepare_preference_dataset(dataset_name_or_path, tokenizer, max_length=512):
    """Prepare preference dataset for DPO training"""
    
    # Load dataset
    if Path(dataset_name_or_path).exists():
        # Local dataset
        dataset = load_dataset('json', data_files={
            'train': f'{dataset_name_or_path}/train.json',
            'validation': f'{dataset_name_or_path}/validation.json'
        })
    else:
        # HuggingFace dataset (e.g., "Anthropic/hh-rlhf")
        dataset = load_dataset(dataset_name_or_path)
    
    def format_preference_data(example):
        """Format data for DPO training"""
        # Expected format: prompt, chosen, rejected
        formatted = {}
        
        # Handle different dataset formats
        if 'prompt' in example and 'chosen' in example and 'rejected' in example:
            formatted['prompt'] = example['prompt']
            formatted['chosen'] = example['chosen']
            formatted['rejected'] = example['rejected']
        elif 'question' in example:
            # Handle instruction-following format
            formatted['prompt'] = example['question']
            formatted['chosen'] = example.get('chosen_response', example.get('response_j', ''))
            formatted['rejected'] = example.get('rejected_response', example.get('response_k', ''))
        else:
            # Try to extract from other common formats
            formatted['prompt'] = example.get('input', example.get('context', ''))
            formatted['chosen'] = example.get('output_1', '')
            formatted['rejected'] = example.get('output_2', '')
        
        return formatted
    
    # Format dataset
    dataset = dataset.map(format_preference_data)
    
    # Filter out examples with empty fields
    dataset = dataset.filter(
        lambda x: len(x['prompt']) > 0 and len(x['chosen']) > 0 and len(x['rejected']) > 0
    )
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="DPO Training for MoE++")
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--ref-model-path', type=str, default=None,
                        help='Path to reference model (uses model-path if not specified)')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Preference dataset name or path')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--max-prompt-length', type=int, default=256,
                        help='Maximum prompt length')
    
    # DPO arguments
    parser.add_argument('--beta', type=float, default=0.1,
                        help='KL regularization coefficient')
    parser.add_argument('--loss-type', type=str, default='sigmoid',
                        choices=['sigmoid', 'hinge', 'ipo'],
                        help='DPO loss type')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing for DPO loss')
    parser.add_argument('--ipo-tau', type=float, default=0.05,
                        help='Temperature for IPO loss')
    parser.add_argument('--reference-free', action='store_true',
                        help='Use reference-free DPO')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=5e-7,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4,
                        help='Gradient accumulation steps')
    parser.add_argument('--num-epochs', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--warmup-steps', type=int, default=150,
                        help='Number of warmup steps')
    parser.add_argument('--eval-steps', type=int, default=100,
                        help='Evaluation interval')
    parser.add_argument('--save-steps', type=int, default=100,
                        help='Save checkpoint interval')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='outputs/dpo_model',
                        help='Output directory for checkpoints')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this training run')
    
    # Other arguments
    parser.add_argument('--mixed-precision', type=str, default='bf16',
                        choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision training')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing')
    parser.add_argument('--use-peft', action='store_true',
                        help='Use parameter-efficient fine-tuning')
    parser.add_argument('--push-to-hub', action='store_true',
                        help='Push model to HuggingFace Hub')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    logger.info("Starting DPO Training")
    logger.info(f"Arguments: {args}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    logger.info(f"Loading and preparing preference dataset: {args.dataset}")
    dataset = prepare_preference_dataset(args.dataset, tokenizer, args.max_length)
    
    logger.info(f"Dataset sizes - Train: {len(dataset['train'])}, Val: {len(dataset.get('validation', []))}")
    
    # Show example
    if len(dataset['train']) > 0:
        example = dataset['train'][0]
        logger.info("Example preference data:")
        logger.info(f"  Prompt: {example['prompt'][:100]}...")
        logger.info(f"  Chosen: {example['chosen'][:100]}...")
        logger.info(f"  Rejected: {example['rejected'][:100]}...")
    
    # Configure DPO training
    dpo_config = DPOConfig(
        model_name_or_path=args.model_path,
        ref_model_name_or_path=args.ref_model_path or args.model_path,
        beta=args.beta,
        loss_type=args.loss_type,
        label_smoothing=args.label_smoothing,
        ipo_tau=args.ipo_tau,
        reference_free=args.reference_free,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=10,
        output_dir=args.output_dir,
        run_name=args.run_name,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        use_peft=args.use_peft,
        push_to_hub=args.push_to_hub,
    )
    
    # PEFT configuration if requested
    peft_config = None
    if args.use_peft:
        from peft import LoraConfig, TaskType
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        dpo_config.peft_config = peft_config.__dict__
    
    # Create trainer
    logger.info("Initializing DPO trainer...")
    trainer = DPOTrainer(
        config=dpo_config,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('validation'),
    )
    
    # Start training
    logger.info("Starting DPO training...")
    try:
        trainer.train()
        logger.info("DPO training completed successfully!")
        
        # Save final model
        final_path = Path(args.output_dir) / "final_model"
        trainer.save_model(final_path)
        logger.info(f"Model saved to {final_path}")
        
        # Push to hub if requested
        if args.push_to_hub:
            logger.info("Pushing model to HuggingFace Hub...")
            trainer.push_to_hub()
        
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