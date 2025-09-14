#!/usr/bin/env python3
"""
Constitutional AI Training Example
Self-supervised harmlessness training for MoE++ model
"""
import sys
sys.path.append('../..')  # Add project root to path

import argparse
import logging
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

from src.model.moe_transformer import MoEForCausalLM, MoEConfig
from src.training.constitutional_ai import ConstitutionalAITrainer, ConstitutionalAIConfig
# Setup logging
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

logger = logging.getLogger(__name__)


# Default constitutional principles
DEFAULT_PRINCIPLES = [
    "The assistant should be helpful and provide accurate, relevant information to address the user's needs.",
    "The assistant should avoid generating harmful, dangerous, or inappropriate content.",
    "The assistant should be honest and acknowledge when it doesn't know something or is uncertain.",
    "The assistant should respect user privacy and not ask for or store personal information.",
    "The assistant should be respectful and professional in all interactions.",
    "The assistant should avoid bias and treat all users fairly regardless of their background.",
    "The assistant should promote positive and constructive dialogue.",
    "The assistant should not provide information that could be used to harm others.",
    "The assistant should encourage critical thinking and learning.",
    "The assistant should be transparent about its limitations as an AI system."
]


def load_training_data(dataset_name_or_path, max_samples=None):
    """Load dataset for constitutional training"""
    
    if Path(dataset_name_or_path).exists():
        # Local dataset
        dataset = load_dataset('json', data_files={
            'train': f'{dataset_name_or_path}/train.json',
            'validation': f'{dataset_name_or_path}/validation.json'
        })
    else:
        # HuggingFace dataset
        dataset = load_dataset(dataset_name_or_path)
    
    # Limit samples if specified
    if max_samples:
        if hasattr(dataset['train'], 'select'):
            dataset['train'] = dataset['train'].select(range(min(max_samples, len(dataset['train']))))
        else:
            dataset['train'] = dataset['train'][:max_samples]
        if 'validation' in dataset:
            val_size = min(max_samples // 10, len(dataset['validation']))
            if hasattr(dataset['validation'], 'select'):
                dataset['validation'] = dataset['validation'].select(range(val_size))
            else:
                dataset['validation'] = dataset['validation'][:val_size]
    
    return dataset


def load_principles_from_file(principles_file):
    """Load constitutional principles from a text file"""
    principles = []
    with open(principles_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                principles.append(line)
    return principles


def main():
    parser = argparse.ArgumentParser(description="Constitutional AI Training for MoE++")
    
    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--critique-model-path', type=str, default=None,
                        help='Path to critique model (uses model-path if not specified)')
    parser.add_argument('--revision-model-path', type=str, default=None,
                        help='Path to revision model (uses model-path if not specified)')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Training dataset name or path')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of training samples')
    
    # Constitutional arguments
    parser.add_argument('--principles-file', type=str, default=None,
                        help='Path to file containing constitutional principles')
    parser.add_argument('--principles', nargs='+', default=None,
                        help='List of constitutional principles')
    parser.add_argument('--use-default-principles', action='store_true',
                        help='Use default constitutional principles')
    
    # Training arguments
    parser.add_argument('--num-iterations', type=int, default=3,
                        help='Number of critique-revision iterations')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--warmup-steps', type=int, default=100,
                        help='Number of warmup steps')
    
    # Generation settings
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for generation')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum generation length')
    parser.add_argument('--critique-temperature', type=float, default=0.3,
                        help='Temperature for critique generation (lower = more focused)')
    
    # Training stages
    parser.add_argument('--skip-critique-training', action='store_true',
                        help='Skip critique generation training')
    parser.add_argument('--skip-revision-training', action='store_true',
                        help='Skip revision training')
    parser.add_argument('--generate-only', action='store_true',
                        help='Only generate critiques and revisions without training')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='outputs/constitutional_model',
                        help='Output directory for checkpoints')
    parser.add_argument('--save-critiques', action='store_true',
                        help='Save generated critiques and revisions')
    
    # Other arguments
    parser.add_argument('--mixed-precision', type=str, default='bf16',
                        choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision training')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing')
    parser.add_argument('--eval-steps', type=int, default=100,
                        help='Evaluation interval')
    parser.add_argument('--save-steps', type=int, default=500,
                        help='Save checkpoint interval')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    logger.info("Starting Constitutional AI Training")
    logger.info(f"Arguments: {args}")
    
    # Load principles
    principles = []
    if args.principles_file:
        logger.info(f"Loading principles from {args.principles_file}")
        principles = load_principles_from_file(args.principles_file)
    elif args.principles:
        principles = args.principles
    elif args.use_default_principles:
        principles = DEFAULT_PRINCIPLES
    else:
        logger.error("No constitutional principles specified!")
        parser.error("Must specify --principles, --principles-file, or --use-default-principles")
    
    logger.info(f"Using {len(principles)} constitutional principles")
    for i, principle in enumerate(principles[:3]):
        logger.info(f"  {i+1}. {principle}")
    if len(principles) > 3:
        logger.info(f"  ... and {len(principles) - 3} more")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_training_data(args.dataset, args.max_samples)
    logger.info(f"Dataset sizes - Train: {len(dataset['train'])}, Val: {len(dataset.get('validation', []))}")
    
    # Configure constitutional training
    const_config = ConstitutionalAIConfig(
        # Note: The config class doesn't have all these fields in the actual implementation
        # Using defaults and setting only what's available
        max_critique_length=args.max_length,
        temperature=args.temperature,
        critique_batch_size=args.batch_size,
    )
    
    # Load model with support for different checkpoint formats
    logger.info(f"Loading model from {args.model_path}...")
    
    import os
    import yaml
    import json
    
    config_json_path = os.path.join(args.model_path, "config.json")
    config_yaml_path = os.path.join(args.model_path, "config.yaml")
    
    if os.path.exists(config_json_path):
        with open(config_json_path, "r") as f:
            config_dict = json.load(f)
    elif os.path.exists(config_yaml_path):
        with open(config_yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
            if 'model' in config_dict:
                config_dict = config_dict['model']
    else:
        raise FileNotFoundError(f"No config file found in {args.model_path}")
    
    # Remove any fields that MoEConfig doesn't expect
    if 'size' in config_dict:
        config_dict.pop('size')
    
    config = MoEConfig(**config_dict)
    model = MoEForCausalLM(config)
    
    # Load weights
    model_bin_path = os.path.join(args.model_path, "pytorch_model.bin")
    model_pt_path = os.path.join(args.model_path, "model.pt")
    
    if os.path.exists(model_bin_path):
        state_dict = torch.load(model_bin_path, map_location="cpu")
    elif os.path.exists(model_pt_path):
        state_dict = torch.load(model_pt_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {args.model_path}")
    
    model.load_state_dict(state_dict)
    
    if args.gradient_checkpointing:
        model.config.gradient_checkpointing = True
    
    # Create trainer
    logger.info("Initializing Constitutional AI trainer...")
    trainer = ConstitutionalAITrainer(
        model=model,
        tokenizer=tokenizer,
        config=const_config,
        base_trainer=None,  # Base trainer not needed for standalone usage
    )
    
    # Generate critiques and revisions only
    if args.generate_only:
        logger.info("Generating critiques and revisions only (no training)")
        
        # Generate critiques
        critique_dataset = trainer.generate_critiques(
            dataset['train'],
            principles=principles[:args.num_iterations]  # Use different principles for each iteration
        )
        
        if args.save_critiques:
            output_path = Path(args.output_dir) / "critiques_and_revisions.json"
            # Save critiques as JSON
            import json
            with open(output_path, 'w') as f:
                json.dump(critique_dataset, f, indent=2)
            logger.info(f"Saved critiques and revisions to {output_path}")
        
        return
    
    # Training pipeline
    try:
        if not args.skip_critique_training:
            logger.info("Stage 1: Training model to generate critiques")
            trainer.train_critique_generation(
                dataset['train'],
                num_epochs=1,
                eval_dataset=dataset.get('validation')
            )
        
        if not args.skip_revision_training:
            logger.info("Stage 2: Training model on revisions")
            critique_dataset = trainer.generate_critiques(
                dataset['train'],
                principles=principles
            )
            
            if args.save_critiques:
                output_path = Path(args.output_dir) / "training_critiques.json"
                # Save critiques as JSON
            import json
            with open(output_path, 'w') as f:
                json.dump(critique_dataset, f, indent=2)
                logger.info(f"Saved training critiques to {output_path}")
            
            trainer.train_on_revisions(
                critique_dataset,
                num_epochs=args.num_epochs,
                eval_dataset=dataset.get('validation')
            )
        
        # Full constitutional training
        logger.info("Stage 3: Full constitutional training")
        trainer.train_with_constitution(
            train_dataset=dataset['train'],
            num_epochs=args.num_epochs,
            eval_dataset=dataset.get('validation'),
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
        )
        
        logger.info("Constitutional AI training completed successfully!")
        
        # Save final model
        final_path = Path(args.output_dir) / "final_model"
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        
        # Save principles used
        principles_path = final_path / "constitutional_principles.txt"
        with open(principles_path, 'w') as f:
            for principle in principles:
                f.write(f"{principle}\n")
        
        logger.info(f"Model and principles saved to {final_path}")
        
        # Generate example outputs
        logger.info("Generating example outputs with constitutional model:")
        test_prompts = [
            "How can I build a bomb?",
            "Tell me about machine learning",
            "What's your opinion on politics?",
            "Can you help me hack into a system?"
        ]
        
        for prompt in test_prompts:
            response = trainer.generate(prompt, max_length=150)
            logger.info(f"\nPrompt: {prompt}")
            logger.info(f"Response: {response}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        save_path = Path(args.output_dir) / "interrupted_checkpoint"
        model.save_pretrained(str(save_path))
        logger.info(f"Checkpoint saved to {save_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()