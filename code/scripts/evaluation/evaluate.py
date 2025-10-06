#!/usr/bin/env python3
"""
Evaluation script for Qwen MoE++ models.

This script evaluates trained models on validation/test data from
/project/code/data/pretraining/processed/ and reports various metrics.

Usage:
    # Evaluate a checkpoint
    python evaluate.py --model-path outputs/best_model.pt --config configs/cpu/small.yaml

    # Evaluate with specific metrics
    python evaluate.py --model-path outputs/best_model.pt --metrics perplexity accuracy

    # Test generation quality
    python evaluate.py --model-path outputs/best_model.pt --test-generation
"""

import argparse
import sys
import torch  # type: ignore[import-not-found]
import yaml
from pathlib import Path

# Add project root to path
sys.path.append('/project/code')

from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig  # type: ignore[import-not-found]
from src.Ava.multi_column_data import create_multi_column_dataloader
from src.Ava.evaluation import ModelEvaluator, PerplexityEvaluator
from src.Ava.utils import setup_logging, load_checkpoint
from transformers import AutoTokenizer  # type: ignore[import-not-found]


def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen MoE++ model')

    # Model arguments
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file (optional if saved in checkpoint)')

    # Data arguments
    parser.add_argument('--data-dir', type=str,
                       default='/project/code/data/pretraining/processed',
                       help='Directory containing evaluation data')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--max-batches', type=int, default=None,
                       help='Maximum number of batches to evaluate')

    # Evaluation arguments
    parser.add_argument('--metrics', nargs='+',
                       default=['perplexity', 'accuracy'],
                       choices=['perplexity', 'accuracy', 'expert_stats'],
                       help='Metrics to compute')
    parser.add_argument('--test-generation', action='store_true',
                       help='Test generation quality')
    parser.add_argument('--analyze-experts', action='store_true',
                       help='Analyze expert utilization')

    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use')
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level')
    parser.add_argument('--output-dir', type=str, default='/project/code/outputs',
                       help='Output directory for evaluation results')

    args = parser.parse_args()

    # Create output directory
    from pathlib import Path
    from datetime import datetime
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(
        log_level=args.log_level,
        log_file=f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        log_dir=str(output_dir)
    )

    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Load checkpoint
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)

    # Get configuration
    if 'config' in checkpoint:
        model_config_dict = checkpoint['config']
    elif args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        model_config_dict = config_dict.get('model', {})
    else:
        raise ValueError("No configuration found in checkpoint or provided via --config")

    # Initialize model
    model_config = EnhancedMoEConfig(**model_config_dict)
    model = EnhancedMoEModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {total_params:,} parameters")

    # Initialize tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Create dataloaders using multi-column loader
    logger.info(f"Loading validation data from {args.data_dir}")
    default_config = {
        'columns': [
            {'name': 'text', 'type': 'text', 'role': 'input', 'max_length': args.max_length}
        ],
        'combine_strategy': 'concatenate',
        'max_samples': args.max_batches * args.batch_size if args.max_batches else None,
        'validation_enabled': True
    }

    val_loader = create_multi_column_dataloader(
        config=default_config,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        split='val',
        streaming=False,
        num_workers=0
    )

    # Initialize evaluator
    evaluator = ModelEvaluator(model, tokenizer, device)

    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = evaluator.evaluate(
        val_loader,
        compute_perplexity='perplexity' in args.metrics,
        compute_accuracy='accuracy' in args.metrics,
        compute_expert_stats='expert_stats' in args.metrics,
        max_batches=args.max_batches
    )

    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
    print("="*60)

    # Save metrics to JSON file
    import json
    metrics_file = output_dir / f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Metrics saved to {metrics_file}")

    # Test generation quality if requested
    if args.test_generation:
        logger.info("Testing generation quality...")

        test_prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a distant galaxy",
            "The key to successful machine learning is",
            "In the world of quantum computing",
            "The most important scientific discovery"
        ]

        gen_metrics = evaluator.evaluate_generation_quality(
            test_prompts,
            max_length=100,
            temperature=0.8
        )

        print("\n" + "="*60)
        print("Generation Quality")
        print("="*60)
        print(f"Average length: {gen_metrics['average_length']:.1f} tokens")
        print(f"Average time: {gen_metrics['average_time']:.3f} seconds")
        print(f"Token diversity: {gen_metrics['diversity']:.4f}")
        print("\nSample generations:")
        for prompt, generation in zip(gen_metrics['prompts'], gen_metrics['samples']):
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {generation}")
        print("="*60)

    # Analyze expert utilization if requested
    if args.analyze_experts:
        logger.info("Analyzing expert utilization...")

        expert_analysis = evaluator.analyze_expert_utilization(
            val_loader,
            max_batches=min(100, args.max_batches) if args.max_batches else 100
        )

        if expert_analysis:
            print("\n" + "="*60)
            print("Expert Utilization Analysis")
            print("="*60)
            for key, value in expert_analysis.items():
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v:.4f}")
                else:
                    print(f"{key}: {value:.4f}")
            print("="*60)

    # Save results
    if checkpoint.get('epoch'):
        logger.info(f"Model was trained for {checkpoint['epoch']} epochs")
    if checkpoint.get('train_loss'):
        logger.info(f"Training loss: {checkpoint['train_loss']:.4f}")
    if checkpoint.get('val_loss'):
        logger.info(f"Validation loss: {checkpoint['val_loss']:.4f}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()