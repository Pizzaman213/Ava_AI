#!/usr/bin/env python3
"""
Test All Latest Checkpoints

This script tests all the latest model checkpoints to verify they work correctly.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from Ava.models import EnhancedMoEModel
from Ava.config.training_config import TrainingConfig


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizer:
    """
    Load tokenizer from path.

    Args:
        tokenizer_path: Path to tokenizer

    Returns:
        Loaded tokenizer
    """
    try:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    return tokenizer


def setup_tokenizer(tokenizer: Optional[PreTrainedTokenizer]) -> PreTrainedTokenizer:
    """
    Setup tokenizer with proper special tokens.

    Args:
        tokenizer: Tokenizer to setup (can be None)

    Returns:
        Configured tokenizer

    Raises:
        ValueError: If tokenizer is None
    """
    if tokenizer is None:
        raise ValueError("Tokenizer cannot be None")

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Add pad token if neither exists
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer


def load_checkpoint(checkpoint_path: str, config_path: str) -> Dict[str, Any]:
    """
    Load a checkpoint and test it.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Path to config file

    Returns:
        Dictionary with test results
    """
    print(f"Testing checkpoint: {checkpoint_path}")

    # Load config - use yaml loading since from_yaml doesn't exist
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Create TrainingConfig from dict (it has __init__)
    from Ava.config.training_config import EnhancedTrainingConfig
    try:
        config = EnhancedTrainingConfig(**config_dict)  # type: ignore[call-arg]
    except (TypeError, KeyError):
        # Fallback: add config_file if missing
        config_dict['config_file'] = config_path
        config = EnhancedTrainingConfig(**config_dict)  # type: ignore[call-arg]

    # Load tokenizer
    tokenizer_path = config.data.tokenizer_name or '/project/code/models/tokenizer/enhanced-65536'  # type: ignore[attr-defined]
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer = setup_tokenizer(tokenizer)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if EnhancedMoEModel is None:
        raise ImportError("EnhancedMoEModel is not available")

    model = EnhancedMoEModel(config.model)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        step = checkpoint.get('step', 0)
        epoch = checkpoint.get('epoch', 0)
    else:
        model.load_state_dict(checkpoint)
        step = 0
        epoch = 0

    model.to(device)
    model.eval()

    # Test generation
    test_prompt = "Once upon a time"
    # Ensure pad_token and eos_token are set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        test_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(  # type: ignore[misc]
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=50,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    results = {
        'checkpoint_path': checkpoint_path,
        'step': step,
        'epoch': epoch,
        'test_prompt': test_prompt,
        'generated_text': generated_text,
        'success': True
    }

    print(f"✅ Checkpoint test passed")
    print(f"   Step: {step}, Epoch: {epoch}")
    print(f"   Generated: {generated_text[:100]}...")

    return results


def find_latest_checkpoints(checkpoint_dir: str, num_checkpoints: int = 5) -> List[Path]:
    """
    Find the latest checkpoints in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        num_checkpoints: Number of latest checkpoints to find

    Returns:
        List of checkpoint paths sorted by modification time (newest first)
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
        return []

    # Find all .pt files
    checkpoints = list(checkpoint_path.glob("**/*.pt"))

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return checkpoints[:num_checkpoints]


def test_all_checkpoints(
    checkpoint_dir: str,
    config_path: str,
    num_checkpoints: int = 5
) -> Dict[int, Any]:
    """
    Test multiple checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints
        config_path: Path to config file
        num_checkpoints: Number of checkpoints to test

    Returns:
        Dictionary mapping checkpoint index to test results
    """
    checkpoints = find_latest_checkpoints(checkpoint_dir, num_checkpoints)

    if not checkpoints:
        print("No checkpoints found")
        return {}

    print(f"\nFound {len(checkpoints)} checkpoints to test")

    all_results: Dict[int, Any] = {}

    for i, checkpoint_path in enumerate(checkpoints):
        print(f"\n{'='*80}")
        print(f"Testing checkpoint {i+1}/{len(checkpoints)}")
        print(f"{'='*80}")

        try:
            results = load_checkpoint(str(checkpoint_path), config_path)
            if results is not None:
                all_results[i] = results
            else:
                all_results[i] = {
                    'checkpoint_path': str(checkpoint_path),
                    'error': 'load_checkpoint returned None'
                }
        except Exception as e:
            print(f"❌ Error testing checkpoint {checkpoint_path}: {e}")
            all_results[i] = {
                'checkpoint_path': str(checkpoint_path),
                'success': False,
                'error': str(e)
            }

    return all_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test all latest checkpoints")
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='/project/code/outputs/runs',
        help='Directory containing checkpoints'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='/project/code/configs/gpu/small.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--num-checkpoints',
        type=int,
        default=5,
        help='Number of latest checkpoints to test'
    )

    args = parser.parse_args()

    print("="*80)
    print("TESTING ALL LATEST CHECKPOINTS")
    print("="*80)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Config: {args.config}")
    print(f"Number of checkpoints: {args.num_checkpoints}")

    results = test_all_checkpoints(
        args.checkpoint_dir,
        args.config,
        args.num_checkpoints
    )

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    successful = sum(1 for r in results.values() if r.get('success', False))
    failed = len(results) - successful

    print(f"Total checkpoints tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if successful == len(results):
        print("\n✅ All checkpoints passed!")
    else:
        print(f"\n⚠️  {failed} checkpoint(s) failed")


if __name__ == '__main__':
    main()
