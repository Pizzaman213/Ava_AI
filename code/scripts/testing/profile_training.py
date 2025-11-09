#!/usr/bin/env python3
"""
Training Profiler - Profile one training step to identify bottlenecks

Usage:
    python profile_training.py --config ../../configs/moe/small_moe.yaml
    python profile_training.py --config ../../configs/moe/deepseek_style.yaml --steps 5

This script profiles the training loop to identify performance bottlenecks in:
- Data loading
- Forward pass
- Backward pass
- Optimizer step
- MoE routing
- Expert computation

Output:
- Console summary of top bottlenecks
- Detailed trace saved to outputs/profiling/trace.json (viewable in chrome://tracing)
- Tensorboard-compatible profiling data
"""

import argparse
import sys
import torch
import torch.profiler
from pathlib import Path
import json
import time

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.Ava.config.training_config import TrainingConfigManager


def profile_training_step(config_path: str, num_steps: int = 3, output_dir: str = "/project/code/outputs/profiling"):
    """
    Profile training for specified number of steps.

    Args:
        config_path: Path to training config YAML
        num_steps: Number of steps to profile (default: 3)
        output_dir: Directory to save profiling results
    """
    print(f"🔍 Profiling Training Pipeline")
    print(f"   Config: {config_path}")
    print(f"   Steps to profile: {num_steps}")
    print(f"   Output: {output_dir}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    config_manager = TrainingConfigManager()
    config = config_manager.load_yaml_config(config_path)

    # Import training components (after config is loaded)
    from src.Ava.data.dataloader import create_dataloaders
    from src.Ava.models.moe_model import OptimizedMoETransformer
    from src.Ava.training.core.trainer import EnhancedModularTrainer

    print("📦 Initializing components...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")

    # Create small subset of data for profiling
    config_dict = config.to_dict()
    config_dict['data']['max_samples'] = 100  # Small dataset for profiling
    config_dict['training']['batch_size'] = min(config_dict['training']['batch_size'], 4)

    # Create dataloaders
    print("   Loading data...")
    train_loader, val_loader = create_dataloaders(config_dict)

    # Create model
    print("   Creating model...")
    model_config = config_dict['model']
    model = SparseTransformerMoE(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_attention_heads=model_config['num_attention_heads'],
        intermediate_size=model_config['intermediate_size'],
        max_position_embeddings=model_config['max_position_embeddings'],
        num_experts=model_config.get('num_experts', 8),
        num_experts_per_token=model_config.get('num_experts_per_token', 2),
        router_type=model_config.get('router_type', 'mixtral'),
    )
    model = model.to(device)
    model.train()

    # Create optimizer
    print("   Creating optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_dict['training']['learning_rate'],
        weight_decay=config_dict['training'].get('weight_decay', 0.1),
    )

    # Get first few batches
    print(f"\n🚀 Starting profiling for {num_steps} steps...")
    data_iter = iter(train_loader)
    batches = [next(data_iter) for _ in range(min(num_steps, len(train_loader)))]

    # Warm-up step (not profiled)
    print("   Warming up...")
    batch = batches[0]
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch.get('labels', input_ids).to(device)

    outputs = model(input_ids, attention_mask=attention_mask)
    loss = torch.nn.functional.cross_entropy(
        outputs.logits.view(-1, model_config['vocab_size']),
        labels.view(-1)
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Profile actual steps
    print(f"   Profiling {num_steps} training steps...")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
    ) as prof:
        for step_idx, batch in enumerate(batches[:num_steps]):
            with torch.profiler.record_function(f"step_{step_idx}"):
                # Move data to device
                with torch.profiler.record_function("data_transfer"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch.get('labels', input_ids).to(device)

                # Forward pass
                with torch.profiler.record_function("forward"):
                    outputs = model(input_ids, attention_mask=attention_mask)

                # Compute loss
                with torch.profiler.record_function("loss_computation"):
                    loss = torch.nn.functional.cross_entropy(
                        outputs.logits.view(-1, model_config['vocab_size']),
                        labels.view(-1)
                    )

                # Backward pass
                with torch.profiler.record_function("backward"):
                    loss.backward()

                # Optimizer step
                with torch.profiler.record_function("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad()

                prof.step()

    # Save results
    print(f"\n💾 Saving profiling results to {output_dir}...")

    # Save trace (viewable in chrome://tracing)
    trace_path = output_path / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"   ✅ Chrome trace: {trace_path}")
    print(f"      View at: chrome://tracing")

    # Save tensorboard data
    tb_path = output_path / "tensorboard"
    prof.export_stacks(str(tb_path / "profiler_stacks.txt"), "self_cuda_time_total")
    print(f"   ✅ Stack traces: {tb_path}/profiler_stacks.txt")

    # Print summary
    print("\n" + "="*80)
    print("📊 PROFILING SUMMARY")
    print("="*80)

    print("\n🔥 Top 10 Operations by CUDA Time:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=10
    ))

    print("\n💾 Top 10 Memory Intensive Operations:")
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage" if torch.cuda.is_available() else "cpu_memory_usage",
        row_limit=10
    ))

    print("\n⚡ Top 10 Operations by Self Time (excluding children):")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",
        row_limit=10
    ))

    # Extract key metrics
    key_averages = prof.key_averages()

    # Find specific operations
    operations = {
        'forward': None,
        'backward': None,
        'optimizer_step': None,
        'data_transfer': None,
    }

    for event in key_averages:
        for op_name in operations.keys():
            if op_name in event.key:
                operations[op_name] = event

    print("\n" + "="*80)
    print("🎯 KEY OPERATION BREAKDOWN")
    print("="*80)

    for op_name, event in operations.items():
        if event:
            cuda_time = event.cuda_time_total if hasattr(event, 'cuda_time_total') else 0
            cpu_time = event.cpu_time_total if hasattr(event, 'cpu_time_total') else 0
            time_ms = (cuda_time or cpu_time) / 1000  # Convert to ms
            print(f"   {op_name:20s}: {time_ms:8.2f} ms")

    # Save summary to JSON
    summary_path = output_path / "summary.json"
    summary = {
        'config': config_path,
        'num_steps_profiled': num_steps,
        'device': str(device),
        'operations': {
            name: {
                'cuda_time_ms': (event.cuda_time_total / 1000) if event and hasattr(event, 'cuda_time_total') else 0,
                'cpu_time_ms': (event.cpu_time_total / 1000) if event and hasattr(event, 'cpu_time_total') else 0,
            }
            for name, event in operations.items() if event
        }
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Summary saved to: {summary_path}")

    print("\n" + "="*80)
    print("✨ Profiling complete!")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Profile training to identify bottlenecks")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training config YAML'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=3,
        help='Number of steps to profile (default: 3)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/project/code/outputs/profiling',
        help='Output directory for profiling results'
    )

    args = parser.parse_args()

    profile_training_step(
        config_path=args.config,
        num_steps=args.steps,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
