#!/usr/bin/env python3
"""
Show summary of outputs directory.

This script displays a summary of all outputs including:
- Training checkpoints
- Log files
- Evaluation metrics
- Generated outputs
"""

import os
from pathlib import Path
from datetime import datetime
import json

def format_size(size):
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def main():
    output_dir = Path('/project/code/outputs')

    if not output_dir.exists():
        print("❌ Output directory does not exist: /project/code/outputs")
        return

    print("=" * 60)
    print("📊 OUTPUTS SUMMARY")
    print("=" * 60)
    print(f"📁 Output Directory: {output_dir}")
    print()

    # Find checkpoints
    print("🔹 Model Checkpoints:")
    checkpoints = list(output_dir.glob("**/*.pt")) + list(output_dir.glob("**/*.pth"))
    if checkpoints:
        for ckpt in sorted(checkpoints)[:10]:  # Show max 10
            size = format_size(ckpt.stat().st_size)
            rel_path = ckpt.relative_to(output_dir)
            print(f"   • {rel_path} ({size})")
        if len(checkpoints) > 10:
            print(f"   ... and {len(checkpoints) - 10} more")
    else:
        print("   No checkpoints found")
    print()

    # Find log files
    print("🔹 Training Logs:")
    logs = list(output_dir.glob("**/*.log"))
    if logs:
        for log in sorted(logs, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            size = format_size(log.stat().st_size)
            rel_path = log.relative_to(output_dir)
            mod_time = datetime.fromtimestamp(log.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            print(f"   • {rel_path} ({size}) - {mod_time}")
        if len(logs) > 5:
            print(f"   ... and {len(logs) - 5} more")
    else:
        print("   No log files found")
    print()

    # Find metrics files
    print("🔹 Evaluation Metrics:")
    metrics = list(output_dir.glob("**/metrics*.json"))
    if metrics:
        for metric_file in sorted(metrics, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            rel_path = metric_file.relative_to(output_dir)
            try:
                with open(metric_file, 'r') as f:
                    data = json.load(f)
                    if 'perplexity' in data:
                        print(f"   • {rel_path} - Perplexity: {data['perplexity']:.2f}")
                    else:
                        print(f"   • {rel_path}")
            except:
                print(f"   • {rel_path}")
        if len(metrics) > 5:
            print(f"   ... and {len(metrics) - 5} more")
    else:
        print("   No metrics files found")
    print()

    # Find generated text files
    print("🔹 Generated Outputs:")
    gen_files = list(output_dir.glob("**/generated*.txt")) + list(output_dir.glob("**/output*.txt"))
    if gen_files:
        for gen in sorted(gen_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
            size = format_size(gen.stat().st_size)
            rel_path = gen.relative_to(output_dir)
            print(f"   • {rel_path} ({size})")
        if len(gen_files) > 5:
            print(f"   ... and {len(gen_files) - 5} more")
    else:
        print("   No generated text files found")
    print()

    # Directory statistics
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    total_files = sum(1 for f in output_dir.rglob("*") if f.is_file())
    total_dirs = sum(1 for d in output_dir.rglob("*") if d.is_dir())

    print("📈 Statistics:")
    print(f"   • Total files: {total_files}")
    print(f"   • Total directories: {total_dirs}")
    print(f"   • Total size: {format_size(total_size)}")
    print()

    # Recent runs
    print("🔹 Recent Training Runs:")
    run_dirs = sorted([d for d in output_dir.glob("run_*") if d.is_dir()],
                      key=lambda x: x.stat().st_mtime, reverse=True)
    if run_dirs:
        for run_dir in run_dirs[:5]:
            mod_time = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            num_files = sum(1 for _ in run_dir.rglob("*") if _.is_file())
            print(f"   • {run_dir.name} - {mod_time} ({num_files} files)")
        if len(run_dirs) > 5:
            print(f"   ... and {len(run_dirs) - 5} more")
    else:
        print("   No training runs found")

    print()
    print("=" * 60)
    print("✅ Use these paths with generation and evaluation scripts:")
    print("   python scripts/generation/generate.py --model-path outputs/best_model.pt")
    print("   python scripts/evaluation/evaluate.py --model-path outputs/best_model.pt")
    print("=" * 60)

if __name__ == "__main__":
    main()