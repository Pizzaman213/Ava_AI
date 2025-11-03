#!/usr/bin/env python3
"""
Script to update memory-related settings in all GPU/hardware config files.
This enables gradient checkpointing, flash attention, and optimizes memory usage.
"""

import re
from pathlib import Path

def update_config_file(filepath):
    """Update memory settings in a single config file."""
    print(f"\n{'='*60}")
    print(f"Processing: {filepath}")
    print(f"{'='*60}")

    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content
    changes = []

    # 1. Enable gradient checkpointing (various locations)
    if re.search(r'gradient_checkpointing:\s*false', content):
        content = re.sub(
            r'(gradient_checkpointing:)\s*false',
            r'\1 true',
            content
        )
        changes.append("✓ Enabled gradient_checkpointing")

    # 2. Enable flash attention
    if re.search(r'use_flash_attention:\s*false', content):
        content = re.sub(
            r'(use_flash_attention:)\s*false',
            r'\1 true',
            content
        )
        changes.append("✓ Enabled use_flash_attention")

    # 3. Enable DeepSpeed activation checkpointing
    if re.search(r'deepspeed_activation_checkpointing:\s*false', content):
        content = re.sub(
            r'(deepspeed_activation_checkpointing:)\s*false',
            r'\1 true',
            content
        )
        changes.append("✓ Enabled deepspeed_activation_checkpointing")

    # 4. Enable activation checkpointing in deepspeed section
    if re.search(r'^\s*activation_checkpointing:\s*false', content, re.MULTILINE):
        content = re.sub(
            r'(^\s*activation_checkpointing:)\s*false',
            r'\1 true',
            content,
            flags=re.MULTILINE
        )
        changes.append("✓ Enabled activation_checkpointing")

    # 5. Reduce prefetch_factor from 4 to 2
    if re.search(r'prefetch_factor:\s*4', content):
        content = re.sub(
            r'(prefetch_factor:)\s*4',
            r'\1 2',
            content
        )
        changes.append("✓ Reduced prefetch_factor: 4 → 2")

    if re.search(r'dataloader_prefetch_factor:\s*4', content):
        content = re.sub(
            r'(dataloader_prefetch_factor:)\s*4',
            r'\1 2',
            content
        )
        changes.append("✓ Reduced dataloader_prefetch_factor: 4 → 2")

    # 6. Increase target_utilization
    target_util_match = re.search(r'target_utilization:\s*0\.(\d+)', content)
    if target_util_match:
        current_val = float(f"0.{target_util_match.group(1)}")
        if current_val < 0.90:
            content = re.sub(
                r'(target_utilization:)\s*0\.\d+',
                r'\1 0.90',
                content
            )
            changes.append(f"✓ Increased target_utilization: {current_val} → 0.90")

    # 7. Adjust warning thresholds
    warning_match = re.search(r'warning_threshold:\s*0\.(\d+)', content)
    if warning_match:
        current_val = float(f"0.{warning_match.group(1)}")
        if current_val < 0.92:
            content = re.sub(
                r'(warning_threshold:)\s*0\.\d+',
                r'\1 0.92',
                content
            )
            changes.append(f"✓ Increased warning_threshold: {current_val} → 0.92")

    # Write back if changes were made
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)

        print(f"\n📝 Changes applied ({len(changes)} updates):")
        for change in changes:
            print(f"  {change}")
        return True
    else:
        print("  ℹ️  No changes needed (already optimized or different structure)")
        return False

def main():
    """Process all GPU and hardware config files."""
    print("\n" + "="*60)
    print("MEMORY OPTIMIZATION CONFIG UPDATER")
    print("="*60)

    # Define config directories
    config_paths = [
        Path("/project/code/configs/gpu"),
        Path("/project/code/configs/hardware"),
    ]

    updated_files = []
    skipped_files = []

    for config_dir in config_paths:
        if not config_dir.exists():
            print(f"\n⚠️  Directory not found: {config_dir}")
            continue

        # Find all YAML files
        yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

        for yaml_file in sorted(yaml_files):
            if update_config_file(yaml_file):
                updated_files.append(yaml_file)
            else:
                skipped_files.append(yaml_file)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Updated: {len(updated_files)} files")
    for f in updated_files:
        print(f"   - {f.name}")

    if skipped_files:
        print(f"\n⏭️  Skipped: {len(skipped_files)} files (already optimized)")
        for f in skipped_files:
            print(f"   - {f.name}")

    print("\n✨ Memory optimization configuration complete!\n")

if __name__ == "__main__":
    main()
