#!/usr/bin/env python3
"""
Script to create all necessary folders for the MoE LLM project after GitHub download.
This ensures all required directory structure is present even if not included in the repository.

Note: Most scripts in this project (prepare_data.py, download_datasets.py, etc.) automatically 
create their own output directories as needed with parents=True and exist_ok=True flags.
This script is primarily for initial setup or to ensure all standard directories exist.
"""

import os
import sys

def create_folders():
    """Create all necessary folders for the project."""
    
    # Define all required folders
    folders = [
        # Main data directories
        "data",
        "data/finetuning",
        "data/finetuning/cache",
        "data/finetuning/processed",
        "data/pretraining",
        "data/pretraining/cache", 
        "data/pretraining/processed",
        "data/pretraining/processed/train",
        "data/pretraining/processed/validation",
        "data/pretraining/raw",
        "data/pretraining/raw/train",
        "data/pretraining/raw/validation",
        
        # MLX specific cache directories
        "cache",
        "cache/mlx_data",
        "cache/mlx_data/train",
        "cache/mlx_data/validation",
        
        # Output and checkpoint directories
        "outputs",
        "outputs/models",
        "outputs/logs",
        "checkpoints",
        "checkpoints/pretrained",
        "checkpoints/finetuned",
        "checkpoints/mlx",
        
        # Wandb directories
        "wandb",
        
        # Test directories
        "tests",
        "tests/data",
        "tests/models",
        
        # Script directories
        "scripts/data_download",
        "scripts/data_prep", 
        "scripts/training",
        "scripts/generation",
        "scripts/evaluation",
        "scripts/utils",
        
        # Example directories
        "examples/fine_tuning",
        "examples/generation",
        "examples/generation/rlhf",
        
        # Documentation
        "docs",
        
        # Temporary and build directories
        "tmp",
        "build",
    ]
    
    # Use relative path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.normpath(os.path.join(script_dir, '..', '..'))
    
    created_count = 0
    existing_count = 0
    
    print("Creating necessary folders for MoE LLM project...")
    print("-" * 50)
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path, exist_ok=True)
                print(f"✓ Created: {folder}")
                created_count += 1
            except OSError as e:
                print(f"✗ Failed to create {folder}: {e}")
        else:
            existing_count += 1
    
    print("-" * 50)
    print(f"Summary:")
    print(f"  Created: {created_count} folders")
    print(f"  Already existed: {existing_count} folders")
    print(f"  Total processed: {len(folders)} folders")
    
    if created_count > 0:
        print(f"\n✓ Setup complete! Created {created_count} missing folders.")
    else:
        print(f"\n✓ All folders already exist. No action needed.")

if __name__ == "__main__":
    try:
        create_folders()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)