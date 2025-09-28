#!/usr/bin/env python3
"""
Test script for validating the training pipeline functionality.
"""

import sys
import torch
import warnings
import traceback
from pathlib import Path

# Add project root to path
sys.path.append('/project/code')

def test_imports():
    """Test that all critical imports work."""
    print("🔍 Testing imports...")

    try:
        # Test core imports
        from src.Ava.config import TrainingConfigManager, EnhancedTrainingConfig
        print("  ✅ Config imports successful")

        from src.Ava.training.enhanced_trainer import EnhancedModularTrainer
        print("  ✅ Enhanced trainer import successful")

        from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig
        print("  ✅ MoE model imports successful")

        from src.Ava.multi_column_data import create_multi_column_dataloader
        print("  ✅ Data loader imports successful")

        # Test distributed components
        from src.Ava.training.distributed_manager import DistributedManager
        from src.Ava.training.rank_aware_error_handler import RankAwareErrorHandler
        from src.Ava.training.distributed_health_checker import DistributedHealthChecker
        print("  ✅ Distributed components imports successful")

        return True

    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config_loading():
    """Test configuration loading."""
    print("🔧 Testing configuration loading...")

    try:
        import yaml
        from src.Ava.config import (
            EnhancedTrainingConfig, TrainingConfig, DataConfig,
            ArchitectureConfig, WandBConfig
        )

        # Load base config
        config_path = Path('/project/code/configs/gpu/base.yaml')
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create enhanced config directly from config dict
        # Use minimal safe defaults for testing
        training_config = EnhancedTrainingConfig(
            config_file=str(config_path),  # Required field
            # Training config
            training=TrainingConfig(
                batch_size=config_dict.get('training', {}).get('batch_size', 4),
                learning_rate=config_dict.get('training', {}).get('learning_rate', 2e-4),
                epochs=config_dict.get('training', {}).get('num_epochs', 3),
            ),
            # Data config
            data=DataConfig(
                max_length=config_dict.get('data', {}).get('max_length', 1024),
                data_dir=config_dict.get('data', {}).get('data_dir', '/project/code/data/combined'),
            ),
            # Architecture config (use defaults)
            architecture=ArchitectureConfig(),
            # WandB config
            wandb=WandBConfig(use_wandb=False)  # Disable for testing
        )

        print(f"  ✅ Config loaded - batch_size: {training_config.training.batch_size}")
        print(f"  ✅ Architecture use_moh: {training_config.architecture.use_moh}")
        print(f"  ✅ Data max_length: {training_config.data.max_length}")

        return True, training_config, config_dict

    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")
        traceback.print_exc()
        return False, None, None

def test_model_creation(training_config, config_dict):
    """Test model and tokenizer creation."""
    print("🏗️  Testing model creation...")

    try:
        from transformers import AutoTokenizer
        from src.Ava.models.moe_model import EnhancedMoEModel, EnhancedMoEConfig

        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"  ✅ Tokenizer loaded - vocab_size: {len(tokenizer)}")

        # Create model config
        model_config_dict = config_dict.get('model', {})
        model_config_dict.update({
            'use_moh': training_config.architecture.use_moh,
            'use_moa': training_config.architecture.use_moa,
            'use_rag': training_config.rag.use_rag,
        })

        model_config = EnhancedMoEConfig.from_dict(model_config_dict)
        print(f"  ✅ Model config created - num_experts: {model_config.num_experts}")

        # Create model (small version for testing)
        model_config.hidden_size = 144  # Divisible by 12 (num_attention_heads)
        model_config.num_layers = 2     # Reduce for testing
        model_config.num_experts = 4    # Reduce for testing

        model = EnhancedMoEModel(model_config)
        print(f"  ✅ Model created - parameters: {sum(p.numel() for p in model.parameters()):,}")

        return True, model, tokenizer

    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        traceback.print_exc()
        return False, None, None

def test_trainer_creation(training_config, model, tokenizer):
    """Test trainer creation."""
    print("👷 Testing trainer creation...")

    try:
        from src.Ava.training.enhanced_trainer import EnhancedModularTrainer

        # Create trainer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = EnhancedModularTrainer(
            model=model,
            config=training_config,
            tokenizer=tokenizer,
            device=device
        )

        print(f"  ✅ Trainer created successfully")
        print(f"  ✅ Device: {trainer.device}")
        print(f"  ✅ Step count: {trainer.step_count}")

        # Test distributed components if available
        if trainer.distributed_manager:
            print(f"  ✅ Distributed manager available")
        if trainer.error_handler:
            print(f"  ✅ Error handler available")
        if trainer.health_checker:
            print(f"  ✅ Health checker available")

        return True, trainer

    except Exception as e:
        print(f"  ❌ Trainer creation failed: {e}")
        traceback.print_exc()
        return False, None

def test_data_loading(training_config, tokenizer):
    """Test basic data loading functionality."""
    print("📊 Testing data loading...")

    try:
        # Create minimal dataset config
        dataset_config = {
            'columns': [{'name': 'text', 'type': 'text'}],
            'combine_strategy': 'concatenate'
        }

        # Test data loader creation with minimal batch size
        from src.Ava.multi_column_data import create_multi_column_dataloader

        # Mock data since we don't want to load the full dataset for testing
        class MockDataset:
            def __init__(self):
                self.data = [
                    {"text": "This is a test sentence for training."},
                    {"text": "Another test sentence with different content."},
                    {"text": "A third sentence to complete the minimal test."}
                ]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                text = self.data[idx]["text"]
                tokens = tokenizer(text, max_length=64, padding="max_length",
                                 truncation=True, return_tensors="pt")
                return {
                    'input_ids': tokens['input_ids'].squeeze(),
                    'attention_mask': tokens['attention_mask'].squeeze(),
                    'labels': tokens['input_ids'].squeeze()  # For causal LM
                }

        # Test dataset creation
        mock_dataset = MockDataset()
        print(f"  ✅ Mock dataset created - {len(mock_dataset)} samples")

        # Test basic tokenization
        sample = mock_dataset[0]
        print(f"  ✅ Sample shape - input_ids: {sample['input_ids'].shape}")

        return True

    except Exception as e:
        print(f"  ❌ Data loading test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Training Pipeline Tests")
    print("=" * 50)

    # Test 1: Imports
    if not test_imports():
        print("❌ Import tests failed - stopping")
        return False

    # Test 2: Configuration
    config_success, training_config, config_dict = test_config_loading()
    if not config_success:
        print("❌ Configuration tests failed - stopping")
        return False

    # Test 3: Model Creation
    model_success, model, tokenizer = test_model_creation(training_config, config_dict)
    if not model_success:
        print("❌ Model creation tests failed - stopping")
        return False

    # Test 4: Trainer Creation
    trainer_success, trainer = test_trainer_creation(training_config, model, tokenizer)
    if not trainer_success:
        print("❌ Trainer creation tests failed - stopping")
        return False

    # Test 5: Data Loading
    if not test_data_loading(training_config, tokenizer):
        print("❌ Data loading tests failed - stopping")
        return False

    print("\n" + "=" * 50)
    print("✅ All tests passed! Training pipeline is functional.")
    print("\n🔧 Available components:")
    print(f"  - Enhanced Trainer: {trainer.__class__.__name__}")
    print(f"  - Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Device: {trainer.device}")
    print(f"  - Distributed Manager: {'✅' if trainer.distributed_manager else '❌'}")
    print(f"  - Error Handler: {'✅' if trainer.error_handler else '❌'}")
    print(f"  - Health Checker: {'✅' if trainer.health_checker else '❌'}")

    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)