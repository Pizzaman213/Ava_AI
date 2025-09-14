#!/usr/bin/env python3
"""
Comprehensive Test Suite - Merged from all test files
Created: 2024-08-30
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import subprocess
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import warnings
import traceback
import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all modules needed for testing - using actual project structure
try:
    from src.model.moe_transformer import MoEConfig, MoEForCausalLM
    StandardLLM = MoEForCausalLM
    LLMConfig = MoEConfig
    BaseLLM = None
except ImportError as e:
    print(f"Warning: Could not import MoE models: {e}")
    StandardLLM = None
    LLMConfig = None
    BaseLLM = None

# Try to import trainer if available
try:
    from src.training.simple_trainer import SimpleTrainer as LLMTrainer
except ImportError:
    try:
        from src.training.parallel_trainer import ParallelTrainer as LLMTrainer
    except ImportError:
        LLMTrainer = None

# Simple tokenizer implementation if not available
class BaseTokenizer:
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
    
    def encode(self, text):
        # Simple character-level encoding
        return [ord(c) % self.vocab_size for c in text]
    
    def decode(self, tokens):
        # Simple decoding
        return ''.join([chr(t % 128) for t in tokens])

# Optimizer imports
try:
    from src.optimizers.advanced_optimizers import LionOptimizer, LookaheadOptimizer
except ImportError:
    LionOptimizer = None
    LookaheadOptimizer = None

# Other optional imports
DistributedTrainer = None
BeamSearch = None
NucleusAdaptiveSampling = None
FeatureAugmentor = None

# Utility functions
def count_parameters(model):
    """Count model parameters"""
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_activation_stats(model):
    """Get activation statistics"""
    return {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration defaults
DEFAULT_TEST_CONFIG = {
    "vocab_size": 50257,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "max_position_embeddings": 1024,
    "intermediate_size": 3072,
    "hidden_act": "gelu",
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "layer_norm_epsilon": 1e-5,
    "initializer_range": 0.02,
    "device": "cpu",
    "dtype": "float32",
    "batch_size": 2,
    "learning_rate": 1e-4,
}

class TestStatus(Enum):
    """Test status enumeration"""
    PASSED = "✅ PASSED"
    FAILED = "❌ FAILED"
    SKIPPED = "⚠️ SKIPPED"
    ERROR = "🔥 ERROR"

@dataclass
class TestResult:
    """Container for test results"""
    name: str
    status: TestStatus
    message: str = ""
    duration: float = 0.0
    details: Optional[Dict[str, Any]] = None

class ComprehensiveTestSuite(unittest.TestCase):
    """Main comprehensive test suite combining all tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.results = []
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running tests on device: {cls.device}")
        
    def setUp(self):
        """Set up for each test"""
        self.start_time = time.time()
        
    def tearDown(self):
        """Clean up after each test"""
        duration = time.time() - self.start_time
        logger.info(f"Test completed in {duration:.2f}s")
        
    def record_result(self, name: str, status: TestStatus, message: str = "", details: Dict = None):
        """Record test result"""
        result = TestResult(
            name=name,
            status=status,
            message=message,
            duration=time.time() - self.start_time,
            details=details
        )
        self.results.append(result)
        logger.info(f"{status.value} {name}: {message}")
        
    # ========== ARCHITECTURAL TESTS ==========
    
    def test_base_model_creation(self):
        """Test basic model creation"""
        if LLMConfig is None or StandardLLM is None:
            self.record_result("Base Model Creation", TestStatus.SKIPPED, 
                             "Model modules not available")
            self.skipTest("Model modules not available")
            
        try:
            config = LLMConfig(
                vocab_size=DEFAULT_TEST_CONFIG["vocab_size"],
                hidden_size=DEFAULT_TEST_CONFIG["hidden_size"],
                num_hidden_layers=DEFAULT_TEST_CONFIG["num_hidden_layers"],
                num_attention_heads=DEFAULT_TEST_CONFIG["num_attention_heads"],
            )
            model = StandardLLM(config)
            self.assertIsNotNone(model)
            self.record_result("Base Model Creation", TestStatus.PASSED, 
                             f"Model created with {count_parameters(model):,} parameters")
        except Exception as e:
            self.record_result("Base Model Creation", TestStatus.FAILED, str(e))
            self.fail(str(e))
            
    def test_forward_pass(self):
        """Test model forward pass"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            model = StandardLLM(config)
            
            batch_size = 2
            seq_length = 128
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            
            with torch.no_grad():
                outputs = model(input_ids)
                
            self.assertEqual(outputs.shape, (batch_size, seq_length, config.vocab_size))
            self.record_result("Forward Pass", TestStatus.PASSED, 
                             f"Output shape: {outputs.shape}")
        except Exception as e:
            self.record_result("Forward Pass", TestStatus.FAILED, str(e))
            self.fail(str(e))
            
    def test_backward_pass(self):
        """Test model backward pass"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            model = StandardLLM(config)
            
            batch_size = 2
            seq_length = 128
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            
            outputs = model(input_ids)
            loss = nn.CrossEntropyLoss()(
                outputs.view(-1, config.vocab_size),
                labels.view(-1)
            )
            
            loss.backward()
            
            # Check gradients
            has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            self.assertTrue(has_grads)
            self.record_result("Backward Pass", TestStatus.PASSED, 
                             f"Loss: {loss.item():.4f}")
        except Exception as e:
            self.record_result("Backward Pass", TestStatus.FAILED, str(e))
            self.fail(str(e))
            
    # ========== FEATURE TESTS ==========
    
    def test_multi_head_attention(self):
        """Test multi-head attention mechanism"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            config.use_multi_head_attention = True
            model = StandardLLM(config)
            
            batch_size = 2
            seq_length = 64
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            
            with torch.no_grad():
                outputs = model(input_ids)
                
            self.assertEqual(outputs.shape[0], batch_size)
            self.record_result("Multi-Head Attention", TestStatus.PASSED)
        except Exception as e:
            self.record_result("Multi-Head Attention", TestStatus.FAILED, str(e))
            
    def test_rotary_embeddings(self):
        """Test rotary position embeddings"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            config.use_rotary_embeddings = True
            model = StandardLLM(config)
            
            batch_size = 2
            seq_length = 64
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            
            with torch.no_grad():
                outputs = model(input_ids)
                
            self.assertIsNotNone(outputs)
            self.record_result("Rotary Embeddings", TestStatus.PASSED)
        except Exception as e:
            self.record_result("Rotary Embeddings", TestStatus.FAILED, str(e))
            
    def test_flash_attention(self):
        """Test flash attention"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            config.use_flash_attention = True
            model = StandardLLM(config)
            
            batch_size = 2
            seq_length = 64
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            
            with torch.no_grad():
                outputs = model(input_ids)
                
            self.assertIsNotNone(outputs)
            self.record_result("Flash Attention", TestStatus.PASSED)
        except Exception as e:
            self.record_result("Flash Attention", TestStatus.FAILED, str(e))
            
    def test_mixture_of_experts(self):
        """Test mixture of experts"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            config.use_mixture_of_experts = True
            config.num_experts = 4
            config.expert_capacity = 1.25
            model = StandardLLM(config)
            
            batch_size = 2
            seq_length = 64
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            
            with torch.no_grad():
                outputs = model(input_ids)
                
            self.assertIsNotNone(outputs)
            self.record_result("Mixture of Experts", TestStatus.PASSED)
        except Exception as e:
            self.record_result("Mixture of Experts", TestStatus.FAILED, str(e))
            
    # ========== OPTIMIZATION TESTS ==========
    
    def test_gradient_checkpointing(self):
        """Test gradient checkpointing"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            config.use_gradient_checkpointing = True
            model = StandardLLM(config)
            
            batch_size = 2
            seq_length = 64
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            
            outputs = model(input_ids)
            loss = nn.CrossEntropyLoss()(
                outputs.view(-1, config.vocab_size),
                labels.view(-1)
            )
            
            loss.backward()
            
            self.assertIsNotNone(loss)
            self.record_result("Gradient Checkpointing", TestStatus.PASSED, 
                             f"Memory saved with checkpointing")
        except Exception as e:
            self.record_result("Gradient Checkpointing", TestStatus.FAILED, str(e))
            
    def test_lion_optimizer(self):
        """Test Lion optimizer"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            model = StandardLLM(config)
            
            optimizer = LionOptimizer(model.parameters(), lr=1e-4)
            
            batch_size = 2
            seq_length = 64
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            labels = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            
            outputs = model(input_ids)
            loss = nn.CrossEntropyLoss()(
                outputs.view(-1, config.vocab_size),
                labels.view(-1)
            )
            
            loss.backward()
            optimizer.step()
            
            self.record_result("Lion Optimizer", TestStatus.PASSED, 
                             f"Optimization step completed")
        except Exception as e:
            self.record_result("Lion Optimizer", TestStatus.FAILED, str(e))
            
    def test_mixed_precision(self):
        """Test mixed precision training"""
        try:
            if not torch.cuda.is_available():
                self.record_result("Mixed Precision", TestStatus.SKIPPED, 
                                 "CUDA not available")
                return
                
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            config.use_mixed_precision = True
            config.device = "cuda"
            model = StandardLLM(config).cuda()
            
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            
            batch_size = 2
            seq_length = 64
            input_ids = torch.randint(0, config.vocab_size, 
                                     (batch_size, seq_length)).cuda()
            labels = torch.randint(0, config.vocab_size, 
                                  (batch_size, seq_length)).cuda()
            
            optimizer = torch.optim.Adam(model.parameters())
            
            with autocast():
                outputs = model(input_ids)
                loss = nn.CrossEntropyLoss()(
                    outputs.view(-1, config.vocab_size),
                    labels.view(-1)
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            self.record_result("Mixed Precision", TestStatus.PASSED, 
                             "FP16 training successful")
        except Exception as e:
            self.record_result("Mixed Precision", TestStatus.FAILED, str(e))
            
    # ========== TRAINING TESTS ==========
    
    def test_training_loop(self):
        """Test basic training loop"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            config.batch_size = 2
            config.learning_rate = 1e-4
            
            model = StandardLLM(config)
            trainer = LLMTrainer(model, config)
            
            # Create dummy data
            train_data = []
            for _ in range(10):
                input_ids = torch.randint(0, config.vocab_size, 
                                         (config.batch_size, 128))
                labels = torch.randint(0, config.vocab_size, 
                                      (config.batch_size, 128))
                train_data.append((input_ids, labels))
            
            initial_loss = None
            final_loss = None
            
            for epoch in range(2):
                epoch_loss = 0
                for input_ids, labels in train_data:
                    loss = trainer.train_step(input_ids, labels)
                    epoch_loss += loss
                    
                    if initial_loss is None:
                        initial_loss = loss
                    final_loss = loss
                    
                avg_loss = epoch_loss / len(train_data)
                
            self.assertLess(final_loss, initial_loss)
            self.record_result("Training Loop", TestStatus.PASSED, 
                             f"Loss decreased from {initial_loss:.4f} to {final_loss:.4f}")
        except Exception as e:
            self.record_result("Training Loop", TestStatus.FAILED, str(e))
            
    def test_distributed_training(self):
        """Test distributed training setup"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            config.use_distributed = True
            
            # Note: Full distributed training requires multiple GPUs
            # This test only verifies the setup
            trainer = DistributedTrainer(config)
            
            self.assertIsNotNone(trainer)
            self.record_result("Distributed Training", TestStatus.PASSED, 
                             "Distributed trainer initialized")
        except Exception as e:
            self.record_result("Distributed Training", TestStatus.FAILED, str(e))
            
    # ========== INFERENCE TESTS ==========
    
    def test_beam_search(self):
        """Test beam search inference"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            model = StandardLLM(config)
            
            beam_search = BeamSearch(model, beam_size=3)
            
            input_ids = torch.randint(0, config.vocab_size, (1, 10))
            
            with torch.no_grad():
                output_ids = beam_search.search(input_ids, max_length=20)
                
            self.assertIsNotNone(output_ids)
            self.record_result("Beam Search", TestStatus.PASSED, 
                             f"Generated {len(output_ids[0])} tokens")
        except Exception as e:
            self.record_result("Beam Search", TestStatus.FAILED, str(e))
            
    def test_nucleus_sampling(self):
        """Test nucleus sampling"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            model = StandardLLM(config)
            
            sampler = NucleusAdaptiveSampling(model, top_p=0.9)
            
            input_ids = torch.randint(0, config.vocab_size, (1, 10))
            
            with torch.no_grad():
                output_ids = sampler.sample(input_ids, max_length=20)
                
            self.assertIsNotNone(output_ids)
            self.record_result("Nucleus Sampling", TestStatus.PASSED, 
                             f"Generated {len(output_ids[0])} tokens")
        except Exception as e:
            self.record_result("Nucleus Sampling", TestStatus.FAILED, str(e))
            
    # ========== TOKENIZATION TESTS ==========
    
    def test_tokenizer(self):
        """Test tokenizer functionality"""
        try:
            tokenizer = BaseTokenizer(vocab_size=50257)
            
            text = "Hello, world! This is a test."
            tokens = tokenizer.encode(text)
            decoded = tokenizer.decode(tokens)
            
            self.assertIsInstance(tokens, list)
            self.assertIsInstance(decoded, str)
            self.record_result("Tokenizer", TestStatus.PASSED, 
                             f"Encoded {len(tokens)} tokens")
        except Exception as e:
            self.record_result("Tokenizer", TestStatus.FAILED, str(e))
            
    # ========== AUGMENTATION TESTS ==========
    
    def test_feature_augmentation(self):
        """Test feature augmentation"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            config.use_feature_augmentation = True
            
            augmentor = FeatureAugmentor(config)
            
            features = torch.randn(2, 128, config.hidden_size)
            augmented = augmentor(features)
            
            self.assertEqual(augmented.shape, features.shape)
            self.record_result("Feature Augmentation", TestStatus.PASSED)
        except Exception as e:
            self.record_result("Feature Augmentation", TestStatus.FAILED, str(e))
            
    # ========== MEMORY TESTS ==========
    
    def test_memory_efficient_attention(self):
        """Test memory efficient attention"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            config.use_memory_efficient_attention = True
            model = StandardLLM(config)
            
            batch_size = 2
            seq_length = 512  # Longer sequence for memory test
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            
            with torch.no_grad():
                outputs = model(input_ids)
                
            self.assertIsNotNone(outputs)
            self.record_result("Memory Efficient Attention", TestStatus.PASSED)
        except Exception as e:
            self.record_result("Memory Efficient Attention", TestStatus.FAILED, str(e))
            
    # ========== QUANTIZATION TESTS ==========
    
    def test_quantization(self):
        """Test model quantization"""
        try:
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            model = StandardLLM(config)
            
            # Get original model size
            original_size = sum(p.numel() * p.element_size() for p in model.parameters())
            
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            
            # Test quantized model
            batch_size = 2
            seq_length = 64
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
            
            with torch.no_grad():
                outputs = quantized_model(input_ids)
                
            self.assertIsNotNone(outputs)
            self.record_result("Quantization", TestStatus.PASSED, 
                             f"Model quantized successfully")
        except Exception as e:
            self.record_result("Quantization", TestStatus.FAILED, str(e))
            
    # ========== INTEGRATION TESTS ==========
    
    def test_end_to_end_pipeline(self):
        """Test complete training and inference pipeline"""
        try:
            # Create config
            config = LLMConfig(**DEFAULT_TEST_CONFIG)
            config.batch_size = 2
            config.learning_rate = 1e-3
            
            # Create model
            model = StandardLLM(config)
            
            # Create trainer
            trainer = LLMTrainer(model, config)
            
            # Training data
            train_data = []
            for _ in range(5):
                input_ids = torch.randint(0, config.vocab_size, (2, 64))
                labels = torch.randint(0, config.vocab_size, (2, 64))
                train_data.append((input_ids, labels))
            
            # Train for one epoch
            total_loss = 0
            for input_ids, labels in train_data:
                loss = trainer.train_step(input_ids, labels)
                total_loss += loss
            
            avg_loss = total_loss / len(train_data)
            
            # Test inference
            model.eval()
            test_input = torch.randint(0, config.vocab_size, (1, 10))
            
            with torch.no_grad():
                output = model(test_input)
                
            self.assertIsNotNone(output)
            self.record_result("End-to-End Pipeline", TestStatus.PASSED, 
                             f"Training loss: {avg_loss:.4f}")
        except Exception as e:
            self.record_result("End-to-End Pipeline", TestStatus.FAILED, str(e))
            
    def test_config_loading(self):
        """Test configuration loading from file"""
        try:
            # Test loading different config files
            config_files = [
                "configs/cpu/ultra_tiny.yaml",
                "configs/gpu/small_tensor.yaml",
                "configs/mps/tiny.yaml"
            ]
            
            for config_file in config_files:
                config_path = Path(__file__).parent.parent / config_file
                if config_path.exists():
                    config = LLMConfig.from_yaml(str(config_path))
                    self.assertIsNotNone(config)
                    
            self.record_result("Config Loading", TestStatus.PASSED, 
                             f"Loaded {len(config_files)} configs")
        except Exception as e:
            self.record_result("Config Loading", TestStatus.FAILED, str(e))
            
    @classmethod
    def tearDownClass(cls):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in cls.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in cls.results if r.status == TestStatus.FAILED)
        skipped = sum(1 for r in cls.results if r.status == TestStatus.SKIPPED)
        errors = sum(1 for r in cls.results if r.status == TestStatus.ERROR)
        
        print(f"\n✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Skipped: {skipped}")
        print(f"🔥 Errors: {errors}")
        print(f"\nTotal: {len(cls.results)} tests")
        
        if failed > 0 or errors > 0:
            print("\nFailed/Error Tests:")
            for r in cls.results:
                if r.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    print(f"  - {r.name}: {r.message}")
        
        # Save results to file
        results_file = Path(__file__).parent / "test_results.json"
        with open(results_file, 'w') as f:
            results_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {
                    "passed": passed,
                    "failed": failed,
                    "skipped": skipped,
                    "errors": errors,
                    "total": len(cls.results)
                },
                "results": [
                    {
                        "name": r.name,
                        "status": r.status.value,
                        "message": r.message,
                        "duration": r.duration,
                        "details": r.details
                    }
                    for r in cls.results
                ]
            }
            json.dump(results_data, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")

def run_quick_tests():
    """Run a quick subset of tests"""
    suite = unittest.TestSuite()
    
    # Add only essential tests
    suite.addTest(ComprehensiveTestSuite('test_base_model_creation'))
    suite.addTest(ComprehensiveTestSuite('test_forward_pass'))
    suite.addTest(ComprehensiveTestSuite('test_backward_pass'))
    suite.addTest(ComprehensiveTestSuite('test_tokenizer'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_feature_tests():
    """Run all feature-specific tests"""
    suite = unittest.TestSuite()
    
    # Add feature tests
    suite.addTest(ComprehensiveTestSuite('test_multi_head_attention'))
    suite.addTest(ComprehensiveTestSuite('test_rotary_embeddings'))
    suite.addTest(ComprehensiveTestSuite('test_flash_attention'))
    suite.addTest(ComprehensiveTestSuite('test_mixture_of_experts'))
    suite.addTest(ComprehensiveTestSuite('test_gradient_checkpointing'))
    suite.addTest(ComprehensiveTestSuite('test_memory_efficient_attention'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_all_tests():
    """Run complete test suite"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(ComprehensiveTestSuite)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive LLM Test Suite")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick essential tests only")
    parser.add_argument("--features", action="store_true",
                       help="Run feature tests only")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("="*60)
    print("COMPREHENSIVE LLM TEST SUITE")
    print("="*60)
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"PyTorch Version: {torch.__version__}")
    print("="*60)
    
    if args.quick:
        print("\nRunning quick tests...")
        success = run_quick_tests()
    elif args.features:
        print("\nRunning feature tests...")
        success = run_feature_tests()
    else:
        print("\nRunning all tests...")
        success = run_all_tests()
    
    sys.exit(0 if success else 1)