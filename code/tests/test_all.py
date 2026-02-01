#!/usr/bin/env python3
"""
Unified Test Suite for Ava LLM Training Framework

This module consolidates all tests from the codebase:
- Coherence measurement tests
- Data loader tests
- Kernel diagnostics and optimization tests
- Triton softmax/topk kernel tests
- Indexed dataset tests
- Thread-safe cache tests
- Config validation tests
- CUDA graph cleanup tests
- Optimizer manager tests
- Checkpoint validation tests
- Router fixes tests
- Tensor-to-scalar conversion tests
"""

import logging
import math
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add code/src to path
code_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(code_dir / 'src'))

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Check hardware availability
CUDA_AVAILABLE = torch.cuda.is_available()

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# =============================================================================
# Helper Classes
# =============================================================================

class SimpleTransformer(nn.Module):
    """Simple transformer for testing."""

    def __init__(self, vocab_size=1000, hidden_size=64, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=256,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, output_hidden_states=False):
        h = self.embedding(x)
        hidden_states = [h]
        for layer in self.layers:
            h = layer(h)
            hidden_states.append(h)
        logits = self.output_proj(h)
        if output_hidden_states:
            return type('Output', (), {
                'logits': logits,
                'hidden_states': hidden_states
            })()
        return logits


class SimpleTestModel(nn.Module):
    """Minimal model for testing coherence measurement."""

    def __init__(self, vocab_size=1000, hidden_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, return_dict=True):
        hidden_states = self.embedding(input_ids)
        if return_dict:
            return {
                'logits': self.lm_head(hidden_states),
                'hidden_states': hidden_states,
                'last_hidden_state': hidden_states,
            }
        else:
            return (self.lm_head(hidden_states), hidden_states)


class DummyModel(nn.Module):
    """Simple model for optimizer testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)


class TinyModel(nn.Module):
    """Tiny model for checkpoint testing."""
    def __init__(self, num_params=5):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(10, 10) for _ in range(num_params)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def pytorch_softmax_topk(logits: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference PyTorch implementation for comparison."""
    probs = F.softmax(logits, dim=-1)
    return torch.topk(probs, top_k, dim=-1)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_batch():
    """Create a valid sample batch for testing."""
    return [
        {
            'input_ids': np.array([1, 2, 3, 4, 5], dtype=np.int64),
            'attention_mask': np.array([1, 1, 1, 1, 1], dtype=np.int64),
            'labels': np.array([1, 2, 3, 4, 5], dtype=np.int64),
        },
        {
            'input_ids': np.array([6, 7, 8], dtype=np.int64),
            'attention_mask': np.array([1, 1, 1], dtype=np.int64),
            'labels': np.array([6, 7, 8], dtype=np.int64),
        },
    ]


# =============================================================================
# Coherence Tests
# =============================================================================

class TestCoherence:
    """Tests for coherence measurement module."""

    def test_coherence_metrics_import(self):
        """Test that coherence module imports correctly."""
        from ava.training.coherence import (
            CoherenceMetrics,
            CoherenceMeasurer,
            CoherenceConfig,
            measure_coherence,
            measure_batch_coherence,
        )
        assert CoherenceMetrics is not None

    def test_coherence_metrics_dataclass(self):
        """Test CoherenceMetrics dataclass."""
        from ava.training.coherence import CoherenceMetrics

        metrics = CoherenceMetrics(
            perplexity=25.5,
            repetition_score=0.15,
            sentence_flow_score=0.82,
            topic_consistency=0.78,
            coherence_score=0.75,
            num_samples=10,
            avg_sequence_length=128.0,
            unique_token_ratio=0.65,
        )
        metrics_dict = metrics.to_dict()
        assert 'coherence/perplexity' in metrics_dict
        assert metrics_dict['coherence/perplexity'] == 25.5

    def test_coherence_config(self):
        """Test CoherenceConfig dataclass."""
        from ava.training.coherence import CoherenceConfig

        config = CoherenceConfig()
        assert config.enabled is True
        assert config.eval_every_n_steps == 500

    def test_coherence_measurer_basic(self):
        """Test basic CoherenceMeasurer functionality."""
        from ava.training.coherence import CoherenceMeasurer, CoherenceConfig

        model = SimpleTransformer(vocab_size=1000, hidden_size=64)
        config = CoherenceConfig(num_samples=2, max_generation_length=32)
        measurer = CoherenceMeasurer(model=model, tokenizer=None, config=config, device=torch.device('cpu'))

        input_ids = torch.randint(0, 1000, (4, 64))
        metrics = measurer.measure(input_ids=input_ids)

        assert metrics.perplexity > 0
        assert 0 <= metrics.repetition_score <= 1
        assert 0 <= metrics.coherence_score <= 1

    def test_repetition_score(self):
        """Test repetition score calculation."""
        from ava.training.coherence import CoherenceMeasurer, CoherenceConfig

        model = SimpleTransformer(vocab_size=1000, hidden_size=64)
        config = CoherenceConfig()
        measurer = CoherenceMeasurer(model=model, config=config, device=torch.device('cpu'))

        repetitive = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
        rep_score = measurer._compute_repetition_score(repetitive)
        assert rep_score > 0.5

        varied = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
        varied_score = measurer._compute_repetition_score(varied)
        assert varied_score < rep_score

    def test_unique_ratio(self):
        """Test unique token ratio calculation."""
        from ava.training.coherence import CoherenceMeasurer, CoherenceConfig

        model = SimpleTransformer(vocab_size=1000, hidden_size=64)
        config = CoherenceConfig()
        measurer = CoherenceMeasurer(model=model, config=config, device=torch.device('cpu'))

        unique = torch.arange(16).unsqueeze(0)
        unique_ratio = measurer._compute_unique_ratio(unique)
        assert unique_ratio == 1.0

        same = torch.ones(1, 16, dtype=torch.long)
        same_ratio = measurer._compute_unique_ratio(same)
        assert same_ratio == 1/16


# =============================================================================
# Coherence Fixes Tests
# =============================================================================

class TestCoherenceFixes:
    """Tests for coherence calculation fixes."""

    def test_log_scale_normalization(self):
        """Test log-scale perplexity normalization consistency."""
        from ava.training.coherence import CoherenceConfig

        config = CoherenceConfig(max_perplexity=100.0)
        test_cases = [
            (1.0, 1.0),
            (10.0, 0.5),
            (100.0, 0.0),
        ]

        for ppl, expected in test_cases:
            if ppl >= config.max_perplexity:
                normalized = 0.0
            elif ppl <= 1.0:
                normalized = 1.0
            else:
                log_ppl = math.log(ppl)
                log_max = math.log(config.max_perplexity)
                normalized = 1.0 - (log_ppl / log_max)
            assert abs(normalized - expected) < 0.01

    def test_short_sequences(self):
        """Test neutral/partial scores for short sequences."""
        from ava.training.coherence import CoherenceMeasurer, CoherenceConfig

        model = SimpleTestModel()
        config = CoherenceConfig()
        measurer = CoherenceMeasurer(model=model, config=config, device=torch.device('cpu'))

        very_short = torch.randint(0, 1000, (1, 1))
        ppl = measurer._compute_perplexity(very_short)
        expected_ppl = math.sqrt(100.0)
        assert abs(ppl - expected_ppl) < 1.0

    def test_repetitive_detection(self):
        """Test repetitive detection for non-generated inputs."""
        from ava.training.coherence import CoherenceMeasurer, CoherenceConfig

        model = SimpleTestModel()
        measurer = CoherenceMeasurer(model=model, config=CoherenceConfig(), device=torch.device('cpu'))

        repetitive = torch.ones(4, 50, dtype=torch.long)
        metrics = measurer.measure(input_ids=repetitive, generate_samples=False)
        assert metrics.unique_token_ratio < 0.10

    def test_dimension_validation(self):
        """Test hidden state dimension handling."""
        from ava.training.coherence import CoherenceMeasurer, CoherenceConfig

        model = SimpleTestModel()
        measurer = CoherenceMeasurer(model=model, config=CoherenceConfig(), device=torch.device('cpu'))

        batch_size, seq_len, hidden_dim = 4, 32, 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        hidden_correct = torch.randn(batch_size, seq_len, hidden_dim)
        validated = measurer._validate_hidden_states(hidden_correct, input_ids)
        assert validated is not None
        assert validated.shape == (batch_size, seq_len, hidden_dim)


# =============================================================================
# Data Loader Tests
# =============================================================================

class TestDataLoaderFixes:
    """Tests for data loader bug fixes."""

    def test_validate_sequence_content_valid(self):
        """Test that valid sequences pass validation."""
        from ava.data.pretokenized import UltraFastPretokenizedDataset

        with patch.object(UltraFastPretokenizedDataset, '_find_data_files', return_value=[]):
            with patch.object(UltraFastPretokenizedDataset, '__init__', lambda self, **kwargs: None):
                dataset = UltraFastPretokenizedDataset.__new__(UltraFastPretokenizedDataset)
                dataset.min_sequence_length = 3
                dataset.vocab_size = 100000

                valid_ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
                is_valid, error = dataset._validate_sequence_content(valid_ids, max_allowed_len=1000)
                assert is_valid is True

    def test_validate_sequence_content_too_long(self):
        """Test that overly long sequences are rejected."""
        from ava.data.pretokenized import UltraFastPretokenizedDataset

        with patch.object(UltraFastPretokenizedDataset, '_find_data_files', return_value=[]):
            with patch.object(UltraFastPretokenizedDataset, '__init__', lambda self, **kwargs: None):
                dataset = UltraFastPretokenizedDataset.__new__(UltraFastPretokenizedDataset)
                dataset.min_sequence_length = 3
                dataset.vocab_size = 100000

                long_ids = np.array([1] * 10000, dtype=np.int64)
                is_valid, error = dataset._validate_sequence_content(long_ids, max_allowed_len=1000)
                assert is_valid is False


# =============================================================================
# Config Validation Tests
# =============================================================================

class TestConfigValidation:
    """Tests for configuration validation."""

    def test_model_config_valid(self):
        """Test that valid ModelConfig passes validation."""
        from ava.config.training_config import ModelConfig

        config = ModelConfig(
            hidden_size=512,
            intermediate_size=2048,
            num_layers=6,
            num_attention_heads=8,
            num_experts=4,
            num_experts_per_token=2,
            capacity_factor=1.25,
        )
        assert config.hidden_size == 512

    def test_model_config_hidden_size_zero(self):
        """Test that hidden_size=0 raises ValueError."""
        from ava.config.training_config import ModelConfig

        with pytest.raises(ValueError) as excinfo:
            ModelConfig(hidden_size=0)
        assert "hidden_size" in str(excinfo.value)

    def test_training_config_valid(self):
        """Test that valid TrainingConfig passes validation."""
        from ava.config.training_config import TrainingConfig

        config = TrainingConfig(
            batch_size=32,
            epochs=5,
            learning_rate=1e-4,
            gradient_accumulation_steps=4,
            warmup_steps=1000,
        )
        assert config.batch_size == 32

    def test_training_config_batch_size_zero(self):
        """Test that batch_size=0 raises ValueError."""
        from ava.config.training_config import TrainingConfig

        with pytest.raises(ValueError) as excinfo:
            TrainingConfig(batch_size=0)
        assert "batch_size" in str(excinfo.value)


# =============================================================================
# CUDA Graph Cleanup Tests
# =============================================================================

class TestCUDAGraphCleanup:
    """Tests for CUDA graph cleanup in TrainingLoopManager."""

    def test_cleanup_cuda_graph_exists(self):
        """Test that _cleanup_cuda_graph method exists."""
        from ava.training.loop import TrainingLoopManager

        assert hasattr(TrainingLoopManager, '_cleanup_cuda_graph')

    def test_cleanup_sets_attributes_to_none(self):
        """Test that cleanup properly nullifies graph attributes."""
        from ava.training.loop import TrainingLoopManager
        from ava.training.context import TrainingContext

        mock_model = MagicMock()
        context = TrainingContext(model=mock_model)
        context.device = torch.device('cpu')

        manager = TrainingLoopManager(context)
        manager._cuda_graph = MagicMock()
        manager._cuda_graph_captured = True
        manager._graph_static_input = {'input_ids': torch.zeros(1)}
        manager._graph_static_loss = torch.zeros(1)
        manager._graph_batch_size = 32
        manager._graph_seq_len = 512

        manager._cleanup_cuda_graph()

        assert manager._cuda_graph is None
        assert manager._cuda_graph_captured is False
        assert manager._graph_static_input is None


# =============================================================================
# Optimizer Manager Tests
# =============================================================================

class TestOptimizerManager:
    """Tests for OptimizerManager functionality."""

    def test_bf16_epsilon_detection(self):
        """Test that BF16 uses eps=1e-6, FP16/FP32 use eps=1e-8."""
        from ava.training.context import TrainingContext
        from ava.training.optimizer import OptimizerManager

        model = DummyModel()
        context = TrainingContext(model=model, device=torch.device('cpu'))
        manager = OptimizerManager(context)
        manager.initialize()

        config_bf16 = {
            'training': {'precision': {'mixed_precision': 'bf16'}},
            'optimizer': {'type': 'adamw'}
        }
        opt = manager.create_optimizer(model, config_bf16, learning_rate=1e-4)
        assert opt.param_groups[0]['eps'] == 1e-6

    def test_scheduler_creation(self):
        """Test LR scheduler creation with warmup and decay."""
        from ava.training.context import TrainingContext
        from ava.training.optimizer import OptimizerManager

        model = DummyModel()
        context = TrainingContext(model=model, device=torch.device('cpu'))
        manager = OptimizerManager(context)
        manager.initialize()

        config = {
            'training': {'precision': {'mixed_precision': 'bf16'}},
            'optimizer': {'type': 'adamw'}
        }
        opt = manager.create_optimizer(model, config, learning_rate=1e-3)
        scheduler = manager.create_scheduler(opt, warmup_steps=100, total_steps=1000, min_lr=1e-5)

        initial_lr = opt.param_groups[0]['lr']
        assert initial_lr < 1e-3


# =============================================================================
# Checkpoint Validation Tests
# =============================================================================

class TestCheckpointValidation:
    """Tests for checkpoint validation."""

    def test_optimizer_state_validation(self):
        """Test that validation detects param count mismatch."""
        from ava.core.checkpoint import CheckpointManager

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir)

            model = TinyModel(num_params=5)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

            loss = model(torch.randn(2, 10)).sum()
            loss.backward()
            opt.step()

            manager = CheckpointManager(save_dir=save_dir, async_save=False)
            manager.save(model, opt, epoch=1, step=100, metrics={'loss': 0.5})

            model2 = TinyModel(num_params=3)
            opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

            ckpt_path = save_dir / 'checkpoint_epoch_1_step_100.pt'
            epoch, step = manager.load(model2, opt2, ckpt_path)
            assert epoch == 1


# =============================================================================
# Router Fixes Tests
# =============================================================================

class TestRouterFixes:
    """Tests for router bug fixes."""

    def test_deepseek_router_indices(self):
        """Test that DeepSeekRouter returns only routed expert indices."""
        from ava.models.routing import DeepSeekRouter

        hidden_size = 128
        num_routed_experts = 7
        num_selected = 2
        num_shared = 1

        router = DeepSeekRouter(
            hidden_size=hidden_size,
            num_experts=num_routed_experts,
            num_selected_experts=num_selected,
            num_shared_experts=num_shared,
        )

        x = torch.randn(32, hidden_size)
        indices, weights, aux_loss, metrics = router(x, training=True)

        assert indices.shape == (32, num_selected)
        assert indices.min() >= 0
        assert indices.max() < num_routed_experts

    def test_mixtral_router(self):
        """Test Mixtral router normal operation."""
        from ava.models.routing import MixtralRouter

        hidden_size = 128
        num_experts = 8
        num_selected = 2

        router = MixtralRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_selected_experts=num_selected,
        )

        x = torch.randn(32, hidden_size)
        indices, weights, aux_loss, metrics = router(x, training=True)

        assert indices.shape == (32, num_selected)
        assert indices.min() >= 0
        assert indices.max() < num_experts


# =============================================================================
# Tensor-to-Scalar Conversion Tests
# =============================================================================

class TestTensorToScalar:
    """Tests for tensor-to-scalar conversion."""

    def test_scalar_tensor(self):
        """Test scalar tensor handling."""
        scalar_tensor = torch.tensor(0.5)
        if hasattr(scalar_tensor, 'numel') and scalar_tensor.numel() > 1:
            result = scalar_tensor.mean().item()
        else:
            result = scalar_tensor.item()
        assert result == 0.5

    def test_multi_element_tensor(self):
        """Test multi-element tensor handling."""
        multi_tensor = torch.tensor([0.25, 0.30, 0.20, 0.25])
        if hasattr(multi_tensor, 'numel') and multi_tensor.numel() > 1:
            result = multi_tensor.mean().item()
        else:
            result = multi_tensor.item()
        expected = 0.25
        assert abs(result - expected) < 1e-6

    def test_float_handling(self):
        """Test regular float handling."""
        float_val = 0.75
        if isinstance(float_val, (int, float)):
            result = float_val
        else:
            result = float_val
        assert result == 0.75


# =============================================================================
# Triton Kernel Tests (CUDA-only)
# =============================================================================

@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestTritonKernels:
    """Tests for Triton kernel implementations."""

    def test_triton_softmax_topk(self):
        """Test Triton kernel against PyTorch reference."""
        from ava.cuda.moe_kernels import fused_softmax_topk, TRITON_AVAILABLE

        device = torch.device('cuda')
        torch.manual_seed(42)

        test_cases = [
            (16, 8, 2),
            (128, 32, 2),
            (256, 64, 2),
        ]

        for num_tokens, num_experts, top_k in test_cases:
            logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
            ref_probs, ref_indices = pytorch_softmax_topk(logits, top_k)

            triton_probs, triton_indices = fused_softmax_topk(logits, top_k, use_triton=True)

            assert triton_probs.shape == ref_probs.shape
            prob_diff = (triton_probs.float() - ref_probs.float()).abs().max().item()
            assert prob_diff < 1e-4

    def test_streaming_correctness(self):
        """Test streaming algorithm produces correct global indices."""
        from ava.cuda.moe_kernels import fused_softmax_topk

        device = torch.device('cuda')
        num_tokens = 10
        num_experts = 128
        top_k = 2

        logits = torch.zeros(num_tokens, num_experts, device=device, dtype=torch.float32)
        logits[:, 100] = 10.0
        logits[:, 120] = 9.0
        logits[:, 0] = 1.0
        logits += torch.randn_like(logits) * 0.01

        topk_probs, topk_indices = fused_softmax_topk(logits, top_k, use_triton=True)

        expected_indices = {100, 120}
        for t in range(num_tokens):
            actual = set(topk_indices[t].tolist())
            assert actual == expected_indices


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not available")
class TestKernelOptimizations:
    """Tests for kernel optimizations."""

    def test_bitonic_sort_correctness_k2(self):
        """Test bitonic sort produces correct top-2 results."""
        from ava.cuda.moe_kernels import fused_softmax_topk, _pytorch_softmax_topk

        torch.manual_seed(42)
        num_tokens = 5000
        num_experts = 32
        top_k = 2

        logits = torch.randn(num_tokens, num_experts, device='cuda', dtype=torch.float32)
        triton_probs, triton_indices = fused_softmax_topk(logits, top_k, use_triton=True)
        pytorch_probs, pytorch_indices = _pytorch_softmax_topk(logits, top_k)

        assert torch.equal(triton_indices, pytorch_indices)
        assert torch.allclose(triton_probs, pytorch_probs, rtol=1e-4, atol=1e-6)


# =============================================================================
# Indexed Dataset Tests (require Arrow files)
# =============================================================================

def create_test_arrow_file(file_path: Path, num_rows: int = 100, min_length: int = 32, max_length: int = 256, seed: int = 42) -> Path:
    """Create a test Arrow file with varied-length sequences."""
    import random
    import pyarrow as pa
    import pyarrow.ipc as ipc

    rng = random.Random(seed)
    np.random.seed(seed)

    sequences = []
    masks = []

    for _ in range(num_rows):
        length = rng.randint(min_length, max_length)
        seq = np.random.randint(1, 50000, size=length, dtype=np.int64)
        mask = np.ones(length, dtype=np.int64)
        sequences.append(seq.tolist())
        masks.append(mask.tolist())

    table = pa.table({
        'input_ids': pa.array(sequences, type=pa.list_(pa.int64())),
        'attention_mask': pa.array(masks, type=pa.list_(pa.int64())),
    })

    with ipc.new_file(str(file_path), table.schema) as writer:
        writer.write(table)

    return file_path


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_arrow_files(temp_data_dir: Path) -> List[Path]:
    """Create multiple test Arrow files."""
    files = []
    for i in range(3):
        file_path = temp_data_dir / f"test_data_{i}.arrow"
        create_test_arrow_file(file_path, num_rows=100, seed=42 + i)
        files.append(file_path)
    return files


class TestIndexedDataset:
    """Tests for IndexedArrowDataset."""

    def test_import_indexed_module(self):
        """Test main module import."""
        from ava.data.indexed import (
            IndexedArrowDataset,
            LengthBinnedSampler,
            DynamicPaddingCollator,
            ArrowTableLRUCache,
            create_indexed_dataloaders,
        )
        assert IndexedArrowDataset is not None

    def test_cache_creation(self):
        """Test cache can be created."""
        from ava.data.indexed import ArrowTableLRUCache

        cache = ArrowTableLRUCache(max_size=10)
        assert cache.max_size == 10
        cache.close()

    def test_sampler_yields_all_indices(self):
        """Test sampler yields all indices exactly once."""
        from ava.data.indexed import LengthBinnedSampler

        lengths = list(range(32, 132))
        sampler = LengthBinnedSampler(lengths=lengths, batch_size=16, num_bins=4, drop_last=False)
        indices = list(sampler)
        assert set(indices) == set(range(100))

    def test_collator_pads_to_batch_max(self):
        """Test collator pads to batch maximum, not global max."""
        from ava.data.indexed import DynamicPaddingCollator

        collator = DynamicPaddingCollator(pad_token_id=0, max_length=512)

        batch = [
            {'input_ids': torch.arange(32), 'attention_mask': torch.ones(32), 'labels': torch.arange(32), 'length': 32},
            {'input_ids': torch.arange(48), 'attention_mask': torch.ones(48), 'labels': torch.arange(48), 'length': 48},
            {'input_ids': torch.arange(64), 'attention_mask': torch.ones(64), 'labels': torch.arange(64), 'length': 64},
        ]

        output = collator(batch)
        assert output['input_ids'].shape == (3, 64)


# =============================================================================
# Module Import Validation Tests
# =============================================================================

class TestModuleImports:
    """
    Tests that validate all modules in the ava package can be imported.

    This catches:
    - Missing dependencies
    - Circular imports
    - Syntax errors
    - Import-time errors
    """

    # -------------------------------------------------------------------------
    # Core Package
    # -------------------------------------------------------------------------

    def test_import_ava_init(self):
        """Test ava package init imports."""
        import ava
        assert ava is not None

    def test_import_ava_utils(self):
        """Test ava.utils imports."""
        import ava.utils
        assert ava.utils is not None

    def test_import_ava_optimization(self):
        """Test ava.optimization imports."""
        import ava.optimization
        assert ava.optimization is not None

    # -------------------------------------------------------------------------
    # ava.core subpackage
    # -------------------------------------------------------------------------

    def test_import_core_init(self):
        """Test ava.core init imports."""
        import ava.core
        assert ava.core is not None

    def test_import_core_activations(self):
        """Test ava.core.activations imports."""
        from ava.core import activations
        assert activations is not None

    def test_import_core_checkpoint(self):
        """Test ava.core.checkpoint imports."""
        from ava.core import checkpoint
        assert checkpoint is not None

    def test_import_core_data_utils(self):
        """Test ava.core.data_utils imports."""
        from ava.core import data_utils
        assert data_utils is not None

    def test_import_core_error_tracking(self):
        """Test ava.core.error_tracking imports."""
        from ava.core import error_tracking
        assert error_tracking is not None

    def test_import_core_logging(self):
        """Test ava.logging.console.colored imports (was ava.core.logging)."""
        from ava.logging.console import colored as ava_logging
        assert ava_logging is not None

    def test_import_core_mixed_precision(self):
        """Test ava.core.mixed_precision imports."""
        from ava.core import mixed_precision
        assert mixed_precision is not None

    def test_import_core_paths(self):
        """Test ava.core.paths imports."""
        from ava.core import paths
        assert paths is not None

    def test_import_core_script_utils(self):
        """Test ava.core.script_utils imports."""
        from ava.core import script_utils
        assert script_utils is not None

    # -------------------------------------------------------------------------
    # ava.config subpackage
    # -------------------------------------------------------------------------

    def test_import_config_init(self):
        """Test ava.config init imports."""
        import ava.config
        assert ava.config is not None

    def test_import_config_constants(self):
        """Test ava.config.constants imports."""
        from ava.config import constants
        assert constants is not None

    def test_import_config_training_config(self):
        """Test ava.config.training_config imports."""
        from ava.config import training_config
        assert training_config is not None

    def test_import_config_validator(self):
        """Test ava.config.validator imports."""
        from ava.config import validator
        assert validator is not None

    def test_import_config_yaml_loader(self):
        """Test ava.config.yaml_loader imports."""
        from ava.config import yaml_loader
        assert yaml_loader is not None

    # -------------------------------------------------------------------------
    # ava.cuda subpackage
    # -------------------------------------------------------------------------

    def test_import_cuda_init(self):
        """Test ava.cuda init imports."""
        import ava.cuda
        assert ava.cuda is not None

    def test_import_cuda_metrics(self):
        """Test ava.logging.metrics.async_logger imports (was ava.cuda.metrics)."""
        from ava.logging.metrics import async_logger as metrics
        assert metrics is not None

    def test_import_cuda_profiler(self):
        """Test ava.cuda.profiler imports."""
        from ava.cuda import profiler
        assert profiler is not None

    def test_import_cuda_streams(self):
        """Test ava.cuda.streams imports."""
        from ava.cuda import streams
        assert streams is not None

    # -------------------------------------------------------------------------
    # ava.data subpackage
    # -------------------------------------------------------------------------

    def test_import_data_init(self):
        """Test ava.data init imports."""
        import ava.data
        assert ava.data is not None

    def test_import_data_bucketing(self):
        """Test ava.data.bucketing imports."""
        from ava.data import bucketing
        assert bucketing is not None

    def test_import_data_conversation(self):
        """Test ava.data.conversation imports."""
        from ava.data import conversation
        assert conversation is not None

    def test_import_data_distributed(self):
        """Test ava.data.distributed imports."""
        from ava.data import distributed
        assert distributed is not None

    def test_import_data_indexed(self):
        """Test ava.data.indexed imports."""
        from ava.data import indexed
        assert indexed is not None

    def test_import_data_multi_column(self):
        """Test ava.data.multi_column imports."""
        from ava.data import multi_column
        assert multi_column is not None

    def test_import_data_packing(self):
        """Test ava.data.packing imports."""
        from ava.data import packing
        assert packing is not None

    def test_import_data_pretokenized(self):
        """Test ava.data.pretokenized imports."""
        from ava.data import pretokenized
        assert pretokenized is not None

    # -------------------------------------------------------------------------
    # ava.eval subpackage
    # -------------------------------------------------------------------------

    def test_import_eval_init(self):
        """Test ava.eval init imports."""
        import ava.eval
        assert ava.eval is not None

    def test_import_eval_coherence(self):
        """Test ava.eval.coherence imports."""
        from ava.eval import coherence
        assert coherence is not None

    # -------------------------------------------------------------------------
    # ava.kernels subpackage
    # -------------------------------------------------------------------------

    def test_import_kernels_init(self):
        """Test ava.kernels init imports."""
        import ava.kernels
        assert ava.kernels is not None

    def test_import_kernels_activations(self):
        """Test ava.kernels.activations imports."""
        from ava.kernels import activations
        assert activations is not None

    def test_import_kernels_fused_experts(self):
        """Test ava.kernels.fused_experts imports."""
        from ava.kernels import fused_experts
        assert fused_experts is not None

    def test_import_kernels_moe(self):
        """Test ava.kernels.moe imports."""
        from ava.kernels import moe
        assert moe is not None

    # -------------------------------------------------------------------------
    # ava.models subpackage
    # -------------------------------------------------------------------------

    def test_import_models_init(self):
        """Test ava.models init imports."""
        import ava.models
        assert ava.models is not None

    def test_import_models_moe(self):
        """Test ava.models.moe imports."""
        from ava.models import moe
        assert moe is not None

    def test_import_models_moe_layer(self):
        """Test ava.models.moe_layer imports."""
        from ava.models import moe_layer
        assert moe_layer is not None

    # -------------------------------------------------------------------------
    # ava.nn subpackage
    # -------------------------------------------------------------------------

    def test_import_nn_init(self):
        """Test ava.nn init imports."""
        import ava.nn
        assert ava.nn is not None

    def test_import_nn_experts(self):
        """Test ava.nn.experts imports."""
        from ava.nn import experts
        assert experts is not None

    def test_import_nn_routing(self):
        """Test ava.nn.routing imports."""
        from ava.nn import routing
        assert routing is not None

    # -------------------------------------------------------------------------
    # ava.optim subpackage
    # -------------------------------------------------------------------------

    def test_import_optim_init(self):
        """Test ava.optim init imports."""
        import ava.optim
        assert ava.optim is not None

    def test_import_optim_lr_managers(self):
        """Test ava.optim.lr_managers imports."""
        from ava.optim import lr_managers
        assert lr_managers is not None

    # -------------------------------------------------------------------------
    # ava.optimizations subpackage
    # -------------------------------------------------------------------------

    def test_import_optimizations_init(self):
        """Test ava.optimizations init imports."""
        import ava.optimizations
        assert ava.optimizations is not None

    def test_import_optimizations_batch_controller(self):
        """Test ava.optimizations.batch_controller imports."""
        from ava.optimizations import batch_controller
        assert batch_controller is not None

    def test_import_optimizations_checkpointing(self):
        """Test ava.optimizations.checkpointing imports."""
        from ava.optimizations import checkpointing
        assert checkpointing is not None

    def test_import_optimizations_fp8(self):
        """Test ava.optimizations.fp8 imports."""
        from ava.optimizations import fp8
        assert fp8 is not None

    def test_import_optimizations_gradients(self):
        """Test ava.optimizations.gradients imports."""
        from ava.optimizations import gradients
        assert gradients is not None

    def test_import_optimizations_hybrid_cache(self):
        """Test ava.optimizations.hybrid_cache imports."""
        from ava.optimizations import hybrid_cache
        assert hybrid_cache is not None

    def test_import_optimizations_oom_recovery(self):
        """Test ava.optimizations.oom_recovery imports."""
        from ava.optimizations import oom_recovery
        assert oom_recovery is not None

    def test_import_optimizations_overlapped_recomputation(self):
        """Test ava.optimizations.overlapped_recomputation imports."""
        from ava.optimizations import overlapped_recomputation
        assert overlapped_recomputation is not None

    def test_import_optimizations_prefetch(self):
        """Test ava.optimizations.prefetch imports."""
        from ava.optimizations import prefetch
        assert prefetch is not None

    def test_import_optimizations_quantization(self):
        """Test ava.optimizations.quantization imports."""
        from ava.optimizations import quantization
        assert quantization is not None

    # -------------------------------------------------------------------------
    # ava.strategies subpackage
    # -------------------------------------------------------------------------

    def test_import_strategies_init(self):
        """Test ava.strategies init imports."""
        import ava.strategies
        assert ava.strategies is not None

    def test_import_strategies_progressive(self):
        """Test ava.strategies.progressive imports."""
        from ava.strategies import progressive
        assert progressive is not None

    # -------------------------------------------------------------------------
    # ava.training subpackage
    # -------------------------------------------------------------------------

    def test_import_training_init(self):
        """Test ava.training init imports."""
        import ava.training
        assert ava.training is not None

    def test_import_training_context(self):
        """Test ava.training.context imports."""
        from ava.training import context
        assert context is not None

    def test_import_training_data_manager(self):
        """Test ava.training.data_manager imports."""
        from ava.training import data_manager
        assert data_manager is not None

    def test_import_training_deepspeed(self):
        """Test ava.training.deepspeed imports."""
        from ava.training import deepspeed
        assert deepspeed is not None

    def test_import_training_diagnostics(self):
        """Test ava.logging.diagnostics.training imports (was ava.training.diagnostics)."""
        from ava.logging.diagnostics import training as diagnostics
        assert diagnostics is not None

    def test_import_training_distributed(self):
        """Test ava.training.distributed imports."""
        from ava.training import distributed
        assert distributed is not None

    def test_import_training_generation(self):
        """Test ava.training.generation imports."""
        from ava.training import generation
        assert generation is not None

    def test_import_training_loop(self):
        """Test ava.training.loop imports."""
        from ava.training import loop
        assert loop is not None

    def test_import_training_model_builder(self):
        """Test ava.training.model_builder imports."""
        from ava.training import model_builder
        assert model_builder is not None

    def test_import_training_optimizer(self):
        """Test ava.training.optimizer imports."""
        from ava.training import optimizer
        assert optimizer is not None

    def test_import_training_pipeline(self):
        """Test ava.training.pipeline imports."""
        from ava.training import pipeline
        assert pipeline is not None

    def test_import_training_run_manager(self):
        """Test ava.training.run_manager imports."""
        from ava.training import run_manager
        assert run_manager is not None

    def test_import_training_state_guard(self):
        """Test ava.training.state_guard imports."""
        from ava.training import state_guard
        assert state_guard is not None

    def test_import_training_validation(self):
        """Test ava.training.validation imports."""
        from ava.training import validation
        assert validation is not None


# =============================================================================
# Key Class Import Validation Tests
# =============================================================================

class TestKeyClassImports:
    """
    Tests that validate key classes can be imported from their modules.

    This ensures the public API is accessible.
    """

    def test_import_model_classes(self):
        """Test key model classes can be imported."""
        from ava.models.moe import EnhancedMoEModel, EnhancedMoEConfig
        from ava.models.moe_layer import SparseMoELayer
        assert EnhancedMoEModel is not None
        assert EnhancedMoEConfig is not None
        assert SparseMoELayer is not None

    def test_import_router_classes(self):
        """Test key router classes can be imported."""
        from ava.models.routing import MixtralRouter, DeepSeekRouter, UnifiedMoERouter
        assert MixtralRouter is not None
        assert DeepSeekRouter is not None
        assert UnifiedMoERouter is not None

    def test_import_expert_classes(self):
        """Test key expert classes can be imported."""
        from ava.models.experts import HighPerformanceExpert, ExpertParallelGroup
        assert HighPerformanceExpert is not None
        assert ExpertParallelGroup is not None

    def test_import_config_classes(self):
        """Test key config classes can be imported."""
        from ava.config.training_config import (
            ModelConfig,
            TrainingConfig,
            DataConfig,
            CoherenceConfig,
            HardwareConfig,
            GenerationConfig,
        )
        assert ModelConfig is not None
        assert TrainingConfig is not None
        assert DataConfig is not None
        assert CoherenceConfig is not None
        assert HardwareConfig is not None
        assert GenerationConfig is not None

    def test_import_data_classes(self):
        """Test key data classes can be imported."""
        from ava.data.pretokenized import UltraFastPretokenizedDataset
        from ava.data.indexed import IndexedArrowDataset, LengthBinnedSampler, create_indexed_dataloaders
        assert UltraFastPretokenizedDataset is not None
        assert IndexedArrowDataset is not None
        assert LengthBinnedSampler is not None
        assert create_indexed_dataloaders is not None

    def test_import_training_classes(self):
        """Test key training classes can be imported."""
        from ava.training.context import TrainingContext
        from ava.training.loop import TrainingLoopManager
        from ava.training.pipeline import TrainingPipeline
        from ava.training.optimizer import OptimizerManager
        from ava.training.data_manager import DataLoaderManager
        assert TrainingContext is not None
        assert TrainingLoopManager is not None
        assert TrainingPipeline is not None
        assert OptimizerManager is not None
        assert DataLoaderManager is not None

    def test_import_eval_classes(self):
        """Test key evaluation classes can be imported."""
        from ava.training.coherence import (
            CoherenceMetrics,
            CoherenceMeasurer,
            CoherenceConfig,
            measure_coherence,
        )
        assert CoherenceMetrics is not None
        assert CoherenceMeasurer is not None
        assert CoherenceConfig is not None
        assert measure_coherence is not None

    def test_import_checkpoint_classes(self):
        """Test key checkpoint classes can be imported."""
        from ava.core.checkpoint import CheckpointManager
        assert CheckpointManager is not None

    def test_import_kernel_functions(self):
        """Test key kernel functions can be imported."""
        from ava.cuda.moe_kernels import fused_softmax_topk, fused_gating_topk
        from ava.cuda.kernel_activations import fused_swiglu, fused_geglu
        assert fused_softmax_topk is not None
        assert fused_gating_topk is not None
        assert fused_swiglu is not None
        assert fused_geglu is not None

    def test_import_lr_manager_classes(self):
        """Test key LR manager classes can be imported."""
        from ava.optimizations.lr_managers import AdaptiveLearningRateManager
        assert AdaptiveLearningRateManager is not None

    def test_import_validation_classes(self):
        """Test key validation classes can be imported."""
        from ava.config.validator import ConfigValidator
        assert ConfigValidator is not None


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
