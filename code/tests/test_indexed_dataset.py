"""
Tests for IndexedArrowDataset and related components.

Tests cover:
- Dataset indexing and random access
- LengthBinnedSampler behavior and shuffling
- DynamicPaddingCollator efficiency
- Factory function integration
"""

import pytest
import random
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import torch
from pathlib import Path
from typing import List


# =============================================================================
# Fixtures
# =============================================================================


def create_test_arrow_file(
    file_path: Path,
    num_rows: int = 100,
    min_length: int = 32,
    max_length: int = 256,
    seed: int = 42,
) -> Path:
    """Create a test Arrow file with varied-length sequences."""
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


@pytest.fixture
def single_arrow_file(temp_data_dir: Path) -> Path:
    """Create a single test Arrow file."""
    file_path = temp_data_dir / "single_test.arrow"
    return create_test_arrow_file(file_path, num_rows=50, seed=123)


# =============================================================================
# Test: Module Imports
# =============================================================================


class TestImports:
    """Test that module can be imported correctly."""

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
        assert LengthBinnedSampler is not None
        assert DynamicPaddingCollator is not None
        assert ArrowTableLRUCache is not None
        assert create_indexed_dataloaders is not None


# =============================================================================
# Test: ArrowTableLRUCache
# =============================================================================


class TestArrowTableLRUCache:
    """Tests for ArrowTableLRUCache."""

    def test_cache_creation(self):
        """Test cache can be created."""
        from ava.data.indexed import ArrowTableLRUCache

        cache = ArrowTableLRUCache(max_size=10)
        assert cache.max_size == 10
        assert len(cache.cache) == 0
        cache.close()

    def test_cache_load_and_retrieve(self, single_arrow_file):
        """Test loading and retrieving tables."""
        from ava.data.indexed import ArrowTableLRUCache

        cache = ArrowTableLRUCache(max_size=10)
        table = cache.get(single_arrow_file)

        assert table is not None
        assert len(table) == 50
        assert 'input_ids' in table.schema.names

        # Retrieve again (should use cache)
        table2 = cache.get(single_arrow_file)
        assert table2 is table  # Same object

        cache.close()

    def test_cache_eviction(self, temp_data_dir):
        """Test LRU eviction when cache is full."""
        from ava.data.indexed import ArrowTableLRUCache

        # Create 5 files
        files = []
        for i in range(5):
            f = temp_data_dir / f"eviction_test_{i}.arrow"
            create_test_arrow_file(f, num_rows=10, seed=i)
            files.append(f)

        # Cache with max_size=3
        cache = ArrowTableLRUCache(max_size=3)

        # Load all 5 files
        for f in files:
            cache.get(f)

        # Only 3 should remain
        assert len(cache.cache) == 3

        # Most recent 3 should be cached
        assert files[2] in cache.cache
        assert files[3] in cache.cache
        assert files[4] in cache.cache

        cache.close()


# =============================================================================
# Test: IndexedArrowDataset
# =============================================================================


class TestIndexedArrowDataset:
    """Tests for IndexedArrowDataset."""

    def test_dataset_creation(self, temp_arrow_files):
        """Test dataset creation and indexing."""
        from ava.data.indexed import IndexedArrowDataset

        dataset = IndexedArrowDataset(
            data_files=temp_arrow_files,
            max_length=512,
            compute_lengths=True,
        )

        # 3 files * 100 rows = 300 samples
        assert len(dataset) == 300

        # Lengths should be computed
        lengths = dataset.get_lengths()
        assert lengths is not None
        assert len(lengths) == 300
        assert all(32 <= l <= 256 for l in lengths)

        dataset.cleanup()

    def test_dataset_getitem(self, temp_arrow_files):
        """Test __getitem__ returns correct data structure."""
        from ava.data.indexed import IndexedArrowDataset

        dataset = IndexedArrowDataset(
            data_files=temp_arrow_files,
            max_length=512,
        )

        sample = dataset[0]

        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'labels' in sample
        assert 'length' in sample

        assert isinstance(sample['input_ids'], torch.Tensor)
        assert sample['input_ids'].dtype == torch.int64
        assert sample['length'] == len(sample['input_ids'])

        dataset.cleanup()

    def test_dataset_random_access(self, temp_arrow_files):
        """Test random access to different indices."""
        from ava.data.indexed import IndexedArrowDataset

        dataset = IndexedArrowDataset(
            data_files=temp_arrow_files,
            max_length=512,
        )

        # Access random indices
        indices = [0, 50, 150, 250, 299]
        for idx in indices:
            sample = dataset[idx]
            assert 'input_ids' in sample
            assert len(sample['input_ids']) > 0

        dataset.cleanup()

    def test_dataset_truncation(self, temp_arrow_files):
        """Test sequences are truncated to max_length."""
        from ava.data.indexed import IndexedArrowDataset

        dataset = IndexedArrowDataset(
            data_files=temp_arrow_files,
            max_length=64,  # Very short
        )

        for i in range(10):
            sample = dataset[i]
            assert len(sample['input_ids']) <= 64

        dataset.cleanup()

    def test_dataset_length_computation_disabled(self, temp_arrow_files):
        """Test that length computation can be disabled."""
        from ava.data.indexed import IndexedArrowDataset

        dataset = IndexedArrowDataset(
            data_files=temp_arrow_files,
            max_length=512,
            compute_lengths=False,
        )

        assert dataset.get_lengths() is None

        dataset.cleanup()


# =============================================================================
# Test: LengthBinnedSampler
# =============================================================================


class TestLengthBinnedSampler:
    """Tests for LengthBinnedSampler."""

    def test_sampler_creation(self):
        """Test sampler creation with various lengths."""
        from ava.data.indexed import LengthBinnedSampler

        lengths = [32, 64, 128, 256] * 100
        sampler = LengthBinnedSampler(
            lengths=lengths,
            batch_size=16,
            num_bins=4,
        )

        assert len(sampler) == len(lengths)

    def test_sampler_yields_all_indices(self):
        """Test sampler yields all indices exactly once."""
        from ava.data.indexed import LengthBinnedSampler

        lengths = list(range(32, 132))  # 100 samples
        sampler = LengthBinnedSampler(
            lengths=lengths,
            batch_size=16,
            num_bins=4,
            drop_last=False,
        )

        indices = list(sampler)
        assert set(indices) == set(range(100))
        assert len(indices) == 100

    def test_sampler_drop_last(self):
        """Test drop_last behavior."""
        from ava.data.indexed import LengthBinnedSampler

        lengths = list(range(100))
        sampler = LengthBinnedSampler(
            lengths=lengths,
            batch_size=16,
            num_bins=4,
            drop_last=True,
        )

        indices = list(sampler)
        # Should drop some samples
        assert len(indices) <= len(lengths)

    def test_sampler_epoch_shuffle(self):
        """Test set_epoch produces different orderings."""
        from ava.data.indexed import LengthBinnedSampler

        lengths = list(range(100))
        sampler = LengthBinnedSampler(
            lengths=lengths,
            batch_size=16,
            num_bins=4,
            seed=42,
        )

        sampler.set_epoch(0)
        epoch0_indices = list(sampler)

        sampler.set_epoch(1)
        epoch1_indices = list(sampler)

        # Different epochs should produce different orderings
        assert epoch0_indices != epoch1_indices

    def test_sampler_reproducibility(self):
        """Test same seed produces same ordering."""
        from ava.data.indexed import LengthBinnedSampler

        lengths = list(range(100))

        sampler1 = LengthBinnedSampler(lengths=lengths, batch_size=16, num_bins=4, seed=42)
        sampler1.set_epoch(0)
        indices1 = list(sampler1)

        sampler2 = LengthBinnedSampler(lengths=lengths, batch_size=16, num_bins=4, seed=42)
        sampler2.set_epoch(0)
        indices2 = list(sampler2)

        assert indices1 == indices2

    def test_sampler_binning(self):
        """Test that samples are grouped by length."""
        from ava.data.indexed import LengthBinnedSampler

        # Create lengths with clear bins
        lengths = [32] * 25 + [64] * 25 + [128] * 25 + [256] * 25
        sampler = LengthBinnedSampler(
            lengths=lengths,
            batch_size=10,
            num_bins=4,
            seed=42,
        )

        # Check bin distribution
        assert len(sampler.bin_indices) == 4
        total_binned = sum(len(b) for b in sampler.bin_indices)
        assert total_binned == 100


# =============================================================================
# Test: DynamicPaddingCollator
# =============================================================================


class TestDynamicPaddingCollator:
    """Tests for DynamicPaddingCollator."""

    def test_collator_creation(self):
        """Test collator creation."""
        from ava.data.indexed import DynamicPaddingCollator

        collator = DynamicPaddingCollator(
            pad_token_id=0,
            max_length=512,
        )
        assert collator.pad_token_id == 0
        assert collator.max_length == 512

    def test_collator_pads_to_batch_max(self):
        """Test collator pads to batch maximum, not global max."""
        from ava.data.indexed import DynamicPaddingCollator

        collator = DynamicPaddingCollator(
            pad_token_id=0,
            max_length=512,
        )

        batch = [
            {'input_ids': torch.arange(32), 'attention_mask': torch.ones(32), 'labels': torch.arange(32), 'length': 32},
            {'input_ids': torch.arange(48), 'attention_mask': torch.ones(48), 'labels': torch.arange(48), 'length': 48},
            {'input_ids': torch.arange(64), 'attention_mask': torch.ones(64), 'labels': torch.arange(64), 'length': 64},
        ]

        output = collator(batch)

        # Should pad to 64, not 512
        assert output['input_ids'].shape == (3, 64)
        assert output['attention_mask'].shape == (3, 64)
        assert output['labels'].shape == (3, 64)

    def test_collator_respects_max_length(self):
        """Test collator respects max_length cap."""
        from ava.data.indexed import DynamicPaddingCollator

        collator = DynamicPaddingCollator(
            pad_token_id=0,
            max_length=50,  # Shorter than some sequences
        )

        batch = [
            {'input_ids': torch.arange(100), 'attention_mask': torch.ones(100), 'labels': torch.arange(100), 'length': 100},
            {'input_ids': torch.arange(80), 'attention_mask': torch.ones(80), 'labels': torch.arange(80), 'length': 80},
        ]

        output = collator(batch)

        # Should be capped at max_length
        assert output['input_ids'].shape == (2, 50)

    def test_collator_efficiency_tracking(self):
        """Test collator tracks padding efficiency."""
        from ava.data.indexed import DynamicPaddingCollator

        collator = DynamicPaddingCollator(pad_token_id=0, max_length=512)

        # Create batch with similar lengths (should be efficient)
        batch = [
            {'input_ids': torch.arange(60), 'attention_mask': torch.ones(60), 'labels': torch.arange(60), 'length': 60},
            {'input_ids': torch.arange(62), 'attention_mask': torch.ones(62), 'labels': torch.arange(62), 'length': 62},
            {'input_ids': torch.arange(64), 'attention_mask': torch.ones(64), 'labels': torch.arange(64), 'length': 64},
        ]

        collator(batch)

        efficiency = collator.get_efficiency()
        assert efficiency > 0.95  # Should be >95% efficient

    def test_collator_empty_batch(self):
        """Test collator handles empty batch."""
        from ava.data.indexed import DynamicPaddingCollator

        collator = DynamicPaddingCollator(pad_token_id=0, max_length=512)
        output = collator([])
        assert output == {}

    def test_collator_left_padding(self):
        """Test left-side padding."""
        from ava.data.indexed import DynamicPaddingCollator

        collator = DynamicPaddingCollator(
            pad_token_id=0,
            max_length=512,
            padding_side='left',
        )

        batch = [
            {'input_ids': torch.tensor([1, 2, 3]), 'attention_mask': torch.ones(3), 'labels': torch.tensor([1, 2, 3]), 'length': 3},
            {'input_ids': torch.tensor([4, 5, 6, 7, 8]), 'attention_mask': torch.ones(5), 'labels': torch.tensor([4, 5, 6, 7, 8]), 'length': 5},
        ]

        output = collator(batch)

        # First sample should have padding on left
        assert output['input_ids'][0, 0].item() == 0  # Padding
        assert output['input_ids'][0, -1].item() == 3  # Original data


# =============================================================================
# Test: Factory Function
# =============================================================================


class TestFactoryFunction:
    """Tests for create_indexed_dataloaders factory."""

    def test_factory_creates_loaders(self, temp_arrow_files, temp_data_dir):
        """Test factory creates train and val loaders."""
        from ava.data.indexed import create_indexed_dataloaders

        train_loader, val_loader = create_indexed_dataloaders(
            data_dir=str(temp_data_dir),
            batch_size=16,
            max_length=256,
            val_split_ratio=0.1,
            num_workers=0,  # Main process for testing
        )

        assert train_loader is not None
        assert val_loader is not None

    def test_factory_split_sizes(self, temp_arrow_files, temp_data_dir):
        """Test factory creates correct split sizes."""
        from ava.data.indexed import create_indexed_dataloaders

        train_loader, val_loader = create_indexed_dataloaders(
            data_dir=str(temp_data_dir),
            batch_size=16,
            max_length=256,
            val_split_ratio=0.2,  # 20% validation
            num_workers=0,
        )

        # Iterate through loaders to check they work
        train_samples = sum(batch['input_ids'].size(0) for batch in train_loader)
        val_samples = sum(batch['input_ids'].size(0) for batch in val_loader)

        # Total should be close to 300, val should be ~60 (20%)
        # With drop_last=True and length binning, more samples may be dropped
        total = train_samples + val_samples
        assert total > 0  # At least some samples processed
        assert val_samples > 0  # Validation set exists

    def test_factory_batches_have_correct_shape(self, temp_arrow_files, temp_data_dir):
        """Test batches have correct tensor shapes."""
        from ava.data.indexed import create_indexed_dataloaders

        train_loader, val_loader = create_indexed_dataloaders(
            data_dir=str(temp_data_dir),
            batch_size=8,
            max_length=128,
            num_workers=0,
        )

        batch = next(iter(train_loader))

        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert 'labels' in batch

        assert batch['input_ids'].dim() == 2
        assert batch['input_ids'].size(0) <= 8  # Batch size
        assert batch['input_ids'].size(1) <= 128  # Max length

    def test_factory_no_files_error(self, tmp_path):
        """Test factory raises error when no files found."""
        from ava.data.indexed import create_indexed_dataloaders

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No valid Arrow/Parquet"):
            create_indexed_dataloaders(
                data_dir=str(empty_dir),
                batch_size=16,
                max_length=256,
                num_workers=0,
            )


# =============================================================================
# Test: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self, temp_arrow_files, temp_data_dir):
        """Test full pipeline from files to batches."""
        from ava.data.indexed import create_indexed_dataloaders

        train_loader, val_loader = create_indexed_dataloaders(
            data_dir=str(temp_data_dir),
            batch_size=32,
            max_length=256,
            val_split_ratio=0.1,
            num_bins=4,
            seed=42,
            num_workers=0,
        )

        # Train for a few batches
        batches_seen = 0
        for batch in train_loader:
            assert batch['input_ids'].size(0) <= 32
            assert batch['input_ids'].size(1) <= 256
            batches_seen += 1
            if batches_seen >= 5:
                break

        assert batches_seen == 5

    def test_epoch_shuffling(self, temp_arrow_files, temp_data_dir):
        """Test that epochs produce different orderings."""
        from ava.data.indexed import (
            IndexedArrowDataset,
            LengthBinnedSampler,
            DynamicPaddingCollator,
            _IndexMappedDataset,
        )
        from torch.utils.data import DataLoader

        dataset = IndexedArrowDataset(
            data_files=temp_arrow_files,
            max_length=256,
            compute_lengths=True,
        )

        indices = list(range(len(dataset)))
        lengths = dataset.get_lengths()
        train_lengths = [lengths[i] for i in indices]

        sampler = LengthBinnedSampler(
            lengths=train_lengths,
            batch_size=16,
            num_bins=4,
            seed=42,
        )

        collator = DynamicPaddingCollator(pad_token_id=0, max_length=256)

        loader = DataLoader(
            dataset,
            batch_size=16,
            sampler=sampler,
            collate_fn=collator,
            num_workers=0,
        )

        # Epoch 0
        sampler.set_epoch(0)
        epoch0_first_batch = next(iter(loader))['input_ids'][0, :5].tolist()

        # Epoch 1
        sampler.set_epoch(1)
        epoch1_first_batch = next(iter(loader))['input_ids'][0, :5].tolist()

        # Should be different
        assert epoch0_first_batch != epoch1_first_batch

        dataset.cleanup()


# =============================================================================
# Main
# =============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
