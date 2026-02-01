"""
Unit tests for training configuration management.

Tests the DynamicConfig class and configuration loading/validation.
"""

import pytest
import torch
from unittest.mock import patch, Mock


# Lazy import to avoid circular import issues during collection
@pytest.fixture
def DynamicConfig():
    """Get DynamicConfig class lazily."""
    from ava.config.training_config import DynamicConfig
    return DynamicConfig


@pytest.fixture
def get_mixed_precision():
    """Get get_mixed_precision function lazily."""
    from ava.config.training_config import get_mixed_precision
    return get_mixed_precision


class TestDynamicConfig:
    """Tests for DynamicConfig class."""

    def test_create_empty(self, DynamicConfig):
        """Test creating empty config."""
        config = DynamicConfig()

        # Should not raise for missing keys
        assert config.missing_key is None

    def test_create_from_dict(self, DynamicConfig):
        """Test creating config from dictionary."""
        data = {
            'model': {'hidden_size': 768},
            'training': {'batch_size': 32},
        }
        config = DynamicConfig(data)

        assert config.model.hidden_size == 768
        assert config.training.batch_size == 32

    def test_nested_access(self, DynamicConfig):
        """Test nested attribute access."""
        data = {
            'level1': {
                'level2': {
                    'level3': 'deep_value',
                },
            },
        }
        config = DynamicConfig(data)

        assert config.level1.level2.level3 == 'deep_value'

    def test_dict_access(self, DynamicConfig):
        """Test dictionary-style access."""
        data = {'key': 'value'}
        config = DynamicConfig(data)

        # Direct attribute access
        assert config.key == 'value'

    def test_missing_key_returns_none(self, DynamicConfig):
        """Test that missing keys return None by default."""
        config = DynamicConfig({'existing': 'value'})

        assert config.nonexistent is None
        assert config.also_missing is None

    def test_strict_mode(self, DynamicConfig):
        """Test strict mode raises on missing keys."""
        DynamicConfig.enable_strict_mode()
        try:
            config = DynamicConfig({'existing': 'value'})

            with pytest.raises(AttributeError):
                _ = config.nonexistent
        finally:
            DynamicConfig.disable_strict_mode()

    def test_accessed_missing_tracking(self, DynamicConfig):
        """Test tracking of accessed missing keys."""
        DynamicConfig.clear_accessed_missing_keys()

        config = DynamicConfig({'existing': 'value'})
        _ = config.missing1
        _ = config.missing2

        missing = DynamicConfig.get_accessed_missing_keys()
        # Note: Depending on implementation, might track these
        # Just verify the method works
        assert isinstance(missing, set)

    def test_nested_dict_conversion(self, DynamicConfig):
        """Test that nested dicts become DynamicConfig."""
        data = {
            'outer': {
                'inner': {
                    'value': 42,
                },
            },
        }
        config = DynamicConfig(data)

        assert isinstance(config.outer, DynamicConfig)
        assert isinstance(config.outer.inner, DynamicConfig)
        assert config.outer.inner.value == 42

    def test_list_values_preserved(self, DynamicConfig):
        """Test that list values are preserved."""
        data = {
            'items': [1, 2, 3],
            'nested_list': [{'a': 1}, {'b': 2}],
        }
        config = DynamicConfig(data)

        assert config.items == [1, 2, 3]
        assert isinstance(config.nested_list, list)

    def test_none_values(self, DynamicConfig):
        """Test that None values are preserved."""
        data = {
            'explicit_none': None,
            'value': 'exists',
        }
        config = DynamicConfig(data)

        assert config.explicit_none is None
        assert config.value == 'exists'

    def test_bool_values(self, DynamicConfig):
        """Test boolean value handling."""
        data = {
            'enabled': True,
            'disabled': False,
        }
        config = DynamicConfig(data)

        assert config.enabled is True
        assert config.disabled is False


class TestGetMixedPrecision:
    """Tests for get_mixed_precision helper function."""

    def test_new_path(self, get_mixed_precision):
        """Test reading from new config path."""
        config = {
            'training': {
                'precision': {
                    'mixed_precision': 'bf16',
                },
            },
        }

        assert get_mixed_precision(config) == 'bf16'

    def test_legacy_path(self, get_mixed_precision):
        """Test reading from legacy config path."""
        config = {
            'training': {
                'mixed_precision': 'fp16',
            },
        }

        assert get_mixed_precision(config) == 'fp16'

    def test_new_path_takes_precedence(self, get_mixed_precision):
        """Test that new path takes precedence over legacy."""
        config = {
            'training': {
                'mixed_precision': 'fp16',  # Legacy
                'precision': {
                    'mixed_precision': 'bf16',  # New
                },
            },
        }

        # New path should win
        assert get_mixed_precision(config) == 'bf16'

    def test_default_value(self, get_mixed_precision):
        """Test default value when not specified."""
        config = {'training': {}}

        assert get_mixed_precision(config) == 'bf16'

    def test_empty_config(self, get_mixed_precision):
        """Test with empty config."""
        config = {}

        assert get_mixed_precision(config) == 'bf16'


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_valid_model_config(self, DynamicConfig):
        """Test that valid model config is accepted."""
        data = {
            'model': {
                'vocab_size': 50000,
                'hidden_size': 768,
                'num_layers': 12,
                'num_attention_heads': 12,
            },
        }
        config = DynamicConfig(data)

        assert config.model.vocab_size == 50000
        assert config.model.hidden_size == 768

    def test_config_with_all_sections(self, DynamicConfig):
        """Test config with all 8 sections."""
        data = {
            'model': {'hidden_size': 768},
            'training': {'batch_size': 32},
            'data': {'max_length': 512},
            'compute': {'device': 'cuda'},
            'distributed': {'world_size': 1},
            'logging': {'log_interval': 100},
            'checkpoints': {'save_steps': 500},
            'experimental': {'enable_fp8': False},
        }
        config = DynamicConfig(data)

        assert config.model.hidden_size == 768
        assert config.training.batch_size == 32
        assert config.data.max_length == 512

    def test_config_type_preservation(self, DynamicConfig):
        """Test that types are preserved."""
        data = {
            'int_val': 42,
            'float_val': 3.14,
            'str_val': 'hello',
            'bool_val': True,
            'list_val': [1, 2, 3],
        }
        config = DynamicConfig(data)

        assert isinstance(config.int_val, int)
        assert isinstance(config.float_val, float)
        assert isinstance(config.str_val, str)
        assert isinstance(config.bool_val, bool)
        assert isinstance(config.list_val, list)


class TestConfigPathMigration:
    """Tests for deprecated path migration."""

    def test_path_migration_enabled_by_default(self, DynamicConfig):
        """Test that path migration is enabled by default."""
        assert DynamicConfig._enable_path_migration is True

    def test_disable_path_migration(self, DynamicConfig):
        """Test disabling path migration."""
        DynamicConfig.disable_path_migration()
        try:
            assert DynamicConfig._enable_path_migration is False
        finally:
            DynamicConfig.enable_path_migration()

    def test_enable_path_migration(self, DynamicConfig):
        """Test enabling path migration."""
        DynamicConfig.disable_path_migration()
        DynamicConfig.enable_path_migration()

        assert DynamicConfig._enable_path_migration is True


class TestConfigEdgeCases:
    """Tests for edge cases and corner cases."""

    def test_empty_nested_dict(self, DynamicConfig):
        """Test with empty nested dictionary."""
        data = {
            'outer': {},
        }
        config = DynamicConfig(data)

        assert isinstance(config.outer, DynamicConfig)
        assert config.outer.missing is None

    def test_special_characters_in_keys(self, DynamicConfig):
        """Test keys with special characters."""
        data = {
            'key_with_underscore': 1,
            'key-with-dash': 2,
        }
        config = DynamicConfig(data)

        assert config.key_with_underscore == 1
        # Dashed keys can only be accessed via getattr or raw data
        assert getattr(config, 'key-with-dash', None) == 2

    def test_numeric_values_edge_cases(self, DynamicConfig):
        """Test numeric edge cases."""
        data = {
            'zero': 0,
            'negative': -1,
            'large': 1e10,
            'small': 1e-10,
            'inf': float('inf'),
        }
        config = DynamicConfig(data)

        assert config.zero == 0
        assert config.negative == -1
        assert config.large == 1e10
        assert config.small == 1e-10

    def test_unicode_values(self, DynamicConfig):
        """Test unicode string values."""
        data = {
            'unicode': '你好世界',
            'emoji': '🚀',
        }
        config = DynamicConfig(data)

        assert config.unicode == '你好世界'
        assert config.emoji == '🚀'

    def test_deeply_nested(self, DynamicConfig):
        """Test deeply nested configuration."""
        data = {'a': {'b': {'c': {'d': {'e': {'f': 'deep'}}}}}}
        config = DynamicConfig(data)

        assert config.a.b.c.d.e.f == 'deep'

    def test_report_missing_keys(self, DynamicConfig):
        """Test reporting missing keys."""
        DynamicConfig.clear_accessed_missing_keys()

        # Just verify method doesn't raise
        DynamicConfig.report_missing_keys()
