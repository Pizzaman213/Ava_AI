"""
Tests for configuration backward compatibility.

These tests ensure that:
1. Old config paths still resolve correctly
2. Deprecation warnings are emitted for old paths
3. New paths work as expected
4. Mixed old/new configs work together
"""

import os
import sys
import warnings
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ava.config.training_config import DynamicConfig
from ava.config.validator import ConfigValidator
from ava.config.path_mapping import (
    CONFIG_PATH_MAPPINGS,
    is_deprecated_path,
    get_new_path,
    warn_deprecated_path,
    clear_warning_cache,
)


class TestPathMapping:
    """Tests for the path mapping module."""

    def test_is_deprecated_path_returns_true_for_old_paths(self):
        """Known deprecated paths should be detected."""
        assert is_deprecated_path('hardware')
        assert is_deprecated_path('hardware.device')
        assert is_deprecated_path('performance')
        assert is_deprecated_path('wandb')
        assert is_deprecated_path('progressive')

    def test_is_deprecated_path_returns_false_for_new_paths(self):
        """New canonical paths should not be deprecated."""
        assert not is_deprecated_path('compute')
        assert not is_deprecated_path('compute.device.type')
        assert not is_deprecated_path('logging.wandb')
        assert not is_deprecated_path('experimental.progressive')

    def test_is_deprecated_path_returns_false_for_unknown_paths(self):
        """Unknown paths should not be deprecated."""
        assert not is_deprecated_path('unknown_section')
        assert not is_deprecated_path('model.hidden_size')  # Not deprecated

    def test_get_new_path_returns_correct_mapping(self):
        """Old paths should map to correct new paths."""
        assert get_new_path('hardware') == 'compute.device'
        assert get_new_path('hardware.device') == 'compute.device.type'
        assert get_new_path('performance') == 'compute.performance'
        assert get_new_path('wandb') == 'logging.wandb'
        assert get_new_path('progressive') == 'experimental.progressive'

    def test_get_new_path_returns_none_for_unknown(self):
        """Unknown paths should return None."""
        assert get_new_path('unknown_section') is None
        assert get_new_path('model.hidden_size') is None

    def test_warn_deprecated_path_emits_warning(self):
        """Deprecated paths should emit warnings."""
        clear_warning_cache()
        os.environ.pop('SUPPRESS_CONFIG_DEPRECATION', None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated_path('hardware', 'compute.device')
            assert len(w) == 1
            assert 'deprecated' in str(w[0].message).lower()
            assert 'hardware' in str(w[0].message)
            assert 'compute.device' in str(w[0].message)

    def test_warn_deprecated_path_warns_only_once(self):
        """Each path should only warn once per session."""
        clear_warning_cache()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_deprecated_path('performance', 'compute.performance')
            warn_deprecated_path('performance', 'compute.performance')
            warn_deprecated_path('performance', 'compute.performance')
            # Should only have 1 warning
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 1


class TestDynamicConfigCompatibility:
    """Tests for DynamicConfig backward compatibility."""

    def test_new_paths_work_directly(self):
        """New canonical paths should work."""
        config = DynamicConfig({
            'compute': {
                'device': {
                    'type': 'cuda',
                    'num_gpus': 4
                }
            }
        })
        assert config.compute.device.type == 'cuda'
        assert config.compute.device.num_gpus == 4

    def test_model_section_unchanged(self):
        """Model section should work as before."""
        config = DynamicConfig({
            'model': {
                'vocab_size': 50000,
                'hidden_size': 1024,
                'num_layers': 12
            }
        })
        assert config.model.vocab_size == 50000
        assert config.model.hidden_size == 1024
        assert config.model.num_layers == 12

    def test_training_section_unchanged(self):
        """Training section should work as before."""
        config = DynamicConfig({
            'training': {
                'precision': {
                    'mixed_precision': 'bf16'
                },
                'optimizer': {
                    'learning_rate': 0.0001
                }
            }
        })
        assert config.training.precision.mixed_precision == 'bf16'
        assert config.training.optimizer.learning_rate == 0.0001

    def test_data_section_unchanged(self):
        """Data section should work as before."""
        config = DynamicConfig({
            'data': {
                'data_dir': '/path/to/data',
                'max_length': 512
            }
        })
        assert config.data.data_dir == '/path/to/data'
        assert config.data.max_length == 512

    def test_missing_keys_return_none(self):
        """Missing keys should return None (default behavior)."""
        config = DynamicConfig({'model': {'hidden_size': 1024}})
        assert config.nonexistent is None
        assert config.model.nonexistent is None

    def test_to_dict_excludes_internal_attributes(self):
        """to_dict should not include internal attributes."""
        config = DynamicConfig({
            'model': {'hidden_size': 1024}
        })
        d = config.to_dict()
        assert '_config_name' not in d
        assert '_raw_data' not in d
        assert 'model' in d


class TestConfigValidatorCompatibility:
    """Tests for ConfigValidator backward compatibility."""

    def test_new_paths_work(self):
        """New canonical paths should work."""
        config = {
            'compute': {
                'device': {'type': 'cuda'},
                'precision': {'mixed_precision': 'bf16'}
            }
        }
        validator = ConfigValidator(config)
        assert validator.get('compute.device.type') == 'cuda'
        assert validator.get('compute.precision.mixed_precision') == 'bf16'

    def test_model_paths_work(self):
        """Model paths should work (not deprecated)."""
        config = {
            'model': {
                'vocab_size': 50000,
                'hidden_size': 1024
            }
        }
        validator = ConfigValidator(config)
        assert validator.get('model.vocab_size') == 50000
        assert validator.get('model.hidden_size') == 1024

    def test_resolve_path_method(self):
        """resolve_path should map old paths to new paths."""
        validator = ConfigValidator({})

        # Deprecated paths should resolve
        assert validator.resolve_path('hardware') == 'compute.device'
        assert validator.resolve_path('hardware.device') == 'compute.device.type'

        # Non-deprecated paths should stay the same
        assert validator.resolve_path('model.hidden_size') == 'model.hidden_size'
        assert validator.resolve_path('training.optimizer') == 'training.optimizer'

    def test_is_deprecated_path_method(self):
        """is_deprecated_path should detect deprecated paths."""
        validator = ConfigValidator({})

        assert validator.is_deprecated_path('hardware')
        assert validator.is_deprecated_path('performance')
        assert not validator.is_deprecated_path('model')
        assert not validator.is_deprecated_path('training')

    def test_required_fields_validation(self):
        """Required fields should be validated."""
        config = {
            'model': {
                'vocab_size': 50000,
                'hidden_size': 1024
            }
        }
        validator = ConfigValidator(config)
        is_valid, errors = validator.validate(raise_on_error=False)
        assert is_valid

    def test_missing_required_fields(self):
        """Missing required fields should fail validation."""
        config = {'model': {}}  # Missing vocab_size, hidden_size
        validator = ConfigValidator(config, strict=True)
        is_valid, errors = validator.validate(raise_on_error=False)
        assert not is_valid
        assert len(errors) >= 2


class TestMixedConfigs:
    """Tests for configs that mix old and new paths."""

    def test_mixed_config_loads(self):
        """Config with both old and new paths should load."""
        config = DynamicConfig({
            # New structure
            'model': {'hidden_size': 1024},
            'training': {'optimizer': {'learning_rate': 0.001}},
            # Old structure (deprecated)
            'hardware': {'device': 'cuda'},
            'wandb': {'enabled': True}
        })

        # New paths should work
        assert config.model.hidden_size == 1024
        assert config.training.optimizer.learning_rate == 0.001

        # Old paths should still exist in the config
        assert config.hardware.device == 'cuda'
        assert config.wandb.enabled == True


class TestExistingConfigsLoad:
    """Tests that existing config files load without errors."""

    @pytest.fixture
    def config_dir(self):
        return Path(__file__).parent.parent / 'configs' / 'moe'

    def test_minimal_working_loads(self, config_dir):
        """minimal_working.yaml should load."""
        config_path = config_dir / 'minimal_working.yaml'
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            config = DynamicConfig(config_dict)
            assert config.model is not None

    def test_large_loads(self, config_dir):
        """large.yaml should load."""
        config_path = config_dir / 'large.yaml'
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            config = DynamicConfig(config_dict)
            assert config.model is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
