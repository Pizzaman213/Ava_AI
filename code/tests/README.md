# Advanced LLM Training Test Suite

This comprehensive test suite validates all advanced training optimizations including optimizers, progressive training, schedulers, FP8 support, and their integrations.

## Test Structure

### Core Test Modules

1. **`test_advanced_optimizers.py`** - Unit tests for Lion, Sophia, and AdaFactor optimizers
   - Parameter update validation
   - Memory efficiency testing
   - Convergence analysis
   - Factory pattern testing

2. **`test_progressive_training.py`** - Tests for progressive training framework
   - Curriculum learning validation
   - GrowLength training progression
   - Dynamic batch sizing adaptation
   - Orchestrator integration

3. **`test_advanced_schedulers.py`** - Tests for enhanced learning rate schedulers
   - Cosine Annealing with Restarts
   - OneCycle scheduler validation
   - Adaptive LR scheduler testing
   - Warmup integration

4. **`test_fp8_training.py`** - Tests for FP8 training support
   - FP8 layer implementations
   - Scaling factor management
   - H100/B200 GPU optimizations
   - Transformer Engine integration

5. **`test_integration.py`** - End-to-end integration tests
   - Optimizer + scheduler combinations
   - Progressive training workflows
   - Full system integration
   - Error handling validation

6. **`test_benchmarks.py`** - Performance benchmarking suite
   - Throughput measurements
   - Memory efficiency analysis
   - Convergence speed comparison
   - System-level performance

7. **`test_stress_edge_cases.py`** - Stress testing and edge case validation
   - Extreme parameter configurations
   - Memory pressure scenarios
   - Error condition handling
   - Long-running stability

### Testing Infrastructure

- **`__init__.py`** - Test configuration and utilities
- **`test_runner.py`** - Automated test runner with reporting
- **`../run_tests.py`** - Simple test execution script

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py

# Run quick test suite (skip benchmarks)
python run_tests.py --quick

# Run specific test suite
python run_tests.py --suite optimizers
```

### Advanced Usage

```bash
# Run with custom output directory
python run_tests.py --output-dir ./my_test_reports

# Run with verbose output
python run_tests.py --verbose

# Run specific test modules
python -m pytest tests/test_advanced_optimizers.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Using the Test Runner Directly

```bash
# Run all tests with detailed reporting
python -m tests.test_runner

# Run specific suites
python -m tests.test_runner --suites optimizers schedulers

# Quick test run
python -m tests.test_runner --quick

# Custom output directory
python -m tests.test_runner --output-dir /path/to/reports
```

## Test Categories

### Unit Tests
- Individual component validation
- Parameter boundary testing
- Function correctness verification
- Error condition handling

### Integration Tests
- Component interaction validation
- End-to-end workflow testing
- Configuration compatibility
- State management verification

### Performance Tests
- Throughput benchmarking
- Memory usage analysis
- Convergence speed measurement
- Scalability assessment

### Stress Tests
- Extreme parameter validation
- Memory pressure testing
- Long-running stability
- Error recovery verification

## Test Configuration

The test suite uses configuration from `tests/__init__.py`:

```python
TEST_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "precision": torch.float32,
    "seed": 42,
    "temp_dir": "/tmp/llm_tests",
    "max_test_time": 3600,
    "memory_threshold_gb": 8.0,
}
```

### Environment Variables

- `CUDA_VISIBLE_DEVICES` - Control GPU usage
- `PYTORCH_CUDA_ALLOC_CONF` - CUDA memory management
- `OMP_NUM_THREADS` - CPU parallelism control

## Test Reports

The test runner generates comprehensive reports:

### HTML Report (`test_report.html`)
- Interactive web-based report
- Test suite summaries
- Individual test results
- System information
- Performance metrics

### JSON Report (`test_report.json`)
- Machine-readable test data
- Complete test results
- Timing information
- Error details

### Text Summary (`test_summary.txt`)
- Quick overview of results
- Failed test summaries
- System configuration
- Success rates

### Performance Analysis (`performance_analysis.json`)
- Benchmark results
- Performance trends
- Resource utilization
- Optimization recommendations

## Writing New Tests

### Test Structure Template

```python
import unittest
import torch
import torch.nn as nn
from tests import TEST_CONFIG

class TestNewFeature(unittest.TestCase):
    """Test new feature implementation"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        # Setup test fixtures

    def test_basic_functionality(self):
        """Test basic feature operation"""
        # Test implementation
        pass

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Edge case testing
        pass

    def tearDown(self):
        # Cleanup if needed
        pass
```

### Best Practices

1. **Use descriptive test names** - Clearly indicate what is being tested
2. **Test both success and failure cases** - Include error condition validation
3. **Use subtests for parameter variations** - Test multiple configurations efficiently
4. **Clean up resources** - Prevent memory leaks and resource conflicts
5. **Mock external dependencies** - Isolate components under test
6. **Use appropriate assertions** - Choose the most specific assertion method

### Adding Benchmarks

```python
def benchmark_new_feature(self):
    """Benchmark new feature performance"""
    timer = BenchmarkTimer()

    with timer:
        # Feature operation
        pass

    # Validate performance
    self.assertLess(timer.elapsed, expected_time)
```

## Continuous Integration

### GitHub Actions Integration

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python run_tests.py --quick
      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: test-reports
          path: test_reports/
```

### Local Pre-commit Hook

```bash
#!/bin/sh
# .git/hooks/pre-commit
python run_tests.py --quick
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch sizes in test configuration
   - Skip GPU-intensive tests: `--suites optimizers schedulers`
   - Set `CUDA_VISIBLE_DEVICES=""` to force CPU testing

2. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify project structure

3. **Test Timeouts**
   - Increase `max_test_time` in configuration
   - Skip benchmark tests with `--quick`
   - Use faster test variants

### Debug Mode

```bash
# Run with Python debugger
python -m pdb -m tests.test_runner

# Verbose pytest output
python -m pytest tests/ -v -s

# Specific test debugging
python -m pytest tests/test_advanced_optimizers.py::TestLionOptimizer::test_lion_convergence -v -s
```

## Contributing

### Adding New Test Suites

1. Create new test file: `tests/test_new_feature.py`
2. Add module to `test_runner.py` test_modules list
3. Update this README with test description
4. Ensure proper test coverage and documentation

### Test Review Checklist

- [ ] Tests cover all public APIs
- [ ] Edge cases and error conditions tested
- [ ] Performance implications validated
- [ ] Documentation updated
- [ ] CI integration verified
- [ ] Cleanup properly implemented

## Performance Expectations

### Benchmark Targets

- **Optimizer step time**: < 5ms for medium models
- **Scheduler overhead**: < 100μs per step
- **Progressive training overhead**: < 20%
- **Memory efficiency**: AdaFactor < 70% of Adam memory
- **FP8 speedup**: 1.2x over FP16 (when supported)

### Test Execution Times

- **Quick suite**: ~5-10 minutes
- **Full suite**: ~30-60 minutes
- **Benchmark suite**: ~15-30 minutes
- **Stress tests**: ~20-45 minutes

## License

This test suite is part of the Advanced LLM Training framework and follows the same licensing terms as the main project.