"""
Performance Benchmarking Suite

Comprehensive benchmarks for all advanced training optimizations including
throughput, memory efficiency, convergence speed, and scalability testing.
"""

import unittest
import torch
import torch.nn as nn
import time
import psutil
import gc
from typing import Dict, List, Tuple, Any
import json
import tempfile
import os

from tests import TEST_CONFIG
from src.Ava.optimization.advanced_optimizers import (
    LionOptimizer, SophiaOptimizer, AdaFactorOptimizer, OptimizerFactory, OptimizerConfig
)
from src.Ava.training.progressive_training import (
    ProgressiveTrainingOrchestrator, ProgressiveConfig,
    CurriculumConfig, GrowLengthConfig, DynamicBatchConfig
)
from src.Ava.training.advanced_schedulers import (
    CosineAnnealingWithRestarts, OneCycleScheduler, AdaptiveLRScheduler
)
from src.Ava.optimization.fp8_training import (
    FP8Config, FP8Optimizer, convert_model_to_fp8
)


class BenchmarkTimer:
    """Utility class for timing operations"""

    def __init__(self):
        self.times = {}
        self.start_time = None

    def start(self, name: str):
        self.start_time = time.perf_counter()
        return self

    def end(self, name: str):
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            self.times[name] = elapsed
            self.start_time = None
            return elapsed
        return 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start_time


class MemoryTracker:
    """Utility class for tracking memory usage"""

    def __init__(self):
        self.peak_memory = 0
        self.initial_memory = 0

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory = torch.cuda.memory_allocated()
        else:
            self.initial_memory = psutil.Process().memory_info().rss

    def peak(self) -> float:
        if torch.cuda.is_available():
            return (torch.cuda.max_memory_allocated() - self.initial_memory) / (1024**3)
        else:
            return (psutil.Process().memory_info().rss - self.initial_memory) / (1024**3)


class TestOptimizerBenchmarks(unittest.TestCase):
    """Benchmark different optimizers"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model_configs = {
            'small': (128, 64, 32, 1),
            'medium': (512, 256, 128, 1),
            'large': (1024, 512, 256, 1)
        }

    def create_model(self, config: Tuple[int, ...]) -> nn.Module:
        """Create model based on configuration"""
        layers = []
        for i in range(len(config) - 1):
            layers.append(nn.Linear(config[i], config[i+1]))
            if i < len(config) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers).to(self.device)

    def benchmark_optimizer_step_time(self):
        """Benchmark optimizer step time"""
        results = {}

        for model_size, config in self.model_configs.items():
            model = self.create_model(config)

            optimizers = {
                'Lion': LionOptimizer(model.parameters(), lr=1e-3),
                'Sophia': SophiaOptimizer(model.parameters(), lr=1e-3),
                'AdaFactor': AdaFactorOptimizer(model.parameters(), relative_step=True),
                'Adam': torch.optim.Adam(model.parameters(), lr=1e-3),
                'AdamW': torch.optim.AdamW(model.parameters(), lr=1e-3)
            }

            results[model_size] = {}

            for opt_name, optimizer in optimizers.items():
                # Warmup
                for _ in range(5):
                    x = torch.randn(32, config[0], device=self.device)
                    y = torch.randn(32, config[-1], device=self.device)

                    optimizer.zero_grad()
                    output = model(x)
                    loss = nn.MSELoss()(output, y)
                    loss.backward()
                    optimizer.step()

                # Benchmark
                timer = BenchmarkTimer()
                num_steps = 50

                with timer:
                    for _ in range(num_steps):
                        x = torch.randn(32, config[0], device=self.device)
                        y = torch.randn(32, config[-1], device=self.device)

                        optimizer.zero_grad()
                        output = model(x)
                        loss = nn.MSELoss()(output, y)
                        loss.backward()
                        optimizer.step()

                avg_step_time = timer.elapsed / num_steps * 1000  # ms
                results[model_size][opt_name] = avg_step_time

        return results

    def benchmark_optimizer_memory_usage(self):
        """Benchmark optimizer memory usage"""
        results = {}

        for model_size, config in self.model_configs.items():
            if not torch.cuda.is_available():
                continue

            results[model_size] = {}

            optimizers_config = [
                ('Lion', lambda p: LionOptimizer(p, lr=1e-3)),
                ('Sophia', lambda p: SophiaOptimizer(p, lr=1e-3)),
                ('AdaFactor', lambda p: AdaFactorOptimizer(p, relative_step=True)),
                ('Adam', lambda p: torch.optim.Adam(p, lr=1e-3)),
                ('AdamW', lambda p: torch.optim.AdamW(p, lr=1e-3))
            ]

            for opt_name, opt_factory in optimizers_config:
                # Clean memory
                torch.cuda.empty_cache()
                gc.collect()

                model = self.create_model(config)
                optimizer = opt_factory(model.parameters())

                memory_tracker = MemoryTracker()
                memory_tracker.start()

                # Initialize optimizer state
                for _ in range(10):
                    x = torch.randn(64, config[0], device=self.device)
                    y = torch.randn(64, config[-1], device=self.device)

                    optimizer.zero_grad()
                    output = model(x)
                    loss = nn.MSELoss()(output, y)
                    loss.backward()
                    optimizer.step()

                peak_memory = memory_tracker.peak()
                results[model_size][opt_name] = peak_memory

                # Cleanup
                del model, optimizer
                torch.cuda.empty_cache()

        return results

    def benchmark_optimizer_convergence(self):
        """Benchmark optimizer convergence speed"""
        results = {}

        # Simple quadratic optimization problem
        def quadratic_loss(x, target=2.0):
            return (x - target) ** 2

        optimizers_config = [
            ('Lion', lambda p: LionOptimizer(p, lr=0.1)),
            ('Sophia', lambda p: SophiaOptimizer(p, lr=0.1)),
            ('AdaFactor', lambda p: AdaFactorOptimizer(p, lr=0.1, relative_step=False)),
            ('Adam', lambda p: torch.optim.Adam(p, lr=0.1)),
            ('SGD', lambda p: torch.optim.SGD(p, lr=0.1))
        ]

        for opt_name, opt_factory in optimizers_config:
            x = torch.tensor([0.0], requires_grad=True, device=self.device)
            optimizer = opt_factory([x])

            losses = []
            steps_to_converge = None

            for step in range(200):
                optimizer.zero_grad()
                loss = quadratic_loss(x)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                # Check convergence (within 1% of optimal)
                if abs(x.item() - 2.0) < 0.02 and steps_to_converge is None:
                    steps_to_converge = step

            results[opt_name] = {
                'steps_to_converge': steps_to_converge or 200,
                'final_loss': losses[-1],
                'convergence_rate': losses
            }

        return results

    def test_optimizer_performance_comparison(self):
        """Compare optimizer performance across metrics"""
        print("\n=== Optimizer Benchmark Results ===")

        # Step time benchmark
        step_times = self.benchmark_optimizer_step_time()
        print("\n1. Average Step Time (ms):")
        for model_size, times in step_times.items():
            print(f"\n{model_size.capitalize()} Model:")
            for opt_name, time_ms in sorted(times.items(), key=lambda x: x[1]):
                print(f"  {opt_name:12}: {time_ms:.2f} ms")

        # Memory usage benchmark
        if torch.cuda.is_available():
            memory_usage = self.benchmark_optimizer_memory_usage()
            print("\n2. Peak Memory Usage (GB):")
            for model_size, memory in memory_usage.items():
                print(f"\n{model_size.capitalize()} Model:")
                for opt_name, mem_gb in sorted(memory.items(), key=lambda x: x[1]):
                    print(f"  {opt_name:12}: {mem_gb:.3f} GB")

        # Convergence benchmark
        convergence = self.benchmark_optimizer_convergence()
        print("\n3. Convergence Performance:")
        for opt_name, metrics in convergence.items():
            print(f"  {opt_name:12}: {metrics['steps_to_converge']:3d} steps, "
                  f"final loss: {metrics['final_loss']:.6f}")

        # All benchmarks should complete
        self.assertTrue(True)


class TestProgressiveTrainingBenchmarks(unittest.TestCase):
    """Benchmark progressive training components"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]

    def benchmark_curriculum_overhead(self):
        """Benchmark curriculum learning overhead"""
        model = nn.Sequential(
            nn.Embedding(1000, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 1000)
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Without curriculum
        timer_baseline = BenchmarkTimer()
        with timer_baseline:
            for step in range(50):
                x = torch.randint(0, 1000, (16, 32), device=self.device)
                y = torch.randint(0, 1000, (16, 32), device=self.device)

                optimizer.zero_grad()
                output = model(x.view(-1, 1)).view(16, 32, -1)
                loss = nn.CrossEntropyLoss()(output.view(-1, 1000), y.view(-1))
                loss.backward()
                optimizer.step()

        baseline_time = timer_baseline.elapsed

        # With curriculum
        progressive_config = ProgressiveConfig(
            curriculum_config=CurriculumConfig(
                strategy="length_based",
                initial_max_length=16,
                final_max_length=64,
                growth_steps=50
            )
        )
        orchestrator = ProgressiveTrainingOrchestrator(progressive_config)

        timer_curriculum = BenchmarkTimer()
        with timer_curriculum:
            for step in range(50):
                max_length = orchestrator.get_current_max_length()

                x = torch.randint(0, 1000, (16, max_length), device=self.device)
                y = torch.randint(0, 1000, (16, max_length), device=self.device)

                optimizer.zero_grad()
                output = model(x.view(-1, 1)).view(16, max_length, -1)
                loss = nn.CrossEntropyLoss()(output.view(-1, 1000), y.view(-1))
                loss.backward()
                optimizer.step()

                orchestrator.update(step=step, loss=loss.item(), gradient_norm=1.0)

        curriculum_time = timer_curriculum.elapsed
        overhead_percent = ((curriculum_time - baseline_time) / baseline_time) * 100

        return {
            'baseline_time': baseline_time,
            'curriculum_time': curriculum_time,
            'overhead_percent': overhead_percent
        }

    def benchmark_dynamic_batch_adaptation(self):
        """Benchmark dynamic batch sizing adaptation speed"""
        model = nn.Linear(256, 1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        progressive_config = ProgressiveConfig(
            dynamic_batch_config=DynamicBatchConfig(
                initial_batch_size=8,
                max_batch_size=64,
                memory_threshold_gb=4.0,
                adaptation_strategy="memory_based"
            ),
            enable_dynamic_batch=True
        )
        orchestrator = ProgressiveTrainingOrchestrator(progressive_config)

        batch_history = []
        adaptation_times = []

        for step in range(100):
            timer = BenchmarkTimer()

            with timer:
                batch_size = orchestrator.get_current_batch_size()

                x = torch.randn(batch_size, 256, device=self.device)
                y = torch.randn(batch_size, 1, device=self.device)

                optimizer.zero_grad()
                output = model(x)
                loss = nn.MSELoss()(output, y)
                loss.backward()
                optimizer.step()

                # Simulate varying memory usage
                memory_usage = batch_size * 0.05 + torch.randn(1).item() * 0.5
                memory_usage = max(0.1, memory_usage)

                orchestrator.update(
                    step=step,
                    loss=loss.item(),
                    memory_usage_gb=memory_usage
                )

            batch_history.append(batch_size)
            adaptation_times.append(timer.elapsed)

        return {
            'avg_adaptation_time': sum(adaptation_times) / len(adaptation_times),
            'batch_adaptations': len(set(batch_history)),
            'batch_range': (min(batch_history), max(batch_history))
        }

    def test_progressive_training_performance(self):
        """Test progressive training performance metrics"""
        print("\n=== Progressive Training Benchmark Results ===")

        # Curriculum overhead
        curriculum_metrics = self.benchmark_curriculum_overhead()
        print(f"\n1. Curriculum Learning Overhead:")
        print(f"  Baseline time: {curriculum_metrics['baseline_time']:.3f}s")
        print(f"  Curriculum time: {curriculum_metrics['curriculum_time']:.3f}s")
        print(f"  Overhead: {curriculum_metrics['overhead_percent']:.1f}%")

        # Dynamic batch adaptation
        batch_metrics = self.benchmark_dynamic_batch_adaptation()
        print(f"\n2. Dynamic Batch Sizing:")
        print(f"  Avg adaptation time: {batch_metrics['avg_adaptation_time']*1000:.2f}ms")
        print(f"  Batch adaptations: {batch_metrics['batch_adaptations']}")
        print(f"  Batch size range: {batch_metrics['batch_range']}")

        # Overhead should be reasonable (< 20%)
        self.assertLess(curriculum_metrics['overhead_percent'], 20.0)

        # Should adapt batch sizes
        self.assertGreater(batch_metrics['batch_adaptations'], 1)


class TestSchedulerBenchmarks(unittest.TestCase):
    """Benchmark learning rate schedulers"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]
        self.model = nn.Linear(64, 1).to(self.device)

    def benchmark_scheduler_step_time(self):
        """Benchmark scheduler step overhead"""
        results = {}

        schedulers = {
            'CosineAnnealing': lambda opt: CosineAnnealingWithRestarts(opt, T_0=20, T_mult=2),
            'OneCycle': lambda opt: OneCycleScheduler(opt, max_lr=0.1, total_steps=100),
            'Adaptive': lambda opt: AdaptiveLRScheduler(opt, patience=5, factor=0.5),
            'StepLR': lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=10),
            'ExponentialLR': lambda opt: torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
        }

        for sched_name, sched_factory in schedulers.items():
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            scheduler = sched_factory(optimizer)

            # Warmup
            for _ in range(10):
                if sched_name == 'Adaptive':
                    scheduler.step(1.0)  # Adaptive needs metric
                else:
                    scheduler.step()

            # Benchmark
            timer = BenchmarkTimer()
            num_steps = 1000

            with timer:
                for step in range(num_steps):
                    if sched_name == 'Adaptive':
                        scheduler.step(1.0 - step * 0.001)  # Decreasing loss
                    else:
                        scheduler.step()

            avg_step_time = timer.elapsed / num_steps * 1000000  # microseconds
            results[sched_name] = avg_step_time

        return results

    def benchmark_scheduler_convergence_impact(self):
        """Benchmark scheduler impact on convergence"""
        results = {}

        # Simple optimization problem
        def create_model_and_data():
            model = nn.Linear(10, 1).to(self.device)
            x = torch.randn(100, 10, device=self.device)
            y = torch.randn(100, 1, device=self.device)
            return model, x, y

        schedulers = {
            'None': lambda opt: None,
            'CosineAnnealing': lambda opt: CosineAnnealingWithRestarts(opt, T_0=20),
            'OneCycle': lambda opt: OneCycleScheduler(opt, max_lr=0.01, total_steps=100),
            'Adaptive': lambda opt: AdaptiveLRScheduler(opt, patience=10, factor=0.5)
        }

        for sched_name, sched_factory in schedulers.items():
            model, x, y = create_model_and_data()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = sched_factory(optimizer)

            losses = []

            for step in range(100):
                optimizer.zero_grad()
                output = model(x)
                loss = nn.MSELoss()(output, y)
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    if sched_name == 'Adaptive':
                        scheduler.step(loss.item())
                    else:
                        scheduler.step()

                losses.append(loss.item())

            results[sched_name] = {
                'final_loss': losses[-1],
                'convergence_speed': sum(1 for i in range(1, len(losses))
                                       if losses[i] < losses[i-1]) / len(losses)
            }

        return results

    def test_scheduler_performance(self):
        """Test scheduler performance metrics"""
        print("\n=== Scheduler Benchmark Results ===")

        # Step time benchmark
        step_times = self.benchmark_scheduler_step_time()
        print("\n1. Scheduler Step Time (μs):")
        for sched_name, time_us in sorted(step_times.items(), key=lambda x: x[1]):
            print(f"  {sched_name:15}: {time_us:.2f} μs")

        # Convergence impact
        convergence = self.benchmark_scheduler_convergence_impact()
        print("\n2. Convergence Impact:")
        for sched_name, metrics in convergence.items():
            print(f"  {sched_name:15}: final_loss={metrics['final_loss']:.6f}, "
                  f"convergence_rate={metrics['convergence_speed']:.3f}")

        # Step times should be reasonable (< 100μs)
        for time_us in step_times.values():
            self.assertLess(time_us, 100.0)


class TestFP8Benchmarks(unittest.TestCase):
    """Benchmark FP8 training performance"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]

    def benchmark_fp8_vs_fp16_throughput(self):
        """Benchmark FP8 vs FP16 training throughput"""
        if not torch.cuda.is_available():
            return {"skipped": "CUDA not available"}

        model_configs = [
            (256, 128, 1),
            (512, 256, 1),
            (1024, 512, 1)
        ]

        results = {}

        for config in model_configs:
            model_name = f"{config[0]}x{config[1]}x{config[2]}"
            results[model_name] = {}

            # FP16 baseline
            model_fp16 = nn.Sequential(
                nn.Linear(config[0], config[1]),
                nn.ReLU(),
                nn.Linear(config[1], config[2])
            ).to(self.device).half()

            optimizer_fp16 = torch.optim.Adam(model_fp16.parameters(), lr=1e-3)

            # Warmup
            for _ in range(5):
                x = torch.randn(32, config[0], device=self.device, dtype=torch.float16)
                y = torch.randn(32, config[2], device=self.device, dtype=torch.float16)

                optimizer_fp16.zero_grad()
                output = model_fp16(x)
                loss = nn.MSELoss()(output, y)
                loss.backward()
                optimizer_fp16.step()

            # Benchmark FP16
            timer = BenchmarkTimer()
            steps = 50

            with timer:
                for _ in range(steps):
                    x = torch.randn(32, config[0], device=self.device, dtype=torch.float16)
                    y = torch.randn(32, config[2], device=self.device, dtype=torch.float16)

                    optimizer_fp16.zero_grad()
                    output = model_fp16(x)
                    loss = nn.MSELoss()(output, y)
                    loss.backward()
                    optimizer_fp16.step()

            fp16_time = timer.elapsed
            fp16_throughput = steps / fp16_time

            results[model_name]['FP16'] = {
                'time': fp16_time,
                'throughput': fp16_throughput
            }

            # FP8 (with fallback behavior)
            fp8_config = FP8Config(enabled=True)

            model_fp8 = nn.Sequential(
                nn.Linear(config[0], config[1]),
                nn.ReLU(),
                nn.Linear(config[1], config[2])
            ).to(self.device)

            # Convert to FP8 (may fall back to FP16/FP32)
            model_fp8 = convert_model_to_fp8(model_fp8, fp8_config)
            optimizer_fp8 = torch.optim.Adam(model_fp8.parameters(), lr=1e-3)

            # Warmup
            for _ in range(5):
                x = torch.randn(32, config[0], device=self.device)
                y = torch.randn(32, config[2], device=self.device)

                optimizer_fp8.zero_grad()
                output = model_fp8(x)
                loss = nn.MSELoss()(output, y)
                loss.backward()
                optimizer_fp8.step()

            # Benchmark FP8
            timer = BenchmarkTimer()

            with timer:
                for _ in range(steps):
                    x = torch.randn(32, config[0], device=self.device)
                    y = torch.randn(32, config[2], device=self.device)

                    optimizer_fp8.zero_grad()
                    output = model_fp8(x)
                    loss = nn.MSELoss()(output, y)
                    loss.backward()
                    optimizer_fp8.step()

            fp8_time = timer.elapsed
            fp8_throughput = steps / fp8_time

            results[model_name]['FP8'] = {
                'time': fp8_time,
                'throughput': fp8_throughput
            }

            # Cleanup
            del model_fp16, model_fp8, optimizer_fp16, optimizer_fp8
            torch.cuda.empty_cache()

        return results

    def test_fp8_performance(self):
        """Test FP8 performance compared to FP16"""
        print("\n=== FP8 Benchmark Results ===")

        throughput_results = self.benchmark_fp8_vs_fp16_throughput()

        if "skipped" in throughput_results:
            print("Skipped: CUDA not available")
            return

        print("\n1. Training Throughput (steps/sec):")
        for model_name, results in throughput_results.items():
            print(f"\n{model_name}:")
            for precision, metrics in results.items():
                print(f"  {precision}: {metrics['throughput']:.2f} steps/sec "
                      f"({metrics['time']:.3f}s total)")

            # Calculate speedup
            if 'FP16' in results and 'FP8' in results:
                speedup = results['FP8']['throughput'] / results['FP16']['throughput']
                print(f"  FP8 speedup: {speedup:.2f}x")


class TestSystemBenchmarks(unittest.TestCase):
    """System-level performance benchmarks"""

    def setUp(self):
        self.device = TEST_CONFIG["device"]

    def benchmark_full_training_pipeline(self):
        """Benchmark complete training pipeline with all optimizations"""
        # Create model
        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)

        # Setup configurations
        configurations = {
            'baseline': {
                'optimizer': torch.optim.Adam(model.parameters(), lr=1e-3),
                'scheduler': None,
                'progressive': None,
                'fp8': False
            },
            'optimized': {
                'optimizer': LionOptimizer(model.parameters(), lr=1e-3),
                'scheduler': OneCycleScheduler(
                    LionOptimizer(model.parameters(), lr=1e-3),
                    max_lr=1e-2, total_steps=100
                ),
                'progressive': ProgressiveTrainingOrchestrator(
                    ProgressiveConfig(
                        dynamic_batch_config=DynamicBatchConfig(
                            initial_batch_size=16,
                            max_batch_size=64
                        ),
                        enable_dynamic_batch=True
                    )
                ),
                'fp8': True
            }
        }

        results = {}

        for config_name, config in configurations.items():
            # Setup model
            test_model = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ).to(self.device)

            if config['fp8']:
                fp8_config = FP8Config(enabled=True)
                test_model = convert_model_to_fp8(test_model, fp8_config)

            optimizer = config['optimizer']
            scheduler = config['scheduler']
            progressive = config['progressive']

            # Memory tracking
            memory_tracker = MemoryTracker()
            memory_tracker.start()

            # Training loop
            timer = BenchmarkTimer()
            losses = []

            with timer:
                for step in range(100):
                    # Get batch size
                    if progressive:
                        batch_size = progressive.get_current_batch_size()
                    else:
                        batch_size = 32

                    x = torch.randn(batch_size, 256, device=self.device)
                    y = torch.randn(batch_size, 1, device=self.device)

                    optimizer.zero_grad()
                    output = test_model(x)
                    loss = nn.MSELoss()(output, y)
                    loss.backward()
                    optimizer.step()

                    if scheduler:
                        scheduler.step()

                    if progressive:
                        progressive.update(
                            step=step,
                            loss=loss.item(),
                            memory_usage_gb=memory_tracker.peak()
                        )

                    losses.append(loss.item())

            results[config_name] = {
                'total_time': timer.elapsed,
                'avg_step_time': timer.elapsed / 100,
                'throughput': 100 / timer.elapsed,
                'peak_memory': memory_tracker.peak(),
                'final_loss': losses[-1],
                'convergence_rate': sum(1 for i in range(1, len(losses))
                                      if losses[i] < losses[i-1]) / len(losses)
            }

            # Cleanup
            del test_model, optimizer
            if scheduler:
                del scheduler
            if progressive:
                del progressive
            torch.cuda.empty_cache()

        return results

    def test_system_performance(self):
        """Test complete system performance"""
        print("\n=== System Benchmark Results ===")

        pipeline_results = self.benchmark_full_training_pipeline()

        print("\n1. Full Training Pipeline Comparison:")
        for config_name, metrics in pipeline_results.items():
            print(f"\n{config_name.capitalize()} Configuration:")
            print(f"  Total time: {metrics['total_time']:.3f}s")
            print(f"  Avg step time: {metrics['avg_step_time']*1000:.2f}ms")
            print(f"  Throughput: {metrics['throughput']:.2f} steps/sec")
            print(f"  Peak memory: {metrics['peak_memory']:.3f} GB")
            print(f"  Final loss: {metrics['final_loss']:.6f}")
            print(f"  Convergence rate: {metrics['convergence_rate']:.3f}")

        # Calculate improvements
        if 'baseline' in pipeline_results and 'optimized' in pipeline_results:
            baseline = pipeline_results['baseline']
            optimized = pipeline_results['optimized']

            throughput_improvement = optimized['throughput'] / baseline['throughput']
            memory_efficiency = baseline['peak_memory'] / optimized['peak_memory'] if optimized['peak_memory'] > 0 else 1

            print(f"\n2. Overall Improvements:")
            print(f"  Throughput: {throughput_improvement:.2f}x")
            print(f"  Memory efficiency: {memory_efficiency:.2f}x")

        # All configurations should complete successfully
        self.assertTrue(True)


if __name__ == '__main__':
    # Set environment for benchmarking
    torch.backends.cudnn.benchmark = True

    # Run benchmarks
    unittest.main(verbosity=2)