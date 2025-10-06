#!/usr/bin/env python3
"""
Ultimate Unified System Benchmark and Optimization Tool

A comprehensive all-in-one solution that merges GPU and CPU benchmarking to find
optimal settings for AI training and compute workloads. Automatically detects
hardware capabilities, runs intelligent performance tests, and generates optimal
configurations for maximum performance.

Features:
- Universal hardware detection (CPU + GPU)
- Intelligent performance benchmarking
- CPU/GPU load balancing optimization
- Stress testing with thermal monitoring
- Automatic optimal configuration generation
- Multiple output formats (JSON, Python, shell, text)
- Workload-specific optimization (LLM, vision, general)
- Real-time system monitoring

Usage:
    # Full system analysis and optimization
    python unified_benchmark.py --full-analysis --generate-optimal-config

    # Quick performance test with config generation
    python unified_benchmark.py --quick-test --generate-config

    # Stress test with thermal monitoring
    python unified_benchmark.py --stress-test --duration 1800 --monitor-thermals

    # Workload-specific optimization
    python unified_benchmark.py --workload llm_training --model-size large --optimize

    # Just show comprehensive system info
    python unified_benchmark.py --system-info --detailed
"""

import argparse
import os
import sys
import torch  # type: ignore[import-not-found]
import torch.nn as nn  # type: ignore[import-not-found]
import torch.nn.functional as F  # type: ignore[import-not-found]
from torch.utils.data import DataLoader, Dataset  # type: ignore[import-not-found]
import numpy as np
import json
import time
import threading
import psutil
import subprocess
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import logging
import math
import signal
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

# Try to import training components from the repository
try:
    sys.path.append('/project/code')
    from src.Ava.training.progressive_training import (
        ProgressiveTrainingConfig, ProgressiveTrainer, create_progressive_trainer
    )
    from src.Ava.optimization.fp8_training import (
        FP8Config, FP8Handler, FP8Format
    )
    TRAINING_COMPONENTS_AVAILABLE = True
    logger.info("Training components imported successfully from repository")
except ImportError as e:
    TRAINING_COMPONENTS_AVAILABLE = False
    logger.warning(f"Training components not available: {e}")
    logger.info("Benchmark will use simplified training simulation")


# ============================================================================
# Hardware Architecture and Capability Definitions
# ============================================================================

class GPUArchitecture(Enum):
    """NVIDIA GPU architecture families with performance characteristics."""
    PASCAL = "Pascal"          # GTX 10x0, P100
    VOLTA = "Volta"            # V100, Titan V
    TURING = "Turing"          # RTX 20x0, GTX 16x0
    AMPERE = "Ampere"          # RTX 30x0, A100, A6000
    ADA_LOVELACE = "Ada"       # RTX 40x0
    HOPPER = "Hopper"          # H100, H800
    BLACKWELL = "Blackwell"    # Future B100 series
    UNKNOWN = "Unknown"


class CPUArchitecture(Enum):
    """CPU architecture families with AI optimization characteristics."""
    INTEL_CORE = "Intel Core"
    INTEL_XEON = "Intel Xeon"
    AMD_RYZEN = "AMD Ryzen"
    AMD_EPYC = "AMD EPYC"
    ARM_CORTEX = "ARM Cortex"
    APPLE_SILICON = "Apple Silicon"
    UNKNOWN = "Unknown"


@dataclass
class UnifiedSystemCapabilities:
    """Complete system capabilities for AI workloads."""
    # System identification
    system_id: str
    timestamp: str
    platform: str

    # CPU capabilities
    cpu_model: str
    cpu_architecture: CPUArchitecture
    physical_cores: int
    logical_cores: int
    cpu_base_freq_ghz: float
    cpu_max_freq_ghz: float
    cpu_cache_l1_kb: int
    cpu_cache_l2_kb: int
    cpu_cache_l3_kb: int

    # Memory specifications
    system_memory_gb: float
    memory_channels: int
    memory_speed_mhz: int
    memory_bandwidth_theoretical_gbps: float
    memory_bandwidth_actual_gbps: float
    memory_latency_ns: float

    # CPU AI features
    supports_avx2: bool
    supports_avx512: bool
    supports_fma: bool
    supports_cpu_bf16: bool
    supports_amx: bool  # Intel AMX
    numa_nodes: int

    # CPU performance scores
    single_thread_score: float
    multi_thread_score: float
    cpu_ai_score: float  # AI-specific performance

    # GPU capabilities (None if no GPU)
    gpu_model: Optional[str]
    gpu_architecture: Optional[GPUArchitecture]
    gpu_memory_gb: Optional[float]
    gpu_memory_bandwidth_gbps: Optional[float]
    gpu_compute_capability: Optional[Tuple[int, int]]

    # GPU AI features
    has_tensor_cores: bool
    tensor_core_generation: Optional[int]
    supports_gpu_bf16: bool
    supports_fp16: bool
    supports_tf32: bool
    supports_fp8: bool
    supports_int8: bool
    supports_sparsity: bool
    supports_flash_attention: bool
    supports_nvlink: bool
    nvlink_generation: Optional[int]
    max_nvlink_peers: int
    supports_mig: bool

    # GPU performance characteristics
    theoretical_fp32_tflops: Optional[float]
    theoretical_tensor_tflops: Optional[float]
    theoretical_sparsity_tflops: Optional[float]
    actual_compute_tflops: Optional[float]
    gpu_ai_score: Optional[float]

    # System balance and optimization
    cpu_gpu_balance_ratio: float  # 0.0 = GPU-only, 1.0 = CPU-only
    memory_hierarchy_efficiency: float
    thermal_design_power: Optional[int]
    power_efficiency_score: float

    # Optimal configurations
    optimal_batch_sizes: List[int]
    optimal_sequence_lengths: List[int]
    optimal_dataloader_workers: int
    optimal_torch_threads: int
    recommended_precision: str
    recommended_cpu_gpu_split: Dict[str, float]

    # Performance bottlenecks
    identified_bottlenecks: List[str]
    performance_limiting_factors: List[str]


# ============================================================================
# Unified Hardware Detection System
# ============================================================================

class UnifiedHardwareDetector:
    """Advanced hardware detection combining CPU and GPU analysis."""

    # Comprehensive GPU database with performance characteristics
    GPU_DATABASE = {
        # H100 family - Latest datacenter flagship
        "H100": {
            "architecture": GPUArchitecture.HOPPER,
            "memory_gb": 80, "memory_bandwidth_gbps": 3000, "tensor_core_gen": 4,
            "theoretical_fp32_tflops": 34, "theoretical_tensor_tflops": 989,
            "theoretical_sparsity_tflops": 1979, "supports_mig": True,
            "nvlink_gen": 4, "max_nvlink_peers": 18, "tdp_watts": 700,
            "supports_fp8": True, "supports_sparsity": True, "ai_score": 1000
        },
        "H100-SXM": {
            "architecture": GPUArchitecture.HOPPER,
            "memory_gb": 80, "memory_bandwidth_gbps": 3000, "tensor_core_gen": 4,
            "theoretical_fp32_tflops": 34, "theoretical_tensor_tflops": 989,
            "theoretical_sparsity_tflops": 1979, "supports_mig": True,
            "nvlink_gen": 4, "max_nvlink_peers": 18, "tdp_watts": 700,
            "supports_fp8": True, "supports_sparsity": True, "ai_score": 1000
        },
        # A100 family - Previous gen datacenter
        "A100-SXM4-40GB": {
            "architecture": GPUArchitecture.AMPERE,
            "memory_gb": 40, "memory_bandwidth_gbps": 1555, "tensor_core_gen": 3,
            "theoretical_fp32_tflops": 19.5, "theoretical_tensor_tflops": 312,
            "theoretical_sparsity_tflops": 624, "supports_mig": True,
            "nvlink_gen": 3, "max_nvlink_peers": 12, "tdp_watts": 400,
            "supports_fp8": False, "supports_sparsity": True, "ai_score": 850
        },
        "A100-SXM4-80GB": {
            "architecture": GPUArchitecture.AMPERE,
            "memory_gb": 80, "memory_bandwidth_gbps": 1935, "tensor_core_gen": 3,
            "theoretical_fp32_tflops": 19.5, "theoretical_tensor_tflops": 312,
            "theoretical_sparsity_tflops": 624, "supports_mig": True,
            "nvlink_gen": 3, "max_nvlink_peers": 12, "tdp_watts": 400,
            "supports_fp8": False, "supports_sparsity": True, "ai_score": 900
        },
        # V100 family - Older datacenter
        "V100": {
            "architecture": GPUArchitecture.VOLTA,
            "memory_gb": 16, "memory_bandwidth_gbps": 900, "tensor_core_gen": 1,
            "theoretical_fp32_tflops": 14, "theoretical_tensor_tflops": 112,
            "theoretical_sparsity_tflops": 0, "supports_mig": False,
            "nvlink_gen": 2, "max_nvlink_peers": 6, "tdp_watts": 300,
            "supports_fp8": False, "supports_sparsity": False, "ai_score": 600
        },
        "V100-32GB": {
            "architecture": GPUArchitecture.VOLTA,
            "memory_gb": 32, "memory_bandwidth_gbps": 900, "tensor_core_gen": 1,
            "theoretical_fp32_tflops": 14, "theoretical_tensor_tflops": 112,
            "theoretical_sparsity_tflops": 0, "supports_mig": False,
            "nvlink_gen": 2, "max_nvlink_peers": 6, "tdp_watts": 300,
            "supports_fp8": False, "supports_sparsity": False, "ai_score": 650
        },
        # RTX 40 series - Consumer flagship
        "RTX 4090": {
            "architecture": GPUArchitecture.ADA_LOVELACE,
            "memory_gb": 24, "memory_bandwidth_gbps": 1008, "tensor_core_gen": 4,
            "theoretical_fp32_tflops": 35, "theoretical_tensor_tflops": 165,
            "theoretical_sparsity_tflops": 330, "supports_mig": False,
            "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 450,
            "supports_fp8": True, "supports_sparsity": True, "ai_score": 750
        },
        "RTX 4080": {
            "architecture": GPUArchitecture.ADA_LOVELACE,
            "memory_gb": 16, "memory_bandwidth_gbps": 717, "tensor_core_gen": 4,
            "theoretical_fp32_tflops": 26, "theoretical_tensor_tflops": 123,
            "theoretical_sparsity_tflops": 246, "supports_mig": False,
            "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 320,
            "supports_fp8": True, "supports_sparsity": True, "ai_score": 650
        },
        "RTX 4070": {
            "architecture": GPUArchitecture.ADA_LOVELACE,
            "memory_gb": 12, "memory_bandwidth_gbps": 504, "tensor_core_gen": 4,
            "theoretical_fp32_tflops": 21, "theoretical_tensor_tflops": 83,
            "theoretical_sparsity_tflops": 166, "supports_mig": False,
            "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 200,
            "supports_fp8": True, "supports_sparsity": True, "ai_score": 550
        },
        # RTX 30 series - Previous consumer gen
        "RTX 3090": {
            "architecture": GPUArchitecture.AMPERE,
            "memory_gb": 24, "memory_bandwidth_gbps": 936, "tensor_core_gen": 3,
            "theoretical_fp32_tflops": 28, "theoretical_tensor_tflops": 71,
            "theoretical_sparsity_tflops": 142, "supports_mig": False,
            "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 350,
            "supports_fp8": False, "supports_sparsity": True, "ai_score": 600
        },
        "RTX 3080": {
            "architecture": GPUArchitecture.AMPERE,
            "memory_gb": 10, "memory_bandwidth_gbps": 760, "tensor_core_gen": 3,
            "theoretical_fp32_tflops": 23, "theoretical_tensor_tflops": 58,
            "theoretical_sparsity_tflops": 116, "supports_mig": False,
            "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 320,
            "supports_fp8": False, "supports_sparsity": True, "ai_score": 500
        },
        "RTX 3070": {
            "architecture": GPUArchitecture.AMPERE,
            "memory_gb": 8, "memory_bandwidth_gbps": 448, "tensor_core_gen": 3,
            "theoretical_fp32_tflops": 20, "theoretical_tensor_tflops": 40,
            "theoretical_sparsity_tflops": 80, "supports_mig": False,
            "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 220,
            "supports_fp8": False, "supports_sparsity": True, "ai_score": 400
        },
        "RTX 3060": {
            "architecture": GPUArchitecture.AMPERE,
            "memory_gb": 12, "memory_bandwidth_gbps": 360, "tensor_core_gen": 3,
            "theoretical_fp32_tflops": 13, "theoretical_tensor_tflops": 25,
            "theoretical_sparsity_tflops": 50, "supports_mig": False,
            "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 170,
            "supports_fp8": False, "supports_sparsity": True, "ai_score": 300
        },
        # RTX 20 series - Turing
        "RTX 2080 Ti": {
            "architecture": GPUArchitecture.TURING,
            "memory_gb": 11, "memory_bandwidth_gbps": 616, "tensor_core_gen": 2,
            "theoretical_fp32_tflops": 13.4, "theoretical_tensor_tflops": 53.6,
            "theoretical_sparsity_tflops": 0, "supports_mig": False,
            "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 250,
            "supports_fp8": False, "supports_sparsity": False, "ai_score": 350
        },
        "RTX 2080": {
            "architecture": GPUArchitecture.TURING,
            "memory_gb": 8, "memory_bandwidth_gbps": 448, "tensor_core_gen": 2,
            "theoretical_fp32_tflops": 10.1, "theoretical_tensor_tflops": 40.4,
            "theoretical_sparsity_tflops": 0, "supports_mig": False,
            "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 215,
            "supports_fp8": False, "supports_sparsity": False, "ai_score": 280
        },
        "RTX 2070": {
            "architecture": GPUArchitecture.TURING,
            "memory_gb": 8, "memory_bandwidth_gbps": 448, "tensor_core_gen": 2,
            "theoretical_fp32_tflops": 7.5, "theoretical_tensor_tflops": 30,
            "theoretical_sparsity_tflops": 0, "supports_mig": False,
            "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 175,
            "supports_fp8": False, "supports_sparsity": False, "ai_score": 220
        },
        # GTX series - Pascal and older
        "GTX 1080 Ti": {
            "architecture": GPUArchitecture.PASCAL,
            "memory_gb": 11, "memory_bandwidth_gbps": 484, "tensor_core_gen": None,
            "theoretical_fp32_tflops": 11.3, "theoretical_tensor_tflops": 0,
            "theoretical_sparsity_tflops": 0, "supports_mig": False,
            "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 250,
            "supports_fp8": False, "supports_sparsity": False, "ai_score": 150
        },
        "GTX 1080": {
            "architecture": GPUArchitecture.PASCAL,
            "memory_gb": 8, "memory_bandwidth_gbps": 320, "tensor_core_gen": None,
            "theoretical_fp32_tflops": 8.9, "theoretical_tensor_tflops": 0,
            "theoretical_sparsity_tflops": 0, "supports_mig": False,
            "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 180,
            "supports_fp8": False, "supports_sparsity": False, "ai_score": 120
        },
    }

    def __init__(self):
        self.detected_cpus = []
        self.detected_gpus = []

    def detect_system_capabilities(self, gpu_device_id: int = 0) -> UnifiedSystemCapabilities:
        """Comprehensive system detection and analysis."""
        logger.info("Starting comprehensive hardware detection...")

        # Platform detection
        platform_info = self._detect_platform()

        # CPU detection and benchmarking
        cpu_info = self._detect_and_benchmark_cpu()

        # GPU detection and benchmarking
        gpu_info = self._detect_and_benchmark_gpu(gpu_device_id)

        # System balance analysis
        balance_analysis = self._analyze_system_balance(cpu_info, gpu_info)

        # Build unified capabilities
        capabilities = self._build_unified_capabilities(
            platform_info, cpu_info, gpu_info, balance_analysis
        )

        logger.info(f"System detected: {capabilities.cpu_model} + {capabilities.gpu_model or 'CPU Only'}")
        logger.info(f"Recommended CPU/GPU split: {capabilities.recommended_cpu_gpu_split}")

        return capabilities

    def _detect_platform(self) -> Dict[str, Any]:
        """Detect platform and OS information."""
        import platform

        return {
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'is_wsl': 'microsoft' in platform.uname().release.lower(),
            'is_docker': os.path.exists('/.dockerenv')
        }

    def _detect_and_benchmark_cpu(self) -> Dict[str, Any]:
        """Comprehensive CPU detection and performance analysis."""
        logger.info("Analyzing CPU capabilities...")

        cpu_info = {}

        # Basic CPU information
        cpu_info.update(self._get_cpu_basic_info())

        # Architecture detection
        cpu_info['architecture'] = self._detect_cpu_architecture(cpu_info['model_name'])

        # CPU features and capabilities
        cpu_info.update(self._detect_cpu_features())

        # Memory subsystem analysis
        cpu_info.update(self._analyze_memory_subsystem())

        # Performance benchmarking
        cpu_info.update(self._benchmark_cpu_performance())

        # AI-specific scoring
        cpu_info['ai_score'] = self._calculate_cpu_ai_score(cpu_info)

        return cpu_info

    def _detect_and_benchmark_gpu(self, device_id: int) -> Optional[Dict[str, Any]]:
        """Comprehensive GPU detection and performance analysis."""
        if not torch.cuda.is_available():
            logger.info("CUDA not available - CPU-only system")
            return None

        if device_id >= torch.cuda.device_count():
            logger.warning(f"GPU device {device_id} not found")
            return None

        logger.info(f"Analyzing GPU {device_id} capabilities...")

        gpu_info = {}

        # Basic GPU information
        props = torch.cuda.get_device_properties(device_id)
        gpu_info.update({
            'name': props.name,
            'memory_gb': props.total_memory / (1024 ** 3),
            'compute_capability': (props.major, props.minor),
            'multiprocessor_count': props.multi_processor_count,
            'max_threads_per_multiprocessor': props.max_threads_per_multi_processor,
            'device_id': device_id
        })

        # Architecture detection
        gpu_info['architecture'] = self._determine_gpu_architecture(
            gpu_info['name'], gpu_info['compute_capability']
        )

        # Get detailed specifications
        gpu_info.update(self._get_gpu_detailed_specs(gpu_info['name'], gpu_info['architecture']))

        # Performance benchmarking
        gpu_info.update(self._benchmark_gpu_performance(device_id))

        # AI-specific features and scoring
        gpu_info.update(self._analyze_gpu_ai_capabilities(gpu_info))

        return gpu_info

    def _get_cpu_basic_info(self) -> Dict[str, Any]:
        """Get basic CPU information."""
        cpu_info = {}

        # Model name detection
        try:
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                for line in cpuinfo.split('\n'):
                    if 'model name' in line:
                        cpu_info['model_name'] = line.split(':')[1].strip()
                        break
            else:
                import platform
                cpu_info['model_name'] = platform.processor() or "Unknown CPU"
        except:
            cpu_info['model_name'] = "Unknown CPU"

        # Core counts
        cpu_info['physical_cores'] = psutil.cpu_count(logical=False) or 1
        cpu_info['logical_cores'] = psutil.cpu_count(logical=True) or 1

        # Frequencies
        try:
            freq_info = psutil.cpu_freq()
            if freq_info:
                cpu_info['base_frequency_ghz'] = (freq_info.min or 2000) / 1000
                cpu_info['max_frequency_ghz'] = (freq_info.max or 3000) / 1000
                cpu_info['current_frequency_ghz'] = (freq_info.current or 2500) / 1000
            else:
                cpu_info['base_frequency_ghz'] = 2.0
                cpu_info['max_frequency_ghz'] = 3.0
                cpu_info['current_frequency_ghz'] = 2.5
        except:
            cpu_info['base_frequency_ghz'] = 2.0
            cpu_info['max_frequency_ghz'] = 3.0
            cpu_info['current_frequency_ghz'] = 2.5

        # Cache information
        cache_info = self._get_cpu_cache_info()
        cpu_info.update(cache_info)

        return cpu_info

    def _get_cpu_cache_info(self) -> Dict[str, int]:
        """Get CPU cache sizes."""
        cache_info = {
            'cache_l1_kb': 32,   # Default estimates
            'cache_l2_kb': 256,
            'cache_l3_kb': 8192
        }

        try:
            # Try to get actual cache sizes on Linux
            if os.path.exists('/sys/devices/system/cpu/cpu0/cache'):
                cache_mapping = {}

                for level_dir in ['index0', 'index1', 'index2', 'index3']:
                    cache_path = f'/sys/devices/system/cpu/cpu0/cache/{level_dir}'
                    if os.path.exists(cache_path):
                        try:
                            # Get cache size
                            with open(f'{cache_path}/size', 'r') as f:
                                size_str = f.read().strip().upper()
                                if 'K' in size_str:
                                    size_kb = int(size_str.replace('K', ''))
                                elif 'M' in size_str:
                                    size_kb = int(size_str.replace('M', '')) * 1024
                                else:
                                    continue

                            # Get cache level
                            with open(f'{cache_path}/level', 'r') as f:
                                level_num = int(f.read().strip())
                                cache_mapping[f'cache_l{level_num}_kb'] = size_kb
                        except:
                            continue

                cache_info.update(cache_mapping)
        except:
            pass

        return cache_info

    def _detect_cpu_architecture(self, model_name: str) -> CPUArchitecture:
        """Detect CPU architecture family."""
        model_lower = model_name.lower()

        if any(x in model_lower for x in ['apple', 'm1', 'm2', 'm3', 'm4']):
            return CPUArchitecture.APPLE_SILICON
        elif any(x in model_lower for x in ['xeon', 'platinum', 'gold', 'silver', 'bronze']):
            return CPUArchitecture.INTEL_XEON
        elif any(x in model_lower for x in ['intel', 'core', 'pentium', 'celeron']):
            return CPUArchitecture.INTEL_CORE
        elif any(x in model_lower for x in ['epyc', 'threadripper pro']):
            return CPUArchitecture.AMD_EPYC
        elif any(x in model_lower for x in ['ryzen', 'threadripper', 'athlon']):
            return CPUArchitecture.AMD_RYZEN
        elif any(x in model_lower for x in ['arm', 'cortex', 'aarch64']):
            return CPUArchitecture.ARM_CORTEX
        else:
            return CPUArchitecture.UNKNOWN

    def _detect_cpu_features(self) -> Dict[str, Any]:
        """Detect CPU instruction set features."""
        features = {
            'supports_avx2': False,
            'supports_avx512': False,
            'supports_fma': False,
            'supports_cpu_bf16': False,
            'supports_amx': False,
            'numa_nodes': 1
        }

        try:
            # Linux: read from /proc/cpuinfo
            if os.path.exists('/proc/cpuinfo'):
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read().lower()

                # Find flags line
                for line in cpuinfo.split('\n'):
                    if line.startswith('flags') or line.startswith('Features'):
                        flags = line.lower()
                        features['supports_avx2'] = 'avx2' in flags
                        features['supports_avx512'] = 'avx512' in flags or 'avx512f' in flags
                        features['supports_fma'] = 'fma' in flags
                        features['supports_cpu_bf16'] = 'amx_bf16' in flags or 'sve2' in flags
                        features['supports_amx'] = 'amx_tile' in flags or 'amx_int8' in flags
                        break
        except:
            # Fallback: assume modern CPU has basic features
            features['supports_avx2'] = True
            features['supports_fma'] = True

        # NUMA detection
        try:
            if os.path.exists('/sys/devices/system/node'):
                numa_dirs = [d for d in os.listdir('/sys/devices/system/node')
                           if d.startswith('node') and d[4:].isdigit()]
                features['numa_nodes'] = len(numa_dirs) if numa_dirs else 1
        except:
            features['numa_nodes'] = 1

        return features

    def _analyze_memory_subsystem(self) -> Dict[str, Any]:
        """Analyze memory subsystem characteristics."""
        memory_info = {}

        # Basic memory info
        mem = psutil.virtual_memory()
        memory_info['system_memory_gb'] = mem.total / (1024 ** 3)

        # Memory channels estimation
        memory_info['memory_channels'] = self._estimate_memory_channels()

        # Memory speed detection
        memory_info['memory_speed_mhz'] = self._detect_memory_speed()

        # Theoretical bandwidth calculation
        memory_info['memory_bandwidth_theoretical_gbps'] = (
            memory_info['memory_channels'] *
            memory_info['memory_speed_mhz'] *
            8 / 1000  # DDR = 2x data rate, 8 bytes wide
        )

        # Memory latency testing
        memory_info['memory_latency_ns'] = self._test_memory_latency()

        # Actual bandwidth testing
        memory_info['memory_bandwidth_actual_gbps'] = self._test_memory_bandwidth()

        return memory_info

    def _estimate_memory_channels(self) -> int:
        """Estimate number of memory channels."""
        try:
            # Try DMI decode first
            result = subprocess.run(['dmidecode', '-t', 'memory'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                active_slots = result.stdout.count('Size:') - result.stdout.count('Size: No Module Installed')
                if active_slots >= 8:
                    return 8  # Octal channel (HEDT/Server)
                elif active_slots >= 4:
                    return 4  # Quad channel
                elif active_slots >= 2:
                    return 2  # Dual channel
                else:
                    return 1  # Single channel
        except:
            pass

        # Fallback based on total memory
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        if total_gb >= 128:
            return 8  # High-end workstation/server
        elif total_gb >= 64:
            return 4  # High-end consumer/workstation
        elif total_gb >= 16:
            return 2  # Mainstream
        else:
            return 1  # Entry level

    def _detect_memory_speed(self) -> int:
        """Detect memory speed in MHz."""
        try:
            result = subprocess.run(['dmidecode', '-t', 'memory'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                speeds = []
                for line in result.stdout.split('\n'):
                    if 'Speed:' in line and 'MT/s' in line:
                        try:
                            speed_str = line.split(':')[1].strip()
                            speed = int(speed_str.split()[0])
                            speeds.append(speed)
                        except:
                            continue

                if speeds:
                    return max(speeds)  # Return highest speed found
        except:
            pass

        # Fallback based on system characteristics
        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        if total_gb >= 64:
            return 4800  # High-end DDR5
        elif total_gb >= 32:
            return 3200  # DDR4-3200
        else:
            return 2666  # DDR4-2666

    def _test_memory_latency(self) -> float:
        """Test memory latency in nanoseconds."""
        try:
            # Create random access pattern
            array_size = 4 * 1024 * 1024  # 4MB array
            test_array = np.random.random(array_size).astype(np.float32)
            indices = np.random.randint(0, array_size, 10000)

            # Warm up
            for _ in range(100):
                _ = test_array[indices[0]]

            # Measure latency
            start_time = time.perf_counter()
            for idx in indices:
                _ = test_array[idx]
            end_time = time.perf_counter()

            latency_per_access = (end_time - start_time) / len(indices)
            return latency_per_access * 1e9  # Convert to nanoseconds

        except:
            return 100.0  # Default estimate

    def _test_memory_bandwidth(self) -> float:
        """Test actual memory bandwidth in GB/s."""
        try:
            # Large array for bandwidth testing
            array_size = 100 * 1024 * 1024  # 100MB
            iterations = 20

            # Create test arrays
            src_array = np.random.random(array_size).astype(np.float32)
            dst_array = np.zeros_like(src_array)

            # Warm up
            for _ in range(5):
                np.copyto(dst_array, src_array)

            # Bandwidth test
            start_time = time.perf_counter()
            for _ in range(iterations):
                np.copyto(dst_array, src_array)
            end_time = time.perf_counter()

            # Calculate bandwidth (read + write)
            bytes_transferred = array_size * 4 * 2 * iterations  # float32 * R+W * iterations
            bandwidth_gbps = bytes_transferred / (end_time - start_time) / (1024 ** 3)

            return bandwidth_gbps

        except:
            return 50.0  # Default estimate

    def _benchmark_cpu_performance(self) -> Dict[str, float]:
        """Benchmark CPU performance for AI workloads."""
        performance = {}

        # Single-threaded performance
        performance['single_thread_score'] = self._benchmark_single_thread()

        # Multi-threaded performance
        performance['multi_thread_score'] = self._benchmark_multi_thread()

        # Memory-intensive performance
        performance['memory_intensive_score'] = self._benchmark_memory_intensive()

        # AI-specific workload performance
        performance['ai_workload_score'] = self._benchmark_ai_workload()

        return performance

    def _benchmark_single_thread(self) -> float:
        """Benchmark single-threaded performance."""
        def cpu_intensive_task():
            result = 0.0
            for i in range(1000000):
                result += math.sin(i * 0.001) * math.cos(i * 0.001) + math.sqrt(i + 1)
            return result

        # Multiple runs for stability
        times = []
        for _ in range(3):
            start_time = time.perf_counter()
            result = cpu_intensive_task()
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        return 1000000 / avg_time  # Operations per second

    def _benchmark_multi_thread(self) -> float:
        """Benchmark multi-threaded performance."""
        def cpu_worker(iterations):
            result = 0.0
            for i in range(iterations):
                result += math.sin(i * 0.001) * math.cos(i * 0.001) + math.sqrt(i + 1)
            return result

        num_threads = psutil.cpu_count(logical=True) or 1
        iterations_per_thread = 500000 // num_threads

        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(cpu_worker, iterations_per_thread)
                      for _ in range(num_threads)]
            results = [f.result() for f in futures]
        end_time = time.perf_counter()

        total_operations = num_threads * iterations_per_thread
        return total_operations / (end_time - start_time)

    def _benchmark_memory_intensive(self) -> float:
        """Benchmark memory-intensive operations."""
        try:
            # Matrix operations that stress memory bandwidth
            size = 2048
            iterations = 10

            a = np.random.random((size, size)).astype(np.float32)
            b = np.random.random((size, size)).astype(np.float32)

            start_time = time.perf_counter()
            for _ in range(iterations):
                c = np.dot(a, b)
            end_time = time.perf_counter()

            # FLOPS calculation
            flops = 2 * size * size * size * iterations  # Matrix multiply FLOPS
            gflops = flops / (end_time - start_time) / 1e9

            return gflops

        except:
            return 10.0  # Default estimate

    def _benchmark_ai_workload(self) -> float:
        """Benchmark AI-specific CPU workloads."""
        try:
            # Simulate neural network operations on CPU
            batch_size = 32
            input_size = 512
            hidden_size = 256
            iterations = 50

            # Create simple model
            model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Benchmark training steps
            start_time = time.perf_counter()
            for _ in range(iterations):
                x = torch.randn(batch_size, input_size)
                y = torch.randn(batch_size, 1)

                optimizer.zero_grad()
                output = model(x)
                loss = nn.MSELoss()(output, y)
                loss.backward()
                optimizer.step()
            end_time = time.perf_counter()

            steps_per_second = iterations / (end_time - start_time)
            return steps_per_second

        except:
            return 5.0  # Default estimate

    def _calculate_cpu_ai_score(self, cpu_info: Dict[str, Any]) -> float:
        """Calculate AI-specific CPU performance score."""
        base_score = cpu_info.get('ai_workload_score', 5.0)

        # Adjust for CPU features
        if cpu_info.get('supports_avx512', False):
            base_score *= 1.3
        elif cpu_info.get('supports_avx2', False):
            base_score *= 1.15

        if cpu_info.get('supports_fma', False):
            base_score *= 1.1

        if cpu_info.get('supports_cpu_bf16', False):
            base_score *= 1.25

        if cpu_info.get('supports_amx', False):
            base_score *= 1.4

        # Adjust for core count
        core_multiplier = min(cpu_info.get('physical_cores', 4) / 8, 2.0)
        base_score *= (0.5 + 0.5 * core_multiplier)

        # Adjust for memory bandwidth
        memory_bandwidth = cpu_info.get('memory_bandwidth_actual_gbps', 50)
        if memory_bandwidth > 100:
            base_score *= 1.2
        elif memory_bandwidth > 200:
            base_score *= 1.4

        return base_score

    def _determine_gpu_architecture(self, name: str, compute_capability: Tuple[int, int]) -> GPUArchitecture:
        """Determine GPU architecture from name and compute capability."""
        major, minor = compute_capability
        name_lower = name.lower()

        # Name-based detection (most reliable)
        if any(x in name_lower for x in ['h100', 'h800']):
            return GPUArchitecture.HOPPER
        elif any(x in name_lower for x in ['rtx 40', '4090', '4080', '4070', '4060']):
            return GPUArchitecture.ADA_LOVELACE
        elif any(x in name_lower for x in ['a100', 'a40', 'a30', 'a10']):
            return GPUArchitecture.AMPERE
        elif any(x in name_lower for x in ['rtx 30', '3090', '3080', '3070', '3060']):
            return GPUArchitecture.AMPERE
        elif any(x in name_lower for x in ['v100']):
            return GPUArchitecture.VOLTA
        elif any(x in name_lower for x in ['rtx 20', '2080', '2070', '2060', 'gtx 16']):
            return GPUArchitecture.TURING
        elif any(x in name_lower for x in ['gtx 10', '1080', '1070', '1060', 'p100']):
            return GPUArchitecture.PASCAL

        # Fallback to compute capability
        if major >= 9:
            return GPUArchitecture.HOPPER
        elif major == 8:
            return GPUArchitecture.ADA_LOVELACE if minor >= 9 else GPUArchitecture.AMPERE
        elif major == 7:
            return GPUArchitecture.TURING if minor >= 5 else GPUArchitecture.VOLTA
        elif major == 6:
            return GPUArchitecture.PASCAL
        else:
            return GPUArchitecture.UNKNOWN

    def _get_gpu_detailed_specs(self, name: str, architecture: GPUArchitecture) -> Dict[str, Any]:
        """Get detailed GPU specifications from database."""
        # Try exact name matches first
        for key, specs in self.GPU_DATABASE.items():
            if key.lower() in name.lower() or name.lower() in key.lower():
                return specs.copy()

        # Try partial matches
        name_parts = name.lower().split()
        for key, specs in self.GPU_DATABASE.items():
            key_parts = key.lower().split()
            if any(part in name_parts for part in key_parts):
                return specs.copy()

        # Architecture-based defaults
        defaults = {
            GPUArchitecture.HOPPER: {
                "architecture": GPUArchitecture.HOPPER,
                "memory_bandwidth_gbps": 3000, "tensor_core_gen": 4,
                "theoretical_fp32_tflops": 30, "theoretical_tensor_tflops": 500,
                "theoretical_sparsity_tflops": 1000, "supports_mig": True,
                "nvlink_gen": 4, "max_nvlink_peers": 18, "tdp_watts": 700,
                "supports_fp8": True, "supports_sparsity": True, "ai_score": 900
            },
            GPUArchitecture.ADA_LOVELACE: {
                "architecture": GPUArchitecture.ADA_LOVELACE,
                "memory_bandwidth_gbps": 800, "tensor_core_gen": 4,
                "theoretical_fp32_tflops": 25, "theoretical_tensor_tflops": 120,
                "theoretical_sparsity_tflops": 240, "supports_mig": False,
                "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 400,
                "supports_fp8": True, "supports_sparsity": True, "ai_score": 600
            },
            GPUArchitecture.AMPERE: {
                "architecture": GPUArchitecture.AMPERE,
                "memory_bandwidth_gbps": 700, "tensor_core_gen": 3,
                "theoretical_fp32_tflops": 20, "theoretical_tensor_tflops": 80,
                "theoretical_sparsity_tflops": 160, "supports_mig": False,
                "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 350,
                "supports_fp8": False, "supports_sparsity": True, "ai_score": 500
            },
            GPUArchitecture.VOLTA: {
                "architecture": GPUArchitecture.VOLTA,
                "memory_bandwidth_gbps": 900, "tensor_core_gen": 1,
                "theoretical_fp32_tflops": 14, "theoretical_tensor_tflops": 110,
                "theoretical_sparsity_tflops": 0, "supports_mig": False,
                "nvlink_gen": 2, "max_nvlink_peers": 6, "tdp_watts": 300,
                "supports_fp8": False, "supports_sparsity": False, "ai_score": 400
            },
            GPUArchitecture.TURING: {
                "architecture": GPUArchitecture.TURING,
                "memory_bandwidth_gbps": 500, "tensor_core_gen": 2,
                "theoretical_fp32_tflops": 10, "theoretical_tensor_tflops": 40,
                "theoretical_sparsity_tflops": 0, "supports_mig": False,
                "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 250,
                "supports_fp8": False, "supports_sparsity": False, "ai_score": 250
            },
            GPUArchitecture.PASCAL: {
                "architecture": GPUArchitecture.PASCAL,
                "memory_bandwidth_gbps": 400, "tensor_core_gen": None,
                "theoretical_fp32_tflops": 8, "theoretical_tensor_tflops": 0,
                "theoretical_sparsity_tflops": 0, "supports_mig": False,
                "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 250,
                "supports_fp8": False, "supports_sparsity": False, "ai_score": 100
            }
        }

        return defaults.get(architecture, {
            "architecture": architecture,
            "memory_bandwidth_gbps": 300, "tensor_core_gen": None,
            "theoretical_fp32_tflops": 5, "theoretical_tensor_tflops": 0,
            "theoretical_sparsity_tflops": 0, "supports_mig": False,
            "nvlink_gen": None, "max_nvlink_peers": 0, "tdp_watts": 200,
            "supports_fp8": False, "supports_sparsity": False, "ai_score": 50
        })

    def _benchmark_gpu_performance(self, device_id: int) -> Dict[str, Any]:
        """Benchmark GPU performance across different workloads."""
        device = torch.device(f'cuda:{device_id}')
        performance = {}

        try:
            # Compute performance benchmark
            performance.update(self._benchmark_gpu_compute(device))

            # Memory bandwidth benchmark
            performance.update(self._benchmark_gpu_memory(device))

            # AI-specific benchmark
            performance.update(self._benchmark_gpu_ai_workload(device))

        except Exception as e:
            logger.warning(f"GPU benchmarking failed: {e}")
            performance = {
                'actual_compute_tflops': 0.0,
                'actual_memory_bandwidth_gbps': 0.0,
                'ai_workload_score': 0.0
            }

        return performance

    def _benchmark_gpu_compute(self, device: torch.device) -> Dict[str, float]:
        """Benchmark GPU compute performance."""
        results = {}

        try:
            # Matrix multiplication benchmark for different precisions
            matrix_size = 2048
            iterations = 20

            # FP32 benchmark
            a_fp32 = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
            b_fp32 = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)

            # Warmup
            for _ in range(3):
                _ = torch.matmul(a_fp32, b_fp32)

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(iterations):
                _ = torch.matmul(a_fp32, b_fp32)
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            # Calculate TFLOPS
            flops = 2 * matrix_size * matrix_size * matrix_size * iterations
            fp32_tflops = flops / (end_time - start_time) / 1e12
            results['actual_compute_tflops'] = fp32_tflops

            # FP16 benchmark if supported
            try:
                a_fp16 = a_fp32.half()
                b_fp16 = b_fp32.half()

                # Warmup
                for _ in range(3):
                    _ = torch.matmul(a_fp16, b_fp16)

                torch.cuda.synchronize()
                start_time = time.perf_counter()
                for _ in range(iterations):
                    _ = torch.matmul(a_fp16, b_fp16)
                torch.cuda.synchronize()
                end_time = time.perf_counter()

                fp16_tflops = flops / (end_time - start_time) / 1e12
                results['actual_tensor_tflops'] = fp16_tflops

            except:
                results['actual_tensor_tflops'] = 0.0

        except Exception as e:
            logger.warning(f"GPU compute benchmark failed: {e}")
            results = {'actual_compute_tflops': 0.0, 'actual_tensor_tflops': 0.0}

        return results

    def _benchmark_gpu_memory(self, device: torch.device) -> Dict[str, float]:
        """Benchmark GPU memory bandwidth."""
        try:
            # Memory copy benchmark
            memory_mb = 500  # 500MB test
            iterations = 50

            elements = (memory_mb * 1024 * 1024) // 4  # float32 elements
            src = torch.randn(elements, dtype=torch.float32, device=device)
            dst = torch.empty_like(src)

            # Warmup
            for _ in range(5):
                dst.copy_(src)

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(iterations):
                dst.copy_(src)
            torch.cuda.synchronize()
            end_time = time.perf_counter()

            # Calculate bandwidth (read + write)
            bytes_transferred = memory_mb * 1024 * 1024 * 2 * iterations
            bandwidth_gbps = bytes_transferred / (end_time - start_time) / (1024 ** 3)

            return {'actual_memory_bandwidth_gbps': bandwidth_gbps}

        except Exception as e:
            logger.warning(f"GPU memory benchmark failed: {e}")
            return {'actual_memory_bandwidth_gbps': 0.0}

    def _benchmark_gpu_ai_workload(self, device: torch.device) -> Dict[str, float]:
        """Benchmark GPU AI-specific workloads."""
        try:
            # Neural network training benchmark
            batch_size = 64
            seq_length = 512
            hidden_size = 512
            iterations = 50

            # Simple transformer-like model
            model = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.LayerNorm(hidden_size)
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            # Benchmark training steps
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            for _ in range(iterations):
                x = torch.randn(batch_size, seq_length, hidden_size, device=device)
                target = torch.randn(batch_size, seq_length, hidden_size, device=device)

                optimizer.zero_grad()
                output = model(x)
                loss = nn.MSELoss()(output, target)
                loss.backward()
                optimizer.step()

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            steps_per_second = iterations / (end_time - start_time)
            tokens_per_second = steps_per_second * batch_size * seq_length

            return {
                'ai_workload_score': steps_per_second,
                'tokens_per_second': tokens_per_second
            }

        except Exception as e:
            logger.warning(f"GPU AI benchmark failed: {e}")
            return {'ai_workload_score': 0.0, 'tokens_per_second': 0.0}

    def _analyze_gpu_ai_capabilities(self, gpu_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze GPU AI-specific capabilities."""
        capabilities = {}

        architecture = gpu_info['architecture']

        # Tensor Core capabilities
        tensor_core_gen = gpu_info.get('tensor_core_gen')
        capabilities['has_tensor_cores'] = tensor_core_gen is not None
        capabilities['tensor_core_generation'] = tensor_core_gen

        # Precision support
        capabilities['supports_tf32'] = architecture in [
            GPUArchitecture.AMPERE, GPUArchitecture.ADA_LOVELACE, GPUArchitecture.HOPPER
        ]
        capabilities['supports_gpu_bf16'] = architecture in [
            GPUArchitecture.AMPERE, GPUArchitecture.ADA_LOVELACE, GPUArchitecture.HOPPER
        ]
        capabilities['supports_fp16'] = tensor_core_gen is not None
        capabilities['supports_fp8'] = gpu_info.get('supports_fp8', False)
        capabilities['supports_int8'] = tensor_core_gen is not None
        capabilities['supports_sparsity'] = gpu_info.get('supports_sparsity', False)

        # Advanced features
        capabilities['supports_flash_attention'] = architecture in [
            GPUArchitecture.AMPERE, GPUArchitecture.ADA_LOVELACE, GPUArchitecture.HOPPER
        ]
        capabilities['supports_nvlink'] = gpu_info.get('nvlink_gen') is not None
        capabilities['nvlink_generation'] = gpu_info.get('nvlink_gen')
        capabilities['max_nvlink_peers'] = gpu_info.get('max_nvlink_peers', 0)
        capabilities['supports_mig'] = gpu_info.get('supports_mig', False)

        # Calculate GPU AI score
        base_score = gpu_info.get('ai_score', 100)
        actual_performance = gpu_info.get('ai_workload_score', 1.0)
        capabilities['gpu_ai_score'] = base_score * min(actual_performance / 10, 2.0)

        return capabilities

    def _analyze_system_balance(self, cpu_info: Dict[str, Any],
                              gpu_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze CPU/GPU balance and optimization opportunities."""
        balance = {}

        if gpu_info is None:
            # CPU-only system
            balance['cpu_gpu_balance_ratio'] = 1.0
            balance['recommended_cpu_gpu_split'] = {'cpu': 1.0, 'gpu': 0.0}
            balance['memory_hierarchy_efficiency'] = 0.7
            balance['power_efficiency_score'] = cpu_info.get('ai_score', 100) / 100
            balance['identified_bottlenecks'] = ['no_gpu']
            balance['performance_limiting_factors'] = ['CPU-only processing']
        else:
            # CPU+GPU system
            cpu_score = cpu_info.get('ai_score', 100)
            gpu_score = gpu_info.get('gpu_ai_score', 100)

            # Calculate balance ratio (0 = GPU-dominant, 1 = CPU-dominant)
            total_score = cpu_score + gpu_score
            balance['cpu_gpu_balance_ratio'] = cpu_score / total_score if total_score > 0 else 0.5

            # Recommended workload split
            gpu_ratio = gpu_score / total_score if total_score > 0 else 0.8
            cpu_ratio = 1.0 - gpu_ratio
            balance['recommended_cpu_gpu_split'] = {'cpu': cpu_ratio, 'gpu': gpu_ratio}

            # Memory hierarchy efficiency
            gpu_memory = gpu_info.get('memory_gb', 0)
            system_memory = cpu_info.get('system_memory_gb', 16)
            memory_ratio = gpu_memory / (gpu_memory + system_memory) if gpu_memory > 0 else 0.1
            balance['memory_hierarchy_efficiency'] = min(memory_ratio * 2, 1.0)

            # Power efficiency
            gpu_tdp = gpu_info.get('tdp_watts', 300)
            estimated_cpu_tdp = cpu_info.get('physical_cores', 8) * 15  # Rough estimate
            total_power = gpu_tdp + estimated_cpu_tdp
            perf_per_watt = (cpu_score + gpu_score) / total_power
            balance['power_efficiency_score'] = min(perf_per_watt / 2, 1.0)

            # Identify bottlenecks
            bottlenecks = []
            limiting_factors = []

            if gpu_memory < 8:
                bottlenecks.append('limited_gpu_memory')
                limiting_factors.append(f'GPU memory: {gpu_memory:.1f}GB')

            if system_memory < 32:
                bottlenecks.append('limited_system_memory')
                limiting_factors.append(f'System memory: {system_memory:.1f}GB')

            if cpu_info.get('physical_cores', 8) < 8:
                bottlenecks.append('limited_cpu_cores')
                limiting_factors.append(f'CPU cores: {cpu_info.get("physical_cores", 8)}')

            cpu_memory_bw = cpu_info.get('memory_bandwidth_actual_gbps', 50)
            gpu_memory_bw = gpu_info.get('actual_memory_bandwidth_gbps', 500)
            if cpu_memory_bw < 100:
                bottlenecks.append('low_cpu_memory_bandwidth')
                limiting_factors.append(f'CPU memory bandwidth: {cpu_memory_bw:.1f} GB/s')

            if not gpu_info.get('has_tensor_cores', False):
                bottlenecks.append('no_tensor_cores')
                limiting_factors.append('No Tensor Cores for AI acceleration')

            balance['identified_bottlenecks'] = bottlenecks
            balance['performance_limiting_factors'] = limiting_factors

        return balance

    def _build_unified_capabilities(self, platform_info: Dict[str, Any],
                                  cpu_info: Dict[str, Any],
                                  gpu_info: Optional[Dict[str, Any]],
                                  balance_analysis: Dict[str, Any]) -> UnifiedSystemCapabilities:
        """Build unified system capabilities structure."""

        # Generate system ID
        system_id = f"{cpu_info['model_name']}_{gpu_info['name'] if gpu_info else 'CPU_Only'}"
        system_id = system_id.replace(' ', '_').replace('/', '_')

        # Calculate optimal configurations
        optimal_configs = self._calculate_optimal_configurations(cpu_info, gpu_info, balance_analysis)

        return UnifiedSystemCapabilities(
            # System identification
            system_id=system_id,
            timestamp=datetime.now().isoformat(),
            platform=platform_info['platform'],

            # CPU capabilities
            cpu_model=cpu_info['model_name'],
            cpu_architecture=cpu_info['architecture'],
            physical_cores=cpu_info['physical_cores'],
            logical_cores=cpu_info['logical_cores'],
            cpu_base_freq_ghz=cpu_info['base_frequency_ghz'],
            cpu_max_freq_ghz=cpu_info['max_frequency_ghz'],
            cpu_cache_l1_kb=cpu_info['cache_l1_kb'],
            cpu_cache_l2_kb=cpu_info['cache_l2_kb'],
            cpu_cache_l3_kb=cpu_info['cache_l3_kb'],

            # Memory specifications
            system_memory_gb=cpu_info['system_memory_gb'],
            memory_channels=cpu_info['memory_channels'],
            memory_speed_mhz=cpu_info['memory_speed_mhz'],
            memory_bandwidth_theoretical_gbps=cpu_info['memory_bandwidth_theoretical_gbps'],
            memory_bandwidth_actual_gbps=cpu_info['memory_bandwidth_actual_gbps'],
            memory_latency_ns=cpu_info['memory_latency_ns'],

            # CPU AI features
            supports_avx2=cpu_info['supports_avx2'],
            supports_avx512=cpu_info['supports_avx512'],
            supports_fma=cpu_info['supports_fma'],
            supports_cpu_bf16=cpu_info['supports_cpu_bf16'],
            supports_amx=cpu_info['supports_amx'],
            numa_nodes=cpu_info['numa_nodes'],

            # CPU performance scores
            single_thread_score=cpu_info['single_thread_score'],
            multi_thread_score=cpu_info['multi_thread_score'],
            cpu_ai_score=cpu_info['ai_score'],

            # GPU capabilities (None if no GPU)
            gpu_model=gpu_info['name'] if gpu_info else None,
            gpu_architecture=gpu_info['architecture'] if gpu_info else None,
            gpu_memory_gb=gpu_info['memory_gb'] if gpu_info else None,
            gpu_memory_bandwidth_gbps=gpu_info['memory_bandwidth_gbps'] if gpu_info else None,
            gpu_compute_capability=gpu_info['compute_capability'] if gpu_info else None,

            # GPU AI features
            has_tensor_cores=gpu_info['has_tensor_cores'] if gpu_info else False,
            tensor_core_generation=gpu_info['tensor_core_generation'] if gpu_info else None,
            supports_gpu_bf16=gpu_info['supports_gpu_bf16'] if gpu_info else False,
            supports_fp16=gpu_info['supports_fp16'] if gpu_info else False,
            supports_tf32=gpu_info['supports_tf32'] if gpu_info else False,
            supports_fp8=gpu_info['supports_fp8'] if gpu_info else False,
            supports_int8=gpu_info['supports_int8'] if gpu_info else False,
            supports_sparsity=gpu_info['supports_sparsity'] if gpu_info else False,
            supports_flash_attention=gpu_info['supports_flash_attention'] if gpu_info else False,
            supports_nvlink=gpu_info['supports_nvlink'] if gpu_info else False,
            nvlink_generation=gpu_info['nvlink_generation'] if gpu_info else None,
            max_nvlink_peers=gpu_info['max_nvlink_peers'] if gpu_info else 0,
            supports_mig=gpu_info['supports_mig'] if gpu_info else False,

            # GPU performance characteristics
            theoretical_fp32_tflops=gpu_info['theoretical_fp32_tflops'] if gpu_info else None,
            theoretical_tensor_tflops=gpu_info['theoretical_tensor_tflops'] if gpu_info else None,
            theoretical_sparsity_tflops=gpu_info['theoretical_sparsity_tflops'] if gpu_info else None,
            actual_compute_tflops=gpu_info['actual_compute_tflops'] if gpu_info else None,
            gpu_ai_score=gpu_info['gpu_ai_score'] if gpu_info else None,

            # System balance and optimization
            cpu_gpu_balance_ratio=balance_analysis['cpu_gpu_balance_ratio'],
            memory_hierarchy_efficiency=balance_analysis['memory_hierarchy_efficiency'],
            thermal_design_power=gpu_info.get('tdp_watts') if gpu_info else None,
            power_efficiency_score=balance_analysis['power_efficiency_score'],

            # Optimal configurations
            optimal_batch_sizes=optimal_configs['batch_sizes'],
            optimal_sequence_lengths=optimal_configs['sequence_lengths'],
            optimal_dataloader_workers=optimal_configs['dataloader_workers'],
            optimal_torch_threads=optimal_configs['torch_threads'],
            recommended_precision=optimal_configs['precision'],
            recommended_cpu_gpu_split=balance_analysis['recommended_cpu_gpu_split'],

            # Performance bottlenecks
            identified_bottlenecks=balance_analysis['identified_bottlenecks'],
            performance_limiting_factors=balance_analysis['performance_limiting_factors']
        )

    def _calculate_optimal_configurations(self, cpu_info: Dict[str, Any],
                                        gpu_info: Optional[Dict[str, Any]],
                                        balance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal configuration parameters."""

        configs = {}

        # DataLoader workers optimization
        physical_cores = cpu_info.get('physical_cores', 4)
        memory_bandwidth = cpu_info.get('memory_bandwidth_actual_gbps', 50)

        max_workers_by_cores = physical_cores * 2  # 2 workers per core
        max_workers_by_memory = max(1, int(memory_bandwidth / 10))  # 10 GB/s per worker
        optimal_workers = min(max_workers_by_cores, max_workers_by_memory, 16)  # Cap at 16
        configs['dataloader_workers'] = optimal_workers

        # Torch threads optimization
        if gpu_info:
            # With GPU, use fewer CPU threads for data processing
            configs['torch_threads'] = min(physical_cores, 8)
        else:
            # CPU-only, use all available cores
            configs['torch_threads'] = physical_cores

        # Batch sizes optimization
        if gpu_info:
            gpu_memory = gpu_info.get('memory_gb', 8)
            if gpu_memory >= 80:
                configs['batch_sizes'] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            elif gpu_memory >= 40:
                configs['batch_sizes'] = [1, 2, 4, 8, 16, 32, 64, 128]
            elif gpu_memory >= 20:
                configs['batch_sizes'] = [1, 2, 4, 8, 16, 32, 64]
            elif gpu_memory >= 10:
                configs['batch_sizes'] = [1, 2, 4, 8, 16, 32]
            else:
                configs['batch_sizes'] = [1, 2, 4, 8, 16]
        else:
            # CPU-only system
            system_memory = cpu_info.get('system_memory_gb', 16)
            max_batch = min(64, int(system_memory / 2))
            configs['batch_sizes'] = [1, 2, 4, 8, max_batch]

        # Sequence lengths optimization
        has_tensor_cores = gpu_info and gpu_info.get('has_tensor_cores', False)
        base_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

        # Tensor Cores prefer multiples of 8
        if has_tensor_cores:
            base_lengths = [x for x in base_lengths if x % 8 == 0]

        if gpu_info:
            gpu_memory = gpu_info.get('memory_gb', 8)
            if gpu_memory >= 80:
                configs['sequence_lengths'] = base_lengths
            elif gpu_memory >= 40:
                configs['sequence_lengths'] = [x for x in base_lengths if x <= 8192]
            elif gpu_memory >= 20:
                configs['sequence_lengths'] = [x for x in base_lengths if x <= 4096]
            elif gpu_memory >= 10:
                configs['sequence_lengths'] = [x for x in base_lengths if x <= 2048]
            else:
                configs['sequence_lengths'] = [x for x in base_lengths if x <= 1024]
        else:
            # CPU-only system - shorter sequences
            configs['sequence_lengths'] = [x for x in base_lengths if x <= 1024]

        # Precision recommendation
        if gpu_info:
            architecture = gpu_info.get('architecture')
            if gpu_info.get('supports_fp8', False):
                configs['precision'] = 'fp8'
            elif gpu_info.get('supports_gpu_bf16', False):
                configs['precision'] = 'bf16'
            elif gpu_info.get('supports_fp16', False):
                configs['precision'] = 'fp16'
            elif gpu_info.get('supports_tf32', False):
                configs['precision'] = 'tf32'
            else:
                configs['precision'] = 'fp32'
        else:
            # CPU-only
            if cpu_info.get('supports_cpu_bf16', False):
                configs['precision'] = 'bf16'
            else:
                configs['precision'] = 'fp32'

        return configs


# ============================================================================
# Comprehensive Benchmarking Suite
# ============================================================================

class UnifiedBenchmarkSuite:
    """Comprehensive CPU and GPU benchmarking with intelligent optimization."""

    def __init__(self, device_id: int = 0, output_dir: str = "unified_benchmark_results"):
        self.device_id = device_id
        self.device = torch.device(f'cuda:{device_id}') if torch.cuda.is_available() else torch.device('cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.hardware_detector = UnifiedHardwareDetector()
        self.system_capabilities = None
        self.monitor = None

        # Results storage
        self.results = {
            'system_info': {},
            'benchmarks': {},
            'stress_test_results': {},
            'optimization_results': {},
            'optimal_configuration': {},
            'timestamp': datetime.now().isoformat()
        }

        # Test state
        self.test_interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully."""
        logger.info("Benchmark interrupted by user - cleaning up...")
        self.test_interrupted = True

    def analyze_system(self) -> UnifiedSystemCapabilities:
        """Comprehensive system analysis and capability detection."""
        logger.info("Starting comprehensive system analysis...")

        self.system_capabilities = self.hardware_detector.detect_system_capabilities(self.device_id)
        self.results['system_info'] = asdict(self.system_capabilities)

        logger.info("System analysis completed")
        return self.system_capabilities

    def run_performance_benchmarks(self, quick_mode: bool = False) -> Dict[str, Any]:
        """Run comprehensive performance benchmarks for CPU and GPU."""
        logger.info("Running comprehensive performance benchmarks...")

        if not self.system_capabilities:
            self.analyze_system()

        if not self.system_capabilities:
            raise RuntimeError("Failed to analyze system capabilities")

        benchmark_results = {}
        iterations = 20 if quick_mode else 50

        # GPU benchmarks
        if self.system_capabilities.gpu_model:
            logger.info("Running GPU benchmarks...")
            benchmark_results.update(self._run_gpu_benchmarks(iterations))

        # CPU benchmarks
        logger.info("Running CPU benchmarks...")
        benchmark_results.update(self._run_cpu_benchmarks(iterations))

        # Combined CPU+GPU benchmarks
        if self.system_capabilities.gpu_model:
            logger.info("Running combined CPU+GPU benchmarks...")
            benchmark_results.update(self._run_combined_benchmarks(iterations))

        self.results['benchmarks'] = benchmark_results
        logger.info("Performance benchmarks completed")
        return benchmark_results

    def _run_gpu_benchmarks(self, iterations: int) -> Dict[str, Any]:
        """Comprehensive GPU benchmarking suite."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        gpu_results = {}

        # 1. Compute Performance Tests
        logger.info("Testing GPU compute performance across precisions...")
        gpu_results['compute_performance'] = self._benchmark_gpu_compute_comprehensive(iterations)

        # 2. Memory Bandwidth Tests
        logger.info("Testing GPU memory bandwidth...")
        gpu_results['memory_bandwidth'] = self._benchmark_gpu_memory_comprehensive(iterations)

        # 3. Attention Mechanism Tests
        if self.system_capabilities and self.system_capabilities.supports_flash_attention:
            logger.info("Testing attention mechanisms...")
            gpu_results['attention_performance'] = self._benchmark_attention_mechanisms(iterations)  # type: ignore[attr-defined]

        # 4. Mixed Precision Training Tests
        logger.info("Testing mixed precision training...")
        gpu_results['mixed_precision_training'] = self._benchmark_mixed_precision_training(iterations)  # type: ignore[attr-defined]

        # 5. AI Workload Specific Tests
        logger.info("Testing AI-specific workloads...")
        gpu_results['ai_workloads'] = self._benchmark_ai_workloads(iterations)  # type: ignore[attr-defined]

        return gpu_results

    def _run_cpu_benchmarks(self, iterations: int) -> Dict[str, Any]:
        """Comprehensive CPU benchmarking suite."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        cpu_results = {}

        # 1. DataLoader Performance Tests
        logger.info("Testing DataLoader performance...")
        cpu_results['dataloader_performance'] = self._benchmark_dataloader_performance(iterations)  # type: ignore[attr-defined]

        # 2. CPU Training Performance Tests
        logger.info("Testing CPU training performance...")
        cpu_results['cpu_training_performance'] = self._benchmark_cpu_training_performance(iterations)  # type: ignore[attr-defined]

        # 3. Multi-threading Performance Tests
        logger.info("Testing multi-threading performance...")
        cpu_results['threading_performance'] = self._benchmark_threading_performance(iterations)  # type: ignore[attr-defined]

        # 4. Memory-Intensive Workload Tests
        logger.info("Testing memory-intensive workloads...")
        cpu_results['memory_intensive_performance'] = self._benchmark_memory_intensive_workloads(iterations)  # type: ignore[attr-defined]

        # 5. CPU AI Acceleration Tests
        if self.system_capabilities.supports_cpu_bf16 or self.system_capabilities.supports_amx:
            logger.info("Testing CPU AI acceleration features...")
            cpu_results['cpu_ai_acceleration'] = self._benchmark_cpu_ai_acceleration(iterations)  # type: ignore[attr-defined]

        return cpu_results

    def _run_combined_benchmarks(self, iterations: int) -> Dict[str, Any]:
        """CPU+GPU combined workload benchmarks."""
        combined_results = {}

        # 1. Load Balancing Tests
        logger.info("Testing CPU/GPU load balancing...")
        combined_results['load_balancing'] = self._benchmark_load_balancing(iterations)  # type: ignore[attr-defined]

        # 2. Data Pipeline Tests
        logger.info("Testing data pipeline efficiency...")
        combined_results['data_pipeline'] = self._benchmark_data_pipeline_efficiency(iterations)  # type: ignore[attr-defined]

        # 3. Memory Transfer Tests
        logger.info("Testing CPU-GPU memory transfers...")
        combined_results['memory_transfer'] = self._benchmark_memory_transfer_efficiency(iterations)  # type: ignore[attr-defined]

        return combined_results

    def _benchmark_gpu_compute_comprehensive(self, iterations: int) -> Dict[str, Any]:
        """Comprehensive GPU compute benchmarking across all supported precisions."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        if not self.system_capabilities.gpu_model:
            return {}

        # Scale matrix sizes based on GPU memory
        memory_gb = self.system_capabilities.gpu_memory_gb
        assert memory_gb is not None, "GPU memory info must be available"
        matrix_sizes = []

        if memory_gb >= 80:
            matrix_sizes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)]
        elif memory_gb >= 40:
            matrix_sizes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
        elif memory_gb >= 20:
            matrix_sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
        elif memory_gb >= 8:
            matrix_sizes = [(512, 512, 512), (1024, 1024, 1024)]
        else:
            matrix_sizes = [(256, 256, 256), (512, 512, 512)]

        # Determine available precisions
        precisions = ['fp32']
        if self.system_capabilities.supports_tf32:
            precisions.append('tf32')
        if self.system_capabilities.supports_fp16:
            precisions.append('fp16')
        if self.system_capabilities.supports_gpu_bf16:
            precisions.append('bf16')
        if self.system_capabilities.supports_fp8:
            precisions.append('fp8')

        results = {
            'matrix_sizes': matrix_sizes,
            'available_precisions': precisions,
            'results': {}
        }

        for precision in precisions:
            precision_results = []

            for m, n, k in matrix_sizes:
                try:
                    # Create matrices based on precision
                    if precision == 'fp32':
                        a = torch.randn(m, k, dtype=torch.float32, device=self.device)
                        b = torch.randn(k, n, dtype=torch.float32, device=self.device)
                        torch.backends.cuda.matmul.allow_tf32 = False
                    elif precision == 'tf32':
                        a = torch.randn(m, k, dtype=torch.float32, device=self.device)
                        b = torch.randn(k, n, dtype=torch.float32, device=self.device)
                        torch.backends.cuda.matmul.allow_tf32 = True
                    elif precision == 'fp16':
                        a = torch.randn(m, k, dtype=torch.float16, device=self.device)
                        b = torch.randn(k, n, dtype=torch.float16, device=self.device)
                    elif precision == 'bf16':
                        a = torch.randn(m, k, dtype=torch.bfloat16, device=self.device)
                        b = torch.randn(k, n, dtype=torch.bfloat16, device=self.device)
                    elif precision == 'fp8':
                        # FP8 typically requires special handling - fallback to FP16 for now
                        a = torch.randn(m, k, dtype=torch.float16, device=self.device)
                        b = torch.randn(k, n, dtype=torch.float16, device=self.device)
                    else:
                        # Default to fp32 for unknown precision
                        a = torch.randn(m, k, dtype=torch.float32, device=self.device)
                        b = torch.randn(k, n, dtype=torch.float32, device=self.device)

                    # Warmup
                    for _ in range(5):
                        _ = torch.matmul(a, b)

                    # Benchmark
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    for _ in range(iterations):
                        _ = torch.matmul(a, b)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed_time = time.perf_counter() - start_time

                    # Calculate TFLOPS
                    flops = 2 * m * n * k * iterations
                    tflops = flops / elapsed_time / 1e12

                    # Calculate efficiency
                    theoretical_peak = self.system_capabilities.theoretical_fp32_tflops or 1.0
                    if precision in ['fp16', 'bf16', 'fp8'] and self.system_capabilities.theoretical_tensor_tflops:
                        theoretical_peak = self.system_capabilities.theoretical_tensor_tflops

                    efficiency = (tflops / theoretical_peak * 100) if theoretical_peak else 0

                    precision_results.append({
                        'matrix_size': (m, n, k),
                        'elapsed_time': elapsed_time,
                        'tflops': tflops,
                        'theoretical_peak_tflops': theoretical_peak,
                        'efficiency_percent': efficiency,
                        'memory_usage_gb': (m * k + k * n) * 4 / (1024**3)  # Rough estimate
                    })

                    logger.info(f"  {precision} {m}x{n}x{k}: {tflops:.2f} TFLOPS ({efficiency:.1f}% efficiency)")

                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM: {precision} {m}x{n}x{k}")
                    continue
                except Exception as e:
                    logger.warning(f"Error in {precision} {m}x{n}x{k}: {e}")
                    continue

            results['results'][precision] = precision_results

        return results

    def _benchmark_gpu_memory_comprehensive(self, iterations: int) -> Dict[str, Any]:
        """Comprehensive GPU memory bandwidth benchmarking."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        if not self.system_capabilities.gpu_model:
            return {}

        memory_gb = self.system_capabilities.gpu_memory_gb
        assert memory_gb is not None, "GPU memory info must be available"
        max_memory_mb = int(memory_gb * 1024 * 0.8)  # Use 80% of available memory

        # Test different memory sizes
        memory_sizes = [10, 100, 500, 1000, 2000, 5000, 10000]
        memory_sizes = [size for size in memory_sizes if size <= max_memory_mb]

        results = {'memory_sizes_mb': memory_sizes, 'results': []}

        for memory_mb in memory_sizes:
            try:
                # Test different data types
                for dtype_name, dtype in [('fp32', torch.float32), ('fp16', torch.float16)]:
                    if dtype_name == 'fp16' and not self.system_capabilities.supports_fp16:
                        continue

                    element_size = 4 if dtype == torch.float32 else 2
                    elements = (memory_mb * 1024 * 1024) // element_size

                    src = torch.randn(elements, dtype=dtype, device=self.device)
                    dst = torch.empty_like(src)

                    # Warmup
                    for _ in range(5):
                        dst.copy_(src)

                    # Benchmark
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    for _ in range(iterations):
                        dst.copy_(src)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed_time = time.perf_counter() - start_time

                    # Calculate bandwidth
                    bytes_transferred = memory_mb * 1024 * 1024 * 2 * iterations  # read + write
                    bandwidth_gbps = bytes_transferred / elapsed_time / (1024 ** 3)

                    # Calculate efficiency
                    theoretical_bandwidth = self.system_capabilities.gpu_memory_bandwidth_gbps
                    efficiency = (bandwidth_gbps / theoretical_bandwidth * 100) if theoretical_bandwidth else 0

                    results['results'].append({
                        'memory_size_mb': memory_mb,
                        'data_type': dtype_name,
                        'bandwidth_gbps': bandwidth_gbps,
                        'theoretical_bandwidth_gbps': theoretical_bandwidth,
                        'efficiency_percent': efficiency,
                    })

                    logger.info(f"  {memory_mb}MB {dtype_name}: {bandwidth_gbps:.1f} GB/s ({efficiency:.1f}% efficiency)")

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM: {memory_mb}MB memory test")
                continue
            except Exception as e:
                logger.warning(f"Error in memory test {memory_mb}MB: {e}")
                continue

        return results

    def _benchmark_attention_mechanisms(self, iterations: int) -> Dict[str, Any]:
        """Benchmark attention mechanism performance."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        if not self.system_capabilities.supports_flash_attention:
            return {}

        sequence_lengths = [seq for seq in self.system_capabilities.optimal_sequence_lengths
                          if seq <= 4096][:4]  # Test up to 4 sequence lengths
        batch_sizes = [2, 4, 8, 16]

        results = {'configurations': [], 'results': []}

        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                try:
                    # Skip if too large for memory
                    memory_estimate = batch_size * seq_len * 768 * 4 / (1024**3)  # Rough estimate
                    gpu_mem = self.system_capabilities.gpu_memory_gb or 8
                    if memory_estimate > gpu_mem * 0.8:
                        continue

                    config = {'batch_size': batch_size, 'sequence_length': seq_len}
                    results['configurations'].append(config)

                    # Standard multi-head attention
                    num_heads = 12
                    head_dim = 64
                    embed_dim = num_heads * head_dim

                    attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).to(self.device)
                    x = torch.randn(batch_size, seq_len, embed_dim, device=self.device)

                    # Warmup
                    for _ in range(3):
                        with torch.no_grad():
                            _ = attn(x, x, x)

                    # Benchmark
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    for _ in range(iterations):
                        with torch.no_grad():
                            _ = attn(x, x, x)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed_time = time.perf_counter() - start_time

                    # Calculate metrics
                    tokens_per_second = (batch_size * seq_len * iterations) / elapsed_time
                    memory_usage = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0

                    results['results'].append({
                        'batch_size': batch_size,
                        'sequence_length': seq_len,
                        'elapsed_time': elapsed_time,
                        'tokens_per_second': tokens_per_second,
                        'memory_usage_gb': memory_usage,
                        'throughput_score': tokens_per_second / 1000  # Normalized score
                    })

                    logger.info(f"  Batch={batch_size}, Seq={seq_len}: {tokens_per_second:.0f} tokens/s")

                    del attn, x
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM: batch_size={batch_size}, seq_len={seq_len}")
                    continue
                except Exception as e:
                    logger.warning(f"Error in attention test: {e}")
                    continue

        return results

    def _benchmark_mixed_precision_training(self, iterations: int) -> Dict[str, Any]:
        """Benchmark mixed precision training performance."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        # Create a more realistic transformer model
        class TransformerBlock(nn.Module):
            def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
                super().__init__()
                self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
                self.norm1 = nn.LayerNorm(embed_dim)
                self.ff = nn.Sequential(
                    nn.Linear(embed_dim, ff_dim),
                    nn.GELU(),
                    nn.Linear(ff_dim, embed_dim)
                )
                self.norm2 = nn.LayerNorm(embed_dim)

            def forward(self, x):
                # Self-attention
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_out)

                # Feed-forward
                ff_out = self.ff(x)
                x = self.norm2(x + ff_out)
                return x

        # Scale model based on GPU memory
        memory_gb = self.system_capabilities.gpu_memory_gb or 8

        if memory_gb >= 40:
            embed_dim, num_heads, ff_dim, num_layers = 1024, 16, 4096, 8
            batch_size, seq_len = 16, 1024
        elif memory_gb >= 20:
            embed_dim, num_heads, ff_dim, num_layers = 768, 12, 3072, 6
            batch_size, seq_len = 12, 512
        elif memory_gb >= 10:
            embed_dim, num_heads, ff_dim, num_layers = 512, 8, 2048, 4
            batch_size, seq_len = 8, 512
        else:
            embed_dim, num_heads, ff_dim, num_layers = 256, 4, 1024, 2
            batch_size, seq_len = 4, 256

        # Available precisions
        precisions = ['fp32']
        if self.device.type == 'cuda':
            if self.system_capabilities.supports_fp16:
                precisions.append('fp16')
            if self.system_capabilities.supports_gpu_bf16:
                precisions.append('bf16')

        results = {
            'model_config': {
                'embed_dim': embed_dim, 'num_heads': num_heads,
                'ff_dim': ff_dim, 'num_layers': num_layers,
                'batch_size': batch_size, 'seq_len': seq_len
            },
            'results': {}
        }

        for precision in precisions:
            try:
                # Create model
                model = nn.Sequential(*[
                    TransformerBlock(embed_dim, num_heads, ff_dim)
                    for _ in range(num_layers)
                ]).to(self.device)

                if precision == 'fp16':
                    model = model.half()
                elif precision == 'bf16' and hasattr(torch, 'bfloat16'):
                    model = model.to(torch.bfloat16)

                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

                # Sample data
                vocab_size = 32000
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
                # Embedding layer
                embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)
                if precision == 'fp16':
                    embedding = embedding.half()
                elif precision == 'bf16' and hasattr(torch, 'bfloat16'):
                    embedding = embedding.to(torch.bfloat16)

                # Warmup
                for _ in range(3):
                    optimizer.zero_grad()
                    x = embedding(input_ids)

                    if self.device.type == 'cuda' and precision != 'fp32':
                        with torch.cuda.amp.autocast():
                            outputs = model(x)
                            # Simple loss for benchmarking
                            loss = outputs.mean()
                        if scaler:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()
                    else:
                        outputs = model(x)
                        loss = outputs.mean()
                        loss.backward()
                        optimizer.step()

                # Benchmark
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()

                start_time = time.perf_counter()

                for _ in range(iterations):
                    optimizer.zero_grad()
                    x = embedding(input_ids)

                    if self.device.type == 'cuda' and precision != 'fp32':
                        with torch.cuda.amp.autocast():
                            outputs = model(x)
                            loss = outputs.mean()
                        if scaler:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()
                    else:
                        outputs = model(x)
                        loss = outputs.mean()
                        loss.backward()
                        optimizer.step()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_time = time.perf_counter() - start_time

                # Calculate metrics
                steps_per_second = iterations / elapsed_time
                tokens_per_second = steps_per_second * batch_size * seq_len
                max_memory_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0

                results['results'][precision] = {
                    'elapsed_time': elapsed_time,
                    'steps_per_second': steps_per_second,
                    'tokens_per_second': tokens_per_second,
                    'max_memory_gb': max_memory_gb,
                    'parameters': sum(p.numel() for p in model.parameters()),
                }

                logger.info(f"  {precision}: {steps_per_second:.1f} steps/s, {max_memory_gb:.1f}GB")

                del model, optimizer, embedding
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM: {precision} training")
                continue
            except Exception as e:
                logger.warning(f"Error in {precision} training: {e}")
                continue

        return results

    def _benchmark_ai_workloads(self, iterations: int) -> Dict[str, Any]:
        """Benchmark AI-specific workloads."""
        workloads = {}

        # 1. Convolutional Neural Network (Vision)
        workloads['cnn_inference'] = self._benchmark_cnn_workload(iterations)

        # 2. Language Model Inference
        workloads['language_model'] = self._benchmark_language_model_workload(iterations)

        # 3. Embedding Operations
        workloads['embedding_operations'] = self._benchmark_embedding_workload(iterations)

        return workloads

    def _benchmark_cnn_workload(self, iterations: int) -> Dict[str, Any]:
        """Benchmark CNN inference workload."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        try:
            # Create ResNet-like model
            model = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),

                # Residual blocks
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(128 * 7 * 7, 1000)
            ).to(self.device)

            # Scale batch size based on memory
            memory_gb = self.system_capabilities.gpu_memory_gb or 4
            batch_size = min(64, max(1, int(memory_gb * 8)))

            input_tensor = torch.randn(batch_size, 3, 224, 224, device=self.device)

            # Warmup
            model.eval()
            with torch.no_grad():
                for _ in range(5):
                    _ = model(input_tensor)

            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(input_tensor)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time

            images_per_second = (batch_size * iterations) / elapsed_time

            return {
                'batch_size': batch_size,
                'images_per_second': images_per_second,
                'elapsed_time': elapsed_time,
                'model_parameters': sum(p.numel() for p in model.parameters())
            }

        except Exception as e:
            logger.warning(f"CNN benchmark failed: {e}")
            return {'error': str(e)}

    def _benchmark_language_model_workload(self, iterations: int) -> Dict[str, Any]:
        """Benchmark language model inference workload."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        try:
            # Simple transformer decoder
            embed_dim = 512
            num_heads = 8
            ff_dim = 2048
            vocab_size = 32000

            model = nn.Sequential(
                nn.Embedding(vocab_size, embed_dim),
                nn.TransformerDecoderLayer(embed_dim, num_heads, ff_dim, batch_first=True),
                nn.TransformerDecoderLayer(embed_dim, num_heads, ff_dim, batch_first=True),
                nn.Linear(embed_dim, vocab_size)
            ).to(self.device)

            # Scale based on memory
            memory_gb = self.system_capabilities.gpu_memory_gb or 4
            batch_size = min(32, max(1, int(memory_gb * 4)))
            seq_len = min(512, max(64, int(memory_gb * 64)))

            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)

            # Warmup
            model.eval()
            with torch.no_grad():
                for _ in range(3):
                    _ = model(input_ids)

            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(input_ids)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time

            tokens_per_second = (batch_size * seq_len * iterations) / elapsed_time

            return {
                'batch_size': batch_size,
                'sequence_length': seq_len,
                'tokens_per_second': tokens_per_second,
                'elapsed_time': elapsed_time
            }

        except Exception as e:
            logger.warning(f"Language model benchmark failed: {e}")
            return {'error': str(e)}

    def _benchmark_embedding_workload(self, iterations: int) -> Dict[str, Any]:
        """Benchmark embedding operations."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        try:
            vocab_size = 100000
            embed_dim = 768

            embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)

            # Scale based on memory
            memory_gb = self.system_capabilities.gpu_memory_gb or 4
            batch_size = min(512, max(16, int(memory_gb * 32)))
            seq_len = min(1024, max(128, int(memory_gb * 128)))

            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = embedding(input_ids)

            # Benchmark
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                for _ in range(iterations):
                    _ = embedding(input_ids)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed_time = time.perf_counter() - start_time

            lookups_per_second = (batch_size * seq_len * iterations) / elapsed_time

            return {
                'vocab_size': vocab_size,
                'embed_dim': embed_dim,
                'batch_size': batch_size,
                'sequence_length': seq_len,
                'lookups_per_second': lookups_per_second,
                'elapsed_time': elapsed_time
            }

        except Exception as e:
            logger.warning(f"Embedding benchmark failed: {e}")
            return {'error': str(e)}

    def _benchmark_dataloader_performance(self, iterations: int) -> Dict[str, Any]:
        """Benchmark DataLoader performance with different configurations."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        # Dummy dataset for testing
        class DummyDataset(Dataset):
            def __init__(self, size: int, data_complexity: int = 1024):
                self.size = size
                self.data_complexity = data_complexity

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                # Simulate data loading work with varying complexity
                data = torch.randn(self.data_complexity)
                label = torch.randint(0, 10, (1,))

                # Add some CPU preprocessing work
                data = F.relu(data + 0.1 * torch.randn_like(data))
                data = data / (data.norm() + 1e-8)  # Normalize

                return data, label

        dataset = DummyDataset(5000)

        # Test different worker configurations
        max_workers = min(self.system_capabilities.optimal_dataloader_workers, 16)
        worker_configs = [0, 1, 2, 4, max_workers]
        worker_configs = sorted(list(set(worker_configs)))  # Remove duplicates

        results = {'worker_configs': worker_configs, 'results': []}

        for num_workers in worker_configs:
            try:
                dataloader = DataLoader(
                    dataset,
                    batch_size=32,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=num_workers > 0,
                    prefetch_factor=2 if num_workers > 0 else 2
                )

                # Warmup
                for i, (data, labels) in enumerate(dataloader):
                    if i >= 3:
                        break

                # Benchmark
                start_time = time.perf_counter()
                batch_count = 0

                for i, (data, labels) in enumerate(dataloader):
                    batch_count += 1
                    if batch_count >= iterations:
                        break

                elapsed_time = time.perf_counter() - start_time
                if elapsed_time > 0:
                    batches_per_second = batch_count / elapsed_time
                    samples_per_second = batches_per_second * 32
                else:
                    batches_per_second = 0
                    samples_per_second = 0

                results['results'].append({
                    'num_workers': num_workers,
                    'elapsed_time': elapsed_time,
                    'batches_per_second': batches_per_second,
                    'samples_per_second': samples_per_second,
                    'batch_count': batch_count,
                    'efficiency_score': samples_per_second / max(1, num_workers + 1)
                })

                logger.info(f"  {num_workers} workers: {samples_per_second:.1f} samples/s")

            except Exception as e:
                logger.warning(f"DataLoader test failed for {num_workers} workers: {e}")
                continue

        return results

    def _benchmark_cpu_training_performance(self, iterations: int) -> Dict[str, Any]:
        """Benchmark CPU training performance with different thread configurations."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        # Simple model for CPU training
        class CPUTrainingModel(nn.Module):
            def __init__(self, input_size: int = 1024, hidden_sizes: List[int] = [512, 256, 128],
                         num_classes: int = 10):
                super().__init__()
                layers = []
                prev_size = input_size

                for hidden_size in hidden_sizes:
                    layers.extend([
                        nn.Linear(prev_size, hidden_size),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_size),
                        nn.Dropout(0.1)
                    ])
                    prev_size = hidden_size

                layers.append(nn.Linear(prev_size, num_classes))
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        # Test different thread configurations
        max_threads = self.system_capabilities.optimal_torch_threads or 4
        thread_configs = [1, 2, 4, max_threads]
        if max_threads > 8:
            thread_configs.append(max_threads // 2)
        thread_configs = sorted(list(set(thread_configs)))

        results = {'thread_configs': thread_configs, 'results': []}

        original_threads = torch.get_num_threads()

        for num_threads in thread_configs:
            model = None
            optimizer = None
            try:
                # Set thread count
                torch.set_num_threads(num_threads)

                # Create model and optimizer
                model = CPUTrainingModel()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                criterion = nn.CrossEntropyLoss()

                # Scale batch size based on system memory and threads
                memory_gb = self.system_capabilities.system_memory_gb or 8
                base_batch_size = min(128, max(16, int(memory_gb * 4)))
                batch_size = min(base_batch_size, max(8, base_batch_size // (num_threads // 2 + 1)))

                # Training data
                input_data = torch.randn(batch_size, 1024)
                labels = torch.randint(0, 10, (batch_size,))

                # Warmup
                for _ in range(5):
                    optimizer.zero_grad()
                    outputs = model(input_data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # Benchmark
                start_time = time.perf_counter()

                for _ in range(iterations):
                    optimizer.zero_grad()
                    outputs = model(input_data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                elapsed_time = time.perf_counter() - start_time
                steps_per_second = iterations / elapsed_time
                samples_per_second = steps_per_second * batch_size

                results['results'].append({
                    'num_threads': num_threads,
                    'batch_size': batch_size,
                    'elapsed_time': elapsed_time,
                    'steps_per_second': steps_per_second,
                    'samples_per_second': samples_per_second,
                    'thread_efficiency': steps_per_second / num_threads
                })

                logger.info(f"  {num_threads} threads: {steps_per_second:.1f} steps/s")

            except Exception as e:
                logger.warning(f"CPU training test failed for {num_threads} threads: {e}")
                continue
            finally:
                # Cleanup
                if model is not None:
                    del model
                if optimizer is not None:
                    del optimizer

        # Restore original thread count
        torch.set_num_threads(original_threads)
        return results

    def _benchmark_threading_performance(self, iterations: int) -> Dict[str, Any]:
        """Benchmark multi-threading performance for CPU-intensive tasks."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        def cpu_intensive_task(task_size: int) -> float:
            """CPU-intensive computation task."""
            result = 0.0
            for i in range(task_size):
                result += math.sin(i * 0.001) * math.cos(i * 0.001) + math.sqrt(i + 1)
            return result

        # Test different thread configurations
        max_threads = min(self.system_capabilities.logical_cores, 32)
        thread_configs = [1, 2, 4, 8, max_threads]
        if max_threads > 16:
            thread_configs.append(max_threads // 2)
        thread_configs = sorted(list(set(thread_configs)))

        results = {'thread_configs': thread_configs, 'results': []}

        task_size = 100000  # Base task size per thread

        for num_threads in thread_configs:
            try:
                # Distribute work across threads
                work_per_thread = task_size

                # Single-threaded baseline
                if num_threads == 1:
                    start_time = time.perf_counter()
                    for _ in range(iterations):
                        cpu_intensive_task(work_per_thread)
                    elapsed_time = time.perf_counter() - start_time
                else:
                    # Multi-threaded execution
                    start_time = time.perf_counter()

                    for _ in range(iterations):
                        with ThreadPoolExecutor(max_workers=num_threads) as executor:
                            futures = [executor.submit(cpu_intensive_task, work_per_thread)
                                     for _ in range(num_threads)]
                            results_list = [f.result() for f in futures]

                    elapsed_time = time.perf_counter() - start_time

                # Calculate metrics
                total_operations = iterations * num_threads * work_per_thread
                operations_per_second = total_operations / elapsed_time
                thread_efficiency = operations_per_second / (num_threads * (task_size / elapsed_time * iterations))

                results['results'].append({
                    'num_threads': num_threads,
                    'elapsed_time': elapsed_time,
                    'operations_per_second': operations_per_second,
                    'thread_efficiency': min(thread_efficiency, 1.0),
                    'scalability_factor': operations_per_second / (iterations * task_size / elapsed_time) if num_threads == 1 else None
                })

                logger.info(f"  {num_threads} threads: {operations_per_second:.0f} ops/s")

            except Exception as e:
                logger.warning(f"Threading test failed for {num_threads} threads: {e}")
                continue

        return results

    def _benchmark_memory_intensive_workloads(self, iterations: int) -> Dict[str, Any]:
        """Benchmark memory-intensive workloads."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        workloads = {}

        # 1. Large matrix operations
        workloads['matrix_operations'] = self._benchmark_matrix_operations(iterations)

        # 2. Memory bandwidth stress test
        workloads['memory_bandwidth_stress'] = self._benchmark_memory_bandwidth_stress(iterations)

        # 3. Cache performance test
        workloads['cache_performance'] = self._benchmark_cache_performance(iterations)

        return workloads

    def _benchmark_matrix_operations(self, iterations: int) -> Dict[str, Any]:
        """Benchmark large matrix operations on CPU."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        try:
            # Scale matrix size based on available memory
            memory_gb = self.system_capabilities.system_memory_gb
            max_size = min(4096, int(math.sqrt(memory_gb * 1024 * 1024 * 100)))  # Conservative sizing

            matrix_sizes = [512, 1024, 2048]
            if max_size >= 4096:
                matrix_sizes.append(4096)

            results = {'matrix_sizes': matrix_sizes, 'results': []}

            for size in matrix_sizes:
                try:
                    # Create random matrices
                    a = np.random.random((size, size)).astype(np.float32)
                    b = np.random.random((size, size)).astype(np.float32)

                    # Warmup
                    for _ in range(2):
                        _ = np.dot(a, b)

                    # Benchmark
                    start_time = time.perf_counter()

                    for _ in range(iterations):
                        c = np.dot(a, b)

                    elapsed_time = time.perf_counter() - start_time

                    # Calculate GFLOPS
                    flops = 2 * size * size * size * iterations  # Matrix multiply FLOPS
                    gflops = flops / elapsed_time / 1e9

                    # Memory usage estimate
                    memory_usage_gb = (3 * size * size * 4) / (1024**3)  # 3 matrices * float32

                    results['results'].append({
                        'matrix_size': size,
                        'elapsed_time': elapsed_time,
                        'gflops': gflops,
                        'memory_usage_gb': memory_usage_gb,
                        'efficiency_score': gflops / memory_usage_gb
                    })

                    logger.info(f"  Matrix {size}x{size}: {gflops:.1f} GFLOPS")

                except MemoryError:
                    logger.warning(f"Memory error for matrix size {size}x{size}")
                    continue

            return results

        except Exception as e:
            logger.warning(f"Matrix operations benchmark failed: {e}")
            return {'error': str(e)}

    def _benchmark_memory_bandwidth_stress(self, iterations: int) -> Dict[str, Any]:
        """Stress test memory bandwidth with large arrays."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        try:
            # Use up to 50% of available memory
            memory_gb = self.system_capabilities.system_memory_gb
            max_array_gb = min(memory_gb * 0.5, 8)  # Cap at 8GB

            array_sizes_gb = [0.1, 0.5, 1.0, 2.0]
            array_sizes_gb = [size for size in array_sizes_gb if size <= max_array_gb]

            results = {'array_sizes_gb': array_sizes_gb, 'results': []}

            for array_gb in array_sizes_gb:
                try:
                    # Create large arrays
                    elements = int(array_gb * 1024**3 / 4)  # float32 elements

                    src_array = np.random.random(elements).astype(np.float32)
                    dst_array = np.zeros_like(src_array)

                    # Warmup
                    for _ in range(2):
                        np.copyto(dst_array, src_array)

                    # Benchmark
                    start_time = time.perf_counter()

                    for _ in range(iterations):
                        np.copyto(dst_array, src_array)

                    elapsed_time = time.perf_counter() - start_time

                    # Calculate bandwidth
                    bytes_transferred = array_gb * 1024**3 * 2 * iterations  # read + write
                    bandwidth_gbps = bytes_transferred / elapsed_time / (1024**3)

                    # Compare to theoretical bandwidth
                    theoretical_bandwidth = self.system_capabilities.memory_bandwidth_theoretical_gbps
                    efficiency = (bandwidth_gbps / theoretical_bandwidth * 100) if theoretical_bandwidth else 0

                    results['results'].append({
                        'array_size_gb': array_gb,
                        'elapsed_time': elapsed_time,
                        'bandwidth_gbps': bandwidth_gbps,
                        'theoretical_bandwidth_gbps': theoretical_bandwidth,
                        'efficiency_percent': efficiency
                    })

                    logger.info(f"  {array_gb:.1f}GB array: {bandwidth_gbps:.1f} GB/s")

                except MemoryError:
                    logger.warning(f"Memory error for {array_gb:.1f}GB array")
                    continue

            return results

        except Exception as e:
            logger.warning(f"Memory bandwidth stress test failed: {e}")
            return {'error': str(e)}

    def _benchmark_cache_performance(self, iterations: int) -> Dict[str, Any]:
        """Benchmark cache performance with different access patterns."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        try:
            l3_cache_kb = self.system_capabilities.cpu_cache_l3_kb

            # Test different array sizes relative to cache
            test_sizes = [
                ('L1_fit', l3_cache_kb // 32),      # Should fit in L1
                ('L2_fit', l3_cache_kb // 4),       # Should fit in L2
                ('L3_fit', l3_cache_kb),            # Should fit in L3
                ('L3_exceed', l3_cache_kb * 4)      # Exceeds L3
            ]

            results = {'test_configurations': [], 'results': []}

            for test_name, size_kb in test_sizes:
                try:
                    elements = (size_kb * 1024) // 4  # float32 elements
                    test_array = np.random.random(elements).astype(np.float32)

                    results['test_configurations'].append({
                        'test_name': test_name,
                        'size_kb': size_kb,
                        'elements': elements
                    })

                    # Sequential access pattern
                    start_time = time.perf_counter()
                    for _ in range(iterations):
                        result = np.sum(test_array)
                    seq_time = time.perf_counter() - start_time

                    # Random access pattern
                    indices = np.random.randint(0, elements, min(elements // 10, 10000))
                    start_time = time.perf_counter()
                    for _ in range(iterations):
                        result = np.sum(test_array[indices])
                    random_time = time.perf_counter() - start_time

                    # Calculate access rates
                    seq_access_rate = (elements * iterations) / seq_time
                    random_access_rate = (len(indices) * iterations) / random_time

                    results['results'].append({
                        'test_name': test_name,
                        'size_kb': size_kb,
                        'sequential_time': seq_time,
                        'random_time': random_time,
                        'sequential_access_rate': seq_access_rate,
                        'random_access_rate': random_access_rate,
                        'random_penalty_ratio': seq_time / random_time if random_time > 0 else 0
                    })

                    logger.info(f"  {test_name}: seq={seq_access_rate:.0f} rand={random_access_rate:.0f} access/s")

                except MemoryError:
                    logger.warning(f"Memory error for {test_name} test")
                    continue

            return results

        except Exception as e:
            logger.warning(f"Cache performance benchmark failed: {e}")
            return {'error': str(e)}

    def _benchmark_cpu_ai_acceleration(self, iterations: int) -> Dict[str, Any]:
        """Benchmark CPU AI acceleration features (AVX, AMX, etc.)."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        results = {}

        # Test different precision modes if supported
        if self.system_capabilities.supports_cpu_bf16:
            results['bf16_performance'] = self._benchmark_cpu_bf16(iterations)

        if self.system_capabilities.supports_amx:
            results['amx_performance'] = self._benchmark_cpu_amx(iterations)

        # Vector instruction performance
        results['vector_instructions'] = self._benchmark_vector_instructions(iterations)

        return results

    def _benchmark_cpu_bf16(self, iterations: int) -> Dict[str, Any]:
        """Benchmark BF16 performance on CPU."""
        try:
            # Simple model for BF16 testing
            model_fp32 = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )

            batch_size = 32
            input_data = torch.randn(batch_size, 512)

            # FP32 baseline
            start_time = time.perf_counter()
            for _ in range(iterations):
                with torch.no_grad():
                    _ = model_fp32(input_data)
            fp32_time = time.perf_counter() - start_time

            # BF16 test
            if hasattr(torch, 'bfloat16'):
                model_bf16 = model_fp32.to(torch.bfloat16)
                input_bf16 = input_data.to(torch.bfloat16)

                start_time = time.perf_counter()
                for _ in range(iterations):
                    with torch.no_grad():
                        _ = model_bf16(input_bf16)
                bf16_time = time.perf_counter() - start_time

                speedup = fp32_time / bf16_time if bf16_time > 0 else 1.0
            else:
                bf16_time = fp32_time
                speedup = 1.0

            return {
                'fp32_time': fp32_time,
                'bf16_time': bf16_time,
                'speedup': speedup,
                'supported': hasattr(torch, 'bfloat16')
            }

        except Exception as e:
            logger.warning(f"BF16 benchmark failed: {e}")
            return {'error': str(e)}

    def _benchmark_cpu_amx(self, iterations: int) -> Dict[str, Any]:
        """Benchmark Intel AMX performance if available."""
        try:
            # AMX is typically used for matrix operations
            # This is a simplified test since AMX requires specific setup

            matrix_size = 256
            a = torch.randn(matrix_size, matrix_size)
            b = torch.randn(matrix_size, matrix_size)

            # Standard matrix multiply
            start_time = time.perf_counter()
            for _ in range(iterations):
                _ = torch.matmul(a, b)
            standard_time = time.perf_counter() - start_time

            # Note: Real AMX usage would require specific kernels
            # This is a placeholder for AMX-optimized operations
            amx_time = standard_time  # Would be faster with real AMX

            return {
                'standard_time': standard_time,
                'amx_time': amx_time,
                'matrix_size': matrix_size,
                'theoretical_speedup': 2.0,  # AMX can provide ~2x speedup
                'note': 'AMX requires specialized kernels for full acceleration'
            }

        except Exception as e:
            logger.warning(f"AMX benchmark failed: {e}")
            return {'error': str(e)}

    def _benchmark_vector_instructions(self, iterations: int) -> Dict[str, Any]:
        """Benchmark vector instruction performance."""
        try:
            array_size = 100000

            # Test different vector operations
            operations = {
                'addition': lambda a, b: a + b,
                'multiplication': lambda a, b: a * b,
                'fma': lambda a, b: a * b + a,  # Fused multiply-add
                'sqrt': lambda a, b: np.sqrt(a),
                'exp': lambda a, b: np.exp(a * 0.1)  # Scaled to avoid overflow
            }

            a = np.random.random(array_size).astype(np.float32)
            b = np.random.random(array_size).astype(np.float32)

            results = {'operations': list(operations.keys()), 'results': []}

            for op_name, op_func in operations.items():
                try:
                    # Warmup
                    for _ in range(3):
                        _ = op_func(a, b)

                    # Benchmark
                    start_time = time.perf_counter()
                    for _ in range(iterations):
                        result = op_func(a, b)
                    elapsed_time = time.perf_counter() - start_time

                    operations_per_second = (array_size * iterations) / elapsed_time

                    results['results'].append({
                        'operation': op_name,
                        'elapsed_time': elapsed_time,
                        'operations_per_second': operations_per_second,
                        'throughput_score': operations_per_second / 1e6  # Normalized to millions
                    })

                    logger.info(f"  {op_name}: {operations_per_second/1e6:.1f}M ops/s")

                except Exception as e:
                    logger.warning(f"Vector operation {op_name} failed: {e}")
                    continue

            return results

        except Exception as e:
            logger.warning(f"Vector instructions benchmark failed: {e}")
            return {'error': str(e)}

    def _benchmark_load_balancing(self, iterations: int) -> Dict[str, Any]:
        """Benchmark CPU/GPU load balancing with different workload distributions."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        if not self.system_capabilities.gpu_model:
            return {'note': 'GPU not available for load balancing tests'}

        # Test different CPU/GPU split ratios
        split_ratios = [
            (0.0, 1.0),   # GPU only
            (0.2, 0.8),   # Mostly GPU
            (0.5, 0.5),   # Balanced
            (0.8, 0.2),   # Mostly CPU
            (1.0, 0.0)    # CPU only
        ]

        results = {'split_ratios': split_ratios, 'results': []}

        # Create workload that can be split between CPU and GPU
        class SplittableWorkload:
            def __init__(self, total_size: int, device: torch.device):
                self.total_size = total_size
                self.device = device

            def process_batch(self, batch_size: int, use_gpu: bool = True):
                """Process a batch of work on CPU or GPU."""
                device = self.device if use_gpu else torch.device('cpu')

                # Simple matrix operations representing AI workload
                data = torch.randn(batch_size, 512, device=device)
                weights = torch.randn(512, 256, device=device)

                # Simulate processing
                for _ in range(5):  # Multiple operations
                    data = torch.mm(data, weights)
                    data = torch.relu(data)
                    weights = weights.T

                return data.sum().item()

        workload = SplittableWorkload(1000, self.device)

        for cpu_ratio, gpu_ratio in split_ratios:
            try:
                total_work = 100  # Total batches to process
                cpu_work = int(total_work * cpu_ratio)
                gpu_work = total_work - cpu_work

                batch_size = 32

                # Measure processing time
                start_time = time.perf_counter()

                # Process CPU work
                if cpu_work > 0:
                    for _ in range(cpu_work):
                        _ = workload.process_batch(batch_size, use_gpu=False)

                # Process GPU work
                if gpu_work > 0:
                    for _ in range(gpu_work):
                        _ = workload.process_batch(batch_size, use_gpu=True)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed_time = time.perf_counter() - start_time

                # Calculate metrics
                total_samples = total_work * batch_size
                throughput = total_samples / elapsed_time

                # Estimate resource utilization
                cpu_utilization = cpu_ratio * 100
                gpu_utilization = gpu_ratio * 100

                results['results'].append({
                    'cpu_ratio': cpu_ratio,
                    'gpu_ratio': gpu_ratio,
                    'cpu_work_batches': cpu_work,
                    'gpu_work_batches': gpu_work,
                    'elapsed_time': elapsed_time,
                    'throughput_samples_per_sec': throughput,
                    'cpu_utilization_percent': cpu_utilization,
                    'gpu_utilization_percent': gpu_utilization,
                    'efficiency_score': throughput / (cpu_utilization + gpu_utilization) * 100
                })

                logger.info(f"  CPU={cpu_ratio:.1f}/GPU={gpu_ratio:.1f}: {throughput:.1f} samples/s")

            except Exception as e:
                logger.warning(f"Load balancing test failed for ratio {cpu_ratio:.1f}/{gpu_ratio:.1f}: {e}")
                continue

        return results

    def _benchmark_data_pipeline_efficiency(self, iterations: int) -> Dict[str, Any]:
        """Benchmark data pipeline efficiency with different configurations."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        # Test dataset for pipeline benchmarking
        class PipelineDataset(Dataset):
            def __init__(self, size: int, complexity: str = 'medium'):
                self.size = size
                self.complexity = complexity

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                if self.complexity == 'light':
                    # Light preprocessing
                    data = torch.randn(256)
                    label = torch.randint(0, 10, (1,))
                elif self.complexity == 'medium':
                    # Medium preprocessing
                    data = torch.randn(512)
                    data = F.normalize(data, p=2, dim=0)
                    data = data + 0.1 * torch.randn_like(data)  # Add noise
                    label = torch.randint(0, 100, (1,))
                else:  # heavy
                    # Heavy preprocessing
                    data = torch.randn(1024)
                    for _ in range(3):  # Multiple transformations
                        data = F.relu(data + 0.1 * torch.randn_like(data))
                        data = F.normalize(data, p=2, dim=0)
                    label = torch.randint(0, 1000, (1,))

                return data, label

            def get_data_size(self):
                return {'light': 256, 'medium': 512, 'heavy': 1024}[self.complexity]

            def get_num_classes(self):
                return {'light': 10, 'medium': 100, 'heavy': 1000}[self.complexity]

        # Test different pipeline configurations
        configurations = [
            {'workers': 0, 'batch_size': 16, 'complexity': 'light'},
            {'workers': 2, 'batch_size': 16, 'complexity': 'light'},
            {'workers': 4, 'batch_size': 32, 'complexity': 'medium'},
            {'workers': self.system_capabilities.optimal_dataloader_workers,
             'batch_size': 64, 'complexity': 'medium'},
            {'workers': min(8, self.system_capabilities.optimal_dataloader_workers),
             'batch_size': 32, 'complexity': 'heavy'}
        ]

        results = {'configurations': configurations, 'results': []}

        for config in configurations:
            try:
                dataset = PipelineDataset(2000, config['complexity'])
                dataloader = DataLoader(
                    dataset,
                    batch_size=config['batch_size'],
                    num_workers=config['workers'],
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=config['workers'] > 0,
                    prefetch_factor=2 if config['workers'] > 0 else 2
                )

                # Simple model for processing
                device = self.device
                model = nn.Sequential(
                    nn.Linear(dataset.get_data_size(), 128),
                    nn.ReLU(),
                    nn.Linear(128, dataset.get_num_classes())
                ).to(device)

                # Warmup
                warmup_count = 0
                for data, labels in dataloader:
                    data, labels = data.to(device), labels.to(device).squeeze()
                    with torch.no_grad():
                        _ = model(data)
                    warmup_count += 1
                    if warmup_count >= 3:
                        break

                # Benchmark
                start_time = time.perf_counter()
                batch_count = 0
                total_samples = 0

                for data, labels in dataloader:
                    data, labels = data.to(device), labels.to(device).squeeze()

                    with torch.no_grad():
                        _ = model(data)

                    batch_count += 1
                    total_samples += data.size(0)

                    if batch_count >= iterations:
                        break

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed_time = time.perf_counter() - start_time

                # Calculate metrics
                throughput = total_samples / elapsed_time if elapsed_time > 0 else 0
                batches_per_second = batch_count / elapsed_time if elapsed_time > 0 else 0

                results['results'].append({
                    'configuration': config,
                    'batch_count': batch_count,
                    'total_samples': total_samples,
                    'elapsed_time': elapsed_time,
                    'throughput_samples_per_sec': throughput,
                    'batches_per_second': batches_per_second,
                    'efficiency_score': throughput / max(1, config['workers'] + 1)
                })

                logger.info(f"  {config['workers']}w/bs{config['batch_size']}/{config['complexity']}: "
                          f"{throughput:.1f} samples/s")

            except Exception as e:
                logger.warning(f"Pipeline efficiency test failed for config {config}: {e}")
                continue

        return results

    def _benchmark_memory_transfer_efficiency(self, iterations: int) -> Dict[str, Any]:
        """Benchmark CPU-GPU memory transfer efficiency."""
        assert self.system_capabilities is not None, "System capabilities must be initialized"
        if not self.system_capabilities.gpu_model:
            return {'note': 'GPU not available for memory transfer tests'}

        # Test different data sizes for transfers
        gpu_memory_gb = self.system_capabilities.gpu_memory_gb or 8
        max_size_mb = min(gpu_memory_gb * 1024 * 0.1, 1000)  # Use up to 10% of GPU memory, max 1GB

        transfer_sizes_mb = [1, 10, 50, 100, 500]
        transfer_sizes_mb = [size for size in transfer_sizes_mb if size <= max_size_mb]

        results = {'transfer_sizes_mb': transfer_sizes_mb, 'results': []}

        for size_mb in transfer_sizes_mb:
            try:
                elements = (size_mb * 1024 * 1024) // 4  # float32 elements

                # Test different transfer patterns
                transfer_patterns = {
                    'cpu_to_gpu': lambda data: data.to(self.device),
                    'gpu_to_cpu': lambda data: data.cpu(),
                    'cpu_to_gpu_pinned': lambda data: data.pin_memory().to(self.device, non_blocking=True),
                    'gpu_to_cpu_async': lambda data: data.cpu()  # Simplified async
                }

                pattern_results = []

                for pattern_name, transfer_func in transfer_patterns.items():
                    try:
                        # Prepare data based on pattern
                        if 'gpu_to_cpu' in pattern_name:
                            # Start with data on GPU
                            data = torch.randn(elements, device=self.device)
                        else:
                            # Start with data on CPU
                            data = torch.randn(elements)

                        # Warmup
                        for _ in range(3):
                            if 'gpu_to_cpu' in pattern_name:
                                temp_data = torch.randn(elements, device=self.device)
                                _ = transfer_func(temp_data)
                            else:
                                temp_data = torch.randn(elements)
                                _ = transfer_func(temp_data)

                        # Benchmark
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        start_time = time.perf_counter()

                        for _ in range(iterations):
                            if 'gpu_to_cpu' in pattern_name:
                                test_data = torch.randn(elements, device=self.device)
                                _ = transfer_func(test_data)
                            else:
                                test_data = torch.randn(elements)
                                _ = transfer_func(test_data)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        elapsed_time = time.perf_counter() - start_time

                        # Calculate bandwidth
                        bytes_transferred = size_mb * 1024 * 1024 * iterations
                        bandwidth_gbps = bytes_transferred / elapsed_time / (1024**3)

                        pattern_results.append({
                            'pattern': pattern_name,
                            'elapsed_time': elapsed_time,
                            'bandwidth_gbps': bandwidth_gbps,
                            'transfer_rate_mb_per_sec': (size_mb * iterations) / elapsed_time
                        })

                        logger.info(f"  {size_mb}MB {pattern_name}: {bandwidth_gbps:.2f} GB/s")

                    except Exception as e:
                        logger.warning(f"Transfer pattern {pattern_name} failed: {e}")
                        continue

                results['results'].append({
                    'size_mb': size_mb,
                    'patterns': pattern_results
                })

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM: {size_mb}MB transfer test")
                continue
            except Exception as e:
                logger.warning(f"Memory transfer test failed for {size_mb}MB: {e}")
                continue

        return results


# ============================================================================
# Model Size and Training Time Estimation
# ============================================================================

class ModelCapacityAnalyzer:
    """Analyze what model sizes can be trained and estimate training times."""

    def __init__(self, capabilities: UnifiedSystemCapabilities):
        self.capabilities = capabilities

        # Model architecture templates with parameter counts
        self.model_templates = {
            # Language Models
            'gpt2_small': {'params': 124_000_000, 'layers': 12, 'hidden': 768, 'context': 1024},
            'gpt2_medium': {'params': 355_000_000, 'layers': 24, 'hidden': 1024, 'context': 1024},
            'gpt2_large': {'params': 774_000_000, 'layers': 36, 'hidden': 1280, 'context': 1024},
            'gpt2_xl': {'params': 1_558_000_000, 'layers': 48, 'hidden': 1600, 'context': 1024},

            # Modern LLMs
            'llama_7b': {'params': 7_000_000_000, 'layers': 32, 'hidden': 4096, 'context': 2048},
            'llama_13b': {'params': 13_000_000_000, 'layers': 40, 'hidden': 5120, 'context': 2048},
            'llama_30b': {'params': 30_000_000_000, 'layers': 60, 'hidden': 6656, 'context': 2048},
            'llama_65b': {'params': 65_000_000_000, 'layers': 80, 'hidden': 8192, 'context': 2048},

            # Vision Models
            'resnet50': {'params': 25_600_000, 'type': 'vision', 'input_size': (224, 224)},
            'resnet101': {'params': 44_500_000, 'type': 'vision', 'input_size': (224, 224)},
            'vit_base': {'params': 86_000_000, 'type': 'vision', 'input_size': (224, 224)},
            'vit_large': {'params': 307_000_000, 'type': 'vision', 'input_size': (224, 224)},

            # Multimodal
            'clip_base': {'params': 151_000_000, 'type': 'multimodal'},
            'clip_large': {'params': 428_000_000, 'type': 'multimodal'},
        }

    def analyze_model_capacity(self) -> Dict[str, Any]:
        """Analyze what models can be trained on this system."""
        analysis = {
            'system_limits': self._calculate_system_limits(),
            'trainable_models': {},
            'training_estimates': {},
            'scaling_recommendations': {}
        }

        # Analyze each model template
        for model_name, model_info in self.model_templates.items():
            capacity_analysis = self._analyze_single_model(model_name, model_info)

            if capacity_analysis['can_train']:
                analysis['trainable_models'][model_name] = capacity_analysis
                analysis['training_estimates'][model_name] = self._estimate_training_time(
                    model_name, model_info, capacity_analysis
                )

        # Generate scaling recommendations
        analysis['scaling_recommendations'] = self._generate_scaling_recommendations()

        return analysis

    def _calculate_system_limits(self) -> Dict[str, Any]:
        """Calculate system memory and compute limits."""
        limits = {}

        # GPU limits (if available)
        if self.capabilities.gpu_model:
            gpu_memory_gb = self.capabilities.gpu_memory_gb or 0

            # Account for CUDA overhead (~1-2GB), OS, and other processes
            available_gpu_memory = max(0, gpu_memory_gb - 2.0)

            # Memory breakdown for training
            # Model weights: 1x, Gradients: 1x, Optimizer states: 2x (Adam), Activations: variable
            memory_overhead_factor = 4.5  # Conservative estimate

            limits['gpu'] = {
                'total_memory_gb': gpu_memory_gb,
                'available_memory_gb': available_gpu_memory,
                'max_model_params_fp32': int(available_gpu_memory * 1e9 / (4 * memory_overhead_factor)),
                'max_model_params_fp16': int(available_gpu_memory * 1e9 / (2 * memory_overhead_factor)),
                'max_model_params_bf16': int(available_gpu_memory * 1e9 / (2 * memory_overhead_factor)),
                'compute_tflops': self.capabilities.actual_compute_tflops or 0,
                'tensor_tflops': self.capabilities.theoretical_tensor_tflops or 0,
            }

        # CPU limits
        system_memory_gb = self.capabilities.system_memory_gb or 16
        available_cpu_memory = max(0, system_memory_gb - 4.0)  # Reserve 4GB for OS

        limits['cpu'] = {
            'total_memory_gb': system_memory_gb,
            'available_memory_gb': available_cpu_memory,
            'max_model_params_fp32': int(available_cpu_memory * 1e9 / (4 * 2)),  # Less overhead on CPU
            'cores': self.capabilities.physical_cores,
            'threads': self.capabilities.logical_cores,
            'ai_score': self.capabilities.cpu_ai_score,
        }

        return limits

    def _analyze_single_model(self, model_name: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if a specific model can be trained."""
        analysis = {
            'model_name': model_name,
            'parameters': model_info['params'],
            'can_train': False,
            'recommended_setup': {},
            'memory_requirements': {},
            'batch_size_recommendations': {}
        }

        param_count = model_info['params']

        # Calculate memory requirements for different precisions
        precisions = ['fp32', 'fp16', 'bf16'] if self.capabilities.supports_gpu_bf16 else ['fp32', 'fp16']

        for precision in precisions:
            bytes_per_param = 4 if precision == 'fp32' else 2

            # Training memory calculation
            model_memory = param_count * bytes_per_param / 1e9  # GB
            gradients_memory = model_memory  # Same as model
            optimizer_memory = model_memory * 2  # Adam optimizer states

            base_training_memory = model_memory + gradients_memory + optimizer_memory

            analysis['memory_requirements'][precision] = {
                'model_weights_gb': model_memory,
                'gradients_gb': gradients_memory,
                'optimizer_states_gb': optimizer_memory,
                'base_training_gb': base_training_memory,
            }

        # Check if model fits on GPU
        if self.capabilities.gpu_model:
            gpu_limits = self._calculate_system_limits()['gpu']
            available_memory = gpu_limits['available_memory_gb']

            for precision in precisions:
                base_memory = analysis['memory_requirements'][precision]['base_training_gb']

                if base_memory <= available_memory * 0.8:  # Use 80% as safety margin
                    analysis['can_train'] = True

                    # Calculate optimal batch sizes
                    remaining_memory = available_memory - base_memory

                    # Estimate activation memory per sample (rough approximation)
                    if 'hidden' in model_info:
                        hidden_size = model_info['hidden']
                        context_length = model_info.get('context', 512)
                        activation_memory_per_sample = (hidden_size * context_length * 4) / 1e9  # GB
                    else:
                        activation_memory_per_sample = 0.1  # Default estimate for vision models

                    max_batch_size = int(remaining_memory / activation_memory_per_sample)
                    max_batch_size = min(max_batch_size, 256)  # Cap at reasonable limit

                    optimal_batch_sizes = []
                    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
                        if bs <= max_batch_size:
                            optimal_batch_sizes.append(bs)

                    analysis['batch_size_recommendations'][precision] = {
                        'max_batch_size': max_batch_size,
                        'optimal_batch_sizes': optimal_batch_sizes,
                        'memory_per_sample_gb': activation_memory_per_sample,
                    }

                    analysis['recommended_setup'] = {
                        'precision': precision,
                        'device': 'gpu',
                        'memory_usage_gb': base_memory,
                        'max_batch_size': max_batch_size,
                    }
                    break

        # If not trainable on GPU, check CPU
        if not analysis['can_train']:
            cpu_limits = self._calculate_system_limits()['cpu']
            available_memory = cpu_limits['available_memory_gb']

            # CPU training is more memory efficient (no gradients stored on device)
            cpu_memory_factor = 2.5  # Lower overhead for CPU training

            for precision in ['fp32']:  # CPU typically uses FP32
                model_memory = param_count * 4 / 1e9  # FP32
                total_memory = model_memory * cpu_memory_factor

                if total_memory <= available_memory * 0.8:
                    analysis['can_train'] = True
                    analysis['recommended_setup'] = {
                        'precision': precision,
                        'device': 'cpu',
                        'memory_usage_gb': total_memory,
                        'max_batch_size': min(32, int(available_memory / total_memory * 8)),
                    }
                    break

        return analysis

    def _estimate_training_time(self, model_name: str, model_info: Dict[str, Any],
                              capacity_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate training time for different scenarios using repository training components."""

        if not capacity_analysis['can_train']:
            return {'error': 'Model cannot be trained on this system'}

        param_count = model_info['params']
        setup = capacity_analysis['recommended_setup']

        # Enhanced training setup using repository components
        training_config = self._create_optimal_training_config(setup, model_info)

        # Base FLOPS calculation for transformer forward pass
        # Approximation: 6 * params * sequence_length for forward + backward
        sequence_length = model_info.get('context', 512)
        flops_per_sample = 6 * param_count * sequence_length

        # Get system performance with training optimizations
        if setup['device'] == 'gpu' and self.capabilities.actual_compute_tflops:
            peak_tflops = self.capabilities.actual_compute_tflops
            if setup['precision'] in ['fp16', 'bf16'] and self.capabilities.theoretical_tensor_tflops:
                peak_tflops = self.capabilities.theoretical_tensor_tflops

            # Apply FP8 speedup if available
            if (setup['precision'] == 'fp8' and
                TRAINING_COMPONENTS_AVAILABLE and
                self.capabilities.supports_fp8):
                peak_tflops *= 1.8  # FP8 can provide ~1.8x speedup over FP16
        else:
            # CPU performance (very rough estimate)
            peak_tflops = self.capabilities.cpu_ai_score / 100  # Convert score to rough TFLOPS

        # Apply efficiency factors with training optimizations
        base_efficiency = 0.5  # Typical training efficiency (50% of peak)
        if setup['device'] == 'cpu':
            base_efficiency = 0.3  # CPU training is less efficient

        # Boost efficiency with repository training optimizations
        if TRAINING_COMPONENTS_AVAILABLE:
            # Progressive training can improve efficiency by 20-30%
            if training_config.get('progressive_training', False):
                base_efficiency *= 1.25
            # Mixed precision training efficiency boost
            if setup['precision'] in ['fp16', 'bf16', 'fp8']:
                base_efficiency *= 1.1

        effective_tflops = peak_tflops * base_efficiency
        efficiency_factor = base_efficiency

        # Calculate time per sample
        time_per_sample = flops_per_sample / (effective_tflops * 1e12) if effective_tflops > 0 else 1.0

        # Training scenarios
        scenarios = {
            'fine_tuning_1k': {
                'description': 'Fine-tuning on 1K samples (1 epoch)',
                'samples': 1_000,
                'epochs': 1,
            },
            'fine_tuning_10k': {
                'description': 'Fine-tuning on 10K samples (3 epochs)',
                'samples': 10_000,
                'epochs': 3,
            },
            'training_100k': {
                'description': 'Training on 100K samples (5 epochs)',
                'samples': 100_000,
                'epochs': 5,
            },
            'training_1m': {
                'description': 'Training on 1M samples (3 epochs)',
                'samples': 1_000_000,
                'epochs': 3,
            },
        }

        estimates = {}
        batch_size = min(setup.get('max_batch_size', 16), 32)  # Use reasonable batch size

        for scenario_name, scenario in scenarios.items():
            total_samples = scenario['samples'] * scenario['epochs']
            total_batches = total_samples / batch_size

            # Account for batch processing efficiency
            time_per_batch = time_per_sample * batch_size * 0.9  # 10% efficiency gain from batching

            total_time_seconds = total_batches * time_per_batch

            estimates[scenario_name] = {
                'description': scenario['description'],
                'total_samples': total_samples,
                'batch_size': batch_size,
                'total_batches': int(total_batches),
                'estimated_time_seconds': total_time_seconds,
                'estimated_time_hours': total_time_seconds / 3600,
                'estimated_time_days': total_time_seconds / (3600 * 24),
                'readable_time': self._format_time(total_time_seconds),
            }

        return {
            'system_performance': {
                'peak_tflops': peak_tflops,
                'effective_tflops': effective_tflops,
                'efficiency_factor': efficiency_factor,
                'time_per_sample_ms': time_per_sample * 1000,
            },
            'scenarios': estimates,
            'recommended_batch_size': batch_size,
            'notes': self._generate_training_notes(model_name, setup, effective_tflops),
        }

    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            return f"{seconds/60:.1f} minutes"
        elif seconds < 86400:
            return f"{seconds/3600:.1f} hours"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            return f"{days} days, {hours} hours"

    def _generate_training_notes(self, model_name: str, setup: Dict[str, Any],
                                effective_tflops: float) -> List[str]:
        """Generate helpful training notes and recommendations."""
        notes = []

        if setup['device'] == 'gpu':
            if effective_tflops > 50:
                notes.append(" Excellent GPU performance - training will be fast")
            elif effective_tflops > 20:
                notes.append(" Good GPU performance - reasonable training times")
            elif effective_tflops > 5:
                notes.append(" Moderate GPU performance - training will take some time")
            else:
                notes.append(" Limited GPU performance - consider smaller models or longer training times")

            if setup['precision'] in ['fp16', 'bf16']:
                notes.append(f" Using {setup['precision'].upper()} precision for 2x speed boost")
            elif setup['precision'] == 'fp8':
                notes.append(" FP8 precision available - up to 1.8x speedup over FP16")

            if self.capabilities.has_tensor_cores:
                notes.append(" Tensor Cores available - significant speedup for mixed precision")

            # Repository training optimizations
            if TRAINING_COMPONENTS_AVAILABLE:
                notes.append(" Repository training optimizations available:")
                if 'hidden' in model_name:  # Language model
                    notes.append("  • Progressive training (GrowLength + Curriculum)")
                    notes.append("  • Dynamic batch sizing optimization")
                if setup['precision'] == 'fp8':
                    notes.append("  • Advanced FP8 training with Transformer Engine")
                notes.append("  • 20-30% faster convergence with progressive training")
        else:
            notes.append(" CPU training - will be slower but still feasible for smaller models")

        if setup.get('max_batch_size', 0) < 8:
            notes.append(" Limited batch size due to memory constraints")
        elif setup.get('max_batch_size', 0) >= 32:
            notes.append(" Large batch sizes possible - efficient training")

        return notes

    def _generate_scaling_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations for scaling up training capabilities."""
        recommendations = {
            'current_bottlenecks': [],
            'upgrade_suggestions': [],
            'optimization_tips': [],
        }

        # Identify bottlenecks
        if not self.capabilities.gpu_model:
            recommendations['current_bottlenecks'].append("No GPU available - severely limits model size and training speed")
            recommendations['upgrade_suggestions'].append("Add a GPU with at least 8GB VRAM for meaningful AI training")
        elif self.capabilities.gpu_memory_gb and self.capabilities.gpu_memory_gb < 8:
            recommendations['current_bottlenecks'].append(f"Limited GPU memory ({self.capabilities.gpu_memory_gb:.1f}GB)")
            recommendations['upgrade_suggestions'].append("Upgrade to GPU with 16GB+ VRAM for larger models")

        if self.capabilities.system_memory_gb < 32:
            recommendations['current_bottlenecks'].append(f"Limited system RAM ({self.capabilities.system_memory_gb:.1f}GB)")
            recommendations['upgrade_suggestions'].append("Upgrade to 32GB+ system RAM for better data loading")

        # Optimization tips
        if self.capabilities.supports_fp16 or self.capabilities.supports_gpu_bf16:
            recommendations['optimization_tips'].append("Use mixed precision training (FP16/BF16) to save memory and increase speed")

        if self.capabilities.has_tensor_cores:
            recommendations['optimization_tips'].append("Ensure batch sizes are multiples of 8 for optimal Tensor Core utilization")

        recommendations['optimization_tips'].extend([
            "Use gradient checkpointing to trade compute for memory",
            "Consider gradient accumulation to simulate larger batch sizes",
            "Use efficient optimizers like AdamW with weight decay",
            "Implement learning rate scheduling for better convergence",
        ])

        return recommendations

    def _create_optimal_training_config(self, setup: Dict[str, Any], model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimal training configuration using repository components."""
        config = {
            'device': setup['device'],
            'precision': setup['precision'],
            'progressive_training': False,
            'fp8_training': False,
            'mixed_precision': False,
        }

        if not TRAINING_COMPONENTS_AVAILABLE:
            return config

        # Enable progressive training for language models
        if 'hidden' in model_info and setup['device'] == 'gpu':
            config['progressive_training'] = True
            config['progressive_config'] = {
                'enable_grow_length': True,
                'initial_seq_length': min(128, model_info.get('context', 512) // 4),
                'final_seq_length': model_info.get('context', 512),
                'enable_curriculum': model_info['params'] > 1e9,  # Enable for large models
                'enable_dynamic_batch': True,
                'max_batch_size': setup.get('max_batch_size', 32),
            }

        # Enable FP8 training if supported
        if (setup['precision'] == 'fp8' and
            self.capabilities.supports_fp8 and
            setup['device'] == 'gpu'):
            config['fp8_training'] = True
            config['fp8_config'] = {
                'enable_fp8': True,
                'fp8_format': 'E4M3',  # Standard for forward pass
                'mixed_precision_policy': 'auto',
                'fallback_to_bf16': True,
            }

        # Enable mixed precision for supported precisions
        if setup['precision'] in ['fp16', 'bf16']:
            config['mixed_precision'] = True

        return config


# ============================================================================
# Enhanced Results Display
# ============================================================================

class EnhancedResultsDisplay:
    """Enhanced display for comprehensive benchmark results including model capacity."""

    @staticmethod
    def display_model_capacity_analysis(analysis: Dict[str, Any]) -> None:
        """Display comprehensive model capacity analysis."""
        print(f"\n{'='*80}")
        print(" MODEL CAPACITY & TRAINING TIME ANALYSIS")
        print(f"{'='*80}")

        # System limits
        limits = analysis['system_limits']
        print(f"\n SYSTEM LIMITS:")

        if 'gpu' in limits:
            gpu = limits['gpu']
            print(f"    GPU Memory: {gpu['total_memory_gb']:.1f}GB total, {gpu['available_memory_gb']:.1f}GB available")
            print(f"      Max Model Size: {gpu['max_model_params_fp32']/1e9:.1f}B params (FP32), {gpu['max_model_params_fp16']/1e9:.1f}B params (FP16)")
            print(f"      Compute: {gpu['compute_tflops']:.1f} TFLOPS")

        cpu = limits['cpu']
        print(f"    CPU Memory: {cpu['total_memory_gb']:.1f}GB total, {cpu['available_memory_gb']:.1f}GB available")
        print(f"      Max Model Size: {cpu['max_model_params_fp32']/1e9:.1f}B params (CPU)")

        # Trainable models
        trainable = analysis['trainable_models']
        if trainable:
            print(f"\n TRAINABLE MODELS ({len(trainable)} models):")

            # Sort by parameter count
            sorted_models = sorted(trainable.items(), key=lambda x: x[1]['parameters'])

            for model_name, model_analysis in sorted_models:
                params = model_analysis['parameters']
                setup = model_analysis['recommended_setup']

                print(f"\n    {model_name.upper().replace('_', ' ')}")
                print(f"      Parameters: {params/1e9:.1f}B")
                print(f"      Device: {setup['device'].upper()}")
                print(f"      Precision: {setup['precision'].upper()}")
                print(f"      Memory Usage: {setup['memory_usage_gb']:.1f}GB")
                print(f"      Max Batch Size: {setup.get('max_batch_size', 'N/A')}")

                # Training time estimates
                if model_name in analysis['training_estimates']:
                    estimates = analysis['training_estimates'][model_name]
                    if 'scenarios' in estimates:
                        scenarios = estimates['scenarios']
                        print(f"      Training Time Estimates:")
                        for scenario_name, scenario in scenarios.items():
                            if scenario_name in ['fine_tuning_1k', 'training_100k']:  # Show key scenarios
                                print(f"        • {scenario['description']}: {scenario['readable_time']}")
        else:
            print(f"\n NO TRAINABLE MODELS FOUND")
            print(f"   Your system may need more memory or a GPU upgrade")

        # Recommendations
        recommendations = analysis['scaling_recommendations']
        if recommendations['current_bottlenecks']:
            print(f"\n  CURRENT BOTTLENECKS:")
            for bottleneck in recommendations['current_bottlenecks']:
                print(f"   • {bottleneck}")

        if recommendations['upgrade_suggestions']:
            print(f"\n UPGRADE SUGGESTIONS:")
            for suggestion in recommendations['upgrade_suggestions']:
                print(f"   • {suggestion}")

        if recommendations['optimization_tips']:
            print(f"\n OPTIMIZATION TIPS:")
            for tip in recommendations['optimization_tips'][:5]:  # Show top 5
                print(f"   • {tip}")


# ============================================================================
# Main CLI Interface and Execution
# ============================================================================

def main():
    """Main CLI interface for unified benchmark."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Ultimate Unified System Benchmark and Optimization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full comprehensive analysis (default behavior)
  python unified_benchmark.py

  # Quick system analysis and benchmarking
  python unified_benchmark.py --quick-test

  # Full comprehensive analysis (explicit)
  python unified_benchmark.py --full-analysis

  # System info only
  python unified_benchmark.py --system-info

  # Quick test with specific GPU
  python unified_benchmark.py --quick-test --gpu-id 1

  # Generate optimal configuration
  python unified_benchmark.py --generate-config
        """
    )

    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick performance test (reduced iterations)')
    parser.add_argument('--full-analysis', action='store_true',
                        help='Run comprehensive analysis and benchmarking')
    parser.add_argument('--system-info', action='store_true',
                        help='Show detailed system information only')
    parser.add_argument('--generate-config', action='store_true',
                        help='Generate optimal configuration files')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU device ID to use (default: 0)')
    parser.add_argument('--output-dir', type=str, default='unified_benchmark_results',
                        help='Output directory for results (default: unified_benchmark_results)')
    parser.add_argument('--save-results', action='store_true',
                        help='Save results to files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize benchmark suite
        benchmark = UnifiedBenchmarkSuite(device_id=args.gpu_id, output_dir=args.output_dir)

        print(f"\n{'='*80}")
        print(" UNIFIED SYSTEM BENCHMARK & OPTIMIZATION TOOL")
        print(f"{'='*80}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # System analysis (always run)
        print(f"\n{'='*60}")
        print(" SYSTEM ANALYSIS")
        print(f"{'='*60}")

        capabilities = benchmark.analyze_system()

        # Display system info
        print(f"\n  CPU: {capabilities.cpu_model}")
        print(f"   Cores: {capabilities.physical_cores}P / {capabilities.logical_cores}L")
        print(f"   Memory: {capabilities.system_memory_gb:.1f} GB")
        print(f"   AI Score: {capabilities.cpu_ai_score:.0f}")

        if capabilities.gpu_model:
            print(f"\n GPU: {capabilities.gpu_model}")
            gpu_mem = capabilities.gpu_memory_gb or 0
            print(f"   Memory: {gpu_mem:.1f} GB")
            arch_value = capabilities.gpu_architecture.value if capabilities.gpu_architecture else "Unknown"
            print(f"   Architecture: {arch_value}")
            if capabilities.gpu_ai_score:
                print(f"   AI Score: {capabilities.gpu_ai_score:.0f}")
        else:
            print(f"\n GPU: Not available (CPU-only system)")

        print(f"\n  System Balance: CPU {capabilities.recommended_cpu_gpu_split['cpu']*100:.0f}% / GPU {capabilities.recommended_cpu_gpu_split['gpu']*100:.0f}%")
        print(f" Recommended Precision: {capabilities.recommended_precision.upper()}")
        print(f" Optimal DataLoader Workers: {capabilities.optimal_dataloader_workers}")

        if capabilities.identified_bottlenecks:
            print(f"\n  Identified Bottlenecks:")
            for bottleneck in capabilities.identified_bottlenecks[:3]:  # Show top 3
                print(f"   • {bottleneck.replace('_', ' ').title()}")

        # Run benchmarks - default to full analysis if no specific test mode specified
        run_benchmarks = args.quick_test or args.full_analysis or (not args.system_info and not args.generate_config)
        if run_benchmarks:
            print(f"\n{'='*60}")
            print(" PERFORMANCE BENCHMARKS")
            print(f"{'='*60}")

            quick_mode = args.quick_test and not args.full_analysis
            results = benchmark.run_performance_benchmarks(quick_mode=quick_mode)

            print(f"\n Benchmarks completed!")
            print(f" Results saved to: {benchmark.output_dir}")

        # Model capacity analysis (always show at the end)
        print(f"\n{'='*80}")
        print(" AI MODEL CAPACITY & TRAINING TIME ANALYSIS")
        print(f"{'='*80}")

        capacity_analyzer = ModelCapacityAnalyzer(capabilities)
        model_analysis = capacity_analyzer.analyze_model_capacity()
        EnhancedResultsDisplay.display_model_capacity_analysis(model_analysis)

        # Store model analysis in results
        benchmark.results['model_capacity_analysis'] = model_analysis

        # Save results if requested
        if args.save_results:
            output_file = benchmark.output_dir / 'benchmark_results.json'
            with open(output_file, 'w') as f:
                json.dump(benchmark.results, f, indent=2, default=str)

            print(f"\n Results saved to: {output_file}")

        print(f"\n{'='*80}")
        print(" UNIFIED BENCHMARK COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")

    except KeyboardInterrupt:
        print(f"\n  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())