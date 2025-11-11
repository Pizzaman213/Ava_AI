#!/usr/bin/env python3
"""
Comprehensive Training Process Analyzer for Ava MoE++

Analyzes all aspects of the training process including:
- Configuration analysis
- MoE expert utilization
- Memory optimization strategy
- Learning rate schedule
- Data pipeline efficiency
- Resource utilization
"""

import yaml
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import subprocess

class TrainingAnalyzer:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.issues = []
        self.warnings = []
        self.optimizations = []
        self.info = []

    def analyze_all(self):
        """Run all analysis checks"""
        print("=" * 80)
        print("AVA MoE++ TRAINING PROCESS ANALYSIS")
        print("=" * 80)
        print()

        self.analyze_configuration()
        self.analyze_moe_settings()
        self.analyze_memory_strategy()
        self.analyze_learning_rate()
        self.analyze_data_pipeline()
        self.analyze_resource_utilization()
        self.analyze_stability_settings()

        self.print_summary()

    def analyze_configuration(self):
        """Analyze overall training configuration"""
        print("📋 CONFIGURATION ANALYSIS")
        print("-" * 80)

        model = self.config.get('model', {})
        training = self.config.get('training', {})

        # Model size
        hidden_size = model.get('hidden_size', 0)
        num_layers = model.get('num_layers', 0)
        num_experts = model.get('num_experts', 0)
        intermediate_size = model.get('intermediate_size', 0)

        # Estimate model size
        params_per_layer = hidden_size * hidden_size * 4  # attention
        params_per_expert = hidden_size * intermediate_size * 3  # FFN
        total_params = (params_per_layer * num_layers +
                       params_per_expert * num_experts * num_layers)

        self.info.append(f"Model: {num_layers} layers, {num_experts} experts")
        self.info.append(f"Hidden size: {hidden_size}, Intermediate: {intermediate_size}")
        self.info.append(f"Est. params: ~{total_params/1e6:.1f}M parameters")

        # Batch size
        batch_size = training.get('batch_size', 1)
        grad_accum = training.get('gradient_accumulation_steps', 1)
        effective_bs = batch_size * grad_accum

        self.info.append(f"Batch size: {batch_size} (effective: {effective_bs})")

        # Training duration
        max_steps = training.get('max_steps', 0)
        warmup_steps = training.get('warmup_steps', 0)

        self.info.append(f"Training: {max_steps} steps ({warmup_steps} warmup)")

        # Optimizer
        optimizer = training.get('optimizer', 'unknown')
        lr = training.get('learning_rate', 0)

        self.info.append(f"Optimizer: {optimizer.upper()} (LR: {lr:.2e})")

        print()

    def analyze_moe_settings(self):
        """Analyze MoE-specific configuration"""
        print("\n🧠 MoE EXPERT UTILIZATION ANALYSIS")
        print("-" * 80)

        model = self.config.get('model', {})
        moe_mem = self.config.get('moe_memory_optimization', {})

        num_experts = model.get('num_experts', 4)
        experts_per_token = model.get('num_experts_per_token', 1)
        capacity_factor = model.get('capacity_factor', 1.25)

        self.info.append(f"Total experts: {num_experts}")
        self.info.append(f"Experts per token: {experts_per_token}")
        self.info.append(f"Capacity factor: {capacity_factor}")

        # Router settings
        router_type = model.get('router_type', 'mixtral')
        router_z_loss = model.get('router_z_loss_coef', 0)
        load_balance_loss = model.get('load_balance_loss_coef', 0)

        self.info.append(f"Router: {router_type}")
        self.info.append(f"  Z-loss coef: {router_z_loss:.6f}")
        self.info.append(f"  Load balance coef: {load_balance_loss:.6f}")

        # Check for potential issues
        if experts_per_token * 2 > num_experts:
            self.warnings.append("experts_per_token * 2 > num_experts may cause load imbalance")

        if capacity_factor < 1.0:
            self.issues.append(f"capacity_factor={capacity_factor} < 1.0 will drop tokens!")

        # Expert offloading
        use_offloading = moe_mem.get('use_expert_offloading', False)
        max_active = moe_mem.get('max_active_experts_gpu', num_experts)

        if use_offloading:
            self.info.append(f"Expert offloading: ENABLED ({max_active}/{num_experts} on GPU)")
            offload_ratio = (1 - max_active / num_experts) * 100
            self.optimizations.append(f"Offloading {offload_ratio:.0f}% of experts to CPU")
        else:
            self.info.append("Expert offloading: DISABLED")

        # LoRA
        use_lora = moe_mem.get('use_lora_experts', False)
        if use_lora:
            lora_rank = moe_mem.get('lora_rank', 8)
            lora_alpha = moe_mem.get('lora_alpha', 16)
            self.info.append(f"LoRA experts: ENABLED (rank={lora_rank}, alpha={lora_alpha})")

            # Calculate memory savings
            intermediate_size = model.get('intermediate_size', 8192)
            hidden_size = model.get('hidden_size', 1024)
            full_params = hidden_size * intermediate_size * 3
            lora_params = hidden_size * lora_rank * 2 + lora_rank * intermediate_size * 2
            savings = (1 - lora_params / full_params) * 100
            self.optimizations.append(f"LoRA reduces expert params by {savings:.1f}%")

        # Quantization
        use_quant = moe_mem.get('use_expert_quantization', False)
        if use_quant:
            quant_bits = moe_mem.get('expert_quantization_bits', 8)
            self.info.append(f"Quantization: INT{quant_bits} ({32/quant_bits:.0f}x compression)")

        print()

    def analyze_memory_strategy(self):
        """Analyze memory optimization strategy"""
        print("\n💾 MEMORY OPTIMIZATION STRATEGY")
        print("-" * 80)

        hardware = self.config.get('hardware', {})
        model = self.config.get('model', {})
        optimizations = self.config.get('optimizations', {})

        # Precision
        precision = hardware.get('mixed_precision', 'fp32')
        self.info.append(f"Precision: {precision.upper()}")

        if precision == 'fp32':
            self.warnings.append("Using FP32 - consider bf16/fp16 for 2x memory savings")

        # Gradient checkpointing
        use_gc = model.get('gradient_checkpointing', False)
        if use_gc:
            self.info.append("Gradient checkpointing: ENABLED (~25% memory savings)")
        else:
            self.warnings.append("Gradient checkpointing disabled - could save memory")

        # Flash attention
        use_flash = model.get('use_flash_attention', False)
        if use_flash:
            self.info.append("Flash Attention: ENABLED (40-50% faster attention)")

        # Memory headroom
        headroom = optimizations.get('memory_headroom_gb', 2.0)
        self.info.append(f"Reserved headroom: {headroom}GB")

        # Memory thresholds
        thresholds = optimizations.get('memory_cleanup_thresholds', {})
        warning_thresh = thresholds.get('warning', 0.85)
        critical_thresh = thresholds.get('critical', 0.90)

        self.info.append(f"Cleanup thresholds: {warning_thresh*100:.0f}% warn, {critical_thresh*100:.0f}% critical")

        print()

    def analyze_learning_rate(self):
        """Analyze learning rate schedule"""
        print("\n📈 LEARNING RATE SCHEDULE ANALYSIS")
        print("-" * 80)

        training = self.config.get('training', {})
        adaptive_lr = self.config.get('adaptive_lr', {})

        lr = training.get('learning_rate', 0)
        scheduler = training.get('lr_scheduler_type', 'constant')
        warmup_steps = training.get('warmup_steps', 0)
        max_steps = training.get('max_steps', 1)

        self.info.append(f"Base LR: {lr:.2e}")
        self.info.append(f"Scheduler: {scheduler}")
        self.info.append(f"Warmup: {warmup_steps} steps ({warmup_steps/max_steps*100:.1f}% of training)")

        # Check warmup ratio
        warmup_ratio = warmup_steps / max_steps
        if warmup_ratio < 0.05:
            self.warnings.append(f"Warmup ratio {warmup_ratio*100:.1f}% may be too short")
        elif warmup_ratio > 0.20:
            self.warnings.append(f"Warmup ratio {warmup_ratio*100:.1f}% may be too long")

        # Adaptive LR settings
        min_lr = adaptive_lr.get('min_lr', lr * 0.01)
        max_lr = adaptive_lr.get('max_lr', lr * 2)
        plateau_patience = adaptive_lr.get('plateau_patience', 500)

        # Convert to float if string
        if isinstance(min_lr, str):
            min_lr = float(min_lr)
        if isinstance(max_lr, str):
            max_lr = float(max_lr)

        self.info.append(f"Adaptive LR: {min_lr:.2e} to {max_lr:.2e}")
        self.info.append(f"Plateau patience: {plateau_patience} steps")

        # Check optimizer-specific LR
        optimizer = training.get('optimizer', 'adamw')
        if optimizer == 'lion':
            if lr > 0.0002:
                self.warnings.append(f"Lion LR {lr:.2e} may be too high (typical: 5e-5 to 2e-4)")
        elif optimizer == 'adamw':
            if lr < 0.0001:
                self.warnings.append(f"AdamW LR {lr:.2e} may be too low (typical: 1e-4 to 5e-4)")

        print()

    def analyze_data_pipeline(self):
        """Analyze data loading efficiency"""
        print("\n📊 DATA PIPELINE EFFICIENCY")
        print("-" * 80)

        data = self.config.get('data', {})
        training = self.config.get('training', {})

        dataset_name = data.get('dataset_name', 'unknown')
        data_dir = Path(data.get('data_dir', '/project/code/data/processed'))

        self.info.append(f"Dataset: {dataset_name}")

        # Check if dataset exists
        dataset_path = data_dir / dataset_name
        if dataset_path.exists():
            size_mb = dataset_path.stat().st_size / 1024 / 1024
            self.info.append(f"Dataset size: {size_mb:.1f}MB")
        else:
            self.warnings.append(f"Dataset not found: {dataset_path}")

        # Dataloader settings
        num_workers = data.get('num_workers', 0)
        prefetch_factor = data.get('dataloader_prefetch_factor', 2)
        pin_memory = data.get('dataloader_pin_memory', False)

        self.info.append(f"Dataloader workers: {num_workers}")
        self.info.append(f"Prefetch factor: {prefetch_factor}")
        self.info.append(f"Pin memory: {'YES' if pin_memory else 'NO'}")

        if num_workers == 0:
            self.optimizations.append("Consider num_workers=2-4 for faster data loading")

        if not pin_memory:
            self.optimizations.append("Enable pin_memory for faster GPU transfers")

        # Sequence length
        max_length = data.get('max_length', 512)
        batch_size = training.get('batch_size', 1)
        tokens_per_batch = max_length * batch_size

        self.info.append(f"Max length: {max_length} tokens")
        self.info.append(f"Tokens/batch: {tokens_per_batch}")

        print()

    def analyze_resource_utilization(self):
        """Analyze system resource usage"""
        print("\n🖥️  RESOURCE UTILIZATION")
        print("-" * 80)

        try:
            # GPU info
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                gpu_info = result.stdout.strip().split(', ')
                if len(gpu_info) >= 5:
                    name, mem_used, mem_total, util, temp = gpu_info[:5]
                    self.info.append(f"GPU: {name}")
                    self.info.append(f"Memory: {mem_used}MB / {mem_total}MB ({int(mem_used)/int(mem_total)*100:.1f}%)")
                    self.info.append(f"Utilization: {util}%")
                    self.info.append(f"Temperature: {temp}°C")

                    # Check if GPU is underutilized
                    if int(util) < 70:
                        self.warnings.append(f"GPU utilization {util}% - may indicate bottleneck")

                    # Check temperature
                    if int(temp) > 80:
                        self.warnings.append(f"GPU temperature {temp}°C is high - check cooling")
        except:
            self.warnings.append("Could not query GPU status")

        try:
            # Process count
            result = subprocess.run(
                ['pgrep', '-f', 'python train.py'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                process_count = len(result.stdout.strip().split('\n'))
                if process_count > 1:
                    self.warnings.append(f"⚠️  CRITICAL: {process_count} training processes detected!")
                    self.issues.append("Multiple training processes may cause resource contention")
        except:
            pass

        print()

    def analyze_stability_settings(self):
        """Analyze training stability settings"""
        print("\n🔒 STABILITY & CONVERGENCE SETTINGS")
        print("-" * 80)

        training = self.config.get('training', {})
        model = self.config.get('model', {})
        gradient_health = self.config.get('gradient_health', {})
        losses = self.config.get('losses', {})

        # Gradient clipping
        max_grad_norm = training.get('max_grad_norm', 1.0)
        self.info.append(f"Gradient clipping: {max_grad_norm}")

        if max_grad_norm < 0.5:
            self.warnings.append(f"max_grad_norm={max_grad_norm} is very aggressive")

        # Initialization
        init_range = model.get('initializer_range', 0.02)
        self.info.append(f"Init range: {init_range}")

        # Gradient health monitoring
        health_enabled = gradient_health.get('enabled', False)
        conditional = gradient_health.get('conditional_monitoring', False)

        if health_enabled:
            mode = "conditional" if conditional else "continuous"
            self.info.append(f"Gradient monitoring: {mode.upper()}")
        else:
            self.warnings.append("Gradient health monitoring disabled")

        # Loss settings
        primary_loss = losses.get('primary_loss_type', 'cross_entropy')
        label_smoothing = losses.get('label_smoothing', 0.0)

        self.info.append(f"Loss function: {primary_loss}")
        if label_smoothing > 0:
            self.info.append(f"Label smoothing: {label_smoothing}")

        # Check for potential instability
        use_ngram = losses.get('use_ngram_penalty', False)
        use_diversity = losses.get('use_diversity_loss', False)

        if use_ngram or use_diversity:
            self.info.append("Anti-repetition losses: ENABLED")
            if not health_enabled:
                self.warnings.append("Anti-repetition losses without health monitoring may cause instability")

        print()

    def print_summary(self):
        """Print analysis summary"""
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)

        if self.info:
            print("\n✓ Key Information:")
            for item in self.info[-10:]:  # Last 10 items
                print(f"  • {item}")

        if self.optimizations:
            print("\n💡 Optimization Opportunities:")
            for item in self.optimizations:
                print(f"  • {item}")

        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for item in self.warnings:
                print(f"  • {item}")

        if self.issues:
            print(f"\n❌ Critical Issues ({len(self.issues)}):")
            for item in self.issues:
                print(f"  • {item}")

        print("\n" + "=" * 80)

        # Overall health score
        score = 100
        score -= len(self.warnings) * 5
        score -= len(self.issues) * 15
        score = max(0, score)

        if score >= 80:
            status = "✓ EXCELLENT"
        elif score >= 60:
            status = "⚠️  GOOD"
        elif score >= 40:
            status = "⚠️  FAIR"
        else:
            status = "❌ NEEDS ATTENTION"

        print(f"\nTraining Health Score: {score}/100 - {status}")
        print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        config_path = "/project/code/configs/moe/tiny_moe_ultra_low_mem.yaml"
    else:
        config_path = sys.argv[1]

    analyzer = TrainingAnalyzer(config_path)
    analyzer.analyze_all()
