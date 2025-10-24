"""
Configuration Generator
Generates config files from templates to eliminate duplication.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

class ConfigGenerator:
    """Generates configuration files from templates."""

    def __init__(self, template_dir: str = None):
        """Initialize the config generator.

        Args:
            template_dir: Path to template directory. Defaults to code/configs/templates
        """
        if template_dir is None:
            # Get project root
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            template_dir = project_root / "code" / "configs" / "templates"

        self.template_dir = Path(template_dir)

    def load_template(self, template_name: str) -> str:
        """Load a template file.

        Args:
            template_name: Name of the template file

        Returns:
            Template content as string
        """
        template_path = self.template_dir / template_name
        with open(template_path, 'r') as f:
            return f.read()

    def replace_placeholders(self, template: str, params: Dict[str, Any]) -> str:
        """Replace placeholders in template with parameters.

        Args:
            template: Template string with {{PLACEHOLDER}} markers
            params: Dictionary of parameter values

        Returns:
            Template with placeholders replaced
        """
        result = template
        for key, value in params.items():
            placeholder = f"{{{{{key}}}}}"
            if isinstance(value, bool):
                value_str = str(value).lower()
            elif isinstance(value, (list, dict)):
                value_str = str(value)
            elif value is None:
                value_str = "null"
            else:
                value_str = str(value)
            result = result.replace(placeholder, value_str)
        return result

    def generate_deepspeed_config(self, zero_stage: int, output_path: str = None) -> str:
        """Generate a DeepSpeed ZeRO configuration file.

        Args:
            zero_stage: ZeRO stage (1, 2, or 3)
            output_path: Path to save the generated config. If None, returns string.

        Returns:
            Generated config as string
        """
        # Define parameters for each ZeRO stage
        if zero_stage == 1:
            params = {
                'ZERO_STAGE': 1,
                'HIDDEN_SIZE': 768,
                'NUM_LAYERS': 20,
                'NUM_ATTENTION_HEADS': 12,
                'INTERMEDIATE_SIZE': 3072,
                'MAX_POSITION_EMBEDDINGS': 2048,
                'NUM_EXPERTS': 16,
                'NUM_EXPERTS_PER_TOKEN': 2,
                'EXPERT_CAPACITY_FACTOR': 1.25,
                'ROUTER_TYPE': 'switch',
                'USE_MOH': True,
                'USE_MOA': False,
                'USE_RAG': False,
                'USE_CROSS_ATTENTION': True,
                'NUM_CROSS_ATTENTION_LAYERS': 2,
                'USE_EPISODIC_MEMORY': True,
                'MEMORY_SIZE': 1000,
                'PARTITION_ACTIVATIONS': False,
                'BATCH_SIZE': 4,
                'GRADIENT_ACCUMULATION_STEPS': 8,
                'NUM_EPOCHS': 10,
                'WARMUP_STEPS': 1000,
                'LEARNING_RATE': '1e-4',
                'CPU_OFFLOAD': False,
                'TRAIN_BATCH_SIZE': 32,
                'ZERO_STAGE_SPECIFIC_CONFIG': '  zero_force_ds_cpu_optimizer: false',
                'CPU_OFFLOAD_CONFIG': '',
                'USE_ALIBI': False,
                'MAX_RETRIEVED_DOCS': 5,
                'RAG_FUSION_TYPE': '"concatenate"',
                'CONTRASTIVE_LOSS': False,
                'ADAPTIVE_GRADIENT_SURGERY': False,
                'GRADIENT_SURGERY_METHOD': 'pcgrad',
                'MEMORY_REPLAY_RATIO': 0.0,
                'MEMORY_REPLAY_STRATEGY': '',
                'USE_NVFP4': False,
                'BIT_WIDTH': 8,
                'QUANTIZATION_EXTRAS': '',
                'EVAL_BATCH_SIZE': 8,
                'BUFFER_SIZE': 5000,
                'MULTI_COLUMN': False,
                'EXPRESS_MODE': True,
                'POOL_SIZE_GB': 16.0,
                'MEMORY_THRESHOLD_GB': 16.0,
                'CLEAR_CACHE_FREQUENCY': 100,
                'SAVE_TOTAL_LIMIT': 3,
                'EVAL_METRICS': 'perplexity',
                'COMPREHENSIVE_EVAL': False,
                'MODEL_SCALE': 'medium',
                'ADDITIONAL_TAGS': '"distributed", "multi-gpu"',
                'WANDB_NOTES': 'configuration with optimizer state sharding',
                'WANDB_EXTRAS': '',
                'HARDWARE_REQUIREMENTS': 'GPUs: 2-8 GPUs with 16GB+ VRAM each\n# Total VRAM: 32-128GB\n# Network: High-bandwidth interconnect (NVLink, InfiniBand)\n# Use case: Medium-large models with optimizer memory concerns'
            }
        elif zero_stage == 2:
            params = {
                'ZERO_STAGE': 2,
                'HIDDEN_SIZE': 1024,
                'NUM_LAYERS': 24,
                'NUM_ATTENTION_HEADS': 16,
                'INTERMEDIATE_SIZE': 4096,
                'MAX_POSITION_EMBEDDINGS': 2048,
                'NUM_EXPERTS': 32,
                'NUM_EXPERTS_PER_TOKEN': 4,
                'EXPERT_CAPACITY_FACTOR': 1.5,
                'ROUTER_TYPE': 'switch',
                'USE_MOH': True,
                'USE_MOA': True,
                'USE_RAG': True,
                'USE_CROSS_ATTENTION': True,
                'NUM_CROSS_ATTENTION_LAYERS': 4,
                'USE_EPISODIC_MEMORY': True,
                'MEMORY_SIZE': 2000,
                'PARTITION_ACTIVATIONS': True,
                'BATCH_SIZE': 2,
                'GRADIENT_ACCUMULATION_STEPS': 16,
                'NUM_EPOCHS': 5,
                'WARMUP_STEPS': 2000,
                'LEARNING_RATE': '8e-5',
                'CPU_OFFLOAD': False,
                'TRAIN_BATCH_SIZE': 32,
                'ZERO_STAGE_SPECIFIC_CONFIG': '  zero_reduce_bucket_size: 500000000\n  zero_allgather_bucket_size: 500000000',
                'CPU_OFFLOAD_CONFIG': '',
                'USE_ALIBI': False,
                'MAX_RETRIEVED_DOCS': 5,
                'RAG_FUSION_TYPE': '"concatenate"',
                'CONTRASTIVE_LOSS': True,
                'ADAPTIVE_GRADIENT_SURGERY': True,
                'GRADIENT_SURGERY_METHOD': 'cagrad',
                'MEMORY_REPLAY_RATIO': 0.2,
                'MEMORY_REPLAY_STRATEGY': '    memory_replay_strategy: "importance"',
                'USE_NVFP4': True,
                'BIT_WIDTH': 4,
                'QUANTIZATION_EXTRAS': '',
                'EVAL_BATCH_SIZE': 4,
                'BUFFER_SIZE': 10000,
                'MULTI_COLUMN': True,
                'EXPRESS_MODE': False,
                'POOL_SIZE_GB': 20.0,
                'MEMORY_THRESHOLD_GB': 20.0,
                'CLEAR_CACHE_FREQUENCY': 50,
                'SAVE_TOTAL_LIMIT': 2,
                'EVAL_METRICS': 'perplexity,bleu,rouge',
                'COMPREHENSIVE_EVAL': True,
                'MODEL_SCALE': 'large',
                'ADDITIONAL_TAGS': '"distributed", "large-model", "rag"',
                'WANDB_NOTES': 'with optimizer+gradient sharding, RAG enabled',
                'WANDB_EXTRAS': '',
                'HARDWARE_REQUIREMENTS': 'GPUs: 4-16 GPUs with 20GB+ VRAM each\n# Total VRAM: 80-320GB\n# Network: High-bandwidth interconnect required (NVLink/InfiniBand)\n# Use case: Large models (1B+ parameters) with gradient memory concerns'
            }
        elif zero_stage == 3:
            params = {
                'ZERO_STAGE': 3,
                'HIDDEN_SIZE': 1536,
                'NUM_LAYERS': 32,
                'NUM_ATTENTION_HEADS': 24,
                'INTERMEDIATE_SIZE': 6144,
                'MAX_POSITION_EMBEDDINGS': 4096,
                'NUM_EXPERTS': 64,
                'NUM_EXPERTS_PER_TOKEN': 6,
                'EXPERT_CAPACITY_FACTOR': 2.0,
                'ROUTER_TYPE': 'gshard',
                'USE_MOH': True,
                'USE_MOA': True,
                'USE_RAG': True,
                'USE_CROSS_ATTENTION': True,
                'NUM_CROSS_ATTENTION_LAYERS': 8,
                'USE_EPISODIC_MEMORY': True,
                'MEMORY_SIZE': 5000,
                'PARTITION_ACTIVATIONS': True,
                'BATCH_SIZE': 1,
                'GRADIENT_ACCUMULATION_STEPS': 32,
                'NUM_EPOCHS': 3,
                'WARMUP_STEPS': 5000,
                'LEARNING_RATE': '5e-5',
                'CPU_OFFLOAD': True,
                'TRAIN_BATCH_SIZE': 32,
                'ZERO_STAGE_SPECIFIC_CONFIG': '''  zero_stage3_prefetch_bucket_size: 500000000
  zero_stage3_param_persistence_threshold: 1000000
  zero_stage3_max_live_parameters: 1000000000
  zero_stage3_max_reuse_distance: 1000000000
  zero_stage3_gather_16bit_weights_on_model_save: true''',
                'CPU_OFFLOAD_CONFIG': '''
  # CPU offloading configuration
  offload_optimizer:
    device: "cpu"
    pin_memory: true
  offload_param:
    device: "cpu"
    pin_memory: true''',
                'USE_ALIBI': True,
                'MAX_RETRIEVED_DOCS': 10,
                'RAG_FUSION_TYPE': '"attention"',
                'CONTRASTIVE_LOSS': True,
                'ADAPTIVE_GRADIENT_SURGERY': True,
                'GRADIENT_SURGERY_METHOD': 'cagrad',
                'MEMORY_REPLAY_RATIO': 0.3,
                'MEMORY_REPLAY_STRATEGY': '    memory_replay_strategy: "importance"',
                'USE_NVFP4': True,
                'BIT_WIDTH': 4,
                'QUANTIZATION_EXTRAS': '    stochastic_rounding: true\n    use_hadamard_transform: true',
                'EVAL_BATCH_SIZE': 2,
                'BUFFER_SIZE': 20000,
                'MULTI_COLUMN': True,
                'EXPRESS_MODE': False,
                'POOL_SIZE_GB': 30.0,
                'MEMORY_THRESHOLD_GB': 30.0,
                'CLEAR_CACHE_FREQUENCY': 25,
                'SAVE_TOTAL_LIMIT': 1,
                'EVAL_METRICS': 'perplexity,bleu,rouge,bertscore',
                'COMPREHENSIVE_EVAL': True,
                'MODEL_SCALE': 'ultra-large',
                'ADDITIONAL_TAGS': '"ultra-large", "cpu-offload", "all-features"',
                'WANDB_NOTES': 'with full parameter sharding, CPU offloading, all features',
                'WANDB_EXTRAS': '  job_type: "train"',
                'HARDWARE_REQUIREMENTS': 'GPUs: 8+ GPUs with 16GB+ VRAM each (can work with less due to CPU offloading)\n# CPU RAM: 256GB+ recommended for parameter offloading\n# Total VRAM: 128GB+ (much less needed due to parameter sharding)\n# Network: Very high-bandwidth interconnect essential (NVLink, InfiniBand)\n# Storage: Fast SSD for CPU offloading, NVMe for parameter offloading\n# Use case: Ultra-large models (10B+ parameters) that don\'t fit in GPU memory'
            }
        else:
            raise ValueError(f"Invalid zero_stage: {zero_stage}. Must be 1, 2, or 3.")

        # Load template and replace placeholders
        template = self.load_template('deepspeed_template.yaml')
        config_str = self.replace_placeholders(template, params)

        # Save if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(config_str)

        return config_str


def main():
    """Main function to generate all configs."""
    generator = ConfigGenerator()

    # Get project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    config_dir = project_root / "code" / "configs" / "distributed"

    # Generate DeepSpeed configs
    print("Generating DeepSpeed configurations...")
    for stage in [1, 2, 3]:
        output_path = config_dir / f"deepspeed_zero{stage}_generated.yaml"
        generator.generate_deepspeed_config(stage, str(output_path))
        print(f"  ✓ Generated {output_path}")

    print("\nConfiguration generation complete!")
    print("Note: GPU configs require more parameters and will be added in next iteration.")


if __name__ == "__main__":
    main()
