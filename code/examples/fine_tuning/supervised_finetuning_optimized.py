#!/usr/bin/env python3
"""
Optimized Supervised Fine-tuning (SFT) with all speed optimizations

Includes:
- Continuous batching for data loading
- Quantization-aware fine-tuning
- PagedAttention for long sequences
- Fused kernels for faster training
- Hierarchical KV cache for evaluation
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root to path

import argparse
import logging
from pathlib import Path
import torch
import os
from typing import Dict

# Set tokenizer parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import AutoTokenizer
import time

from src.model.moe_transformer import MoEForCausalLM
from src.training.trainer import MoETrainer, TrainingConfig
from src.utils.logging_utils import setup_logging
from src.generation.optimized_generator import OptimizedTextGenerator, GenerationConfig
from src.optimization.quantization import AWQConfig, quantize_model_awq
from src.optimization.hierarchical_kv_cache import HierarchicalKVCache, CacheConfig

logger = logging.getLogger(__name__)


def prepare_sft_dataset(dataset_name_or_path, tokenizer, max_length=512, streaming=True):
    """Prepare dataset for supervised fine-tuning with optimization support"""
    
    # Load dataset with streaming enabled for HuggingFace datasets
    if Path(dataset_name_or_path).exists():
        # Local dataset - streaming not supported for local files
        dataset = load_dataset('json', data_files={
            'train': f'{dataset_name_or_path}/train.json',
            'validation': f'{dataset_name_or_path}/validation.json'
        })
        streaming = False  # Force disable streaming for local files
    else:
        # HuggingFace dataset with streaming
        dataset = load_dataset(dataset_name_or_path, streaming=streaming)
    
    def format_instruction(example):
        """Format instruction-response pairs"""
        if 'instruction' in example and 'response' in example:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
        elif 'prompt' in example and 'completion' in example:
            text = f"{example['prompt']}{example['completion']}"
        else:
            # Assume it's already formatted
            text = example.get('text', '')
        
        return {'text': text}
    
    # For streaming datasets, we need to handle column names differently
    if streaming:
        # For streaming, get column names from first example
        first_example = next(iter(dataset['train']))
        remove_columns = list(first_example.keys())
    else:
        remove_columns = dataset['train'].column_names
    
    # Format dataset
    dataset = dataset.map(format_instruction, remove_columns=remove_columns)
    
    # Tokenize
    def tokenize_function(examples):
        # Handle both single examples (streaming) and batched examples
        if isinstance(examples['text'], str):
            texts = [examples['text']]
            is_single = True
        else:
            texts = examples['text']
            is_single = False
            
        # Tokenize the texts
        outputs = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None  # Don't return tensors yet
        )
        
        # For language modeling, labels are the same as input_ids
        if is_single:
            # For single examples, return without list wrapping
            return {
                'input_ids': outputs['input_ids'][0],
                'attention_mask': outputs['attention_mask'][0],
                'labels': outputs['input_ids'][0]
            }
        else:
            outputs['labels'] = outputs['input_ids'].copy()
            return outputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=not streaming,  # Streaming datasets process one example at a time
        remove_columns=['text']
    )
    
    # For non-streaming datasets, set the format for PyTorch
    if not streaming:
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    return tokenized_dataset


class OptimizedSFTTrainer(MoETrainer):
    """Enhanced trainer with optimization support for fine-tuning"""
    
    def __init__(self, model, config: TrainingConfig, train_dataset, eval_dataset=None, enable_optimizations: bool = True):
        super().__init__(model, config, train_dataset, eval_dataset)
        self.enable_optimizations = enable_optimizations
        self.optimized_generator = None
        self.wandb_run = None  # For external wandb logging
        
    def setup_model(self, model=None):
        """Setup model with optimizations"""
        super().setup_model(model)
        
        if self.enable_optimizations:
            # Setup optimized generator for evaluation
            gen_config = GenerationConfig(
                use_continuous_batching=True,
                use_paged_attention=True,
                use_hierarchical_cache=True,
                use_fused_kernels=True,
                max_batch_size=16,
                kv_cache_gpu_gb=4.0,
                kv_cache_cpu_gb=16.0,
            )
            
            self.optimized_generator = OptimizedTextGenerator(
                model=self.model,
                tokenizer=self.tokenizer,
                config=gen_config,
                device=self.device
            )
            
            logger.info("Optimizations enabled for fine-tuning:")
            logger.info(f"  - Continuous batching: {gen_config.use_continuous_batching}")
            logger.info(f"  - PagedAttention: {gen_config.use_paged_attention}")
            logger.info(f"  - Hierarchical cache: {gen_config.use_hierarchical_cache}")
            logger.info(f"  - Fused kernels: {gen_config.use_fused_kernels}")
    
    def evaluate_generation(self, eval_prompts=None):
        """Evaluate generation quality using optimized generator"""
        if not self.optimized_generator or not eval_prompts:
            return
        
        self.model.eval()
        
        logger.info("\nEvaluating generation quality with optimizations...")
        
        # Use optimized batch generation
        start_time = time.time()
        outputs = self.optimized_generator.generate(
            eval_prompts,
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9
        )
        end_time = time.time()
        
        # Log results
        for i, (prompt, output) in enumerate(zip(eval_prompts, outputs)):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Response: {output[len(prompt):]}")  # Remove prompt from output
        
        # Performance stats
        total_time = end_time - start_time
        avg_time = total_time / len(eval_prompts)
        logger.info(f"\nGeneration performance:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Average time per prompt: {avg_time:.2f}s")
        logger.info(f"  Throughput: {len(eval_prompts) / total_time:.2f} prompts/sec")
        
        # Get optimization stats
        stats = self.optimized_generator.get_optimization_stats()
        logger.info(f"\nOptimization statistics:")
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        # Log generation metrics to WandB
        if self.wandb_run:
            generation_metrics = {
                "generation/total_time": total_time,
                "generation/avg_time_per_prompt": avg_time,
                "generation/throughput": len(eval_prompts) / total_time,
            }
            generation_metrics.update({f"generation/{k}": v for k, v in stats.items()})
            self.wandb_run.log(generation_metrics, step=self.global_step)
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Enhanced metric logging for fine-tuning with WandB support"""
        # Call parent implementation
        super()._log_metrics(metrics)
        
        # Log additional fine-tuning specific metrics to external WandB
        if self.wandb_run and self.accelerator.is_main_process:
            enhanced_metrics = {
                # Training metrics
                "finetune/loss": metrics.get("train_loss", 0),
                "finetune/learning_rate": metrics.get("learning_rate", 0),
                "finetune/grad_norm": metrics.get("grad_norm", 0),
                "finetune/epoch": self.epoch,
                "finetune/global_step": self.global_step,
                
                # Performance metrics
                "performance/samples_per_second": self.config.batch_size / metrics.get("step_time", 1.0) if "step_time" in metrics else 0,
                "performance/tokens_per_second": (self.config.batch_size * self.config.max_length) / metrics.get("step_time", 1.0) if "step_time" in metrics else 0,
            }
            
            # Memory metrics (if available)
            if torch.cuda.is_available():
                enhanced_metrics.update({
                    "memory/gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "memory/gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                })
            
            # LoRA specific metrics if using LoRA
            if hasattr(self.model, 'peft_modules') or hasattr(self.model, 'base_model'):
                try:
                    trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in self.model.parameters())
                    enhanced_metrics.update({
                        "lora/trainable_params": trainable_params,
                        "lora/trainable_params_percentage": (trainable_params / total_params) * 100,
                    })
                except:
                    pass
            
            self.wandb_run.log(enhanced_metrics, step=self.global_step)
    
    def _evaluate(self) -> float:
        """Enhanced evaluation with generation quality metrics"""
        # Get validation loss
        eval_loss = super()._evaluate()
        
        # Log to external WandB
        if self.wandb_run and self.accelerator.is_main_process:
            self.wandb_run.log({
                "finetune/eval_loss": eval_loss,
                "finetune/eval_perplexity": torch.exp(torch.tensor(eval_loss)).item(),
            }, step=self.global_step)
        
        return eval_loss


def main():
    parser = argparse.ArgumentParser(description="Optimized supervised fine-tuning for MoE++ models")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default="outputs/moe_model",
                       help="Path to pre-trained model")
    parser.add_argument("--model-size", type=str, default="small",
                       choices=["small", "medium", "large"],
                       help="Model size configuration")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="databricks/databricks-dolly-15k",
                       help="Dataset name or path")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--streaming", action="store_true", default=True,
                       help="Use streaming dataset loading")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size per device")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--eval-steps", type=int, default=500,
                       help="Evaluation interval")
    parser.add_argument("--save-steps", type=int, default=1000,
                       help="Save checkpoint interval")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                       help="Gradient accumulation steps")
    
    # Optimization arguments
    parser.add_argument("--enable-all-optimizations", action="store_true", default=True,
                       help="Enable all speed optimizations")
    parser.add_argument("--quantization", type=str, choices=["none", "4bit", "8bit"],
                       default="none", help="Quantization for memory efficiency")
    parser.add_argument("--use-paged-attention", action="store_true", default=True,
                       help="Use PagedAttention for memory efficiency")
    parser.add_argument("--use-fused-kernels", action="store_true", default=True,
                       help="Use fused CUDA kernels")
    parser.add_argument("--use-lora", action="store_true", default=False,
                       help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="outputs/sft_model_optimized",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--experiment-name", type=str, default="optimized_sft",
                       help="Experiment name for tracking")
    
    # Other arguments
    parser.add_argument("--mixed-precision", type=str, choices=["no", "fp16", "bf16"],
                       default="fp16", help="Mixed precision training")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="moe-sft-optimized",
                       help="W&B project name")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("=" * 70)
    logger.info("Optimized Supervised Fine-tuning")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Optimizations enabled: {args.enable_all_optimizations}")
    logger.info("=" * 70)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = prepare_sft_dataset(
        args.dataset,
        tokenizer,
        max_length=args.max_length,
        streaming=args.streaming
    )
    
    # Load model
    logger.info("Loading model...")
    if Path(args.model_path).exists():
        model = MoEForCausalLM.from_pretrained(args.model_path)
        logger.info(f"Loaded model from {args.model_path}")
    else:
        # Create new model with size configuration
        from src.model.moe_transformer import MoEConfig
        config = MoEConfig(
            size=args.model_size,
            use_flash_attn=args.enable_all_optimizations,
            use_paged_attention=args.use_paged_attention and args.enable_all_optimizations,
            gradient_checkpointing=args.gradient_checkpointing
        )
        model = MoEForCausalLM(config)
        logger.info(f"Created new {args.model_size} model")
    
    # Apply optimizations
    if args.enable_all_optimizations:
        # Enable fused kernels in model config
        if hasattr(model.config, 'use_fused_kernels'):
            model.config.use_fused_kernels = args.use_fused_kernels
        
        # Apply quantization if requested
        if args.quantization == "4bit":
            logger.info("Applying 4-bit quantization...")
            # Create calibration data
            calibration_data = []
            for i, batch in enumerate(dataset['train']):
                if i >= 128:
                    break
                calibration_data.append(batch)
            
            awq_config = AWQConfig(bits=4, group_size=128)
            model = quantize_model_awq(model, calibration_data, awq_config)
            logger.info("4-bit quantization applied")
    
    # Setup LoRA if requested
    if args.use_lora:
        logger.info(f"Setting up LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Make config compatible with PEFT by adding a get method
        if not hasattr(model.config, 'get'):
            class ConfigWrapper:
                def __init__(self, config):
                    self._config = config
                    # Copy all attributes
                    for attr in dir(config):
                        if not attr.startswith('_'):
                            setattr(self, attr, getattr(config, attr))
                    # Add PEFT-required attributes
                    self.model_type = "moe"  # Custom model type
                
                def get(self, key, default=None):
                    return getattr(self._config, key, default)
                
                def __getattr__(self, name):
                    # Handle special cases
                    if name == 'model_type':
                        return 'moe'
                    return getattr(self._config, name)
            
            model.config = ConfigWrapper(model.config)
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "w1", "w2", "w3"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Get the original config if it's wrapped
    if hasattr(model.config, '_config'):
        original_config = model.config._config
    else:
        original_config = model.config
    
    # Create training configuration
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        wandb_run_name=args.experiment_name,  # Use wandb_run_name instead
        
        # Training hyperparameters
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=args.warmup_steps,
        
        # Evaluation and saving
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=10,
        
        # Optimizations
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        
        # Model configuration
        model_config=original_config,
        
        # Disable distributed training for single GPU/MPS
        distributed=False,
        local_rank=-1,
        world_size=1,
    )
    
    # Get train and validation datasets
    train_dataset = dataset['train']
    eval_dataset = dataset.get('validation', None) if not args.streaming else None
    
    # Create optimized trainer with all required arguments
    trainer = OptimizedSFTTrainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        enable_optimizations=args.enable_all_optimizations
    )
    
    # Setup W&B if requested
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config={
                "model_size": args.model_size,
                "dataset": args.dataset,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation,
                "num_epochs": args.num_epochs,
                "max_length": args.max_length,
                "warmup_steps": args.warmup_steps,
                "mixed_precision": args.mixed_precision,
                "optimizations": {
                    "enabled": args.enable_all_optimizations,
                    "quantization": args.quantization,
                    "paged_attention": args.use_paged_attention,
                    "fused_kernels": args.use_fused_kernels,
                    "lora": args.use_lora,
                    "lora_r": args.lora_r if args.use_lora else None,
                    "lora_alpha": args.lora_alpha if args.use_lora else None,
                    "gradient_checkpointing": args.gradient_checkpointing,
                },
                "streaming_dataset": args.streaming,
            }
        )
        trainer.wandb_run = wandb.run
    
    # Define evaluation prompts
    eval_prompts = [
        "Write a Python function to sort a list:",
        "Explain quantum computing in simple terms:",
        "What are the benefits of renewable energy?",
        "How do I improve my communication skills?",
        "Create a healthy meal plan for a week:",
    ]
    
    # Train
    logger.info("\nStarting optimized fine-tuning...")
    trainer.train()
    
    # Final evaluation with optimized generation
    logger.info("\nFinal evaluation...")
    trainer.evaluate_generation(eval_prompts)
    
    # Save final model
    logger.info(f"\nSaving fine-tuned model to {args.output_dir}")
    trainer.save_checkpoint(is_final=True)
    
    # Cleanup
    if trainer.optimized_generator:
        trainer.optimized_generator.shutdown()
    
    logger.info("\nOptimized fine-tuning completed!")
    
    # Benchmark improvements
    if args.enable_all_optimizations:
        logger.info("\nBenchmarking optimization improvements...")
        results = trainer.optimized_generator.benchmark(
            num_prompts=10,
            max_new_tokens=100
        )
        logger.info(f"Speedup achieved: {results['speedup']:.2f}x")
        logger.info(f"Optimized throughput: {results['optimized_throughput']:.1f} tokens/sec")


if __name__ == "__main__":
    main()