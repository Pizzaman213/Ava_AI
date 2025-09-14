#!/usr/bin/env python3
"""
Supervised Fine-tuning (SFT) Example
Fine-tune MoE++ model on custom instruction datasets
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root to path

import argparse
import logging
from pathlib import Path
import torch

# Set tokenizer parallelism to false only on CUDA or multi-GPU setups
if torch.cuda.is_available() or torch.cuda.device_count() > 1:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import AutoTokenizer

from src.model.moe_transformer import MoEForCausalLM
from src.training.trainer import MoETrainer, TrainingConfig
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def prepare_sft_dataset(dataset_name_or_path, tokenizer, max_length=512, streaming=True):
    """Prepare dataset for supervised fine-tuning"""
    
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
    
    return tokenized_dataset, streaming


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-tuning for MoE++")
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained checkpoint')
    parser.add_argument('--model-size', type=str, default='small',
                        choices=['small', 'medium', 'large'],
                        help='Model size configuration')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name or path (e.g., "tatsu-lab/alpaca")')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--warmup-steps', type=int, default=500,
                        help='Number of warmup steps')
    parser.add_argument('--eval-steps', type=int, default=500,
                        help='Evaluation interval')
    parser.add_argument('--save-steps', type=int, default=1000,
                        help='Save checkpoint interval')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='outputs/sft_model',
                        help='Output directory for checkpoints')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this training run')
    
    # Other arguments
    parser.add_argument('--mixed-precision', type=str, default=None,
                        choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision training (uses value from checkpoint config if not specified)')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing')
    parser.add_argument('--use-lora', action='store_true',
                        help='Use LoRA for parameter-efficient fine-tuning')
    parser.add_argument('--freeze-base', action='store_true',
                        help='Freeze base model and only train new layers')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    logger.info("Starting Supervised Fine-tuning")
    logger.info(f"Arguments: {args}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    
    # Handle different checkpoint formats
    import os
    import yaml
    import json
    from src.model.moe_transformer import MoEConfig
    
    config_json_path = os.path.join(args.checkpoint, "config.json")
    config_yaml_path = os.path.join(args.checkpoint, "config.yaml")
    
    # Store the full config dict for later use
    full_config_dict = None
    
    if os.path.exists(config_json_path):
        with open(config_json_path, "r") as f:
            full_config_dict = json.load(f)
            config_dict = full_config_dict
    elif os.path.exists(config_yaml_path):
        with open(config_yaml_path, "r") as f:
            full_config_dict = yaml.safe_load(f)
            # Extract model config if nested
            if 'model' in full_config_dict:
                config_dict = full_config_dict['model']
            else:
                config_dict = full_config_dict
    else:
        raise FileNotFoundError(f"No config file found in {args.checkpoint}")
    
    # Create config and model
    # Remove any fields that MoEConfig doesn't expect
    fields_to_remove = ['size', '_optimization_configs', 'training', 'data', 'generation', 'monitoring']
    for field in fields_to_remove:
        if field in config_dict:
            config_dict.pop(field)
    
    config = MoEConfig(**config_dict)
    model = MoEForCausalLM(config)
    
    # Load weights
    model_bin_path = os.path.join(args.checkpoint, "pytorch_model.bin")
    model_pt_path = os.path.join(args.checkpoint, "model.pt")
    
    if os.path.exists(model_bin_path):
        state_dict = torch.load(model_bin_path, map_location="cpu")
    elif os.path.exists(model_pt_path):
        state_dict = torch.load(model_pt_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No model weights found in {args.checkpoint}")
    
    model.load_state_dict(state_dict)
    
    # Get training config early for LoRA settings
    early_training_dict = {}
    if 'training' in full_config_dict:
        early_training_dict = full_config_dict['training']
    
    # Apply LoRA if requested from command line or config
    use_lora = args.use_lora or early_training_dict.get('use_lora', False)
    
    if use_lora:
        logger.info("Applying LoRA for parameter-efficient fine-tuning...")
        from peft import LoraConfig, get_peft_model, TaskType
        
        # Get LoRA config from training config or use defaults
        lora_config_dict = early_training_dict.get('lora_config', {})
        
        # Check if using QLoRA (4-bit quantization)
        use_4bit = lora_config_dict.get('use_4bit', False)
        
        if use_4bit:
            logger.info("Using QLoRA with 4-bit quantization...")
            # Import bitsandbytes for quantization
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            
            # Create quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if lora_config_dict.get('bnb_4bit_compute_dtype') == 'float16' else torch.bfloat16,
                bnb_4bit_quant_type=lora_config_dict.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=lora_config_dict.get('bnb_4bit_use_double_quant', True),
            )
            
            # Note: For QLoRA, the model needs to be loaded with quantization config
            # This would typically be done during model loading, not after
            logger.warning("QLoRA requires model to be loaded with quantization config. Please use --quantization flag during model loading.")
        
        # Create LoRA config
        peft_config = LoraConfig(
            task_type=getattr(TaskType, lora_config_dict.get('task_type', 'CAUSAL_LM')),
            inference_mode=False,
            r=lora_config_dict.get('r', 16),
            lora_alpha=lora_config_dict.get('lora_alpha', 32),
            lora_dropout=lora_config_dict.get('lora_dropout', 0.1),
            bias=lora_config_dict.get('bias', 'none'),
            target_modules=lora_config_dict.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj", "gate", "w1", "w2", "w3"])
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    # Freeze base model if requested
    if args.freeze_base and not use_lora:
        logger.info("Freezing base model parameters...")
        for param in model.model.parameters():
            param.requires_grad = False
        # Only train the language modeling head
        for param in model.lm_head.parameters():
            param.requires_grad = True
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Load the full config to get training configuration
    if os.path.exists(config_yaml_path):
        with open(config_yaml_path, "r") as f:
            full_config = yaml.safe_load(f)
    else:
        full_config = {}
    
    # Get training and data config from checkpoint or use defaults
    training_dict = full_config.get('training', {})
    data_dict = full_config.get('data', {})
    
    logger.info(f"Using configuration from {config_yaml_path}")
    
    # Use max_length from data config or args, but respect model's max position embeddings
    config_max_length = data_dict.get('max_length', args.max_length)
    model_max_length = getattr(model.config, 'max_position_embeddings', 512)
    max_length = min(config_max_length, model_max_length)
    
    if max_length != config_max_length:
        logger.warning(f"Requested max_length {config_max_length} exceeds model's max_position_embeddings {model_max_length}. Using {max_length}.")
    
    # Prepare dataset with streaming enabled by default
    logger.info(f"Loading and preparing dataset: {args.dataset}")
    dataset, is_streaming = prepare_sft_dataset(args.dataset, tokenizer, max_length, streaming=True)
    
    # Fix dataloader config if needed
    num_workers = training_dict.get('dataloader_num_workers', 0)
    
    # Determine device and appropriate mixed precision
    # First, try to detect the actual device
    if torch.backends.mps.is_available():
        device_type = 'mps'
        # MPS has issues with multiprocessing, so force num_workers to 0
        num_workers = 0
        # MPS doesn't support fp16 or bf16, use no mixed precision
        default_mixed_precision = 'no'
    elif torch.cuda.is_available():
        device_type = 'cuda'
        # CUDA supports both fp16 and bf16
        default_mixed_precision = training_dict.get('mixed_precision', 'fp16')
    else:
        device_type = 'cpu'
        # CPU doesn't support mixed precision
        default_mixed_precision = 'no'
        # CPU also works better with fewer workers
        num_workers = min(num_workers, 2)
    
    # Override device_map if it doesn't match detected device
    device_map = full_config.get('device_map', device_type)
    if device_map != device_type:
        logger.warning(f"Config device_map '{device_map}' doesn't match detected device '{device_type}'. Using detected device.")
        device_map = device_type
    
    # Calculate max_steps to avoid dataloader dependency during init
    # For streaming datasets, we can't get the exact size, so use an estimate or -1
    train_dataset = dataset['train']
    
    # Try to get dataset size
    dataset_size = -1
    if hasattr(train_dataset, '__len__'):
        try:
            dataset_size = len(train_dataset)
        except:
            pass
    
    # If we couldn't get size and it's streaming, try to get from dataset info
    if dataset_size == -1 and is_streaming:
        try:
            # Try to get dataset info from HuggingFace
            from datasets import load_dataset_builder
            builder = load_dataset_builder(args.dataset)
            if hasattr(builder.info, 'splits') and 'train' in builder.info.splits:
                dataset_size = builder.info.splits['train'].num_examples
                logger.info(f"Auto-detected dataset size: {dataset_size} examples")
        except:
            logger.info("Could not auto-detect dataset size, using streaming mode")
    
    # Use batch size from config or args
    batch_size = int(training_dict.get('batch_size', args.batch_size))
    gradient_accumulation_steps = int(training_dict.get('gradient_accumulation_steps', args.gradient_accumulation_steps))
    num_epochs = int(training_dict.get('num_epochs', args.num_epochs))
    
    if dataset_size > 0:
        steps_per_epoch = dataset_size // (batch_size * gradient_accumulation_steps)
        max_steps = steps_per_epoch * num_epochs
    else:
        # For streaming with unknown size, don't set max_steps
        max_steps = -1
    
    # Create training configuration using checkpoint config as base
    # Use checkpoint values by default, allow command line overrides
    training_config = TrainingConfig(
        model_config=model.config,
        # Use checkpoint training config values, with command line overrides
        learning_rate=float(training_dict.get('learning_rate', args.learning_rate)),
        batch_size=int(training_dict.get('batch_size', args.batch_size)),
        gradient_accumulation_steps=int(training_dict.get('gradient_accumulation_steps', args.gradient_accumulation_steps)),
        num_epochs=int(training_dict.get('num_epochs', args.num_epochs)),
        max_steps=max_steps,  # Set max_steps to avoid dataloader issue
        warmup_steps=int(training_dict.get('warmup_steps', args.warmup_steps)),
        eval_steps=int(training_dict.get('eval_steps', args.eval_steps)),
        save_steps=int(training_dict.get('save_steps', args.save_steps)),
        output_dir=args.output_dir,
        mixed_precision=args.mixed_precision or default_mixed_precision,  # Use args if provided, else device-appropriate default
        # Use values from checkpoint config
        max_grad_norm=training_dict.get('max_grad_norm', 1.0),
        logging_steps=training_dict.get('logging_steps', 10),
        save_total_limit=training_dict.get('save_total_limit', 3),
        dataloader_num_workers=num_workers,  # Use the fixed num_workers value
        distributed=False,  # Force single GPU mode
        local_rank=-1,  # No local rank for single GPU
        weight_decay=float(training_dict.get('weight_decay', 0.01)),
        adam_beta1=float(training_dict.get('adam_beta1', 0.9)),
        adam_beta2=float(training_dict.get('adam_beta2', 0.95)),
        adam_epsilon=float(training_dict.get('adam_epsilon', 1e-8)),
        gradient_checkpointing=training_dict.get('gradient_checkpointing', args.gradient_checkpointing),
        use_zero3=False,  # Disable ZeRO-3 for single GPU
        cpu_offload=False,  # Disable CPU offload for single GPU
        use_wandb=False,  # Disable wandb for now
        max_length=int(data_dict.get('max_length', args.max_length)),  # Use checkpoint max_length
    )
    
    # Create trainer
    # Get train and validation datasets
    train_dataset = dataset['train']
    eval_dataset = None
    if 'validation' in dataset:
        eval_dataset = dataset['validation']
    elif 'test' in dataset:
        eval_dataset = dataset['test']
    
    trainer = MoETrainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Start training
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Save final model
        final_path = Path(args.output_dir) / "final_model"
        model.save_pretrained(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        logger.info(f"Model saved to {final_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        save_path = Path(args.output_dir) / "interrupted_checkpoint"
        model.save_pretrained(str(save_path))
        logger.info(f"Checkpoint saved to {save_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()