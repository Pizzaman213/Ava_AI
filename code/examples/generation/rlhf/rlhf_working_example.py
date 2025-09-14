#!/usr/bin/env python3
"""
Working RLHF Example
Demonstrates how to use the UnifiedRLHFTrainer with a trained model
"""
import torch
import sys
from pathlib import Path
import logging
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from src.rlhf import UnifiedRLHFTrainer, TrainingConfig
from src.model.moe_transformer import MoETransformer
from src.utils.config import load_config
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_path=None, config_path="configs/mps/small.yaml"):
    """Load a trained model or create a new one"""
    if model_path and Path(model_path).exists():
        logger.info(f"Loading trained model from: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get config from checkpoint or use provided
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = load_config(config_path)
        
        model = MoETransformer(config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.info(f"Creating new model from config: {config_path}")
        config = load_config(config_path)
        model = MoETransformer(config)
    
    return model, config

def setup_rlhf_config(output_dir="outputs/rlhf_working", num_epochs=1):
    """Setup RLHF training configuration"""
    config = TrainingConfig(
        model_name="rlhf_working",
        num_epochs=num_epochs,
        output_dir=output_dir,
        use_wandb=False,  # Disable wandb for demo
        
        # Reduce training phases for quick demo
        training_phases=["ppo_optimization", "final_tuning"],
        
        # Use smaller batch sizes for demo
        gradient_accumulation_steps=2,
        
        # Save checkpoints
        save_strategy="steps",
        save_steps=100,
        eval_steps=50,
    )
    
    # Configure simplified PPO settings
    config.ppo.learning_rate = 1e-5
    config.ppo.batch_size = 4
    config.ppo.ppo_epochs = 2
    
    # Simplify other components
    config.self_supervised.use_reasoning_chains = False
    config.multi_reward.use_learned_weights = False
    config.reasoning.use_chain_of_thought = True
    config.reasoning.max_reasoning_length = 200
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Working RLHF Example")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--config-path", type=str, default="configs/mps/small.yaml",
                        help="Path to model config")
    parser.add_argument("--output-dir", type=str, default="outputs/rlhf_working",
                        help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--test-only", action="store_true",
                        help="Only test generation without training")
    args = parser.parse_args()
    
    # Check for existing trained models
    if not args.model_path:
        possible_paths = [
            "examples/fine_tuning/outputs/sft_model/final_model/pytorch_model.bin",
            "examples/fine_tuning/outputs/sft_model/checkpoint-best/pytorch_model.bin",
            "outputs/sft_model/final_model/pytorch_model.bin",
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                args.model_path = path
                logger.info(f"Found trained model at: {path}")
                break
    
    # Load model
    model, model_config = load_trained_model(args.model_path, args.config_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test generation before RLHF
    if args.test_only or True:  # Always show before/after
        logger.info("\n=== Testing model BEFORE RLHF ===")
        test_prompts = [
            "Explain quantum computing in simple terms",
            "What are the benefits of exercise?",
            "How does photosynthesis work?",
        ]
        
        for prompt in test_prompts:
            logger.info(f"\nPrompt: {prompt}")
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=100,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Response: {response}\n")
    
    if args.test_only:
        return
    
    # Setup RLHF configuration
    training_config = setup_rlhf_config(args.output_dir, args.num_epochs)
    
    # Initialize RLHF trainer
    logger.info("\n=== Initializing RLHF Trainer ===")
    trainer = UnifiedRLHFTrainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config
    )
    
    # Run RLHF training
    logger.info("\n=== Starting RLHF Training ===")
    logger.info(f"Training phases: {training_config.training_phases}")
    
    try:
        # The train() method handles all phases automatically
        trainer.train()
        
        logger.info("\n=== RLHF Training Completed Successfully! ===")
        
        # Test generation after RLHF
        logger.info("\n=== Testing model AFTER RLHF ===")
        for prompt in test_prompts:
            logger.info(f"\nPrompt: {prompt}")
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=100,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Response: {response}\n")
        
        # Save the final model
        final_path = Path(args.output_dir) / "final_model"
        logger.info(f"\nSaving RLHF-tuned model to: {final_path}")
        trainer._save_checkpoint("final")
        
    except Exception as e:
        logger.error(f"\nRLHF training encountered an error: {e}")
        logger.info("This may be due to missing dependencies or incomplete implementation.")
        logger.info("The example demonstrates the correct usage pattern.")
        
        # Show what would have happened
        logger.info("\n=== Expected RLHF Process ===")
        logger.info("1. Self-supervised pretraining to improve consistency")
        logger.info("2. Constitutional AI alignment for safety")
        logger.info("3. PPO optimization with human preferences")
        logger.info("4. Reasoning enhancement for better logical thinking")
        logger.info("5. Final unified training combining all objectives")

if __name__ == "__main__":
    main()