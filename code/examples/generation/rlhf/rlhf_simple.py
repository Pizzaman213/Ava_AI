#!/usr/bin/env python3
"""
Simple RLHF Example - Quick demonstration of RLHF capabilities
Shows how to apply RLHF to a trained model with minimal setup
"""
import torch
import sys
from pathlib import Path
import logging

sys.path.append(str(Path(__file__).parent.parent))

from src.rlhf import (
    UnifiedRLHFTrainer,
    TrainingConfig,
    EnhancedPPOConfig,
)
from src.model.moe_transformer import MoETransformer
from src.utils.config import load_config
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_rlhf_demo():
    """Quick RLHF demonstration"""
    # Look for existing trained model
    model_paths = [
        "examples/fine_tuning/outputs/sft_model/final_model/pytorch_model.bin",
        "examples/fine_tuning/outputs/sft_model/checkpoint-best/pytorch_model.bin",
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            break
    
    if not model_path:
        logger.error("No trained model found. Please run training first.")
        logger.info("Run: python examples/fine_tuning/supervised_finetuning.py")
        return
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint.get('config', load_config("configs/mps/small.yaml"))
    
    model = MoETransformer(config)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create minimal RLHF config
    training_config = TrainingConfig(
        model_name="rlhf_demo",
        num_epochs=1,
        output_dir="outputs/rlhf_demo",
        use_wandb=False,
    )
    
    # Simple PPO config
    training_config.ppo = EnhancedPPOConfig(
        learning_rate=1e-5,
        batch_size=8,
        ppo_epochs=2,
    )
    
    # Initialize trainer
    trainer = UnifiedRLHFTrainer(
        model=model,
        tokenizer=tokenizer,
        config=training_config
    )
    
    # Simple preference data
    preference_data = [
        {
            "prompt": "Explain machine learning",
            "chosen": "Machine learning is a subset of AI that enables systems to learn from data.",
            "rejected": "Machine learning is complicated stuff with computers."
        },
        {
            "prompt": "What is Python?",
            "chosen": "Python is a high-level, interpreted programming language known for its simplicity.",
            "rejected": "Python is a snake."
        }
    ]
    
    # Quick PPO training
    logger.info("Starting quick RLHF training...")
    try:
        trainer.ppo_optimization(preference_data=preference_data)
        logger.info("RLHF training completed!")
        
        # Test generation
        test_prompt = "Explain artificial intelligence"
        logger.info(f"\nTest prompt: {test_prompt}")
        
        inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=100,
                temperature=0.8,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Model response: {response}")
        
        # Save the RLHF-tuned model
        trainer.save_model("outputs/rlhf_demo/final_model")
        logger.info("Model saved to outputs/rlhf_demo/final_model")
        
    except Exception as e:
        logger.error(f"RLHF training failed: {e}")
        logger.info("This is expected if some RLHF components are not fully implemented.")
        logger.info("The example shows the correct usage pattern.")

if __name__ == "__main__":
    quick_rlhf_demo()