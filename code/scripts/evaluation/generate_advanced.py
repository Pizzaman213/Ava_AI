#!/usr/bin/env python3
"""
Advanced generation script for MoE++ model with speculative decoding,
uncertainty quantification, and expert prefetching
"""
import argparse
import os
import sys
import torch
import yaml
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.moe_transformer import MoEConfig, MoETransformer
from src.model.lora_experts import LoRAMoEModel, LoRAConfig
from src.model.uncertainty_quantification import UncertaintyAwareMoE, UncertaintyConfig
from src.generation.speculative_moe import SpeculativeMoE, SpeculativeConfig
from src.optimization.expert_prefetching import ExpertPrefetcher, PrefetchConfig
from src.optimization.model_pruning import ModelPruner, PruningConfig
from src.utils.logging_utils import setup_logging
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced generation with MoE++ model")
    
    # Model
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint or directory")
    parser.add_argument("--model-config", type=str, help="Path to model config (if not in checkpoint)")
    
    # Generation
    parser.add_argument("--prompt", type=str, help="Generation prompt")
    parser.add_argument("--prompt-file", type=str, help="File containing prompts (one per line)")
    parser.add_argument("--max-length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate per prompt")
    
    # Advanced features
    parser.add_argument("--use-speculation", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--use-uncertainty", action="store_true", help="Enable uncertainty tracking")
    parser.add_argument("--use-prefetching", action="store_true", help="Enable expert prefetching")
    parser.add_argument("--uncertainty-threshold", type=float, default=0.7, help="Uncertainty threshold for warnings")
    parser.add_argument("--abstain-threshold", type=float, default=0.9, help="Uncertainty threshold for abstention")
    
    # Speculation settings
    parser.add_argument("--draft-length", type=int, default=5, help="Number of tokens to draft speculatively")
    parser.add_argument("--acceptance-threshold", type=float, default=0.9, help="Speculation acceptance threshold")
    parser.add_argument("--use-tree-speculation", action="store_true", help="Use tree-based speculation")
    
    # Pruning
    parser.add_argument("--prune-model", action="store_true", help="Prune model before generation")
    parser.add_argument("--pruning-sparsity", type=float, default=0.3, help="Target sparsity for pruning")
    
    # Output
    parser.add_argument("--output-file", type=str, help="Save generation results to file")
    parser.add_argument("--interactive", action="store_true", help="Interactive generation mode")
    parser.add_argument("--show-uncertainty", action="store_true", help="Display uncertainty scores")
    parser.add_argument("--show-stats", action="store_true", help="Display generation statistics")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for optimization")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    return parser.parse_args()

class AdvancedGenerator:
    """Advanced text generator with uncertainty and speculation"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
        use_speculation: bool = False,
        use_uncertainty: bool = False,
        use_prefetching: bool = False,
        speculation_config: Optional[SpeculativeConfig] = None,
        prefetch_config: Optional[PrefetchConfig] = None
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
        
        # Move model to device
        self.base_model = model.to(device)
        
        # Wrap with speculation if enabled
        if use_speculation:
            self.logger.info("Enabling speculative decoding...")
            config = speculation_config or SpeculativeConfig()
            self.model = SpeculativeMoE(self.base_model, config)
        else:
            self.model = self.base_model
            
        # Setup prefetching if enabled
        if use_prefetching:
            self.logger.info("Enabling expert prefetching...")
            config = prefetch_config or PrefetchConfig()
            self.prefetcher = ExpertPrefetcher(self.model, config)
        else:
            self.prefetcher = None
            
        # Check if model supports uncertainty
        self.supports_uncertainty = (
            use_uncertainty and 
            isinstance(self.base_model, UncertaintyAwareMoE)
        )
        
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        num_samples: int = 1,
        show_uncertainty: bool = False,
        uncertainty_threshold: float = 0.7,
        abstain_threshold: float = 0.9
    ) -> List[Tuple[str, Optional[Dict[str, Any]]]]:
        """Generate text with advanced features"""
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        prompt_length = input_ids.shape[1]
        
        results = []
        
        for _ in range(num_samples):
            start_time = time.time()
            
            # Prefetch experts if enabled
            if self.prefetcher:
                self.prefetcher.prefetch_for_sequence(input_ids)
                
            # Generate based on model type
            if isinstance(self.model, SpeculativeMoE):
                # Speculative generation
                generated_ids, stats = self.model.generate(
                    input_ids,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
                generated_text = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )
                
                generation_time = time.time() - start_time
                stats['generation_time'] = generation_time
                stats['tokens_per_second'] = max_length / generation_time
                
                results.append((generated_text, stats))
                
            elif self.supports_uncertainty:
                # Generation with uncertainty tracking
                generated_text, uncertainty_history = self._generate_with_uncertainty(
                    input_ids,
                    max_length,
                    temperature,
                    top_k,
                    top_p,
                    uncertainty_threshold,
                    abstain_threshold,
                    show_uncertainty
                )
                
                generation_time = time.time() - start_time
                stats = {
                    'uncertainty_history': uncertainty_history,
                    'generation_time': generation_time,
                    'tokens_per_second': len(generated_text.split()) / generation_time
                }
                
                results.append((generated_text, stats))
                
            else:
                # Standard generation
                generated_ids = self._standard_generate(
                    input_ids,
                    max_length,
                    temperature,
                    top_k,
                    top_p
                )
                
                generated_text = self.tokenizer.decode(
                    generated_ids[0],
                    skip_special_tokens=True
                )
                
                generation_time = time.time() - start_time
                stats = {
                    'generation_time': generation_time,
                    'tokens_per_second': (generated_ids.shape[1] - prompt_length) / generation_time
                }
                
                results.append((generated_text, stats))
                
        return results
        
    def _generate_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
        uncertainty_threshold: float,
        abstain_threshold: float,
        show_uncertainty: bool
    ) -> Tuple[str, Dict[str, List[float]]]:
        """Generate with uncertainty tracking"""
        
        generated = input_ids
        uncertainty_history = {
            'epistemic': [],
            'aleatoric': [],
            'total': []
        }
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions with uncertainty
                outputs = self.model(generated, return_uncertainty=True)
                
                # Track uncertainty
                uncertainty = outputs['uncertainty']
                token_uncertainty = uncertainty['total'][:, -1].mean().item()
                
                for key in uncertainty_history:
                    if key in uncertainty:
                        uncertainty_history[key].append(
                            uncertainty[key][:, -1].mean().item()
                        )
                        
                # Check abstention threshold
                if token_uncertainty > abstain_threshold:
                    self.logger.warning(
                        f"Stopping generation due to high uncertainty: {token_uncertainty:.3f}"
                    )
                    break
                    
                # Warn about high uncertainty
                if token_uncertainty > uncertainty_threshold and show_uncertainty:
                    self.logger.warning(
                        f"High uncertainty detected: {token_uncertainty:.3f}"
                    )
                    
                # Sample next token
                logits = outputs['logits'][:, -1]
                next_token = self._sample_token(logits, temperature, top_k, top_p)
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated_text, uncertainty_history
        
    def _standard_generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """Standard autoregressive generation"""
        
        generated = input_ids
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(generated)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Sample next token
                next_token = self._sample_token(
                    logits[:, -1],
                    temperature,
                    top_k,
                    top_p
                )
                
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                    
        return generated
        
    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> torch.Tensor:
        """Sample token with temperature, top-k, and top-p"""
        
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy decoding
            return logits.argmax(dim=-1)
            
        # Apply top-k
        if top_k > 0:
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[:, -1, None]] = float('-inf')
            
        # Apply top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
            
        # Sample from distribution
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token

def load_model(args) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load model with advanced features"""
    
    logger = logging.getLogger(__name__)
    
    # Load checkpoint
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Extract configs
    if 'config' in checkpoint:
        model_config = checkpoint['config'].model_config
        advanced_config = checkpoint.get('advanced_config', {})
    else:
        # Load from separate config file
        with open(args.model_config, 'r') as f:
            config = yaml.safe_load(f)
        from scripts.training.train import setup_model_config
        model_config = setup_model_config(config)
        advanced_config = {}
        
    # Create model
    from scripts.training.train_advanced import create_advanced_model
    model = create_advanced_model(model_config, advanced_config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, advanced_config

def prune_model_for_inference(model: torch.nn.Module, sparsity: float) -> torch.nn.Module:
    """Prune model for faster inference"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"Pruning model to {sparsity:.0%} sparsity...")
    
    config = PruningConfig(
        target_sparsity=sparsity,
        prune_experts=True,
        prune_attention=True,
        prune_ffn=True,
        iterative_steps=1,
        finetune_epochs=0  # No fine-tuning for inference
    )
    
    pruner = ModelPruner(model, config)
    
    # Simple evaluation function (perplexity not needed for inference)
    def dummy_eval(model):
        return 1.0
        
    results = pruner.prune(None, dummy_eval)
    
    logger.info(f"Pruning complete. Sparsity: {results['final_sparsity']:.2%}")
    
    return model

def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load model
    model, advanced_config = load_model(args)
    
    # Prune if requested
    if args.prune_model:
        model = prune_model_for_inference(model, args.pruning_sparsity)
        
    # Compile if requested
    if args.compile and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
        
    # Initialize tokenizer
    tokenizer_name = advanced_config.get('tokenizer', 'gpt2')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Setup speculation config
    speculation_config = None
    if args.use_speculation:
        speculation_config = SpeculativeConfig(
            draft_sequence_length=args.draft_length,
            acceptance_threshold=args.acceptance_threshold,
            use_tree_speculation=args.use_tree_speculation
        )
        
    # Setup prefetch config
    prefetch_config = None
    if args.use_prefetching:
        prefetch_config = PrefetchConfig(
            cache_size_gb=2.0,
            prefetch_window=3,
            use_pattern_learning=True
        )
        
    # Create generator
    generator = AdvancedGenerator(
        model,
        tokenizer,
        device=args.device,
        use_speculation=args.use_speculation,
        use_uncertainty=args.use_uncertainty,
        use_prefetching=args.use_prefetching,
        speculation_config=speculation_config,
        prefetch_config=prefetch_config
    )
    
    # Get prompts
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    elif args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompts.extend(line.strip() for line in f if line.strip())
    elif args.interactive:
        pass  # Handle below
    else:
        prompts.append("Once upon a time")  # Default prompt
        
    # Generate
    if args.interactive:
        # Interactive mode
        logger.info("Entering interactive mode. Type 'quit' to exit.")
        
        while True:
            try:
                prompt = input("\nPrompt: ")
                if prompt.lower() in ['quit', 'exit']:
                    break
                    
                results = generator.generate(
                    prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_samples=args.num_samples,
                    show_uncertainty=args.show_uncertainty,
                    uncertainty_threshold=args.uncertainty_threshold,
                    abstain_threshold=args.abstain_threshold
                )
                
                for i, (text, stats) in enumerate(results):
                    print(f"\n--- Sample {i+1} ---")
                    print(text)
                    
                    if args.show_stats:
                        print(f"\nStats:")
                        print(f"  Generation time: {stats.get('generation_time', 0):.2f}s")
                        print(f"  Tokens/second: {stats.get('tokens_per_second', 0):.1f}")
                        if 'effective_speedup' in stats:
                            print(f"  Speculative speedup: {stats['effective_speedup']:.2f}x")
                        if 'acceptance_rate' in stats:
                            print(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")
                            
                    if args.show_uncertainty and 'uncertainty_history' in stats:
                        history = stats['uncertainty_history']
                        print(f"\nUncertainty:")
                        print(f"  Avg epistemic: {sum(history['epistemic'])/len(history['epistemic']):.3f}")
                        print(f"  Avg aleatoric: {sum(history['aleatoric'])/len(history['aleatoric']):.3f}")
                        print(f"  Max total: {max(history['total']):.3f}")
                        
            except KeyboardInterrupt:
                break
                
    else:
        # Batch mode
        all_results = []
        
        for prompt in prompts:
            logger.info(f"Generating for prompt: {prompt[:50]}...")
            
            results = generator.generate(
                prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_samples=args.num_samples,
                show_uncertainty=args.show_uncertainty,
                uncertainty_threshold=args.uncertainty_threshold,
                abstain_threshold=args.abstain_threshold
            )
            
            for text, stats in results:
                result = {
                    'prompt': prompt,
                    'generated_text': text,
                    'stats': stats
                }
                all_results.append(result)
                
                # Print result
                print(f"\nPrompt: {prompt}")
                print(f"Generated: {text}")
                
                if args.show_stats:
                    print(f"Generation time: {stats.get('generation_time', 0):.2f}s")
                    print(f"Tokens/second: {stats.get('tokens_per_second', 0):.1f}")
                    if 'effective_speedup' in stats:
                        print(f"Speculative speedup: {stats['effective_speedup']:.2f}x")
                        
        # Save results if requested
        if args.output_file:
            import json
            with open(args.output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Saved results to {args.output_file}")
            
        # Print summary statistics
        if args.show_stats and len(all_results) > 1:
            avg_time = sum(r['stats']['generation_time'] for r in all_results) / len(all_results)
            avg_tps = sum(r['stats']['tokens_per_second'] for r in all_results) / len(all_results)
            
            print(f"\nSummary:")
            print(f"  Average generation time: {avg_time:.2f}s")
            print(f"  Average tokens/second: {avg_tps:.1f}")
            
            if any('effective_speedup' in r['stats'] for r in all_results):
                speedups = [r['stats']['effective_speedup'] for r in all_results 
                           if 'effective_speedup' in r['stats']]
                avg_speedup = sum(speedups) / len(speedups)
                print(f"  Average speculative speedup: {avg_speedup:.2f}x")

if __name__ == "__main__":
    main()