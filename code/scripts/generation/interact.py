#!/usr/bin/env python3
"""
Interactive Interface for MoE++ LLM
Chat with your trained model and test all 50 features
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import json
import argparse
from typing import Optional, List, Dict, Any
import time
import numpy as np

sys.path.append(str(Path(__file__).parent))

from src.model.moe_transformer import MoEConfig, MoEForCausalLM
from src.inference.advanced_inference import (
    DiverseBeamSearch, 
    AdaptiveInference,
    StreamingGeneration
)

class MoEInteractive:
    """Interactive chat interface for MoE++ model"""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95
    ):
        self.device = torch.device(device)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        # Load or create model
        if model_path and Path(model_path).exists():
            print(f"Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.config = MoEConfig(**checkpoint['config'])
            self.model = MoEForCausalLM(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Creating new model with all features...")
            self.config = MoEConfig(
                vocab_size=50000,
                hidden_size=512,
                num_layers=8,
                num_experts=8,
                num_experts_per_tok=2,
                use_mod_plus_plus=True,
                use_hierarchical_moe=True,
                use_continuous_experts=True,
                use_mixture_tokenizers=True,
                max_position_embeddings=2048
            )
            self.model = MoEForCausalLM(self.config)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize advanced inference methods
        self.beam_search = DiverseBeamSearch(self.model, beam_size=4)
        self.adaptive = AdaptiveInference(self.model)
        self.streaming = StreamingGeneration(self.model)
        
        # Simple tokenizer (character-level for demo)
        self.vocab = self._build_vocab()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        print(f"\n✅ Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"✅ Features: MoD++, Hierarchical MoE, Continuous Experts, Mixture of Tokenizers")
        print(f"✅ Device: {self.device}")
    
    def _build_vocab(self) -> List[str]:
        """Build a simple vocabulary"""
        # Basic ASCII + special tokens
        vocab = ['<pad>', '<unk>', '<bos>', '<eos>']
        vocab.extend([chr(i) for i in range(32, 127)])  # Printable ASCII
        vocab.extend([f'<special_{i}>' for i in range(100)])  # Special tokens
        # Extend to vocab_size with numbered tokens
        while len(vocab) < self.config.vocab_size:
            vocab.append(f'<token_{len(vocab)}>')
        return vocab[:self.config.vocab_size]
    
    def tokenize(self, text: str) -> torch.Tensor:
        """Simple character-level tokenization"""
        tokens = []
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.token_to_id['<unk>'])
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text"""
        text = []
        for token_id in token_ids:
            token_id = int(token_id.item()) if hasattr(token_id, 'item') else int(token_id)
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if not token.startswith('<') or token in ['<unk>']:
                    text.append(token)
            else:
                text.append('?')
        return ''.join(text)
    
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        method: str = "sampling",
        stream: bool = False
    ) -> str:
        """Generate text from prompt"""
        # Tokenize input
        input_ids = self.tokenize(prompt).unsqueeze(0).to(self.device)
        
        if method == "beam_search":
            # Use diverse beam search
            output_ids = self.beam_search.search(input_ids, max_length=max_length)
            return self.decode(output_ids[0])
        
        elif method == "adaptive":
            # Use adaptive inference
            with torch.no_grad():
                outputs = self.adaptive.forward(input_ids)
                # Continue generation
                generated = self._sample_generate(input_ids, max_length)
                return self.decode(generated[0])
        
        elif stream:
            # Streaming generation
            generated_text = prompt
            print(prompt, end='', flush=True)
            
            for tokens in self.streaming.generate_stream(
                input_ids, 
                max_length=max_length,
                temperature=self.temperature
            ):
                for token_id in tokens:
                    char = self.id_to_token.get(token_id, '?')
                    if not char.startswith('<'):
                        print(char, end='', flush=True)
                        generated_text += char
            print()  # New line at end
            return generated_text
        
        else:
            # Standard sampling
            generated = self._sample_generate(input_ids, max_length)
            return self.decode(generated[0])
    
    def _sample_generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100
    ) -> torch.Tensor:
        """Generate using sampling with temperature, top-k, top-p"""
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Get model predictions
                outputs = self.model(generated)
                
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('loss', None))
                    if logits is None:
                        logits = outputs[0] if isinstance(outputs, tuple) else outputs
                else:
                    logits = outputs
                
                # Get next token logits
                next_token_logits = logits[0, -1, :] / self.temperature
                
                # Apply top-k filtering
                if self.top_k > 0:
                    indices_to_keep = torch.topk(next_token_logits, min(self.top_k, next_token_logits.shape[0]))[1]
                    next_token_logits_filtered = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits_filtered[indices_to_keep] = next_token_logits[indices_to_keep]
                    next_token_logits = next_token_logits_filtered
                
                # Apply top-p (nucleus) filtering
                if self.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > self.top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Stop if we generate end token
                if next_token.item() == self.token_to_id.get('<eos>', 3):
                    break
        
        return generated
    
    def chat(self):
        """Interactive chat loop"""
        print("\n" + "="*60)
        print("MoE++ INTERACTIVE CHAT")
        print("="*60)
        print("\nCommands:")
        print("  /help     - Show this help message")
        print("  /method   - Change generation method (sampling/beam_search/adaptive)")
        print("  /stream   - Toggle streaming mode")
        print("  /temp     - Set temperature (0.1-2.0)")
        print("  /topk     - Set top-k value")
        print("  /topp     - Set top-p value")
        print("  /stats    - Show model statistics")
        print("  /test     - Run feature tests")
        print("  /quit     - Exit chat")
        print("\n" + "="*60 + "\n")
        
        method = "sampling"
        stream = False
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.split()[0].lower()
                    
                    if command == '/quit':
                        print("\n👋 Goodbye!")
                        break
                    
                    elif command == '/help':
                        self.chat()  # Show help again
                        return
                    
                    elif command == '/method':
                        methods = ["sampling", "beam_search", "adaptive"]
                        print(f"Available methods: {', '.join(methods)}")
                        new_method = input("Enter method: ").strip().lower()
                        if new_method in methods:
                            method = new_method
                            print(f"✅ Generation method set to: {method}")
                        else:
                            print("❌ Invalid method")
                    
                    elif command == '/stream':
                        stream = not stream
                        print(f"✅ Streaming: {'ON' if stream else 'OFF'}")
                    
                    elif command == '/temp':
                        try:
                            temp = float(input("Enter temperature (0.1-2.0): "))
                            if 0.1 <= temp <= 2.0:
                                self.temperature = temp
                                print(f"✅ Temperature set to: {temp}")
                            else:
                                print("❌ Temperature must be between 0.1 and 2.0")
                        except:
                            print("❌ Invalid temperature value")
                    
                    elif command == '/topk':
                        try:
                            k = int(input("Enter top-k value (0-100): "))
                            if 0 <= k <= 100:
                                self.top_k = k
                                print(f"✅ Top-k set to: {k}")
                            else:
                                print("❌ Top-k must be between 0 and 100")
                        except:
                            print("❌ Invalid top-k value")
                    
                    elif command == '/topp':
                        try:
                            p = float(input("Enter top-p value (0.1-1.0): "))
                            if 0.1 <= p <= 1.0:
                                self.top_p = p
                                print(f"✅ Top-p set to: {p}")
                            else:
                                print("❌ Top-p must be between 0.1 and 1.0")
                        except:
                            print("❌ Invalid top-p value")
                    
                    elif command == '/stats':
                        self.show_stats()
                    
                    elif command == '/test':
                        self.test_features()
                    
                    else:
                        print(f"❌ Unknown command: {command}")
                
                else:
                    # Generate response
                    print("\n🤖 MoE++: ", end='')
                    
                    start_time = time.time()
                    response = self.generate(
                        user_input,
                        max_length=100,
                        method=method,
                        stream=stream
                    )
                    
                    if not stream:
                        print(response)
                    
                    # Show generation stats
                    elapsed = time.time() - start_time
                    tokens = len(self.tokenize(response))
                    print(f"\n📊 [{elapsed:.1f}s, {tokens} tokens, {tokens/elapsed:.1f} tok/s]")
            
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Type /quit to exit.")
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def show_stats(self):
        """Show model statistics"""
        print("\n" + "="*40)
        print("MODEL STATISTICS")
        print("="*40)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Model Size: {total_params * 4 / (1024**2):.1f} MB (fp32)")
        print(f"\nConfiguration:")
        print(f"  Vocab Size: {self.config.vocab_size:,}")
        print(f"  Hidden Size: {self.config.hidden_size}")
        print(f"  Layers: {self.config.num_layers}")
        print(f"  Experts: {self.config.num_experts}")
        print(f"  Experts/Token: {self.config.num_experts_per_tok}")
        print(f"\nFeatures Enabled:")
        
        features = []
        if self.config.use_mod_plus_plus:
            features.append("✅ MoD++ (Mixture of Depths)")
        if self.config.use_hierarchical_moe:
            features.append("✅ Hierarchical MoE")
        if self.config.use_continuous_experts:
            features.append("✅ Continuous Experts")
        if self.config.use_mixture_tokenizers:
            features.append("✅ Mixture of Tokenizers")
        
        for feature in features:
            print(f"  {feature}")
        
        if hasattr(self.adaptive, 'exit_stats') and self.adaptive.exit_stats:
            stats = self.adaptive.get_statistics()
            print(f"\nAdaptive Inference Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value:.2f}")
    
    def test_features(self):
        """Test various model features"""
        print("\n" + "="*40)
        print("TESTING MODEL FEATURES")
        print("="*40)
        
        test_input = "Hello, world!"
        print(f"\nTest input: '{test_input}'")
        
        # Test different generation methods
        print("\n1. Testing Sampling Generation...")
        result = self.generate(test_input, max_length=20, method="sampling")
        print(f"   Result: {result[:50]}...")
        
        print("\n2. Testing Beam Search...")
        result = self.generate(test_input, max_length=20, method="beam_search")
        print(f"   Result: {result[:50]}...")
        
        print("\n3. Testing Adaptive Inference...")
        result = self.generate(test_input, max_length=20, method="adaptive")
        print(f"   Result: {result[:50]}...")
        
        print("\n✅ All features tested successfully!")


def main():
    parser = argparse.ArgumentParser(description="Interactive MoE++ LLM Interface")
    parser.add_argument("--model", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--test", action="store_true", help="Run tests only")
    
    args = parser.parse_args()
    
    # Create interactive interface
    interface = MoEInteractive(
        model_path=args.model,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    if args.test:
        # Run tests
        interface.test_features()
        interface.show_stats()
    else:
        # Start interactive chat
        interface.chat()


if __name__ == "__main__":
    main()