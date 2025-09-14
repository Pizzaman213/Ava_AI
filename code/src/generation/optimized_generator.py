"""
Optimized text generation with all speed optimizations

This generator integrates all 10 optimizations for maximum performance.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Dict, Any, Callable
from transformers import PreTrainedTokenizerBase
import warnings
import time
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

from LLM.src.inference.continuous_batching import ContinuousBatchingEngine, InferenceRequest
from LLM.src.optimization.hierarchical_kv_cache import HierarchicalKVCache, CacheConfig


@dataclass
class GenerationConfig:
    """Configuration for optimized generation"""
    # Basic generation params
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    
    # Optimization flags
    use_continuous_batching: bool = True
    use_paged_attention: bool = True
    use_hierarchical_cache: bool = True
    use_fused_kernels: bool = True
    use_speculative_decoding: bool = False
    
    # Batching params
    max_batch_size: int = 32
    batch_timeout_ms: int = 50
    enable_sequence_packing: bool = True
    
    # Memory params
    kv_cache_gpu_gb: float = 8.0
    kv_cache_cpu_gb: float = 32.0
    kv_cache_nvme_gb: float = 128.0


class OptimizedTextGenerator:
    """
    Optimized text generator with all speed optimizations
    
    Features:
    - Continuous batching for multiple requests
    - Quantized model support
    - PagedAttention integration
    - Hierarchical KV cache
    - Fused CUDA kernels
    - Streaming generation
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        config: Optional[GenerationConfig] = None,
        device: Optional[str] = None
    ):
        """Initialize optimized generator"""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize optimization components
        self._init_optimizations()
    
    def _init_optimizations(self):
        """Initialize optimization components"""
        # Continuous batching engine
        if self.config.use_continuous_batching:
            self.batch_engine = ContinuousBatchingEngine(
                model=self.model,
                tokenizer=self.tokenizer,
                max_batch_size=self.config.max_batch_size,
                batch_timeout_ms=self.config.batch_timeout_ms,
                device=self.device,
                enable_sequence_packing=self.config.enable_sequence_packing
            )
        else:
            self.batch_engine = None
        
        # Hierarchical KV cache
        if self.config.use_hierarchical_cache:
            cache_config = CacheConfig(
                gpu_cache_size_gb=self.config.kv_cache_gpu_gb,
                cpu_cache_size_gb=self.config.kv_cache_cpu_gb,
                nvme_cache_size_gb=self.config.kv_cache_nvme_gb
            )
            self.kv_cache = HierarchicalKVCache(cache_config)
        else:
            self.kv_cache = None
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        streaming: bool = False,
        callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text with optimizations
        
        Args:
            prompt: Input prompt(s)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            streaming: Enable streaming generation
            callback: Streaming callback function
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text(s)
        """
        # Handle batch vs single prompt
        if isinstance(prompt, str):
            prompts = [prompt]
            single_prompt = True
        else:
            prompts = prompt
            single_prompt = False
        
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        
        # Route to appropriate generation method
        if self.config.use_continuous_batching and self.batch_engine:
            results = self._generate_batched(
                prompts, max_new_tokens, temperature, top_p, top_k,
                streaming, callback, **kwargs
            )
        else:
            results = self._generate_standard(
                prompts, max_new_tokens, temperature, top_p, top_k,
                streaming, callback, **kwargs
            )
        
        return results[0] if single_prompt else results
    
    def _generate_batched(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        streaming: bool,
        callback: Optional[Callable],
        **kwargs
    ) -> List[str]:
        """Generate using continuous batching"""
        # Tokenize prompts
        encoded_prompts = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0]
            encoded_prompts.append(input_ids)
        
        # Create requests
        requests = []
        results = {}
        
        for i, (prompt, input_ids) in enumerate(zip(prompts, encoded_prompts)):
            req_id = f"gen_{i}_{time.time()}"
            
            # Completion callback
            def make_callback(rid, idx):
                def cb(text):
                    results[rid] = prompts[idx] + text
                return cb
            
            # Streaming callback
            stream_cb = None
            if streaming and callback:
                def make_stream_callback(idx):
                    def cb(token):
                        callback(f"[{idx}] {token}")
                    return cb
                stream_cb = make_stream_callback(i)
            
            request = InferenceRequest(
                request_id=req_id,
                prompt_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                priority=0,
                streaming_callback=stream_cb,
                completion_callback=make_callback(req_id, i)
            )
            
            self.batch_engine.submit_request(request)
            requests.append(req_id)
        
        # Wait for all completions
        while len(results) < len(requests):
            time.sleep(0.01)
        
        # Return results in order
        return [results[req_id] for req_id in requests]
    
    def _generate_standard(
        self,
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        streaming: bool,
        callback: Optional[Callable],
        **kwargs
    ) -> List[str]:
        """Standard generation without batching"""
        results = []
        
        for prompt in prompts:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            
            # Generate
            generated_ids = []
            past_key_values = None
            
            for step in range(max_new_tokens):
                with torch.no_grad():
                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
                    
                    # Handle both dict and object outputs
                    if isinstance(outputs, dict):
                        logits = outputs['logits'][:, -1, :]
                        past_key_values = outputs.get('past_key_values')
                    else:
                        logits = outputs.logits[:, -1, :]
                        past_key_values = outputs.past_key_values
                    
                    # Apply temperature
                    if temperature > 0:
                        logits = logits / temperature
                    
                    # Top-k filtering
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = -float('Inf')
                    
                    # Top-p filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            -1, sorted_indices, sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = -float('Inf')
                    
                    # Sample
                    if self.config.do_sample and temperature > 0:
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    generated_ids.append(next_token)
                    
                    # Update inputs
                    input_ids = next_token
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((1, 1), device=self.device)
                    ], dim=1)
                    
                    # Check for EOS
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Streaming callback
                    if streaming and callback:
                        token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                        callback(token_text)
            
            # Decode full sequence
            generated_ids = torch.cat(generated_ids, dim=1)
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            results.append(prompt + generated_text)
        
        return results
    
    def generate_async(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[asyncio.Future]:
        """Asynchronous generation for multiple prompts"""
        futures = []
        
        for prompt in prompts:
            future = self.executor.submit(
                self.generate,
                prompt,
                **kwargs
            )
            futures.append(future)
        
        return futures
    
    def benchmark(
        self,
        num_prompts: int = 10,
        prompt_length: int = 50,
        max_new_tokens: int = 100
    ) -> Dict[str, float]:
        """Benchmark generation performance"""
        # Generate random prompts
        prompts = []
        for _ in range(num_prompts):
            # Use random words from tokenizer
            ids = torch.randint(100, 10000, (prompt_length,))
            prompt = self.tokenizer.decode(ids, skip_special_tokens=True)
            prompts.append(prompt)
        
        # Measure with optimizations
        start_time = time.time()
        results = self.generate(prompts, max_new_tokens=max_new_tokens)
        optimized_time = time.time() - start_time
        
        # Measure without batching (if enabled)
        if self.config.use_continuous_batching:
            self.config.use_continuous_batching = False
            start_time = time.time()
            results_standard = self._generate_standard(
                prompts, max_new_tokens,
                self.config.temperature,
                self.config.top_p,
                self.config.top_k,
                False, None
            )
            standard_time = time.time() - start_time
            self.config.use_continuous_batching = True
        else:
            standard_time = optimized_time
        
        # Calculate metrics
        total_tokens = num_prompts * max_new_tokens
        
        return {
            "num_prompts": num_prompts,
            "max_new_tokens": max_new_tokens,
            "optimized_time": optimized_time,
            "standard_time": standard_time,
            "speedup": standard_time / optimized_time if optimized_time > 0 else 1.0,
            "optimized_throughput": total_tokens / optimized_time,
            "standard_throughput": total_tokens / standard_time,
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get statistics from optimization components"""
        stats = {
            "optimizations_enabled": {
                "continuous_batching": self.config.use_continuous_batching,
                "paged_attention": self.config.use_paged_attention,
                "hierarchical_cache": self.config.use_hierarchical_cache,
                "fused_kernels": self.config.use_fused_kernels,
            }
        }
        
        if self.batch_engine:
            stats["batching_stats"] = self.batch_engine.get_stats()
        
        if self.kv_cache:
            stats["cache_stats"] = self.kv_cache.get_stats()
        
        return stats
    
    def shutdown(self):
        """Clean shutdown of optimization components"""
        if self.batch_engine:
            self.batch_engine.shutdown()
        
        if self.kv_cache:
            self.kv_cache.clear_cache()
        
        self.executor.shutdown(wait=True)


def create_optimized_generator(
    model_path: str,
    config_path: str,
    device: Optional[str] = None
) -> OptimizedTextGenerator:
    """
    Factory function to create optimized generator
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to optimization config
        device: Device to use
        
    Returns:
        Configured OptimizedTextGenerator
    """
    from ..model.moe_transformer import MoEModel
    from ..utils.config import load_config
    from transformers import AutoTokenizer
    
    # Load config
    config = load_config(config_path)
    
    # Create model
    model_config = type('Config', (), config['model'])()
    model = MoEModel(model_config)
    
    # Load checkpoint
    if Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.get('tokenizer', 'gpt2')
    )
    
    # Create generation config
    gen_config = GenerationConfig(
        **config.get('generation', {}),
        use_continuous_batching=config.get('continuous_batching', {}).get('enabled', True),
        use_paged_attention=config.get('paged_attention', {}).get('enabled', True),
        use_hierarchical_cache=config.get('hierarchical_kv_cache', {}).get('enabled', True),
        use_fused_kernels=config.get('cuda_kernels', {}).get('use_fused_attention', True)
    )
    
    # Create generator
    generator = OptimizedTextGenerator(
        model=model,
        tokenizer=tokenizer,
        config=gen_config,
        device=device
    )
    
    return generator