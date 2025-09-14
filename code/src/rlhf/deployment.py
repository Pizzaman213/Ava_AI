"""
Production Deployment System for RLHF Models
Implements optimized inference, model serving, and monitoring
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union, AsyncGenerator
import numpy as np
from dataclasses import dataclass, field
import logging
from pathlib import Path
import json
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time
from collections import deque, defaultdict
import pickle
import threading
from queue import Queue, PriorityQueue
import uvloop
from transformers import AutoTokenizer
import triton
import onnx
import tensorrt as trt

logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Configuration for production deployment"""
    # Model settings
    model_path: str
    model_format: str = "pytorch"  # pytorch, onnx, tensorrt
    device: str = "cuda"
    
    # Optimization settings
    use_flash_attention: bool = True
    use_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"
    use_int8_quantization: bool = False
    use_fp16: bool = True
    
    # Batching settings
    max_batch_size: int = 32
    max_sequence_length: int = 2048
    dynamic_batching: bool = True
    batch_timeout_ms: int = 50
    
    # Caching
    use_kv_cache: bool = True
    cache_size_mb: int = 4096
    cache_implementation: str = "torch"  # torch, triton
    
    # Serving settings
    num_workers: int = 4
    max_concurrent_requests: int = 100
    request_timeout_s: int = 30
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 8000
    log_predictions: bool = False
    sample_rate: float = 0.01
    
    # Model optimization
    optimize_for_latency: bool = True
    optimize_for_throughput: bool = False
    target_latency_ms: float = 100.0
    
    # Safety
    max_output_length: int = 512
    temperature_bounds: Tuple[float, float] = (0.1, 2.0)
    enable_content_filtering: bool = True

class OptimizedModel(nn.Module):
    """Optimized model wrapper for production inference"""
    def __init__(
        self,
        base_model: nn.Module,
        config: DeploymentConfig
    ):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Apply optimizations
        self._optimize_model()
        
        # Setup caching
        if config.use_kv_cache:
            self.kv_cache = self._setup_kv_cache()
        
        # Compile model if configured
        if config.use_torch_compile and hasattr(torch, 'compile'):
            self.base_model = torch.compile(
                self.base_model,
                mode=config.compile_mode,
                fullgraph=True
            )
    
    def _optimize_model(self):
        """Apply model optimizations"""
        # Quantization
        if self.config.use_int8_quantization:
            self._apply_int8_quantization()
        
        # Mixed precision
        if self.config.use_fp16:
            self.base_model = self.base_model.half()
        
        # Flash attention
        if self.config.use_flash_attention:
            self._enable_flash_attention()
        
        # Fuse operations
        self._fuse_operations()
    
    def _apply_int8_quantization(self):
        """Apply INT8 quantization to model"""
        import torch.quantization as quant
        
        # Prepare model for quantization
        self.base_model.eval()
        
        # Fuse modules
        self.base_model = quant.fuse_modules(
            self.base_model,
            [['conv', 'bn', 'relu']]
        )
        
        # Quantize
        self.base_model = quant.quantize_dynamic(
            self.base_model,
            {nn.Linear},
            dtype=torch.qint8
        )
    
    def _enable_flash_attention(self):
        """Enable Flash Attention if available"""
        try:
            from flash_attn import flash_attn_func
            
            # Replace attention modules
            for name, module in self.base_model.named_modules():
                if "attention" in name.lower():
                    # Wrap with flash attention
                    module._orig_forward = module.forward
                    module.forward = self._flash_attention_forward(module)
        except ImportError:
            logger.warning("Flash Attention not available")
    
    def _flash_attention_forward(self, module):
        """Create flash attention forward function"""
        def forward(hidden_states, attention_mask=None, **kwargs):
            # Implement flash attention logic
            # This is a simplified version
            return module._orig_forward(hidden_states, attention_mask, **kwargs)
        return forward
    
    def _fuse_operations(self):
        """Fuse operations for better performance"""
        # Fuse linear layers where possible
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Sequential):
                # Look for fuseable patterns
                fused = []
                i = 0
                while i < len(module):
                    if (i + 1 < len(module) and
                        isinstance(module[i], nn.Linear) and
                        isinstance(module[i + 1], nn.ReLU)):
                        # Fuse Linear + ReLU
                        fused.append(nn.Sequential(
                            module[i],
                            module[i + 1]
                        ))
                        i += 2
                    else:
                        fused.append(module[i])
                        i += 1
                
                # Replace with fused version
                if len(fused) < len(module):
                    new_module = nn.Sequential(*fused)
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = self.base_model
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)
                    setattr(parent, child_name, new_module)
    
    def _setup_kv_cache(self) -> Dict[str, torch.Tensor]:
        """Setup key-value cache for inference"""
        cache = {}
        
        # Estimate cache sizes
        num_layers = sum(1 for _ in self.base_model.modules() if "layer" in str(type(_)))
        hidden_size = next(self.base_model.parameters()).shape[-1]
        
        cache_shape = (
            self.config.max_batch_size,
            self.config.max_sequence_length,
            hidden_size
        )
        
        # Pre-allocate cache tensors
        device = next(self.base_model.parameters()).device
        dtype = torch.float16 if self.config.use_fp16 else torch.float32
        
        for i in range(num_layers):
            cache[f"layer_{i}_key"] = torch.zeros(
                cache_shape, device=device, dtype=dtype
            )
            cache[f"layer_{i}_value"] = torch.zeros(
                cache_shape, device=device, dtype=dtype
            )
        
        return cache
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Optimized forward pass"""
        # Use KV cache if available
        if use_cache and hasattr(self, 'kv_cache'):
            kwargs['past_key_values'] = self._get_relevant_cache(input_ids.shape[0])
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.use_fp16):
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # Update cache if used
        if use_cache and hasattr(outputs, 'past_key_values'):
            self._update_cache(outputs.past_key_values)
        
        return outputs
    
    def _get_relevant_cache(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get relevant cache entries for current batch"""
        cache_list = []
        
        for i in range(len(self.kv_cache) // 2):
            key_cache = self.kv_cache[f"layer_{i}_key"][:batch_size]
            value_cache = self.kv_cache[f"layer_{i}_value"][:batch_size]
            cache_list.append((key_cache, value_cache))
        
        return cache_list
    
    def _update_cache(self, new_cache: List[Tuple[torch.Tensor, torch.Tensor]]):
        """Update KV cache with new values"""
        for i, (key, value) in enumerate(new_cache):
            batch_size = key.shape[0]
            self.kv_cache[f"layer_{i}_key"][:batch_size] = key
            self.kv_cache[f"layer_{i}_value"][:batch_size] = value

class DynamicBatcher:
    """Dynamic batching for optimal throughput"""
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.pending_requests = PriorityQueue()
        self.batch_queue = Queue()
        self.request_id_counter = 0
        self.active_batches = {}
        
        # Start batching thread
        self.batching_thread = threading.Thread(target=self._batching_loop, daemon=True)
        self.batching_thread.start()
    
    def add_request(
        self,
        input_ids: torch.Tensor,
        generation_config: Dict[str, Any],
        callback: Callable
    ) -> int:
        """Add request to batching queue"""
        request_id = self.request_id_counter
        self.request_id_counter += 1
        
        request = {
            'id': request_id,
            'input_ids': input_ids,
            'generation_config': generation_config,
            'callback': callback,
            'timestamp': time.time(),
            'priority': generation_config.get('priority', 5)
        }
        
        # Add to priority queue (negative priority for min heap)
        self.pending_requests.put((-request['priority'], request['timestamp'], request))
        
        return request_id
    
    def _batching_loop(self):
        """Main batching loop"""
        while True:
            batch = []
            batch_start_time = time.time()
            
            # Collect requests for batch
            while (len(batch) < self.config.max_batch_size and
                   (time.time() - batch_start_time) * 1000 < self.config.batch_timeout_ms):
                
                try:
                    _, _, request = self.pending_requests.get(timeout=0.001)
                    batch.append(request)
                except:
                    if batch:  # If we have any requests, process them
                        break
                    continue
            
            if batch:
                self._process_batch(batch)
    
    def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of requests"""
        # Pad inputs to same length
        max_length = max(req['input_ids'].shape[-1] for req in batch)
        
        padded_inputs = []
        attention_masks = []
        
        for req in batch:
            input_ids = req['input_ids']
            pad_length = max_length - input_ids.shape[-1]
            
            if pad_length > 0:
                padded = F.pad(input_ids, (0, pad_length), value=0)
                mask = torch.cat([
                    torch.ones_like(input_ids),
                    torch.zeros(input_ids.shape[0], pad_length, device=input_ids.device)
                ], dim=-1)
            else:
                padded = input_ids
                mask = torch.ones_like(input_ids)
            
            padded_inputs.append(padded)
            attention_masks.append(mask)
        
        # Stack batch
        batch_input = torch.cat(padded_inputs, dim=0)
        batch_mask = torch.cat(attention_masks, dim=0)
        
        # Add to processing queue
        self.batch_queue.put({
            'requests': batch,
            'input_ids': batch_input,
            'attention_mask': batch_mask
        })

class StreamingGenerator:
    """Streaming text generation with optimizations"""
    def __init__(
        self,
        model: OptimizedModel,
        tokenizer: AutoTokenizer,
        config: DeploymentConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Setup generation cache
        self.generation_cache = {}
        
        # Beam search optimization
        self.beam_scorer = None
    
    async def generate_stream(
        self,
        input_ids: torch.Tensor,
        generation_config: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens"""
        # Validate generation config
        generation_config = self._validate_generation_config(generation_config)
        
        # Initialize generation
        current_ids = input_ids
        past_key_values = None
        generated_tokens = []
        
        for step in range(generation_config.get('max_new_tokens', 100)):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            temperature = generation_config.get('temperature', 1.0)
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k/top-p filtering
            filtered_logits = self._top_k_top_p_filtering(
                next_token_logits,
                top_k=generation_config.get('top_k', 50),
                top_p=generation_config.get('top_p', 0.9)
            )
            
            # Sample next token
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Update state
            current_ids = next_token
            past_key_values = outputs.past_key_values
            generated_tokens.append(next_token)
            
            # Decode and yield
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text
            
            # Check stopping criteria
            if next_token.item() == self.tokenizer.eos_token_id:
                break
    
    def _validate_generation_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and apply bounds to generation config"""
        validated = config.copy()
        
        # Temperature bounds
        if 'temperature' in validated:
            validated['temperature'] = np.clip(
                validated['temperature'],
                self.config.temperature_bounds[0],
                self.config.temperature_bounds[1]
            )
        
        # Max length
        if 'max_new_tokens' in validated:
            validated['max_new_tokens'] = min(
                validated['max_new_tokens'],
                self.config.max_output_length
            )
        
        return validated
    
    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 0.0,
        filter_value: float = -float('Inf')
    ) -> torch.Tensor:
        """Apply top-k and top-p filtering"""
        assert logits.dim() == 2  # batch_size x vocab_size
        
        top_k = min(top_k, logits.size(-1))
        
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        
        return logits

class ModelServer:
    """Production model server with monitoring"""
    def __init__(
        self,
        model_path: str,
        config: DeploymentConfig
    ):
        self.config = config
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model(model_path)
        
        # Setup components
        self.batcher = DynamicBatcher(config)
        self.generator = StreamingGenerator(self.model, self.tokenizer, config)
        
        # Metrics
        self.metrics = ModelMetrics(config)
        
        # Worker pool
        self.executor = ThreadPoolExecutor(max_workers=config.num_workers)
        
        # Start processing loop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def _load_model(self, model_path: str) -> Tuple[OptimizedModel, AutoTokenizer]:
        """Load and optimize model"""
        # Load base model
        from transformers import AutoModelForCausalLM
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
            device_map="auto" if self.config.device == "cuda" else None
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Wrap with optimized model
        optimized_model = OptimizedModel(base_model, self.config)
        
        # Move to device
        if self.config.device == "cuda":
            optimized_model = optimized_model.cuda()
        
        optimized_model.eval()
        
        return optimized_model, tokenizer
    
    async def generate(
        self,
        prompt: str,
        generation_config: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text from prompt"""
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_sequence_length,
            truncation=True
        )
        
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Default generation config
        if generation_config is None:
            generation_config = {
                'max_new_tokens': 100,
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 50
            }
        
        # Update metrics
        self.metrics.record_request(len(prompt))
        
        try:
            if stream:
                # Streaming generation
                return self.generator.generate_stream(
                    inputs['input_ids'],
                    generation_config
                )
            else:
                # Non-streaming generation
                result = await self._generate_batch(
                    inputs['input_ids'],
                    generation_config
                )
                
                # Update metrics
                generation_time = time.time() - start_time
                self.metrics.record_generation(
                    generation_time,
                    len(result),
                    success=True
                )
                
                return result
        
        except Exception as e:
            self.metrics.record_generation(
                time.time() - start_time,
                0,
                success=False
            )
            raise e
    
    async def _generate_batch(
        self,
        input_ids: torch.Tensor,
        generation_config: Dict[str, Any]
    ) -> str:
        """Generate text using batching"""
        future = asyncio.Future()
        
        # Add to batch queue
        request_id = self.batcher.add_request(
            input_ids,
            generation_config,
            lambda result: future.set_result(result)
        )
        
        # Wait for result
        result = await future
        
        return result
    
    def start_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Start the model server"""
        from aiohttp import web
        
        app = web.Application()
        
        # Add routes
        app.router.add_post('/generate', self._handle_generate)
        app.router.add_get('/health', self._handle_health)
        app.router.add_get('/metrics', self._handle_metrics)
        
        # Start processing thread
        processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        processing_thread.start()
        
        # Run server
        web.run_app(app, host=host, port=port)
    
    async def _handle_generate(self, request):
        """Handle generation request"""
        from aiohttp import web
        
        try:
            data = await request.json()
            prompt = data.get('prompt', '')
            generation_config = data.get('generation_config', {})
            stream = data.get('stream', False)
            
            if stream:
                # Streaming response
                response = web.StreamResponse()
                response.headers['Content-Type'] = 'text/event-stream'
                await response.prepare(request)
                
                async for token in await self.generate(prompt, generation_config, stream=True):
                    await response.write(f"data: {json.dumps({'token': token})}\n\n".encode())
                
                await response.write(b"data: [DONE]\n\n")
                return response
            else:
                # Non-streaming response
                result = await self.generate(prompt, generation_config, stream=False)
                return web.json_response({'generated_text': result})
        
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def _handle_health(self, request):
        """Health check endpoint"""
        from aiohttp import web
        
        health_status = {
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'active_requests': self.batcher.pending_requests.qsize(),
            'uptime': time.time()
        }
        
        return web.json_response(health_status)
    
    async def _handle_metrics(self, request):
        """Metrics endpoint"""
        from aiohttp import web
        
        metrics = self.metrics.get_metrics()
        return web.json_response(metrics)
    
    def _processing_loop(self):
        """Main processing loop for batches"""
        while True:
            try:
                # Get batch from queue
                batch_data = self.batcher.batch_queue.get(timeout=1.0)
                
                # Process batch
                self._process_batch_sync(batch_data)
                
            except:
                continue
    
    def _process_batch_sync(self, batch_data: Dict[str, Any]):
        """Process batch synchronously"""
        requests = batch_data['requests']
        input_ids = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        
        # Generate for batch
        with torch.no_grad():
            outputs = self.model.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,  # Use from requests
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return results
        for i, request in enumerate(requests):
            # Get generated tokens for this request
            generated = outputs[i][input_ids.shape[1]:]
            
            # Decode
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            
            # Call callback
            request['callback'](text)

class ModelMetrics:
    """Metrics collection for model server"""
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.request_count = 0
        self.generation_count = 0
        self.error_count = 0
        self.total_generation_time = 0
        self.total_tokens_generated = 0
        
        # Latency histogram
        self.latency_buckets = [10, 25, 50, 100, 250, 500, 1000, 2500]
        self.latency_histogram = defaultdict(int)
        
        # Recent latencies for percentiles
        self.recent_latencies = deque(maxlen=1000)
    
    def record_request(self, prompt_length: int):
        """Record incoming request"""
        self.request_count += 1
    
    def record_generation(self, generation_time: float, output_length: int, success: bool):
        """Record generation completion"""
        if success:
            self.generation_count += 1
            self.total_generation_time += generation_time
            self.total_tokens_generated += output_length
            
            # Update latency histogram
            latency_ms = generation_time * 1000
            self.recent_latencies.append(latency_ms)
            
            for bucket in self.latency_buckets:
                if latency_ms <= bucket:
                    self.latency_histogram[bucket] += 1
                    break
            else:
                self.latency_histogram['inf'] += 1
        else:
            self.error_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        metrics = {
            'request_count': self.request_count,
            'generation_count': self.generation_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'avg_generation_time_ms': (self.total_generation_time * 1000) / max(self.generation_count, 1),
            'avg_tokens_per_generation': self.total_tokens_generated / max(self.generation_count, 1),
            'latency_histogram': dict(self.latency_histogram)
        }
        
        # Calculate percentiles
        if self.recent_latencies:
            latencies = sorted(self.recent_latencies)
            metrics['latency_p50'] = latencies[len(latencies) // 2]
            metrics['latency_p90'] = latencies[int(len(latencies) * 0.9)]
            metrics['latency_p99'] = latencies[int(len(latencies) * 0.99)]
        
        return metrics

class ContentFilter:
    """Content filtering for safety"""
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.filters = self._load_filters()
    
    def _load_filters(self) -> List[Callable]:
        """Load content filters"""
        filters = []
        
        # Basic profanity filter
        filters.append(self._profanity_filter)
        
        # PII filter
        filters.append(self._pii_filter)
        
        # Custom filters can be added here
        
        return filters
    
    def _profanity_filter(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check for profanity"""
        # Simplified - use proper profanity detection library
        profanity_words = set()  # Load from file
        
        words = text.lower().split()
        for word in words:
            if word in profanity_words:
                return False, "Content contains inappropriate language"
        
        return True, None
    
    def _pii_filter(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check for PII"""
        import re
        
        # Check for common PII patterns
        patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        for pii_type, pattern in patterns.items():
            if re.search(pattern, text):
                return False, f"Content may contain {pii_type}"
        
        return True, None
    
    def filter(self, text: str) -> Tuple[bool, Optional[str]]:
        """Apply all filters"""
        for filter_func in self.filters:
            passed, reason = filter_func(text)
            if not passed:
                return False, reason
        
        return True, None

# Deployment utilities
def export_to_onnx(model: nn.Module, export_path: str, sample_input: torch.Tensor):
    """Export model to ONNX format"""
    model.eval()
    
    torch.onnx.export(
        model,
        sample_input,
        export_path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size', 1: 'sequence'}
        },
        opset_version=14,
        do_constant_folding=True
    )
    
    logger.info(f"Exported model to ONNX: {export_path}")

def optimize_onnx_model(onnx_path: str, optimized_path: str):
    """Optimize ONNX model"""
    import onnxruntime as ort
    from onnxruntime.transformers import optimizer
    
    # Load and optimize
    opt_model = optimizer.optimize_model(
        onnx_path,
        model_type='bert',  # Adjust based on model
        num_heads=12,  # Adjust based on model
        hidden_size=768  # Adjust based on model
    )
    
    opt_model.save_model_to_file(optimized_path)
    logger.info(f"Optimized ONNX model saved to: {optimized_path}")

def create_tensorrt_engine(onnx_path: str, engine_path: str, config: DeploymentConfig):
    """Create TensorRT engine from ONNX model"""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Build configuration
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    if config.use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Build engine
    engine = builder.build_engine(network, config)
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    logger.info(f"TensorRT engine saved to: {engine_path}")
    return engine

def benchmark_model(
    model_server: ModelServer,
    num_requests: int = 1000,
    prompt_length: int = 50
) -> Dict[str, Any]:
    """Benchmark model performance"""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    results = {
        'latencies': [],
        'throughput': 0,
        'errors': 0
    }
    
    # Generate test prompts
    test_prompt = "The quick brown fox " * (prompt_length // 4)
    
    async def run_request():
        start = time.time()
        try:
            result = await model_server.generate(
                test_prompt,
                {'max_new_tokens': 50}
            )
            latency = time.time() - start
            results['latencies'].append(latency)
        except:
            results['errors'] += 1
    
    # Run benchmark
    start_time = time.time()
    
    loop = asyncio.get_event_loop()
    tasks = [run_request() for _ in range(num_requests)]
    loop.run_until_complete(asyncio.gather(*tasks))
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    if results['latencies']:
        latencies = sorted(results['latencies'])
        results['min_latency'] = min(latencies)
        results['max_latency'] = max(latencies)
        results['avg_latency'] = sum(latencies) / len(latencies)
        results['p50_latency'] = latencies[len(latencies) // 2]
        results['p90_latency'] = latencies[int(len(latencies) * 0.9)]
        results['p99_latency'] = latencies[int(len(latencies) * 0.99)]
        results['throughput'] = num_requests / total_time
        results['success_rate'] = len(latencies) / num_requests
    
    return results

# Example usage
if __name__ == "__main__":
    # Configuration
    config = DeploymentConfig(
        model_path="path/to/model",
        use_flash_attention=True,
        use_torch_compile=True,
        use_fp16=True,
        max_batch_size=32,
        dynamic_batching=True
    )
    
    # Create server
    server = ModelServer("path/to/model", config)
    
    # Run benchmark
    benchmark_results = benchmark_model(server)
    print(f"Benchmark results: {benchmark_results}")
    
    # Start server
    server.start_server(host="0.0.0.0", port=8080)