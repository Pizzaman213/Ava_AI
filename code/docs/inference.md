# Inference Guide

## Table of Contents
- [Quick Start](#quick-start)
- [Model Loading](#model-loading)
- [Generation Strategies](#generation-strategies)
- [Speculative Decoding](#speculative-decoding)
- [Optimization Techniques](#optimization-techniques)
- [Deployment Options](#deployment-options)
- [Performance Tuning](#performance-tuning)
- [API Reference](#api-reference)

## Quick Start

### Basic Inference

```python
from moe_llm import MoEModel, TextGenerator

# Load model
model = MoEModel.from_pretrained("path/to/model")
generator = TextGenerator(model)

# Generate text
response = generator.generate(
    "What is the meaning of life?",
    max_length=256,
    temperature=0.7,
    top_p=0.9
)

print(response)
```

### Batch Inference

```python
# Process multiple prompts
prompts = [
    "Explain quantum computing",
    "Write a Python function to sort a list",
    "What are the benefits of exercise?"
]

responses = generator.generate_batch(
    prompts,
    max_length=256,
    temperature=0.7,
    num_return_sequences=1
)

for prompt, response in zip(prompts, responses):
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")
```

### Streaming Generation

```python
# Stream tokens as they're generated
for token in generator.generate_stream(
    "Tell me a story about a robot",
    max_length=512
):
    print(token, end='', flush=True)
```

## Model Loading

### Loading Strategies

```python
# 1. Full precision loading
model = MoEModel.from_pretrained(
    "path/to/model",
    device_map="auto",
    torch_dtype=torch.float32
)

# 2. Half precision loading (faster, less memory)
model = MoEModel.from_pretrained(
    "path/to/model",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 3. Quantized loading (INT8)
model = MoEModel.from_pretrained(
    "path/to/model",
    device_map="auto",
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

# 4. Sharded loading for large models
model = MoEModel.from_pretrained(
    "path/to/model",
    device_map="balanced",
    max_memory={0: "24GB", 1: "24GB", "cpu": "128GB"}
)

# 5. Loading with specific attention variant
model = MoEModel.from_pretrained(
    "path/to/model",
    attention_variant="linear",  # For extreme length
    device_map="auto"
)
```

### Device Management

```python
# Manual device mapping
device_map = {
    "embeddings": 0,
    "layers.0-11": 0,
    "layers.12-23": 1,
    "lm_head": 1
}

model = MoEModel.from_pretrained(
    "path/to/model",
    device_map=device_map
)

# CPU offloading for large models
model = MoEModel.from_pretrained(
    "path/to/model",
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True
)
```

### Expert Loading Optimization

```python
# Load only frequently used experts
from moe_llm import ExpertPruningConfig

pruning_config = ExpertPruningConfig(
    keep_top_k_experts=32,  # Keep only top 32 experts
    pruning_method="usage_frequency",
    load_pruned_only=True
)

model = MoEModel.from_pretrained(
    "path/to/model",
    expert_pruning_config=pruning_config
)
```

## Generation Strategies

### Sampling Methods

```python
# 1. Greedy Decoding (deterministic)
response = generator.generate(
    prompt,
    do_sample=False,
    max_length=256
)

# 2. Top-k Sampling
response = generator.generate(
    prompt,
    do_sample=True,
    top_k=50,
    temperature=0.8
)

# 3. Top-p (Nucleus) Sampling
response = generator.generate(
    prompt,
    do_sample=True,
    top_p=0.9,
    temperature=0.7
)

# 4. Typical Sampling
response = generator.generate(
    prompt,
    do_sample=True,
    typical_p=0.95,
    temperature=0.8
)

# 5. Contrastive Search
response = generator.generate(
    prompt,
    penalty_alpha=0.6,
    top_k=4,
    use_contrastive_search=True
)
```

### Beam Search

```python
# Standard beam search
response = generator.generate(
    prompt,
    num_beams=5,
    early_stopping=True,
    length_penalty=1.2
)

# Diverse beam search
response = generator.generate(
    prompt,
    num_beams=5,
    num_beam_groups=5,
    diversity_penalty=0.5
)
```

### Constrained Generation

```python
# 1. Length constraints
response = generator.generate(
    prompt,
    min_length=100,
    max_length=200,
    length_penalty=1.5
)

# 2. Token constraints
def token_constraint(batch_id, previous_tokens):
    # Disallow certain tokens
    forbidden_tokens = [tokenizer.eos_token_id, tokenizer.pad_token_id]
    return [i for i in range(vocab_size) if i not in forbidden_tokens]

response = generator.generate(
    prompt,
    prefix_allowed_tokens_fn=token_constraint
)

# 3. Structured generation (JSON)
from moe_llm.generation import JSONConstraint

json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}}
    }
}

constraint = JSONConstraint(json_schema)
response = generator.generate(
    "Generate a person profile:",
    constraints=[constraint]
)
```

## Speculative Decoding

### Basic Setup

```python
from moe_llm import SpeculativeDecoder

# Initialize with draft model
speculative_decoder = SpeculativeDecoder(
    target_model=model,
    draft_model_name="small_draft_model",
    gamma=5,  # Number of draft tokens
    temperature=0.7
)

# Generate with speculation
response = speculative_decoder.generate(
    prompt,
    max_length=256,
    acceptance_threshold=0.9
)

print(f"Speedup: {speculative_decoder.get_speedup():.2f}x")
```

### Advanced Speculative Decoding

```python
# Tree-based speculation
from moe_llm import TreeSpeculativeDecoder

tree_decoder = TreeSpeculativeDecoder(
    target_model=model,
    draft_model=draft_model,
    tree_depth=3,
    branching_factor=2,
    temperature_schedule=[1.0, 0.9, 0.8]  # Per depth
)

# Self-speculative decoding (using early layers)
from moe_llm import SelfSpeculativeDecoder

self_decoder = SelfSpeculativeDecoder(
    model=model,
    draft_layers=12,  # Use first 12 layers as draft
    gamma=4
)
```

### Speculation Monitoring

```python
# Track speculation efficiency
stats = speculative_decoder.generate_with_stats(
    prompt,
    max_length=256
)

print(f"Acceptance rate: {stats['acceptance_rate']:.2%}")
print(f"Average accepted tokens: {stats['avg_accepted']:.2f}")
print(f"Time saved: {stats['time_saved']:.2f}s")
```

## Optimization Techniques

### KV-Cache Management

```python
# 1. Static KV-cache allocation
generator = TextGenerator(
    model,
    kv_cache_config={
        "max_batch_size": 32,
        "max_sequence_length": 2048,
        "dtype": torch.float16
    }
)

# 2. Dynamic KV-cache with eviction
from moe_llm import DynamicKVCache

kv_cache = DynamicKVCache(
    max_memory_mb=4096,
    eviction_policy="lru",
    compression_ratio=0.5
)

generator = TextGenerator(model, kv_cache=kv_cache)

# 3. Paged attention (vLLM-style)
from moe_llm import PagedKVCache

paged_cache = PagedKVCache(
    block_size=16,
    num_blocks=1024,
    num_layers=model.config.num_layers
)
```

### Continuous Batching

```python
from moe_llm import ContinuousBatchingEngine

# Initialize engine
engine = ContinuousBatchingEngine(
    model=model,
    max_batch_size=256,
    max_sequence_length=2048,
    scheduling_policy="fcfs"  # first-come-first-serve
)

# Add requests
request_ids = []
for prompt in prompts:
    req_id = engine.add_request(
        prompt=prompt,
        max_tokens=256,
        temperature=0.7
    )
    request_ids.append(req_id)

# Process requests
while not engine.is_empty():
    engine.step()
    
    # Check completed requests
    for req_id in request_ids:
        if engine.has_completed(req_id):
            result = engine.get_result(req_id)
            print(f"Request {req_id}: {result}")
```

### Flash Decoding

```python
# Enable Flash Attention 2 for inference
model.config.use_flash_attention = True
model.config.flash_attention_config = {
    "enable_flash_decoding": True,
    "window_size": 2048,
    "softmax_scale": 1.0 / math.sqrt(head_dim)
}

# Verify Flash Attention is being used
if model.is_using_flash_attention():
    print("Flash Attention 2 enabled for inference")
```

### NEW: Attention Variants for Inference

```python
# 1. Linear Attention for extreme length (100k+ tokens)
model.set_attention_variant("linear")
response = generator.generate(
    very_long_prompt,  # 100k tokens
    max_new_tokens=1000,
    temperature=0.7
)

# 2. Streaming Attention for chat/conversational models
model.set_attention_variant("streaming", 
    num_sink_tokens=4,
    recent_window_size=1024
)
# Maintains context indefinitely with fixed memory
for user_message in conversation:
    response = generator.generate(
        user_message,
        max_new_tokens=256,
        use_cache=True  # Reuse KV cache
    )

# 3. Sparse Attention for long documents
model.set_attention_variant("sparse",
    sparse_global_tokens=128,
    sparse_local_window_size=512,
    sparse_random_blocks=3
)
# Process 64k token documents efficiently
summary = generator.generate(
    f"Summarize this document: {long_document}",
    max_new_tokens=500
)

# 4. Cached Attention for repetitive tasks
model.set_attention_variant("cached",
    cache_size=128,
    cache_refresh_interval=100
)
# 10-50% speedup on similar prompts
for similar_prompt in similar_prompts:
    response = generator.generate(similar_prompt)

# 5. ALiBi for length extrapolation
model.set_attention_variant("alibi")
# Generate beyond training length without quality loss
response = generator.generate(
    prompt,
    max_length=8192  # Even if trained on 2048
)
```

### Choosing the Right Attention Variant

| Use Case | Attention Variant | Benefits | Trade-offs |
|----------|------------------|----------|------------|
| Chat/Conversational | Streaming | Infinite context, fixed memory | Slight quality drop |
| Long Documents (16k+) | Sparse | Efficient long-range | Complex implementation |
| Extreme Length (100k+) | Linear | O(n) complexity | Different attention pattern |
| Code Generation | Sliding Window | Good local context | Limited global view |
| Repetitive Tasks | Cached | 10-50% speedup | Memory for cache |
| Length Extrapolation | ALiBi | No position limit | Slightly different training |

### Model Quantization

```python
# 1. Dynamic quantization
from moe_llm import quantize_model

quantized_model = quantize_model(
    model,
    quantization_config={
        "bits": 8,
        "group_size": 128,
        "damp_percent": 0.01,
        "desc_act": True,
        "sym": False
    }
)

# 2. AWQ quantization
from moe_llm import AWQQuantizer

quantizer = AWQQuantizer(
    w_bit=4,
    group_size=128,
    zero_point=True
)

quantized_model = quantizer.quantize(
    model,
    calibration_data=calibration_dataset
)

# 3. GPTQ quantization
from moe_llm import GPTQQuantizer

gptq_quantizer = GPTQQuantizer(
    bits=4,
    group_size=128,
    desc_act=False,
    sym=True,
    true_sequential=True
)

quantized_model = gptq_quantizer.quantize(model)
```

## Deployment Options

### Local Deployment

```python
# FastAPI server
from fastapi import FastAPI
from moe_llm import InferenceServer

app = FastAPI()
server = InferenceServer(model, max_batch_size=32)

@app.post("/generate")
async def generate(prompt: str, max_tokens: int = 256):
    response = await server.generate_async(
        prompt=prompt,
        max_tokens=max_tokens
    )
    return {"response": response}

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
```

### TorchServe Deployment

```python
# model_handler.py
import torch
from ts.torch_handler.base_handler import BaseHandler

class MoEHandler(BaseHandler):
    def initialize(self, context):
        self.model = MoEModel.from_pretrained(
            context.system_properties.get("model_dir")
        )
        self.generator = TextGenerator(self.model)
    
    def preprocess(self, data):
        return data[0].get("prompt")
    
    def inference(self, prompt):
        return self.generator.generate(
            prompt,
            max_length=256,
            temperature=0.7
        )
    
    def postprocess(self, output):
        return [{"response": output}]
```

### Triton Inference Server

```python
# model.py for Triton
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.model = MoEModel.from_pretrained(
            args["model_repository"]
        )
        self.generator = TextGenerator(self.model)
    
    def execute(self, requests):
        responses = []
        
        for request in requests:
            prompt = pb_utils.get_input_tensor_by_name(
                request, "prompt"
            ).as_numpy()[0].decode()
            
            output = self.generator.generate(prompt)
            
            output_tensor = pb_utils.Tensor(
                "output",
                np.array([output.encode()])
            )
            
            responses.append(
                pb_utils.InferenceResponse([output_tensor])
            )
        
        return responses
```

### Ray Serve Deployment

```python
import ray
from ray import serve

@serve.deployment(
    num_replicas=2,
    ray_actor_options={"num_gpus": 1}
)
class MoEDeployment:
    def __init__(self):
        self.model = MoEModel.from_pretrained("path/to/model")
        self.generator = TextGenerator(self.model)
    
    async def __call__(self, request):
        prompt = await request.json()
        response = self.generator.generate(
            prompt["text"],
            **prompt.get("params", {})
        )
        return {"response": response}

# Deploy
serve.run(MoEDeployment.bind())
```

## Performance Tuning

### Benchmarking

```python
from moe_llm import benchmark_inference

# Run comprehensive benchmark
results = benchmark_inference(
    model=model,
    batch_sizes=[1, 8, 16, 32],
    sequence_lengths=[128, 256, 512, 1024],
    num_iterations=100,
    warmup_iterations=10
)

# Print results
print("Throughput (tokens/sec):")
print(results["throughput_table"])
print("\nLatency (ms/token):")
print(results["latency_table"])
```

### Memory Profiling

```python
import torch.profiler

# Profile memory usage
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    profile_memory=True,
    record_shapes=True
) as prof:
    response = generator.generate(prompt, max_length=256)

# Analyze results
print(prof.key_averages().table(
    sort_by="cuda_memory_usage",
    row_limit=10
))
```

### Optimization Checklist

1. **Model Loading**
   - Use appropriate precision (BF16 for most cases)
   - Enable device_map="auto" for multi-GPU
   - Consider quantization for memory-constrained environments

2. **Generation**
   - Use speculative decoding for long outputs
   - Enable KV-cache optimization
   - Tune sampling parameters

3. **Batching**
   - Use continuous batching for serving
   - Optimize batch sizes for your hardware
   - Enable padding optimization

4. **Hardware**
   - Use Flash Attention 2 when available
   - Enable TF32 for Ampere GPUs
   - Optimize CUDA graphs for static shapes

## API Reference

### TextGenerator Class

```python
class TextGenerator:
    def __init__(
        self,
        model: MoEModel,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        generation_config: Optional[GenerationConfig] = None,
        device: Optional[str] = None
    ):
        """Initialize text generator"""
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, List[str]]:
        """Generate text from prompt(s)"""
    
    def generate_stream(
        self,
        prompt: str,
        **kwargs
    ) -> Iterator[str]:
        """Stream generated tokens"""
    
    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """Batch generation with optimizations"""
```

### SpeculativeDecoder Class

```python
class SpeculativeDecoder:
    def __init__(
        self,
        target_model: MoEModel,
        draft_model: Union[str, nn.Module],
        gamma: int = 5,
        temperature: float = 1.0,
        top_p: float = 0.9,
        device: Optional[str] = None
    ):
        """Initialize speculative decoder"""
    
    def generate(
        self,
        prompt: str,
        max_length: int,
        acceptance_threshold: float = 0.9,
        fallback_on_rejection: bool = True,
        **kwargs
    ) -> str:
        """Generate with speculative decoding"""
    
    def get_speedup(self) -> float:
        """Get average speedup factor"""
```

For more details, see:
- [API Reference](api_reference.md)
- [Performance Tuning](performance_tuning.md)
- [Deployment Guide](deployment.md)