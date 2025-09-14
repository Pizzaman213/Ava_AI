# Text Generation Examples

This guide provides comprehensive examples for generating text with the MoE++ model, including support for both base and fine-tuned models.

## Table of Contents
- [Quick Start](#quick-start)
- [Basic Generation](#basic-generation)
- [Advanced Generation Strategies](#advanced-generation-strategies)
- [Batch Generation](#batch-generation)
- [RLHF Generation](#rlhf-generation)
- [Speculative Decoding](#speculative-decoding)
- [RAG Integration](#rag-integration)
- [Constrained Generation](#constrained-generation)
- [Command Line Usage](#command-line-usage)

## Quick Start

### Using the Generation Scripts

All generation scripts are now in the `examples/generation` folder and support both base and fine-tuned models.

### Main Interactive Generation Script

```bash
# Auto-detect best checkpoint and run interactive mode
python generate.py

# Use specific checkpoint
python generate.py --checkpoint ../../outputs/sft_model/checkpoint-best

# Run test generation
python generate.py --test

# List available checkpoints
python generate.py --list
```

### Interactive Commands

Once in interactive mode:
- Type your prompt and press Enter to generate
- `/help` - Show available commands
- `/set` - Show current generation settings
- `/set temperature 0.9` - Change temperature
- `/set max_tokens 50` - Change max tokens
- `/quit` - Exit

### Advanced Usage

```bash
# Use the full interactive generator directly
python interactive_generation.py ../../outputs/moe_train_20250819_182455/best

# Run simple test script
python simple_test.py

# Use legacy simple generation (being phased out)
python simple_generation.py --checkpoint ../../outputs/checkpoints/best
```

## Basic Generation

### Simple Generation
```python
# Greedy decoding (deterministic)
response = generator.generate(
    "What is machine learning?",
    max_length=200,
    do_sample=False
)

# Sampling with temperature
response = generator.generate(
    "Tell me a story about",
    max_length=300,
    do_sample=True,
    temperature=0.8
)
```

### Generation Parameters
```python
from src.inference.generate import GenerationConfig

# Create custom generation config
config = GenerationConfig(
    max_length=512,
    min_length=50,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.2,
    length_penalty=1.0,
    no_repeat_ngram_size=3,
    do_sample=True,
    early_stopping=True
)

response = generator.generate(
    "The future of artificial intelligence",
    generation_config=config
)
```

## Advanced Generation Strategies

### Top-k Sampling
```python
# Limit vocabulary to top-k tokens
response = generator.generate(
    "Once upon a time",
    max_length=200,
    do_sample=True,
    top_k=50,
    temperature=0.8
)
```

### Top-p (Nucleus) Sampling
```python
# Sample from smallest set with cumulative probability >= p
response = generator.generate(
    "The meaning of life is",
    max_length=200,
    do_sample=True,
    top_p=0.92,
    temperature=0.7
)
```

### Typical Sampling
```python
# Sample based on local typicality
config = GenerationConfig(
    max_length=256,
    do_sample=True,
    typical_p=0.95,
    temperature=0.8
)
response = generator.generate("In the year 2050", generation_config=config)
```

### Contrastive Search
```python
# Use contrastive decoding for more coherent text
config = GenerationConfig(
    max_length=256,
    use_contrastive_search=True,
    penalty_alpha=0.6,
    top_k=5
)
response = generator.generate("Explain neural networks", generation_config=config)
```

### Beam Search
```python
# Generate multiple sequences and select best
config = GenerationConfig(
    max_length=256,
    num_beams=5,
    num_return_sequences=3,
    early_stopping=True,
    length_penalty=1.2,
    no_repeat_ngram_size=2
)
responses = generator.generate("Write a haiku about", generation_config=config)
```

### Diverse Beam Search
```python
# Generate diverse outputs with beam search
config = GenerationConfig(
    max_length=256,
    num_beams=6,
    num_beam_groups=3,
    diversity_penalty=0.5,
    num_return_sequences=3
)
responses = generator.generate("Creative uses for AI include", generation_config=config)
```

## RLHF Generation

For examples using RLHF (Reinforcement Learning from Human Feedback) trained models, see the [RLHF generation examples](rlhf/README.md).

### Quick RLHF Example
```python
# Generate with RLHF-trained model
from src.model.moe_transformer import MoEForCausalLM

# Load RLHF model
model = MoEForCausalLM.from_pretrained("outputs/rlhf_model")
generator = TextGenerator(model, tokenizer)

# Generate with preference-aligned model
response = generator.generate(
    "Explain climate change",
    generation_config=GenerationConfig(
        max_length=256,
        temperature=0.7,
        do_sample=True
    )
)
```

See [rlhf/](rlhf/) for:
- Reward model integration
- Best-of-N sampling
- Preference-based generation
- Safety filtering

## Batch Generation

### Basic Batch Processing
```python
# Generate for multiple prompts efficiently
prompts = [
    "The future of AI is",
    "Quantum computers will",
    "In the year 2050,",
    "The most important scientific discovery"
]

responses = generator.generate(
    prompts,
    generation_config=GenerationConfig(
        max_length=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
)

for prompt, response in zip(prompts, responses):
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")
```

### Streaming Generation
```python
# Stream tokens as they are generated
def stream_example():
    prompt = "Once upon a time in a distant galaxy"
    
    print(f"Prompt: {prompt}")
    print("Response: ", end="")
    
    for token in generator.generate_stream(
        prompt,
        max_length=200,
        temperature=0.8
    ):
        print(token, end="", flush=True)
    print()  # New line at end

stream_example()
```

## Speculative Decoding

### Basic Speculative Decoding
```python
from src.inference.generate import create_text_generator

# Create generator with speculative decoding
generator = create_text_generator(
    model_path="outputs/checkpoints/best",
    use_speculative_decoding=True
)

# Generate with 2-3x speedup
config = GenerationConfig(
    max_length=512,
    temperature=0.8,
    use_speculative_decoding=True
)

import time
start = time.time()
response = generator.generate("Write a detailed explanation of", generation_config=config)
end = time.time()

print(f"Generated in {end - start:.2f} seconds")
print(f"Response: {response}")
```

### Advanced Speculative Decoding
```python
from src.model.speculative import SpeculativeDecoder, SpeculativeConfig

# Configure speculative decoding
spec_config = SpeculativeConfig(
    draft_model_size="small",  # Use smaller draft model
    gamma=5,  # Number of draft tokens
    temperature=0.8,
    acceptance_threshold=0.9,
    fallback_on_rejection=True
)

# Create speculative decoder
spec_decoder = SpeculativeDecoder(
    target_model=model,
    config=spec_config
)

# Generate with statistics
result = spec_decoder.generate_with_stats(
    "Explain the theory of relativity",
    max_length=300
)

print(f"Response: {result['text']}")
print(f"Speedup: {result['speedup']:.2f}x")
print(f"Acceptance rate: {result['acceptance_rate']:.2%}")
```

## RAG Integration

### Basic RAG Generation
```python
from src.inference.generate import create_text_generator

# Create generator with RAG
generator = create_text_generator(
    model_path="outputs/checkpoints/best",
    use_rag=True
)

config = GenerationConfig(
    max_length=512,
    use_rag=True,
    use_chain_of_thought=True
)

# Generate with retrieval augmentation
response = generator.generate(
    "What are the latest developments in quantum computing?",
    generation_config=config
)
print(response)
```

### Custom RAG Pipeline
```python
from src.model.rag_module import RAGModule, RAGConfig

# Configure RAG
rag_config = RAGConfig(
    index_path="data/knowledge_base",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    top_k=5,
    rerank=True
)

# Create RAG module
rag_module = RAGModule(rag_config)

# Generate with custom retrieval
context = rag_module.retrieve("quantum computing breakthroughs 2024")
augmented_prompt = f"Context: {context}\n\nQuestion: What are the latest quantum computing breakthroughs?"

response = generator.generate(
    augmented_prompt,
    max_length=300,
    temperature=0.7
)
```

## Constrained Generation

### Length Constraints
```python
# Generate with specific length requirements
response = generator.generate(
    "Write a summary of machine learning",
    generation_config=GenerationConfig(
        min_length=100,
        max_length=150,
        length_penalty=2.0
    )
)
```

### Token Constraints
```python
# Custom token filtering
def avoid_words_constraint(batch_id, input_ids):
    """Avoid generating certain words"""
    banned_tokens = tokenizer(["hate", "violence", "explicit"], add_special_tokens=False).input_ids
    banned_ids = [id for sublist in banned_tokens for id in sublist]
    
    allowed = list(range(tokenizer.vocab_size))
    for banned_id in banned_ids:
        if banned_id in allowed:
            allowed.remove(banned_id)
    
    return allowed

response = generator.generate(
    "Generate a children's story",
    prefix_allowed_tokens_fn=avoid_words_constraint,
    max_length=200
)
```

### Structured Generation (JSON)
```python
# Generate valid JSON
json_prompt = """Generate a JSON object for a person with name, age, and skills:
{"""

response = generator.generate(
    json_prompt,
    max_length=100,
    temperature=0.7,
    stop_sequences=["}"]
)

# Complete the JSON
full_json = json_prompt + response + "}"
print(full_json)

# Validate
import json
try:
    parsed = json.loads(full_json)
    print("Valid JSON generated:", parsed)
except json.JSONDecodeError:
    print("Invalid JSON generated")
```

## Command Line Usage

### Simple Generation
```bash
# Basic generation
python ../../scripts/generate_simple.py \
  --checkpoint outputs/checkpoints/best \
  --prompt "Explain machine learning" \
  --max-length 256
```

### Advanced Generation
```bash
# With sampling parameters
python ../../scripts/generate.py \
  --checkpoint outputs/checkpoints/best \
  --prompt "What is the future of AI?" \
  --temperature 0.9 \
  --top-p 0.95 \
  --top-k 50 \
  --num-return-sequences 3
```

### Interactive Mode
```bash
# Start interactive session
python ../../scripts/interactive.py --checkpoint outputs/checkpoints/best

# With specific device
python ../../scripts/interactive.py \
  --checkpoint outputs/checkpoints/best \
  --device mps  # For Apple Silicon

python ../../scripts/interactive.py \
  --checkpoint outputs/checkpoints/best \
  --device cuda:0  # For specific GPU
```

## Performance Optimization

### Memory-Efficient Generation
```python
# Enable memory optimizations
config = GenerationConfig(
    max_length=2048,
    use_cache=True,
    use_flash_attention=True,
    use_memory_efficient_attention=True
)

# For long sequences
generator.model.config.gradient_checkpointing = True
```

### Multi-GPU Generation
```python
import torch

# Distributed generation
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    generator = TextGenerator(model, tokenizer)
```

### Generation with Quantization
```python
# Load quantized model for faster inference
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = MoEForCausalLM.from_pretrained(
    "outputs/checkpoints/best",
    quantization_config=quantization_config,
    device_map="auto"
)

generator = TextGenerator(model, tokenizer)
```

## Monitoring and Debugging

### Generation Statistics
```python
# Get detailed generation statistics
config = GenerationConfig(
    max_length=256,
    output_scores=True,
    output_attentions=True,
    return_dict_in_generate=True
)

outputs = generator.model.generate(
    input_ids=tokenizer(prompt, return_tensors="pt").input_ids,
    generation_config=config
)

print(f"Generated tokens: {outputs.sequences.shape[1]}")
print(f"Generation scores shape: {outputs.scores[0].shape if outputs.scores else 'N/A'}")
```

### Expert Usage Analysis
```python
# Monitor which experts are being used during generation
def analyze_expert_usage(prompt, max_length=100):
    model.eval()
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
    
    # Analyze expert routing patterns
    # (Requires custom model modifications to expose routing decisions)
    print(f"Generated {len(outputs.sequences[0])} tokens")

analyze_expert_usage("Explain transformer architecture")
```

## Best Practices

1. **Temperature Selection**:
   - 0.3-0.7: Focused, deterministic responses
   - 0.7-0.9: Balanced creativity
   - 0.9-1.2: More creative, diverse outputs

2. **Sampling Strategy**:
   - Use top-p (0.9-0.95) for general text
   - Use top-k (40-80) for more controlled generation
   - Combine both for best results

3. **Performance Tips**:
   - Enable Flash Attention for long sequences
   - Use speculative decoding for 2-3x speedup
   - Batch prompts when possible
   - Use appropriate precision (bf16/fp16)

4. **Quality Control**:
   - Set appropriate repetition_penalty (1.1-1.3)
   - Use no_repeat_ngram_size for avoiding repetition
   - Set min_length to avoid truncated responses

## Troubleshooting

### Repetitive Output
```python
config = GenerationConfig(
    repetition_penalty=1.3,
    no_repeat_ngram_size=3,
    temperature=0.8,
    top_p=0.95
)
```

### Incoherent Output
```python
# Use lower temperature and contrastive search
config = GenerationConfig(
    temperature=0.6,
    use_contrastive_search=True,
    penalty_alpha=0.6,
    top_k=5
)
```

### Out of Memory
```python
# Reduce batch size or use CPU offloading
generator.model.config.use_cache = False
generator.model.to('cpu')  # Offload to CPU
```

For more examples and advanced usage, see:
- [Fine-tuning Guide](../fine_tuning/README.md)
- [Training Documentation](../../docs/training.md)
- [API Reference](../../docs/api_reference.md)