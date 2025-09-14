# API Reference

## Table of Contents
- [Model Classes](#model-classes)
- [Configuration Classes](#configuration-classes)
- [Training Classes](#training-classes)
- [Data Processing](#data-processing)
- [RLHF Components](#rlhf-components)
- [Optimization Utilities](#optimization-utilities)
- [Inference Components](#inference-components)
- [Distributed Training](#distributed-training)

## Model Classes

### MoEModel

```python
class MoEModel(nn.Module):
    """
    Main MoE++ Transformer model with hierarchical routing and MoD.
    
    Args:
        config (MoEConfig): Model configuration object
    
    Attributes:
        embeddings (MoEEmbeddings): Token and position embeddings
        layers (nn.ModuleList): List of MoE transformer layers
        ln_f (RMSNorm): Final layer normalization
        
    Methods:
        forward(input_ids, attention_mask=None, position_ids=None, 
                past_key_values=None, use_cache=True, output_hidden_states=False,
                output_router_probs=False, return_dict=True)
            Forward pass through the model.
            
        get_input_embeddings() -> nn.Embedding
            Get input embedding layer.
            
        set_input_embeddings(embeddings: nn.Embedding)
            Set input embedding layer.
            
        get_router_probs(hidden_states: torch.Tensor) -> List[torch.Tensor]
            Get routing probabilities for each layer.
            
        save_pretrained(save_directory: str)
            Save model to directory.
            
        from_pretrained(pretrained_model_name_or_path: str, **kwargs)
            Load model from pretrained weights.
    """
```

### MoELayer

```python
class MoELayer(nn.Module):
    """
    Single MoE transformer layer with attention and expert FFN.
    
    Args:
        config (MoEConfig): Model configuration
        layer_idx (int): Layer index for positional encoding
        
    Attributes:
        attention (BaseAttention): Attention module (variant based on config)
        moe_block (MoEBlock): Mixture of experts FFN block
        ln_1 (RMSNorm): Pre-attention layer norm
        ln_2 (RMSNorm): Pre-FFN layer norm
        mod_router (DepthRouter): Mixture of Depths router
        
    Methods:
        forward(hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, use_cache=True, output_router_probs=False)
            Forward pass through the layer.
            
        get_expert_assignments(hidden_states) -> torch.Tensor
            Get expert assignments for tokens.
            
        set_attention_variant(variant: str, **kwargs)
            Dynamically change attention mechanism.
    """
```

### Expert

```python
class Expert(nn.Module):
    """
    Single expert network with SwiGLU activation.
    
    Args:
        config (MoEConfig): Model configuration
        expert_id (int): Unique expert identifier
        
    Attributes:
        w1 (nn.Linear): First linear transformation
        w2 (nn.Linear): Second linear transformation (gate)
        w3 (nn.Linear): Output projection
        
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor
            Apply expert transformation with SwiGLU.
            
        get_weight_norm() -> float
            Get L2 norm of expert weights.
    """
```

### HierarchicalRouter

```python
class HierarchicalRouter(nn.Module):
    """
    Two-level hierarchical routing for experts.
    
    Args:
        config (MoEConfig): Model configuration
        
    Attributes:
        domain_router (nn.Linear): Top-level domain router
        expert_routers (nn.ModuleList): Per-domain expert routers
        temperature (float): Softmax temperature
        
    Methods:
        forward(hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Dict]
            Compute routing probabilities and auxiliary losses.
            
        get_load_balancing_loss(router_probs: torch.Tensor) -> torch.Tensor
            Compute load balancing auxiliary loss.
            
        get_router_z_loss(router_logits: torch.Tensor) -> torch.Tensor
            Compute router z-loss for stability.
    """
```

### Attention Classes

```python
# Base Attention Class
class BaseAttention(nn.Module):
    """
    Base class for all attention variants.
    
    Args:
        config (MoEConfig): Model configuration
        layer_idx (int): Layer index
    """

# Attention Variants
class SlidingWindowAttention(BaseAttention):
    """Local window attention with O(n×w) complexity."""

class SparseAttention(BaseAttention):
    """BigBird-style sparse attention with global tokens + local windows + random blocks."""

class StreamingAttentionWithSinks(BaseAttention):
    """Attention with sink tokens for continuous generation."""

class ALiBiAttention(BaseAttention):
    """Attention with Linear Biases - no position embeddings needed."""

class LinearAttention(BaseAttention):
    """O(n) complexity attention using kernel trick."""

class CachedAttention(BaseAttention):
    """Attention with pattern caching for repetitive tasks."""

# Factory Function
def create_attention_layer(config: MoEConfig, layer_idx: int) -> BaseAttention:
    """Create attention layer based on config.attention_variant."""
```

### Position Embeddings

```python
class xPosRotaryEmbedding(nn.Module):
    """
    Extrapolatable rotary position embeddings.
    
    Args:
        dim (int): Embedding dimension
        max_position (int): Maximum sequence length during training
        base (float): Base for rotary embeddings
        
    Methods:
        forward(x, seq_len) -> Tuple[torch.Tensor, torch.Tensor]
            Apply rotary embeddings with extrapolation support.
    """
```

## Configuration Classes

### MoEConfig

```python
@dataclass
class MoEConfig:
    """
    Configuration for MoE++ model.
    
    Attributes:
        # Model dimensions
        hidden_size: int = 2048
        num_layers: int = 24
        vocab_size: int = 50257
        max_position_embeddings: int = 2048
        
        # Attention configuration
        num_attention_heads: int = 16
        num_key_value_heads: Optional[int] = None  # For GQA
        attention_dropout: float = 0.0
        use_flash_attention: bool = True
        attention_variant: str = "standard"  # NEW
        
        # NEW: Attention variant specific configs
        # Sliding Window
        sliding_window_size: int = 512
        
        # Sparse Attention
        sparse_global_tokens: int = 128
        sparse_random_blocks: int = 3
        sparse_local_window_size: int = 256
        
        # Streaming Attention
        num_sink_tokens: int = 4
        recent_window_size: int = 1024
        
        # Linear Attention
        linear_eps: float = 1e-6
        
        # Cached Attention  
        cache_size: int = 128
        cache_refresh_interval: int = 100
        
        # xPos embeddings
        use_xpos: bool = False
        xpos_decay_base: float = 512.0
        
        # MoE configuration
        num_experts: int = 64
        experts_per_token: int = 4
        expert_capacity_factor: float = 1.25
        use_hierarchical_routing: bool = True
        num_expert_domains: int = 8
        
        # NEW: MoE enhancements
        parallel_expert_processing: bool = True
        use_adaptive_capacity: bool = False
        capacity_warmup_steps: int = 1000
        capacity_ema_decay: float = 0.99
        expert_dropout: float = 0.0
        use_importance_weighting: bool = True
        importance_temp: float = 0.5
        load_balancing_loss_fn: str = "smooth_l1"
        smooth_l1_beta: float = 1.0
        
        # MoD configuration
        use_mod: bool = True
        mod_mode: str = "learned"  # "learned", "entropy", "attention"
        target_mod_rate: float = 0.5
        
        # Activation configuration
        hidden_act: str = "swiglu"
        intermediate_size: int = 8192
        
        # Training configuration
        dropout: float = 0.1
        load_balancing_loss_weight: float = 0.01
        router_z_loss_weight: float = 0.001
        router_noise_epsilon: float = 1e-2
        
        # Optimization
        use_gradient_checkpointing: bool = False
        gradient_checkpointing_policy: str = "nothing"
        
    Methods:
        from_pretrained(model_name_or_path: str) -> MoEConfig
            Load configuration from pretrained model.
            
        save_pretrained(save_directory: str)
            Save configuration to directory.
            
        to_dict() -> Dict[str, Any]
            Convert configuration to dictionary.
            
        from_dict(config_dict: Dict[str, Any]) -> MoEConfig
            Create configuration from dictionary.
    """
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    """
    Configuration for model training.
    
    Attributes:
        # Basic settings
        output_dir: str = "./outputs"
        num_train_epochs: int = 3
        per_device_train_batch_size: int = 8
        per_device_eval_batch_size: int = 8
        gradient_accumulation_steps: int = 1
        
        # Optimizer settings
        learning_rate: float = 5e-5
        weight_decay: float = 0.01
        adam_beta1: float = 0.9
        adam_beta2: float = 0.999
        adam_epsilon: float = 1e-8
        max_grad_norm: float = 1.0
        
        # Scheduler settings
        lr_scheduler_type: str = "cosine"
        warmup_steps: int = 500
        warmup_ratio: float = 0.0
        
        # Logging settings
        logging_dir: str = "./logs"
        logging_steps: int = 10
        save_steps: int = 500
        eval_steps: int = 500
        
        # Advanced settings
        fp16: bool = False
        bf16: bool = True
        gradient_checkpointing: bool = True
        deepspeed: Optional[str] = None
        
    Methods:
        to_dict() -> Dict[str, Any]
            Convert to dictionary for serialization.
    """
```

## Training Classes

### Trainer

```python
class Trainer:
    """
    Main training class for MoE models.
    
    Args:
        model (MoEModel): Model to train
        args (TrainingConfig): Training configuration
        train_dataset (Dataset): Training dataset
        eval_dataset (Optional[Dataset]): Evaluation dataset
        tokenizer (PreTrainedTokenizer): Tokenizer
        data_collator (Optional[DataCollator]): Data collator
        compute_metrics (Optional[Callable]): Metrics computation function
        callbacks (Optional[List[TrainerCallback]]): Training callbacks
        optimizers (Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]): 
            Custom optimizer and scheduler
            
    Methods:
        train() -> TrainOutput
            Main training loop.
            
        evaluate(eval_dataset: Optional[Dataset] = None) -> Dict[str, float]
            Evaluate model on dataset.
            
        predict(test_dataset: Dataset) -> PredictionOutput
            Make predictions on test dataset.
            
        save_model(output_dir: Optional[str] = None)
            Save model checkpoint.
            
        push_to_hub(**kwargs)
            Push model to Hugging Face Hub.
    """
```

### ParallelTrainer

```python
class ParallelTrainer(Trainer):
    """
    Distributed training with model/data/pipeline parallelism.
    
    Additional Args:
        model_parallel_size (int): Tensor parallel size
        pipe_parallel_size (int): Pipeline parallel size
        expert_parallel_size (int): Expert parallel size
        
    Additional Methods:
        setup_distributed() -> None
            Initialize distributed training environment.
            
        shard_model() -> None
            Apply model sharding across devices.
            
        gather_predictions(predictions: List[torch.Tensor]) -> torch.Tensor
            Gather predictions from all processes.
    """
```

## Data Processing

### DataFilter

```python
class DataFilter:
    """
    Filter and score text data for quality.
    
    Args:
        min_length (int): Minimum text length
        max_length (int): Maximum text length
        min_quality_score (float): Minimum BERT quality score
        remove_duplicates (bool): Whether to remove duplicates
        language (str): Target language for filtering
        
    Methods:
        filter_dataset(dataset: Dataset) -> Dataset
            Filter dataset based on criteria.
            
        score_text(text: str) -> float
            Score text quality using BERT.
            
        remove_near_duplicates(texts: List[str], threshold: float = 0.9) -> List[str]
            Remove near-duplicate texts using MinHash.
    """
```

### FastDataLoader

```python
class FastDataLoader:
    """
    Optimized dataloader for large-scale training.
    
    Args:
        dataset (Dataset): Training dataset
        batch_size (int): Batch size
        num_workers (int): Number of data loading workers
        prefetch_factor (int): Number of batches to prefetch
        persistent_workers (bool): Keep workers alive between epochs
        memory_pin (bool): Pin memory for GPU transfer
        
    Methods:
        __iter__() -> Iterator[Dict[str, torch.Tensor]]
            Iterate over batches.
            
        set_epoch(epoch: int) -> None
            Set epoch for proper shuffling.
            
        state_dict() -> Dict[str, Any]
            Get dataloader state.
            
        load_state_dict(state_dict: Dict[str, Any]) -> None
            Load dataloader state.
    """
```

### TokenizerWrapper

```python
class TokenizerWrapper:
    """
    Fast tokenizer with caching and optimization.
    
    Args:
        tokenizer_name_or_path (str): Base tokenizer
        max_length (int): Maximum sequence length
        cache_size (int): Token cache size
        use_fast (bool): Use fast tokenizer implementation
        
    Methods:
        tokenize(texts: Union[str, List[str]], **kwargs) -> BatchEncoding
            Tokenize texts with caching.
            
        batch_encode_plus(texts: List[str], **kwargs) -> BatchEncoding
            Batch tokenization with optimization.
            
        decode(token_ids: Union[int, List[int]], **kwargs) -> str
            Decode token IDs to text.
            
        save_pretrained(save_directory: str) -> None
            Save tokenizer configuration.
    """
```

## RLHF Components

### PPOTrainer

```python
class PPOTrainer:
    """
    Proximal Policy Optimization trainer for RLHF.
    
    Args:
        model (MoEModel): Policy model
        ref_model (MoEModel): Reference model
        reward_model (nn.Module): Reward model
        config (PPOConfig): PPO configuration
        tokenizer (PreTrainedTokenizer): Tokenizer
        
    Methods:
        generate_experience(prompts: List[str]) -> List[PPOExperience]
            Generate experiences using current policy.
            
        compute_rewards(responses: List[str]) -> torch.Tensor
            Compute rewards for generated responses.
            
        train_step(experiences: List[PPOExperience]) -> Dict[str, float]
            Single PPO training step.
            
        train(dataset: Dataset, num_epochs: int = 1) -> None
            Main PPO training loop.
    """
```

### DPOTrainer

```python
class DPOTrainer:
    """
    Direct Preference Optimization trainer.
    
    Args:
        model (MoEModel): Model to train
        ref_model (Optional[MoEModel]): Reference model
        config (DPOConfig): DPO configuration
        tokenizer (PreTrainedTokenizer): Tokenizer
        
    Methods:
        compute_loss(batch: Dict[str, torch.Tensor]) -> torch.Tensor
            Compute DPO loss for batch.
            
        train_step(batch: Dict[str, torch.Tensor]) -> Dict[str, float]
            Single training step.
            
        train(dataset: Dataset) -> None
            Main DPO training loop.
            
        create_preference_dataset(prompts: List[str], 
                                  responses: List[Tuple[str, str]]) -> Dataset
            Create preference dataset from comparisons.
    """
```

### ConstitutionalAITrainer

```python
class ConstitutionalAITrainer:
    """
    Constitutional AI self-training system.
    
    Args:
        model (MoEModel): Model to train
        tokenizer (PreTrainedTokenizer): Tokenizer
        principles (List[ConstitutionalPrinciple]): Constitutional principles
        config (ConstitutionalConfig): Configuration
        
    Methods:
        generate_critique(response: str, principle: ConstitutionalPrinciple) -> str
            Generate critique based on principle.
            
        revise_response(response: str, critique: str, 
                        principle: ConstitutionalPrinciple) -> str
            Revise response based on critique.
            
        self_train(prompts: List[str], num_iterations: int = 3) -> None
            Self-training loop with critiques and revisions.
            
        evaluate_constitutional_adherence(test_prompts: List[str]) -> Dict[str, float]
            Evaluate adherence to constitutional principles.
    """
```

## Optimization Utilities

### ZeROOptimizer

```python
class ZeROOptimizer:
    """
    ZeRO optimizer wrapper for memory efficiency.
    
    Args:
        optimizer (torch.optim.Optimizer): Base optimizer
        model (nn.Module): Model to optimize
        config (ZeROConfig): ZeRO configuration
        
    Methods:
        step() -> None
            Optimizer step with ZeRO optimizations.
            
        zero_grad(set_to_none: bool = True) -> None
            Clear gradients efficiently.
            
        state_dict() -> Dict[str, Any]
            Get optimizer state.
            
        load_state_dict(state_dict: Dict[str, Any]) -> None
            Load optimizer state.
    """
```

### ActivationCheckpointing

```python
class ActivationCheckpointing:
    """
    Gradient checkpointing utilities.
    
    Methods:
        apply_checkpoint(model: nn.Module, policy: CheckpointPolicy) -> None
            Apply checkpointing to model.
            
        checkpoint_sequential(functions: List[Callable], 
                              segments: int, *args) -> Any
            Checkpoint sequential computations.
            
        set_checkpoint_policy(model: nn.Module, 
                              policy: Callable[[nn.Module], bool]) -> None
            Set custom checkpoint policy.
    """
```

## Inference Components

### TextGenerator

```python
class TextGenerator:
    """
    High-level text generation interface.
    
    Args:
        model (MoEModel): Model for generation
        tokenizer (PreTrainedTokenizer): Tokenizer
        config (GenerationConfig): Generation configuration
        
    Methods:
        generate(prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]
            Generate text from prompt(s).
            
        generate_stream(prompt: str, **kwargs) -> Iterator[str]
            Stream generated tokens.
            
        generate_with_scores(prompt: str, **kwargs) -> GenerateOutput
            Generate with token scores and probabilities.
            
        set_generation_config(config: GenerationConfig) -> None
            Update generation configuration.
    """
```

### SpeculativeDecoder

```python
class SpeculativeDecoder:
    """
    Speculative decoding for faster inference.
    
    Args:
        target_model (MoEModel): Target model
        draft_model (Union[MoEModel, str]): Draft model or path
        gamma (int): Number of draft tokens
        temperature (float): Sampling temperature
        
    Methods:
        generate(prompt: str, max_length: int, **kwargs) -> str
            Generate with speculative decoding.
            
        speculate(input_ids: torch.Tensor, gamma: int) -> SpeculationOutput
            Generate draft tokens.
            
        verify(draft_tokens: torch.Tensor, 
                target_probs: torch.Tensor) -> VerificationOutput
            Verify draft tokens against target.
            
        get_acceptance_rate() -> float
            Get average acceptance rate.
    """
```

## Distributed Training

### ModelParallelism

```python
class ModelParallelism:
    """
    Model parallelism utilities.
    
    Args:
        tensor_parallel_size (int): Tensor parallel degree
        pipeline_parallel_size (int): Pipeline parallel degree
        expert_parallel_size (int): Expert parallel degree
        
    Methods:
        parallelize_model(model: MoEModel) -> DistributedModel
            Apply model parallelism.
            
        deparallelize_model(model: DistributedModel) -> MoEModel
            Convert back to single-device model.
            
        synchronize() -> None
            Synchronize across all parallel groups.
    """
```

### ExpertParallelism

```python
class ExpertParallelism:
    """
    Expert-specific parallelism.
    
    Args:
        num_experts (int): Total number of experts
        expert_parallel_size (int): Expert parallel degree
        
    Methods:
        scatter_experts(experts: List[Expert]) -> None
            Distribute experts across devices.
            
        all_to_all(tensor: torch.Tensor) -> torch.Tensor
            All-to-all communication for expert routing.
            
        load_balance_experts(usage_stats: Dict[int, float]) -> None
            Rebalance experts based on usage.
    """
```

## Utility Functions

### Model Utilities

```python
def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """Count model parameters."""

def estimate_flops(model: nn.Module, batch_size: int, seq_length: int) -> float:
    """Estimate FLOPs for forward pass."""

def get_model_memory_footprint(model: nn.Module, 
                               batch_size: int, 
                               seq_length: int) -> Dict[str, float]:
    """Estimate memory usage breakdown."""
```

### Training Utilities

```python
def create_optimizer(model: nn.Module, 
                     config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer with proper parameter groups."""

def create_scheduler(optimizer: torch.optim.Optimizer, 
                     config: TrainingConfig, 
                     num_training_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Create learning rate scheduler."""

def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
    """Compute evaluation metrics."""
```

### Data Utilities

```python
def load_dataset(data_path: str, 
                 split: str = "train", 
                 streaming: bool = False) -> Dataset:
    """Load dataset from various formats."""

def preprocess_function(examples: Dict[str, List], 
                        tokenizer: PreTrainedTokenizer, 
                        max_length: int) -> BatchEncoding:
    """Preprocess text examples for training."""

def create_data_collator(tokenizer: PreTrainedTokenizer, 
                         max_length: int) -> DataCollator:
    """Create appropriate data collator."""
```

For detailed examples and usage patterns, see:
- [Training Guide](training.md)
- [Inference Guide](inference.md)
- [Examples](../examples/)