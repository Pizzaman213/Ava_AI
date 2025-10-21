# Ava LLM Training Framework - Architecture & Features

Complete documentation of the system architecture, model design, and implemented features.

## 📐 Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                  Training Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────┐   │
│  │ Data Loading │ -> │ Tokenization │ -> │  Batching  │   │
│  └──────────────┘    └──────────────┘    └────────────┘   │
│         ↓                   ↓                    ↓          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Model Forward Pass                      │   │
│  │  ┌──────────────────────────────────────────────┐  │   │
│  │  │  Embedding Layer (vocab_size -> hidden)    │  │   │
│  │  └──────────────────────────────────────────────┘  │   │
│  │         ↓                                           │   │
│  │  ┌──────────────────────────────────────────────┐  │   │
│  │  │  Transformer Stack (12 layers)             │  │   │
│  │  │  ├─ Multi-Head Attention (12 heads)        │  │   │
│  │  │  ├─ MoE Router (selects 2 experts)         │  │   │
│  │  │  └─ Feed-Forward Networks                  │  │   │
│  │  └──────────────────────────────────────────────┘  │   │
│  │         ↓                                           │   │
│  │  ┌──────────────────────────────────────────────┐  │   │
│  │  │  Output Layer (hidden -> logits)            │  │   │
│  │  └──────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│         ↓                                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Loss Computation                       │  │
│  │  ├─ Language Modeling Loss                       │  │
│  │  ├─ Multi-Token Prediction (MTP)                 │  │
│  │  └─ Auxiliary Loss (expert balance)              │  │
│  └──────────────────────────────────────────────────────┘  │
│         ↓                                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Backward Pass                        │  │
│  │  ├─ Gradient Computation                         │  │
│  │  ├─ Gradient Clipping                            │  │
│  │  └─ Gradient Accumulation                        │  │
│  └──────────────────────────────────────────────────────┘  │
│         ↓                                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Optimizer Step                      │  │
│  │  ├─ Momentum/Adaptive updates                    │  │
│  │  ├─ Weight decay                                 │  │
│  │  └─ Learning rate schedule                       │  │
│  └──────────────────────────────────────────────────────┘  │
│         ↓                                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Checkpoint & Logging                │  │
│  │  ├─ Save model weights                           │  │
│  │  ├─ Log metrics to W&B                           │  │
│  │  └─ Validation & evaluation                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🧠 Model Architecture

### Base Configuration

For the **small** model (233M parameters):
```
Input: (batch_size, seq_length) -> token IDs

Embedding Layer:
  vocab_size: 65,536
  hidden_size: 768
  output: (batch_size, seq_length, 768)

Transformer Stack: 12 layers
  ├─ Multi-Head Attention (768 dim, 12 heads = 64 per head)
  ├─ MoE Router (select 2 of N experts)
  └─ Feed-Forward (3,072 hidden dim)

Layer Normalization:
  Applied after attention and FFN

Output Layer:
  Linear projection: (hidden_size) -> (vocab_size)
  output: (batch_size, seq_length, 65536) logits
```

### Attention Mechanism

**Standard Multi-Head Attention**:
```
Q = X @ W_q
K = X @ W_k
V = X @ W_v

scores = softmax(Q @ K^T / sqrt(d_k))
output = scores @ V
```

**Multi-Query Attention (MQA)** - Optional optimization:
- Shares key and value across heads
- Memory efficient, slightly faster
- Similar quality to standard attention

**Grouped Query Attention (GQA)** - Optional optimization:
- Groups heads for key and value
- Balance between MQA and standard
- Often the best choice

### Position Encoding

Uses **Rotary Position Embeddings (RoPE)**:
- Applied to Q and K directly
- Enables length extrapolation
- More efficient than absolute position embeddings
- Supports sequences up to 2048 tokens by default

## 🎯 Mixture of Experts (MoE)

### MoE Design

```
Input (batch_size, seq_length, 768)
        ↓
┌─────────────────────────────────┐
│  Router Network (768 -> N)      │
│  Outputs: routing logits        │
└─────────────────────────────────┘
        ↓
    ┌───────────────┐
    │ Top-K Select  │  Select top 2 experts
    │ (K=2)         │
    └───────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│         Expert Networks (Parallel)                      │
│  ┌────────────┬────────────┬───────────┐                │
│  │ Expert 1   │ Expert 2   │ Expert N  │  Each outputs │
│  │ FFN Layer  │ FFN Layer  │ FFN Layer │  (hidden_size)│
│  └────────────┴────────────┴───────────┘                │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────┐
│ Weighted Combination            │
│ out = weight_1 * expert_1 +     │
│       weight_2 * expert_2       │
└─────────────────────────────────┘
        ↓
Output (batch_size, seq_length, 768)
```

### Key Parameters

| Parameter | Value | Impact |
|-----------|-------|--------|
| Number of Experts | 4-8 | More experts = higher capacity, more memory |
| Top-K | 2 | Route to 2 experts per token |
| Expert Dim | 3,072 | Hidden dimension of each expert FFN |
| Router Temp | 1.0 | Softness of routing decisions |
| Load Balance Weight | 0.1 | Prevents expert collapse |

### Load Balancing

MoE has a tendency for all tokens to route to a few "popular" experts:

**Auxiliary Loss** (trained alongside main loss):
```
aux_loss = load_balance_weight * (expert_loss + importance_loss)

expert_loss: Encourages equal expert usage
importance_loss: Prevents specific experts from being too important
```

This automatically balances token distribution across experts.

## 📊 Loss Functions

### Primary Loss: Language Modeling

Standard next-token prediction:
```
logits = model(input_tokens)
loss = cross_entropy(logits, target_tokens)
```

Optimized with:
- Label smoothing (default: 0.1)
- Gradient clipping (max norm: 1.0)
- Mixed precision (bfloat16)

### Multi-Token Prediction (MTP)

Predicts multiple future tokens:
```
For token at position t, predict:
  - t+1 (1 step ahead)
  - t+2 (2 steps ahead)
  - t+3 (3 steps ahead)

loss_mtp = w1 * loss_1step +
           w2 * loss_2step +
           w3 * loss_3step

Total loss = loss_lm + 0.1 * loss_mtp
```

Benefits:
- Improves coherence for longer sequences
- Better learns long-range dependencies
- Can prevent repetition

### Auxiliary Loss (MoE Balance)

Ensures experts are equally utilized:
```
expert_usage = mean(routing_weights)
aux_loss = penalty * (var(expert_usage))
```

## ⚙️ Training Dynamics

### Learning Rate Schedule

**Warmup + Cosine Annealing**:
```
LR(t) = lr_max * min(t / warmup_steps, 
        0.5 * (1 + cos(π * (t - warmup_steps) / (total - warmup))))

Example with lr_max=0.006:
Step 0:      LR = 0.000
Step 1500:   LR = 0.003 (warmup halfway)
Step 3000:   LR = 0.006 (peak)
Step 50000:  LR = 0.003 (cosine decay)
Step 100000: LR = 0.001 (further decay)
```

### Gradient Accumulation

Effective batch size = batch_size * gradient_accumulation_steps

Example:
```
batch_size = 8
gradient_accumulation_steps = 4
effective_batch_size = 32

Optimizer is updated every 4 iterations
Smoother gradient flow
Better stability with large effective batches
```

### Optimization Details

**Default Optimizer: FusedAdam**

When CUDA available:
- 10-15% faster than standard PyTorch Adam
- Lower memory footprint
- In-place operations

Falls back to standard Adam if:
- CUDA not available
- Mixed precision not supported
- Memory constraints

Parameters:
```
beta1: 0.9      (momentum decay)
beta2: 0.95     (adaptive lr decay, higher than 0.999 for stability)
eps: 1e-8       (numerical stability)
weight_decay: 0.01  (L2 regularization)
```

## 🚀 Advanced Features

### RLHF (Reinforcement Learning from Human Feedback)

**Pipeline**:
1. **Experience Collection**
   - Generate responses to prompts using policy model
   - Collect experiences (prompt, response, log_probs)

2. **Reward Computation**
   - Judge model rates response quality (0-10 scale)
   - Compute advantages using GAE
   - Whiten rewards for stability

3. **PPO Update**
   - Compute policy gradient with PPO clipping
   - Update policy with 4 epochs per batch
   - Monitor KL divergence from reference model

4. **Adaptive KL Penalty**
   - Automatically adjust penalty coefficient
   - Keep KL divergence at target (~0.01)
   - Prevents distribution mismatch

### Model-to-Model Reward

Uses language model as judge:
```
Judge Prompt:
"Rate this response on a scale of 0-10.
Prompt: {prompt}
Response: {response}
Rating: "

Judge generates rating token
Extract numerical value: 0-10
Use as reward signal
```

Benefits:
- No human labels needed
- Can be domain-specific
- Scalable to any prompt domain

### Gradient Checkpointing

For memory efficiency:
- Don't store intermediate activations during forward pass
- Recompute activations during backward pass
- 40-60% memory reduction
- ~20% slower training (but enables larger batches)

Can be enabled:
```yaml
gradient_checkpointing: true  # in config
```

### Flash Attention

Efficient attention implementation:
- Reduces memory from O(n²) to O(n)
- IO-aware algorithm optimized for GPUs
- 2-4x faster than standard attention
- No quality loss

Falls back automatically:
```
Flash Attention v2/v3 → xformers → SDPA → manual
```

## 📈 Performance Metrics

### Loss Trajectory

**Healthy pre-training**:
```
Step 100:    Loss ≈ 10.5 (random baseline for 65k vocab)
Step 1,000:  Loss ≈ 6.0-7.0 (learning initial patterns)
Step 10,000: Loss ≈ 3.5-4.0 (coherence emerging)
Step 50,000: Loss ≈ 2.0-2.5 (good performance)
Step 100k:   Loss ≈ 1.8-2.0 (excellent quality)
```

### Generation Quality Metrics

| Metric | Good Range | Method |
|--------|-----------|--------|
| Coherence | > 70% | Manual evaluation or LLM judge |
| Distinct-2 | 0.5-0.8 | Ratio of unique bigrams |
| Repetition | < 0.3 | Ngram repetition frequency |
| Entropy | 4.0-5.0 | Shannon entropy of token dist |
| BLEU (if ref) | > 20 | Precision against references |

### Training Efficiency

**Throughput**: Tokens/second
```
RTX 3090 Ti, small model:
- Baseline: ~2000 tokens/sec
- With optimizations: ~4000-5000 tokens/sec
- With DeepSpeed ZeRO-2: ~6000-7000 tokens/sec
```

**Model FLOPS Utilization (MFU)**:
```
MFU = (actual_throughput) / (theoretical_peak)

RTX 3090 Ti:
- Theoretical: 40 TFLOPS (bfloat16)
- Typical: 30-35% MFU
- With Flash Attention: 40-45% MFU
```

## 🔧 Configuration Hierarchy

Config files override each other:

```
1. Base defaults (in code)
   ↓
2. configs/gpu/base.yaml (general settings)
   ↓
3. configs/gpu/small.yaml (model-specific)
   ↓
4. Command-line arguments --override param=value
```

Example override:
```bash
python train.py \
  --config configs/gpu/small.yaml \
  --batch-size 16 \
  --learning-rate 0.003
```

## 🎯 Training Stages

### Stage 1: Initial Learning (0-10k steps)
- Model learns basic patterns
- Loss drops rapidly
- Loss: 10 → 5
- Quality: Random tokens → Recognizable words

### Stage 2: Stabilization (10k-50k steps)
- Loss decreases more slowly
- Model learns coherence
- Loss: 5 → 2.5
- Quality: Word chains → Sentence structure

### Stage 3: Refinement (50k-100k steps)
- Loss plateaus at low value
- Quality improves incrementally
- Loss: 2.5 → 2.0
- Quality: Sentences → Coherent paragraphs

### Stage 4: Mastery (100k+ steps)
- Very slow improvement
- Diminishing returns
- Model produces high-quality text
- Further training: fine-tuning or RLHF

## 📁 Architecture Files

**Key Model Files**:
```
src/Ava/models/
├── moe_model.py          # Main MoE model
├── attention.py          # Attention implementations
├── moe_router.py         # Expert routing
└── layers.py             # Custom layers

src/Ava/training/
├── enhanced_trainer.py   # Main training loop
├── distributed_optim.py  # Optimization strategies
└── profiling_tools.py    # Performance monitoring

src/Ava/losses/
├── main_loss.py          # Language modeling loss
├── mtp_loss.py           # Multi-token prediction
└── auxiliary_loss.py     # Expert balancing
```

## 🎓 Learning Path

To understand the codebase:

1. **Start**: Read model architecture (moe_model.py)
2. **Attention**: Understand attention layers (attention.py)
3. **MoE**: Study expert routing (moe_router.py)
4. **Training**: Review training loop (enhanced_trainer.py)
5. **Loss**: Study loss functions (losses/)
6. **Advanced**: Explore RLHF (rlhf/)

---

**Status**: ✅ Complete and tested
**Last Updated**: 2025-10-21
