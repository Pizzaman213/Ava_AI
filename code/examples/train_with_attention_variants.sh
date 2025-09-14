#!/bin/bash
# Example training commands showcasing different attention variants

# 1. Standard training with Flash Attention 2 (baseline)
echo "Training with standard attention + Flash Attention 2..."
python scripts/train.py \
    --config configs/gpu/medium.yaml \
    --attention-variant standard \
    --output-dir outputs/standard_attention \
    --experiment-name standard_flash_attn

# 2. Long document processing with sparse attention
echo "Training for long documents with sparse attention..."
python scripts/train.py \
    --config configs/attention_examples/long_context.yaml \
    --attention-variant sparse \
    --sparse-global-tokens 128 \
    --sparse-random-blocks 5 \
    --sliding-window-size 512 \
    --use-xpos \
    --output-dir outputs/long_docs \
    --experiment-name sparse_attention_papers

# 3. Chat model with streaming attention
echo "Training chat model with streaming attention..."
python scripts/train.py \
    --config configs/attention_examples/chat_streaming.yaml \
    --attention-variant streaming \
    --num-sink-tokens 8 \
    --output-dir outputs/chat_model \
    --experiment-name streaming_chat

# 4. Extreme length with linear attention
echo "Training on extreme length sequences with linear attention..."
python scripts/train.py \
    --config configs/attention_examples/extreme_length.yaml \
    --attention-variant linear \
    --linear-attention-feature elu \
    --use-xpos \
    --batch-size 1 \
    --gradient-accumulation-steps 64 \
    --output-dir outputs/extreme_length \
    --experiment-name linear_128k

# 5. Fast inference with sliding window
echo "Training efficient model with sliding window attention..."
python scripts/train.py \
    --config configs/attention_examples/efficient_inference.yaml \
    --attention-variant sliding_window \
    --sliding-window-size 256 \
    --output-dir outputs/fast_inference \
    --experiment-name sliding_window_fast

# 6. ALiBi for length generalization
echo "Training with ALiBi attention for better length generalization..."
python scripts/train.py \
    --config configs/gpu/medium.yaml \
    --attention-variant alibi \
    --max-steps 50000 \
    --output-dir outputs/alibi_model \
    --experiment-name alibi_length_generalization

# 7. Cached attention for repetitive tasks
echo "Training with cached attention for code completion..."
python scripts/train.py \
    --config configs/gpu/small.yaml \
    --attention-variant cached \
    --data-path /path/to/code_dataset \
    --output-dir outputs/cached_attention \
    --experiment-name cached_code_completion

# 8. Multi-GPU distributed training with sparse attention
echo "Distributed training with sparse attention..."
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/gpu/medium.yaml \
    --attention-variant sparse \
    --sparse-global-tokens 64 \
    --distributed \
    --zero-stage 2 \
    --output-dir outputs/distributed_sparse \
    --experiment-name distributed_sparse_4gpu

# 9. Combined features for maximum efficiency
echo "Training with combined optimizations..."
python scripts/train.py \
    --config configs/gpu/medium.yaml \
    --attention-variant sparse \
    --sparse-global-tokens 64 \
    --use-xpos \
    --mixed-precision bf16 \
    --gradient-checkpointing \
    --zero-stage 3 \
    --use-cpu-offload \
    --output-dir outputs/combined_optimizations \
    --experiment-name all_optimizations \
    --wandb \
    --wandb-project moe-attention-experiments