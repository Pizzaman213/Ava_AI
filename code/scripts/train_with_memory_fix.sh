#!/bin/bash

# Training Script with GPU Memory Management Fix
# Prevents cache accumulation and OOM errors

echo "================================================"
echo "Starting Training with Memory Management Fix"
echo "================================================"

# Set PyTorch memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:False'
export CUDA_LAUNCH_BLOCKING=0

# Limit memory fraction
export PYTORCH_CUDA_PER_PROCESS_MEMORY_FRACTION=0.9

# Disable some memory-hungry optimizations
export TORCH_CUDNN_V8_API_ENABLED=0

echo "Memory management environment variables set:"
echo "  - Max split size: 128MB"
echo "  - Garbage collection threshold: 0.6"
echo "  - Memory fraction: 90%"
echo "  - Expandable segments: Disabled"
echo ""

# Navigate to project root
cd /project

# Run training with optimized config
echo "Starting training with batch_size=32, grad_accum=2"
echo "Effective batch size: 64"
echo ""

python code/scripts/5_training/train.py \
    --config code/configs/gpu/small.yaml \
    2>&1 | tee training_output.log

echo ""
echo "Training completed. Check training_output.log for details."