#!/bin/bash
# Test training with memory-safe configuration

set -e

echo "=========================================="
echo "Testing Training with Fixed Configuration"
echo "=========================================="
echo ""

# Check system resources
echo "System Resources:"
echo "-----------------"
free -h | grep -E "Mem:|Swap:"
echo ""

# Clear page cache if possible (best effort)
sync
echo ""

# Set environment variables for memory safety
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Run training with timeout and output capture
echo "Starting training (2 minute test)..."
echo ""

cd /project/code/scripts/5_training

timeout 120 python train.py \
    --config /project/code/configs/moe/single_gpu_optimized.yaml \
    --force-single-gpu \
    2>&1 | tee /tmp/training_test.log || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo ""
        echo "✓ Training ran for 2 minutes without crashes!"
        echo ""
        echo "Memory status after test:"
        free -h | grep -E "Mem:|Swap:"
        exit 0
    elif [ $EXIT_CODE -eq 137 ]; then
        echo ""
        echo "✗ Training was killed (OOM)"
        echo ""
        echo "Memory status:"
        free -h
        tail -50 /tmp/training_test.log
        exit 1
    else
        echo ""
        echo "✗ Training failed with exit code $EXIT_CODE"
        echo ""
        tail -50 /tmp/training_test.log
        exit $EXIT_CODE
    fi
}
