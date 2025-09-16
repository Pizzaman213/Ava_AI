#!/bin/bash
# Test script for ultra-tiny model training

echo "🚀 Testing Ultra-Tiny Model Training"
echo "===================================="

# Set memory limits to prevent system crash
ulimit -v 4000000  # Limit virtual memory to 4GB

# Run training with minimal settings
python3 /project/code/scripts/training/train.py \
    --config /project/code/configs/cpu/ultra_tiny.yaml \
    --max-samples 100 \
    --batch-size 2 \
    --epochs 1 \
    2>&1 | tee training_test.log

echo ""
echo "✅ Test complete! Check training_test.log for details"