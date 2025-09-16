#!/bin/bash
# Test streaming data loader

echo "🌊 Testing Streaming Data Loader"
echo "===================================="

# Test with streaming
python3 /project/code/scripts/training/train.py \
    --config /project/code/configs/cpu/ultra_tiny.yaml \
    --streaming \
    --max-samples 100 \
    --buffer-size 50 \
    --batch-size 2 \
    --epochs 1 \
    2>&1 | tee streaming_test.log

echo ""
echo "✅ Streaming test complete! Check streaming_test.log for details"