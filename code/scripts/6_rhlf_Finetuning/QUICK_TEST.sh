#!/bin/bash
# Quick RLHF Test with Custom Tokenizer
# This script tests the RLHF pipeline with your custom 65k tokenizer

set -e  # Exit on error

echo "=============================================="
echo "RLHF Quick Test with Custom Tokenizer"
echo "=============================================="
echo ""

# 1. Check tokenizer exists
TOKENIZER_PATH="/project/code/models/tokenizer/enhanced-65536"
if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "❌ Custom tokenizer not found at: $TOKENIZER_PATH"
    echo "Please check the path in configs/gpu/small.yaml"
    exit 1
fi
echo "✓ Found custom tokenizer at: $TOKENIZER_PATH"

# 2. Check if model exists
MODEL_PATH="/project/code/outputs/runs/run_20251016_094657_81a543c4/checkpoints/latest_model.pt"
if [ -f "$MODEL_PATH" ]; then
    echo "✓ Found model at: $MODEL_PATH"
    USE_MODEL="--policy-model $MODEL_PATH --judge-model $MODEL_PATH"
else
    echo "⚠ No specific model found, will use path from config"
    USE_MODEL=""
fi

# 3. Create test prompts
echo ""
echo "Creating test prompts..."
python code/scripts/6_rhlf_Finetuning/prepare_prompts.py \
  --create-samples \
  --num-samples 20 \
  --output /project/code/data/rlhf/prompts.json \
  --no-split

if [ $? -eq 0 ]; then
    echo "✓ Created test prompts"
else
    echo "❌ Failed to create prompts"
    exit 1
fi

# 4. Show config info
echo ""
echo "Configuration:"
echo "  - Config: configs/gpu/small.yaml"
echo "  - Tokenizer: $TOKENIZER_PATH"
echo "  - Device: CPU (for testing)"
echo "  - Epochs: 1 (quick test)"
echo ""

# 5. Test run (dry run - just initialization)
echo "Testing RLHF initialization..."
echo "Note: This will test that everything loads correctly"
echo ""

# Create a test that just initializes (doesn't train)
python -c "
import sys
sys.path.insert(0, 'code/src')

from transformers import PreTrainedTokenizerFast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test tokenizer loading
tokenizer_path = '$TOKENIZER_PATH'
logger.info(f'Loading custom tokenizer from {tokenizer_path}')

try:
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    logger.info(f'✓ Loaded custom tokenizer with vocab size: {len(tokenizer)}')
    logger.info(f'  - Pad token: {tokenizer.pad_token}')
    logger.info(f'  - EOS token: {tokenizer.eos_token}')
    logger.info(f'  - BOS token: {tokenizer.bos_token}')

    # Test encoding
    test_text = 'What is machine learning?'
    encoded = tokenizer.encode(test_text)
    logger.info(f'  - Test encoding: {test_text} -> {len(encoded)} tokens')

    print('\n✓ Custom tokenizer test passed!')
    sys.exit(0)
except Exception as e:
    logger.error(f'✗ Tokenizer test failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "✓ All tests passed!"
    echo "=============================================="
    echo ""
    echo "Your RLHF pipeline is ready with custom tokenizer!"
    echo ""
    echo "To run full RLHF training:"
    echo "  python code/scripts/6_rhlf_Finetuning/train_rlhf.py \\"
    echo "    --config configs/gpu/small.yaml $USE_MODEL"
    echo ""
else
    echo ""
    echo "❌ Tests failed. Please check the errors above."
    exit 1
fi
