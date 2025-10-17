#!/bin/bash
# Verification script for LR Finder overfitting fixes

echo "========================================"
echo "LR Finder Fix Verification"
echo "========================================"
echo ""

# Check if fixes are in place
echo "1. Checking advanced_warmup_scheduling.py for correct EMA formula..."
if grep -q "(1 - self.smoothing) \* loss_value + self.smoothing \* smoothed_loss" code/src/Ava/training/advanced_warmup_scheduling.py; then
    echo "   ✓ EMA formula is CORRECT"
else
    echo "   ✗ EMA formula may be incorrect"
fi

echo ""
echo "2. Checking advanced_warmup_scheduling.py for divergence threshold..."
if grep -q "smoothed_loss > 8 \* best_loss" code/src/Ava/training/advanced_warmup_scheduling.py; then
    echo "   ✓ Divergence threshold set to 8x (correct)"
else
    echo "   ⚠ Divergence threshold may need adjustment"
fi

echo ""
echo "3. Checking lr_finder.py for beta value..."
if grep -q "beta: float = 0.9" code/src/Ava/training/lr_finder.py; then
    echo "   ✓ Beta value is 0.9 (correct)"
else
    echo "   ✗ Beta value may be incorrect"
fi

echo ""
echo "4. Checking lr_finder.py for savgol_window..."
if grep -q "savgol_window: int = 31" code/src/Ava/training/lr_finder.py; then
    echo "   ✓ Savgol window is 31 (correct)"
else
    echo "   ✗ Savgol window may be incorrect"
fi

echo ""
echo "5. Checking lr_finder.py for stop_div_threshold..."
if grep -q "stop_div_threshold: float = 4.0" code/src/Ava/training/lr_finder.py; then
    echo "   ✓ Stop divergence threshold is 4.0 (correct)"
else
    echo "   ✗ Stop divergence threshold may be incorrect"
fi

echo ""
echo "6. Checking lr_finder.py for correct loss handling (no scaling)..."
if grep -q "return loss.item()$" code/src/Ava/training/lr_finder.py; then
    echo "   ✓ Loss scaling removed (correct)"
else
    echo "   ✗ Loss may still be scaled incorrectly"
fi

echo ""
echo "7. Checking small.yaml config for correct settings..."
if grep -q "beta: 0.9" code/configs/gpu/small.yaml; then
    echo "   ✓ Config beta is 0.9 (correct)"
else
    echo "   ✗ Config beta may be incorrect"
fi

if grep -q "gradient_accumulation_steps: 1  # FIXED" code/configs/gpu/small.yaml; then
    echo "   ✓ Config gradient_accumulation_steps is 1 (correct)"
else
    echo "   ⚠ Config gradient_accumulation_steps may need adjustment"
fi

echo ""
echo "8. Verifying training pipeline gradient accumulation..."
if grep -q "scaled_loss = total_loss / gradient_accumulation_steps" code/src/Ava/training/enhanced_trainer.py; then
    echo "   ✓ Training pipeline correctly scales loss (correct)"
else
    echo "   ✗ Training pipeline loss scaling may be incorrect"
fi

echo ""
echo "========================================"
echo "Verification Complete"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run: ./TEST_AUTO_LR_FINDER.sh"
echo "2. Check LR finder plot for smooth, responsive curve"
echo "3. Verify suggested LR is in the valley (not at minimum)"
echo "4. Start training with: ./RESTART_TRAINING.sh"
echo ""
