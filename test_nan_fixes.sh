#!/bin/bash
# Quick test to verify NaN/Inf loss fixes
# This script runs training for 100 steps and checks if loss is finite

set -e

echo "========================================="
echo "Testing NaN/Inf Loss Fixes"
echo "========================================="
echo ""

CONFIG="code/configs/moe/tiny_moe_ultra_low_mem.yaml"
LOG_FILE="/tmp/nan_fix_test_$(date +%s).log"

echo "Config: $CONFIG"
echo "Log file: $LOG_FILE"
echo ""

echo "[1/3] Checking configuration..."
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi
echo "✓ Config file exists"
echo ""

echo "[2/3] Running training for 100 steps..."
echo "This will take ~2-3 minutes..."
echo ""

# Run training with timeout and capture output
timeout 300 python code/scripts/5_training/train.py \
    --config "$CONFIG" \
    2>&1 | tee "$LOG_FILE" || true

echo ""
echo "[3/3] Analyzing results..."
echo ""

# Check for NaN/Inf losses
NAN_COUNT=$(grep -c "CRITICAL.*NaN or Inf" "$LOG_FILE" || true)
INF_LOSS_COUNT=$(grep -c "Loss=inf" "$LOG_FILE" || true)
SKIP_COUNT=$(grep -c "Skipping optimizer step" "$LOG_FILE" || true)

# Check for successful training
FINITE_LOSS_COUNT=$(grep -c "Loss=[0-9]" "$LOG_FILE" || true)

echo "Results:"
echo "  - NaN/Inf detections: $NAN_COUNT"
echo "  - Inf loss displays: $INF_LOSS_COUNT"
echo "  - Skipped optimizer steps: $SKIP_COUNT"
echo "  - Finite loss steps: $FINITE_LOSS_COUNT"
echo ""

# Determine success
if [ $NAN_COUNT -eq 0 ] && [ $INF_LOSS_COUNT -eq 0 ] && [ $FINITE_LOSS_COUNT -gt 0 ]; then
    echo "========================================="
    echo "✓ SUCCESS: Training is stable!"
    echo "========================================="
    echo ""
    echo "Loss values are finite and training is progressing."
    echo "You can now run full training with:"
    echo "  python code/scripts/5_training/train.py --config $CONFIG"
    echo ""
    exit 0
elif [ $NAN_COUNT -gt 0 ] || [ $INF_LOSS_COUNT -gt 0 ]; then
    echo "========================================="
    echo "✗ FAILURE: NaN/Inf losses still occurring"
    echo "========================================="
    echo ""
    echo "Further investigation needed. Check the log:"
    echo "  cat $LOG_FILE"
    echo ""
    echo "Possible additional fixes:"
    echo "  1. Reduce learning rate further (to 1e-5)"
    echo "  2. Increase gradient clipping (max_grad_norm: 0.3)"
    echo "  3. Disable DeepSeek loss temporarily"
    echo "  4. Check data preprocessing"
    echo ""
    exit 1
else
    echo "========================================="
    echo "? UNCLEAR: Could not determine status"
    echo "========================================="
    echo ""
    echo "Check the log file for details:"
    echo "  cat $LOG_FILE"
    echo ""
    exit 2
fi
