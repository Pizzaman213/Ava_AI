#!/bin/bash
# Check if current training can continue or needs restart

echo "========================================"
echo "Training Health Check"
echo "========================================"
echo ""

# Get current step from training output
CURRENT_STEP=$(grep -oP 'optimizer_step=\K\d+' code/outputs/small_enhanced/training.log 2>/dev/null | tail -1)
if [ -z "$CURRENT_STEP" ]; then
    CURRENT_STEP="unknown"
fi

echo "Current training step: $CURRENT_STEP"
echo ""

# Check config values
echo "Checking config values..."
echo ""

EOS_PENALTY=$(grep -A1 "eos_penalty_weight:" code/configs/gpu/small.yaml | grep -oP '\d+\.\d+' | head -1)
EOS_BIAS=$(grep -A1 "eos_logit_bias:" code/configs/gpu/small.yaml | grep -oP '[-+]?\d+\.\d+' | head -1)
MIN_SEQ=$(grep -A1 "min_sequence_length:" code/configs/gpu/small.yaml | grep -oP '\d+' | head -1)
LR_END=$(grep -A1 "lr_end:" code/configs/gpu/small.yaml | grep -oP '\d+\.\d+e-\d+' | head -1)

echo "Configuration values:"
echo "  eos_penalty_weight: $EOS_PENALTY (should be 0.3)"
echo "  eos_logit_bias: $EOS_BIAS (should be -0.5)"
echo "  min_sequence_length: $MIN_SEQ (should be 30)"
echo "  lr_end: $LR_END (should be 1.0e-06)"
echo ""

# Determine health status
NEEDS_RESTART=false

if (( $(echo "$EOS_PENALTY > 5.0" | bc -l) )); then
    echo "⚠️  WARNING: EOS penalty is too high ($EOS_PENALTY > 5.0)"
    NEEDS_RESTART=true
fi

if (( $(echo "$EOS_BIAS > 1.0" | bc -l) )); then
    echo "⚠️  WARNING: EOS bias is positive ($EOS_BIAS), should be negative"
    NEEDS_RESTART=true
fi

if [ "$MIN_SEQ" -gt 40 ]; then
    echo "⚠️  WARNING: Min sequence length is too high ($MIN_SEQ > 40)"
fi

echo ""
echo "========================================"
echo "Recommendation:"
echo "========================================"
echo ""

if [ "$NEEDS_RESTART" = true ] && [ "$CURRENT_STEP" != "unknown" ] && [ "$CURRENT_STEP" -lt 5000 ]; then
    echo "🔄 RESTART RECOMMENDED"
    echo ""
    echo "Reasons:"
    echo "  - Training is still early (step $CURRENT_STEP < 5000)"
    echo "  - Config had critical issues that may have caused collapse"
    echo "  - Fixed config will train better from the start"
    echo ""
    echo "Action: Run ./RESTART_TRAINING.sh"

elif [ "$NEEDS_RESTART" = true ]; then
    echo "⚠️  CONTINUE WITH CAUTION"
    echo ""
    echo "Reasons:"
    echo "  - Training is already at step $CURRENT_STEP"
    echo "  - Restarting would lose progress"
    echo "  - Model may recover with fixed config"
    echo ""
    echo "Action: Continue training, monitor generation quality"
    echo "        If quality doesn't improve, consider restarting"

else
    echo "✅ CONTINUE TRAINING"
    echo ""
    echo "Reasons:"
    echo "  - Config values are now correct"
    echo "  - Training can continue safely"
    echo "  - No need to restart"
    echo ""
    echo "Action: Let training continue with fixed config"
fi

echo ""
echo "========================================"
echo ""
