#!/bin/bash
# Test Automatic LR Finder with Outlier Detection
#
# This script demonstrates the NEW automatic outlier detection feature:
# - Automatically excludes methods >5x away from median (like valley)
# - Uses geometric mean of remaining methods
# - Provides clear explanation of what was excluded and why
#
# Example with your previous results:
#   fastai:   7.38e-05
#   valley:   9.93e-06  <- 7.4x from median, AUTOMATICALLY EXCLUDED
#   steepest: 1.12e-04
#
# After outlier removal:
#   fastai:   7.38e-05
#   steepest: 1.12e-04
#   Variance: 1.51x → HIGH confidence! ✅✅
#   Result:   9.06e-05 (geometric mean)

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║           🤖 AUTOMATIC LR FINDER - Test with Outlier Detection 🤖            ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 NEW FEATURES:"
echo "  ✅ Automatic outlier detection (>5x from median)"
echo "  ✅ Smart method selection without manual intervention"
echo "  ✅ Clear explanation of what was excluded and why"
echo "  ✅ Geometric mean of non-outlier methods"
echo ""
echo "📊 Example (your previous results):"
echo "  • fastai:   7.38e-05"
echo "  • valley:   9.93e-06  🚫 EXCLUDED (7.4x from median - outlier)"
echo "  • steepest: 1.12e-04"
echo ""
echo "  After outlier removal:"
echo "  • fastai + steepest variance: 1.51x"
echo "  • Result: 9.06e-05 (geometric mean)"
echo "  • Confidence: HIGH ✅✅"
echo ""

# Check if user wants to run full test
read -p "Run full LR finder test? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Skipped. The automatic outlier detection is already enabled in:"
    echo "  • /project/code/scripts/4_Find_Lr/run_lr_finder_enhanced.py"
    echo ""
    echo "To use it, just run:"
    echo "  bash /project/QUICK_FIND_LR.sh"
    echo ""
    echo "Or directly:"
    echo "  cd /project/code/scripts/4_Find_Lr"
    echo "  python run_lr_finder_enhanced.py --config ../../configs/gpu/small.yaml"
    echo ""
    exit 0
fi

# Run full test
echo ""
echo "Step 1: Freeing GPU memory..."
echo "────────────────────────────────────────────────────────────────────────────────"
pkill -f 'train.py' || echo "No training found"
pkill -f 'lr_finder' || echo "No LR finder found"
sleep 5

python3 << 'CLEAR'
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✓ GPU cache cleared")
CLEAR

echo ""
echo "Step 2: Running LR Finder with Automatic Outlier Detection..."
echo "────────────────────────────────────────────────────────────────────────────────"

cd /project/code/scripts/4_Find_Lr
python3 run_lr_finder_enhanced.py \
    --config ../../configs/gpu/small.yaml \
    --methods fastai valley steepest \
    --output ../../lr_results_auto

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                              ✅ COMPLETE ✅                                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Check the output above for:"
echo "  🚫 EXCLUDED methods (outliers automatically detected)"
echo "  ✅ USED methods (non-outliers for geometric mean)"
echo "  📈 Final variance after outlier removal"
echo ""
echo "The config has been automatically updated with the best LR!"
echo ""
