#!/bin/bash
# Example script showing how to use the Learning Rate Finder

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║   Learning Rate Finder Example - Find Optimal Learning Rate       ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Navigate to scripts/training directory
cd "$(dirname "$0")/../training" || exit 1

echo "📊 Example 1: Run LR Finder and view results (manual inspection)"
echo "────────────────────────────────────────────────────────────────────"
python train.py \
  --config ../../configs/gpu/small.yaml \
  --run-lr-finder \
  --lr-finder-iterations 100 \
  --max-samples 5000

echo ""
echo "✅ LR Finder plot saved! Check the output for suggested learning rate."
echo ""
echo "────────────────────────────────────────────────────────────────────"
echo ""

read -p "Press Enter to run Example 2 (auto-apply suggested LR)..."

echo ""
echo "📊 Example 2: Auto-apply suggested LR and start training"
echo "────────────────────────────────────────────────────────────────────"
python train.py \
  --config ../../configs/gpu/small.yaml \
  --run-lr-finder \
  --lr-finder-use-suggested \
  --lr-finder-iterations 150 \
  --epochs 3 \
  --max-samples 10000

echo ""
echo "✅ Training complete with auto-suggested learning rate!"
echo ""
echo "────────────────────────────────────────────────────────────────────"
echo ""

read -p "Press Enter to run Example 3 (custom LR range)..."

echo ""
echo "📊 Example 3: Custom LR range for fine-tuning"
echo "────────────────────────────────────────────────────────────────────"
python train.py \
  --config ../../configs/gpu/small.yaml \
  --run-lr-finder \
  --lr-finder-start 1e-9 \
  --lr-finder-end 0.01 \
  --lr-finder-iterations 200 \
  --lr-finder-method valley \
  --max-samples 3000

echo ""
echo "✅ Custom LR range test complete!"
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    All Examples Complete!                          ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
