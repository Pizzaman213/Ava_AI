#!/bin/bash
# Compare the three training configurations side-by-side
# Usage: ./compare_configs.sh

echo "======================================================================"
echo "Training Configuration Comparison"
echo "======================================================================"
echo ""

configs=(
    "code/configs/moe/tiny_moe_multi_gpu.yaml:Baseline"
    "code/configs/moe/tiny_moe_multi_gpu_fast.yaml:Fast (Phase 1-3)"
    "code/configs/moe/tiny_moe_multi_gpu_max_speed.yaml:Max Speed (Phase 1-4)"
)

echo "Running validation on all configs..."
echo ""

for config_info in "${configs[@]}"; do
    IFS=':' read -r config_path config_name <<< "$config_info"

    echo "======================================================================"
    echo "Config: $config_name"
    echo "Path: $config_path"
    echo "======================================================================"

    if [ -f "$config_path" ]; then
        python test_optimized_configs.py --config "$config_path" --skip-memory-estimate
        echo ""
    else
        echo "❌ Config file not found: $config_path"
        echo ""
    fi
done

echo "======================================================================"
echo "Summary"
echo "======================================================================"
echo ""
echo "Three configurations are available:"
echo ""
echo "1. Baseline (tiny_moe_multi_gpu.yaml)"
echo "   - Current configuration"
echo "   - Throughput: 1.0x (baseline)"
echo "   - Risk: N/A"
echo ""
echo "2. Fast (tiny_moe_multi_gpu_fast.yaml) ⭐ RECOMMENDED"
echo "   - Phases 1-3 optimizations applied"
echo "   - Throughput: 1.5-1.8x"
echo "   - Risk: Low"
echo "   - No additional dependencies"
echo ""
echo "3. Max Speed (tiny_moe_multi_gpu_max_speed.yaml)"
echo "   - All phases 1-4 optimizations applied"
echo "   - Throughput: 2.0-3.0x"
echo "   - Risk: Medium"
echo "   - Requires: DeepSpeed, flash-attn (optional)"
echo ""
echo "To use a config:"
echo "  python code/scripts/5_training/train.py --config <path>"
echo ""
echo "For detailed comparison, see:"
echo "  TRAINING_SPEED_OPTIMIZATION_GUIDE.md"
echo ""
