#!/bin/bash
# Quick test of step_100000 with various prompts and temperatures

MODEL="/project/code/outputs/runs/run_20251006_015602_9bd4df77/checkpoints/step_100000/model.pt"

echo "=================================="
echo "Testing Step 100000 Checkpoint"
echo "=================================="
echo ""

prompts=(
    "The quick brown fox"
    "Hello, my name is"
    "In the year 2050,"
    "Once upon a time"
)

temps=(0.3 0.7 1.0)

for prompt in "${prompts[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Prompt: '$prompt'"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    for temp in "${temps[@]}"; do
        echo ""
        echo "  Temperature: $temp"
        echo "  ────────────────────────────"
        python generate.py \
            --model-path "$MODEL" \
            --prompt "$prompt" \
            --temperature $temp \
            --cpu \
            --max-length 50 \
            2>/dev/null | grep -A 1 "Generated:"
    done
    echo ""
done

echo "=================================="
echo "Testing Complete"
echo "=================================="
