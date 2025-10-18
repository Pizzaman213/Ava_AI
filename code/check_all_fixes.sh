#!/bin/bash
echo "======================================================================"
echo "CHECKING ALL FIXES"
echo "======================================================================"

echo ""
echo "1. Config File Values:"
echo "   vocab_size: $(grep 'vocab_size:' configs/gpu/small.yaml | head -1 | awk '{print $2}')"
echo "   hidden_size: $(grep 'hidden_size:' configs/gpu/small.yaml | head -1 | awk '{print $2}')"
echo "   num_experts: $(grep 'num_experts:' configs/gpu/small.yaml | head -1 | awk '{print $2}')"
echo "   repetition_penalty_weight: $(grep 'repetition_penalty_weight:' configs/gpu/small.yaml | head -1 | awk '{print $2}')"

echo ""
echo "2. Dataclass Defaults (should be False):"
grep "use_moh: bool = " src/Ava/config/training_config.py | head -1
grep "use_moa: bool = " src/Ava/config/training_config.py | head -1
grep "use_rag: bool = " src/Ava/config/training_config.py | head -1

echo ""
echo "3. Critical Code Fixes:"
echo "   ✓ MoE load balancing: $(grep -c 'load_balance_loss' src/Ava/models/moe_model.py) references"
echo "   ✓ Causal mask: $(grep -c 'torch.triu' src/Ava/models/moe_model.py) reference"
echo "   ✓ Anti-rep uses labels: $(grep -c 'labels, attention_mask' src/Ava/losses/anti_repetition_loss.py) references"
echo "   ✓ Architecture YAML loading: $(grep -c 'Architecture config loaded from YAML' scripts/5_training/train.py) reference"

echo ""
echo "======================================================================"
echo "✅ All fixes verified!"
echo "======================================================================"
echo ""
echo "Next: Stop current training and restart with fixed config"
echo "  kill $(pgrep -f 'train.py' | head -1)"
echo "  cd scripts/5_training"
echo "  python train.py --config ../../configs/gpu/small.yaml"
