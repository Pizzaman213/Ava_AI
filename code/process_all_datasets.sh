#!/bin/bash

echo "🚀 Processing all raw datasets to Arrow format..."
echo "="
echo ""

# Process Alpaca (even if already done, to ensure consistency)
echo "📊 Processing Alpaca..."
python3 scripts/data_prep/prepare_data.py \
    --input-path data/pretraining/raw/alpaca \
    --output-dir data/pretraining/processed/alpaca \
    --input-format json \
    --max-length 512

# Process OpenAssistant
echo "📊 Processing OpenAssistant..."
python3 scripts/data_prep/prepare_data.py \
    --input-path data/pretraining/raw/openassistant \
    --output-dir data/pretraining/processed/openassistant \
    --input-format json \
    --max-length 512

# Process WizardLM
echo ""
echo "📊 Processing WizardLM..."
python3 scripts/data_prep/prepare_data.py \
    --input-path data/pretraining/raw/wizardlm \
    --output-dir data/pretraining/processed/wizardlm \
    --input-format json \
    --max-length 512

# Process TinyStories
echo ""
echo "📊 Processing TinyStories..."
python3 scripts/data_prep/prepare_data.py \
    --input-path data/pretraining/raw/tinystories \
    --output-dir data/pretraining/processed/tinystories \
    --input-format json \
    --max-length 512

# Process Wikipedia Sample if it exists
if [ -d "data/pretraining/raw/wikipedia_sample" ] && [ "$(ls -A data/pretraining/raw/wikipedia_sample)" ]; then
    echo ""
    echo "📊 Processing Wikipedia Sample..."
    python3 scripts/data_prep/prepare_data.py \
        --input-path data/pretraining/raw/wikipedia_sample \
        --output-dir data/pretraining/processed/wikipedia_sample \
        --input-format json \
        --max-length 512
fi

echo ""
echo "="
echo "✅ All datasets processed!"
echo ""
echo "📝 Now updating config file..."

# Update the config file to include all processed datasets
cat > /tmp/config_update.txt << 'EOF'
  train_paths:
    - ./data/pretraining/processed/alpaca/train/shard_*
    - ./data/pretraining/processed/openassistant/train/shard_*
    - ./data/pretraining/processed/wizardlm/train/shard_*
    - ./data/pretraining/processed/tinystories/train/shard_*
    - ./data/pretraining/processed/train/shard_*
EOF

echo "Config should be updated with:"
cat /tmp/config_update.txt
echo ""
echo "🚀 Ready to train with:"
echo "   python3 scripts/training/train_with_data.py --config configs/mps/small.yaml"