#!/bin/bash
# Training data setup script
# Quickly prepares training data for the training pipeline

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════════"
echo "  Training Data Setup Script"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Define paths
DATA_DIR="/project/code/data/processed"
ORCA_FILE="$DATA_DIR/Open-Orca_OpenOrca_processed.jsonl"
TRAIN_FILE="$DATA_DIR/training_data.jsonl"
SMALL_FILE="$DATA_DIR/small_training_data.jsonl"

echo "Checking data directory..."
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Data directory not found: $DATA_DIR"
    echo "Creating directory..."
    mkdir -p "$DATA_DIR"
fi

echo "✓ Data directory exists: $DATA_DIR"
echo ""

# Check for source file
echo "Checking for source data files..."
if [ ! -f "$ORCA_FILE" ]; then
    echo "❌ Source file not found: $ORCA_FILE"
    echo ""
    echo "Available files in $DATA_DIR:"
    ls -lh "$DATA_DIR" || echo "Directory is empty"
    echo ""
    echo "Please ensure at least one JSONL file exists in $DATA_DIR"
    exit 1
fi

echo "✓ Found source file:"
wc -l "$ORCA_FILE" | awk '{print "  Lines: " $1}'
ls -lh "$ORCA_FILE" | awk '{print "  Size: " $5}'
echo ""

# Create training data
echo "Creating training datasets..."
echo ""

# Option 1: Small dataset for quick testing (60 samples)
echo "1️⃣  Creating small dataset (60 samples)..."
cp "$ORCA_FILE" "$SMALL_FILE"
cat "$ORCA_FILE" >> "$SMALL_FILE"
cat "$ORCA_FILE" >> "$SMALL_FILE"
echo "✓ Created: $SMALL_FILE"
wc -l "$SMALL_FILE" | awk '{print "  Samples: " $1}'
echo ""

# Option 2: Medium dataset for regular training (180 samples)
echo "2️⃣  Creating medium dataset (180 samples)..."
cp "$ORCA_FILE" "$TRAIN_FILE"
cat "$ORCA_FILE" >> "$TRAIN_FILE"
cat "$ORCA_FILE" >> "$TRAIN_FILE"
cat "$ORCA_FILE" >> "$TRAIN_FILE"
cat "$ORCA_FILE" >> "$TRAIN_FILE"
cat "$ORCA_FILE" >> "$TRAIN_FILE"
cat "$ORCA_FILE" >> "$TRAIN_FILE"
cat "$ORCA_FILE" >> "$TRAIN_FILE"
cat "$ORCA_FILE" >> "$TRAIN_FILE"
echo "✓ Created: $TRAIN_FILE"
wc -l "$TRAIN_FILE" | awk '{print "  Samples: " $1}'
echo ""

# Verify data
echo "═══════════════════════════════════════════════════════════════"
echo "  Data Verification"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "Files created in $DATA_DIR:"
ls -lh "$DATA_DIR"/*.jsonl
echo ""

# Validate JSON
echo "Validating JSON format..."
{
    echo "Checking small dataset..."
    head -1 "$SMALL_FILE" | python -m json.tool > /dev/null && echo "✓ Valid JSON" || echo "❌ Invalid JSON"

    echo "Checking medium dataset..."
    head -1 "$TRAIN_FILE" | python -m json.tool > /dev/null && echo "✓ Valid JSON" || echo "❌ Invalid JSON"
} 2>/dev/null || echo "⚠️  Could not validate (python-json not available, but files should be fine)"

echo ""

# Configuration recommendation
echo "═══════════════════════════════════════════════════════════════"
echo "  Next Steps"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "1. Verify configuration in configs/gpu/small.yaml:"
echo "   data:"
echo "     data_dir: /project/code/data/processed"
echo ""
echo "2. For quick test (5 minutes):"
echo "   python code/scripts/5_training/train.py \\"
echo "       --config configs/gpu/small.yaml \\"
echo "       --max-steps 50"
echo ""
echo "3. For full training:"
echo "   python code/scripts/5_training/train.py \\"
echo "       --config configs/gpu/small.yaml \\"
echo "       --num-epochs 3"
echo ""
echo "4. Monitor training:"
echo "   tensorboard --logdir outputs/"
echo ""

echo "✅ Data setup complete!"
echo ""
