#!/bin/bash
# Setup AG News dataset - Download and Process

set -e  # Exit on error

echo "========================================"
echo "AG News Dataset Setup"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "Project directory: $PROJECT_DIR"
echo ""

# Step 1: Download AG News
echo "Step 1: Downloading AG News dataset..."
echo "----------------------------------------"
python "$SCRIPT_DIR/download_ag_news.py"

if [ $? -ne 0 ]; then
    echo "Error: Failed to download AG News dataset"
    exit 1
fi

echo ""
echo "Step 2: Processing AG News dataset..."
echo "----------------------------------------"
python "$SCRIPT_DIR/process_ag_news.py"

if [ $? -ne 0 ]; then
    echo "Error: Failed to process AG News dataset"
    exit 1
fi

echo ""
echo "========================================"
echo "AG News Setup Complete!"
echo "========================================"
echo ""
echo "Dataset location: $PROJECT_DIR/data/ag_news/"
echo "  - Raw data: $PROJECT_DIR/data/ag_news/raw/"
echo "  - Processed data: $PROJECT_DIR/data/ag_news/processed/"
echo ""
echo "Training config updated: configs/gpu/small.yaml"
echo "  data_dir: /project/code/data/ag_news/processed"
echo ""
echo "You can now start training with:"
echo "  python scripts/5_training/train.py --config configs/gpu/small.yaml"
echo ""
