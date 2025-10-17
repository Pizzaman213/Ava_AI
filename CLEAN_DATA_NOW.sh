#!/bin/bash
# Quick data cleaning script
# Removes conversational AI artifacts from training data

set -e

echo "=========================================="
echo "🧹 CLEANING TRAINING DATA"
echo "=========================================="
echo ""

DATA_DIR="/project/code/data/processed"
BACKUP_DIR="/project/code/data/processed_backup_$(date +%Y%m%d_%H%M%S)"

echo "📋 Data files to clean:"
ls -lh "$DATA_DIR"/*.jsonl | awk '{print "  ", $9, "(" $5 ")"}'
echo ""

# Files we DEFINITELY need to clean (conversational datasets)
CONTAMINATED_FILES=(
    "Open-Orca_SlimOrca_processed.jsonl"
    "Anthropic_hh-rlhf_processed.jsonl"
    "HuggingFaceH4_ultrachat_200k_processed.jsonl"
)

echo "🎯 High-priority contaminated files:"
for file in "${CONTAMINATED_FILES[@]}"; do
    if [ -f "$DATA_DIR/$file" ]; then
        size=$(ls -lh "$DATA_DIR/$file" | awk '{print $5}')
        echo "  ❌ $file ($size) - Contains conversational AI data"
    fi
done
echo ""

read -p "Do you want to REMOVE these contaminated files? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "📦 Creating backup first..."
    mkdir -p "$BACKUP_DIR"

    for file in "${CONTAMINATED_FILES[@]}"; do
        if [ -f "$DATA_DIR/$file" ]; then
            echo "  Backing up: $file"
            cp "$DATA_DIR/$file" "$BACKUP_DIR/"
            echo "  Removing: $file"
            rm "$DATA_DIR/$file"
        fi
    done

    echo ""
    echo "✅ Removed contaminated files!"
    echo "   Backup saved to: $BACKUP_DIR"
    echo ""
    echo "📊 Remaining clean datasets:"
    ls -lh "$DATA_DIR"/*.jsonl | awk '{print "  ✓", $9, "(" $5 ")"}'
    echo ""
    echo "🎯 These datasets should be clean:"
    echo "   • HuggingFaceFW/fineweb-edu (educational content)"
    echo "   • wikimedia/wikipedia (encyclopedia)"
    echo "   • allenai/c4 (web crawl)"
    echo "   • microsoft/orca-math (math problems)"
    echo ""
    echo "⚠️  IMPORTANT: You should restart training from scratch now"
    echo "   The current model has learned bad patterns from contaminated data"
else
    echo ""
    echo "❌ Cancelled - no files were modified"
fi
