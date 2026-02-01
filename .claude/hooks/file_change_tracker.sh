#!/bin/bash
# File change tracker hook for Edit/Write tool uses
# Tracks modifications to critical training, model, and config files

LOG_DIR="/root/Ava_AI/logs/claude_activity"
LOG_FILE="$LOG_DIR/file_changes.jsonl"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Get input from stdin (Claude Code passes tool input via stdin as JSON)
INPUT=$(cat)

# Extract file path from the JSON input
FILE_PATH=$(echo "$INPUT" | grep -oP '"file_path"\s*:\s*"[^"]*"' | sed 's/"file_path"\s*:\s*"//' | sed 's/"$//' | head -1)

# If no file path found, try alternate field names
if [ -z "$FILE_PATH" ]; then
    FILE_PATH=$(echo "$INPUT" | grep -oP '"path"\s*:\s*"[^"]*"' | sed 's/"path"\s*:\s*"//' | sed 's/"$//' | head -1)
fi

# Critical paths to track (training, models, CUDA, configs, scripts)
CRITICAL_PATTERNS=(
    "code/src/ava/training/"
    "code/src/ava/models/"
    "code/src/ava/cuda/"
    "code/src/ava/optimizations/"
    "code/src/ava/data/"
    "code/src/ava/config/"
    "code/configs/moe/"
    "code/configs/rlhf/"
    "code/configs/distributed/"
    "code/scripts/5_training/"
    "code/scripts/6_rhlf_Finetuning/"
    "code/tests/"
)

# Check if the file matches any critical pattern
IS_CRITICAL=false
for pattern in "${CRITICAL_PATTERNS[@]}"; do
    if echo "$FILE_PATH" | grep -q "$pattern"; then
        IS_CRITICAL=true
        break
    fi
done

if [ "$IS_CRITICAL" = true ] && [ -n "$FILE_PATH" ]; then
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    SESSION_ID="${CLAUDE_SESSION_ID:-$(date +%s)}"

    # Determine the category
    CATEGORY="other"
    if echo "$FILE_PATH" | grep -q "training/"; then
        CATEGORY="training"
    elif echo "$FILE_PATH" | grep -q "models/"; then
        CATEGORY="models"
    elif echo "$FILE_PATH" | grep -q "cuda/"; then
        CATEGORY="cuda"
    elif echo "$FILE_PATH" | grep -q "optimizations/"; then
        CATEGORY="optimizations"
    elif echo "$FILE_PATH" | grep -q "data/"; then
        CATEGORY="data"
    elif echo "$FILE_PATH" | grep -q "config"; then
        CATEGORY="config"
    elif echo "$FILE_PATH" | grep -q "scripts/"; then
        CATEGORY="scripts"
    elif echo "$FILE_PATH" | grep -q "tests/"; then
        CATEGORY="tests"
    fi

    # Determine tool type (Edit vs Write)
    TOOL_TYPE="unknown"
    if echo "$INPUT" | grep -q '"old_string"'; then
        TOOL_TYPE="Edit"
    elif echo "$INPUT" | grep -q '"content"'; then
        TOOL_TYPE="Write"
    fi

    # Extract filename
    FILENAME=$(basename "$FILE_PATH" 2>/dev/null || echo "$FILE_PATH")

    # Escape file path for JSON
    ESCAPED_PATH=$(echo "$FILE_PATH" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g')

    # Log the change
    echo "{\"timestamp\":\"$TIMESTAMP\",\"session_id\":\"$SESSION_ID\",\"file_path\":\"$ESCAPED_PATH\",\"filename\":\"$FILENAME\",\"category\":\"$CATEGORY\",\"tool\":\"$TOOL_TYPE\"}" >> "$LOG_FILE"
fi

# Always exit 0 - don't block
exit 0
