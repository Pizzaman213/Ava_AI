#!/bin/bash
# Pre-tool-use hook for benchmark/training commands
# Captures GPU state before execution

LOG_DIR="/root/Ava_AI/logs/claude_activity"
LOG_FILE="$LOG_DIR/benchmarks.jsonl"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Get command from stdin (Claude Code passes tool input via stdin as JSON)
INPUT=$(cat)

# Extract command from the JSON input
COMMAND=$(echo "$INPUT" | grep -oP '"command"\s*:\s*"[^"]*"' | sed 's/"command"\s*:\s*"//' | sed 's/"$//' | head -1)

# If no command found in JSON, try to get it directly
if [ -z "$COMMAND" ]; then
    COMMAND="$INPUT"
fi

# Check if this is a training/benchmark related command
if echo "$COMMAND" | grep -qiE "(train_pipeline|finetune|benchmark|test_|pytest|generate\.py|train_rlhf)"; then
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    SESSION_ID="${CLAUDE_SESSION_ID:-$(date +%s)}"

    # Get GPU memory state
    GPU_MEM="unknown"
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' | tr ',' '/')
    fi

    # Determine command type
    CMD_TYPE="other"
    if echo "$COMMAND" | grep -qi "benchmark"; then
        CMD_TYPE="benchmark"
    elif echo "$COMMAND" | grep -qi "train"; then
        CMD_TYPE="training"
    elif echo "$COMMAND" | grep -qi "test_\|pytest"; then
        CMD_TYPE="test"
    elif echo "$COMMAND" | grep -qi "generate"; then
        CMD_TYPE="generation"
    fi

    # Escape command for JSON
    ESCAPED_CMD=$(echo "$COMMAND" | sed 's/\\/\\\\/g' | sed 's/"/\\"/g' | tr '\n' ' ' | head -c 500)

    # Log pre-execution state
    echo "{\"timestamp\":\"$TIMESTAMP\",\"phase\":\"pre\",\"command\":\"$ESCAPED_CMD\",\"session_id\":\"$SESSION_ID\",\"gpu_memory_mb\":\"$GPU_MEM\",\"type\":\"$CMD_TYPE\"}" >> "$LOG_FILE"
fi

# Always exit 0 - don't block execution
exit 0
