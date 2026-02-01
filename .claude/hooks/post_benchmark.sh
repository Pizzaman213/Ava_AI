#!/bin/bash
# Post-tool-use hook for benchmark/training commands
# Extracts metrics from command output and logs results

LOG_DIR="/root/Ava_AI/logs/claude_activity"
LOG_FILE="$LOG_DIR/benchmarks.jsonl"
SUMMARY_FILE="$LOG_DIR/benchmark_summary.md"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Get input from stdin (Claude Code passes tool result via stdin as JSON)
INPUT=$(cat)

# Extract command and output from the JSON input
COMMAND=$(echo "$INPUT" | grep -oP '"command"\s*:\s*"[^"]*"' | sed 's/"command"\s*:\s*"//' | sed 's/"$//' | head -1)
EXIT_CODE=$(echo "$INPUT" | grep -oP '"exit_code"\s*:\s*[0-9]+' | grep -oP '[0-9]+' | head -1)
STDOUT=$(echo "$INPUT" | grep -oP '"stdout"\s*:\s*"[^"]*"' | sed 's/"stdout"\s*:\s*"//' | sed 's/"$//')

# Default exit code if not found
EXIT_CODE="${EXIT_CODE:-0}"

# Check if this is a training/benchmark related command
if echo "$COMMAND" | grep -qiE "(train_pipeline|finetune|benchmark|test_|pytest|generate\.py|train_rlhf)"; then
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    SESSION_ID="${CLAUDE_SESSION_ID:-$(date +%s)}"

    # Get GPU memory state after execution
    GPU_MEM="unknown"
    if command -v nvidia-smi &> /dev/null; then
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' | tr ',' '/')
    fi

    # Extract metrics from output
    TOKENS_PER_SEC=""
    THROUGHPUT=""
    LOSS=""
    MEMORY_GB=""

    # Try to extract tokens/sec (various formats)
    if [ -n "$STDOUT" ]; then
        TOKENS_PER_SEC=$(echo "$STDOUT" | grep -oiP '(tokens?[/_]s(ec)?|tok/s)\s*[=:]\s*[0-9.]+' | grep -oP '[0-9.]+$' | tail -1)
        if [ -z "$TOKENS_PER_SEC" ]; then
            TOKENS_PER_SEC=$(echo "$STDOUT" | grep -oiP '[0-9.]+\s*(tokens?[/_]s|tok/s)' | grep -oP '^[0-9.]+' | tail -1)
        fi

        # Extract throughput
        THROUGHPUT=$(echo "$STDOUT" | grep -oiP 'throughput\s*[=:]\s*[0-9.]+' | grep -oP '[0-9.]+$' | tail -1)

        # Extract loss
        LOSS=$(echo "$STDOUT" | grep -oiP '(loss|train_loss)\s*[=:]\s*[0-9.]+' | grep -oP '[0-9.]+$' | tail -1)

        # Extract memory usage
        MEMORY_GB=$(echo "$STDOUT" | grep -oiP '(memory|mem|gpu_mem)\s*[=:]\s*[0-9.]+\s*gb' | grep -oP '[0-9.]+' | tail -1)
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

    # Build metrics JSON
    METRICS="{"
    if [ -n "$TOKENS_PER_SEC" ]; then
        METRICS="$METRICS\"tokens_per_sec\":$TOKENS_PER_SEC"
    fi
    if [ -n "$THROUGHPUT" ]; then
        [ "$METRICS" != "{" ] && METRICS="$METRICS,"
        METRICS="$METRICS\"throughput\":$THROUGHPUT"
    fi
    if [ -n "$LOSS" ]; then
        [ "$METRICS" != "{" ] && METRICS="$METRICS,"
        METRICS="$METRICS\"loss\":$LOSS"
    fi
    if [ -n "$MEMORY_GB" ]; then
        [ "$METRICS" != "{" ] && METRICS="$METRICS,"
        METRICS="$METRICS\"memory_gb\":$MEMORY_GB"
    fi
    METRICS="$METRICS}"

    # Log post-execution state
    echo "{\"timestamp\":\"$TIMESTAMP\",\"phase\":\"post\",\"command\":\"$ESCAPED_CMD\",\"session_id\":\"$SESSION_ID\",\"exit_code\":$EXIT_CODE,\"gpu_memory_mb\":\"$GPU_MEM\",\"metrics\":$METRICS,\"type\":\"$CMD_TYPE\"}" >> "$LOG_FILE"

    # Update human-readable summary
    if [ ! -f "$SUMMARY_FILE" ]; then
        echo "# Benchmark Summary" > "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        echo "Auto-generated log of benchmark and training runs." >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        echo "| Timestamp | Type | Command | Tokens/s | Memory | Exit |" >> "$SUMMARY_FILE"
        echo "|-----------|------|---------|----------|--------|------|" >> "$SUMMARY_FILE"
    fi

    # Add row to summary table
    SHORT_CMD=$(echo "$COMMAND" | sed 's/.*\///' | head -c 40)
    TOKENS_DISPLAY="${TOKENS_PER_SEC:-N/A}"
    MEM_DISPLAY="${GPU_MEM:-N/A}"
    echo "| $TIMESTAMP | $CMD_TYPE | \`$SHORT_CMD\` | $TOKENS_DISPLAY | $MEM_DISPLAY | $EXIT_CODE |" >> "$SUMMARY_FILE"
fi

# Always exit 0 - don't block
exit 0
