#!/bin/bash
# Session start hook - displays project status and context
# Claude Code hook for Ava AI project

LOG_DIR="/root/Ava_AI/logs/claude_activity"
LOG_FILE="$LOG_DIR/sessions.jsonl"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Get session ID from environment or generate one
SESSION_ID="${CLAUDE_SESSION_ID:-$(date +%s)}"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Log session start
echo "{\"timestamp\":\"$TIMESTAMP\",\"event\":\"session_start\",\"session_id\":\"$SESSION_ID\"}" >> "$LOG_FILE"

# Output project context
echo "=========================================="
echo "  Ava AI Training Framework - Session Start"
echo "=========================================="
echo ""

# Git status
echo "📌 Git Status:"
cd /root/Ava_AI
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
echo "   Branch: $BRANCH"
MODIFIED=$(git status --porcelain 2>/dev/null | wc -l)
echo "   Modified files: $MODIFIED"
echo ""

# GPU status
echo "🖥️  GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | while read line; do
        echo "   $line"
    done
else
    echo "   nvidia-smi not available"
fi
echo ""

# Recent benchmarks
echo "📊 Recent Benchmarks:"
BENCHMARK_FILE="$LOG_DIR/benchmarks.jsonl"
if [ -f "$BENCHMARK_FILE" ]; then
    # Show last 3 benchmark results
    tail -n 10 "$BENCHMARK_FILE" 2>/dev/null | grep '"phase":"post"' | tail -n 3 | while read line; do
        CMD=$(echo "$line" | grep -oP '"command":"[^"]*"' | cut -d'"' -f4 | head -c 50)
        TOKENS=$(echo "$line" | grep -oP '"tokens_per_sec":[0-9.]+' | cut -d':' -f2)
        if [ -n "$TOKENS" ]; then
            echo "   • $CMD... → ${TOKENS} tok/s"
        fi
    done
    if [ $(wc -l < "$BENCHMARK_FILE" 2>/dev/null || echo 0) -eq 0 ]; then
        echo "   No benchmarks recorded yet"
    fi
else
    echo "   No benchmarks recorded yet"
fi
echo ""

# Reminders from CLAUDE.md
echo "⚠️  Development Guidelines:"
echo "   • Testing: Use RTX 3060 only (RTX 3090 Ti reserved for production)"
echo "   • Benchmarking: Always compare before/after for training code changes"
echo "   • Metrics: Log tokens/sec, memory usage, throughput"
echo ""
echo "=========================================="
