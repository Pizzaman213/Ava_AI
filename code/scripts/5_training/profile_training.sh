#!/bin/bash
#
# Ava Training Profiler - Automatic Nsight Systems Profiling
#
# This script wraps train_pipeline.py with nsys to capture full GPU profiling data.
# The nsys profile is automatically saved to the training run's profiles/ folder.
#
# Usage:
#   ./profile_training.sh --config /project/code/configs/moe/large.yaml
#   ./profile_training.sh --config /project/code/configs/moe/large.yaml --epochs 1
#   MAX_STEPS=1000 ./profile_training.sh --config /project/code/configs/moe/large.yaml
#
# Multi-GPU:
#   By default, profiling uses only GPU 0 for consistent results.
#   To use multiple GPUs: PROFILE_ALL_GPUS=1 ./profile_training.sh ...
#   To select specific GPU: CUDA_VISIBLE_DEVICES=1 ./profile_training.sh ...
#
# Output:
#   - Nsight Systems report saved to the training run's profiles/ folder
#   - PyTorch profiler traces also in the run's profiles/ folder
#   - Summary statistics printed to console
#
# Note: Profiling is DISABLED by default in normal training.
#       Use this script or --enable-profiling to enable it.
#

set -e

# Default to single GPU (GPU 0) unless PROFILE_ALL_GPUS is set
if [ -z "${PROFILE_ALL_GPUS}" ] && [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "Note: Using single GPU (GPU 0) by default for profiling."
    echo "      To use all GPUs: PROFILE_ALL_GPUS=1 $0 $@"
    echo "      To select specific GPU: CUDA_VISIBLE_DEVICES=1 $0 $@"
    echo ""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_NAME="nsys_profile_${TIMESTAMP}"

# Output directories
# RunManager creates runs in <base>/pretraining/<run_id>
# Navigate to code/outputs (two levels up from scripts/5_training/)
BASE_OUTPUT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)/outputs"
RUNS_DIR="${BASE_OUTPUT_DIR}/pretraining"
TEMP_NSYS_DIR="/tmp/nsys_profile_${TIMESTAMP}"
mkdir -p "${TEMP_NSYS_DIR}"
mkdir -p "${RUNS_DIR}"

echo "============================================================"
echo "Ava Training Profiler"
echo "============================================================"
echo "Timestamp: ${TIMESTAMP}"
echo "Script dir: ${SCRIPT_DIR}"
echo "Output dir: ${BASE_OUTPUT_DIR}"
echo "Runs dir: ${RUNS_DIR}"
echo "Temp nsys dir: ${TEMP_NSYS_DIR}"
echo ""

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "ERROR: nsys (Nsight Systems) not found in PATH"
    echo "Please install NVIDIA Nsight Systems or add it to your PATH"
    exit 1
fi

# Check for heterogeneous GPU setup and warn user
if command -v nvidia-smi &> /dev/null; then
    GPU_MEMORIES=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -u | wc -l)
    if [ "$GPU_MEMORIES" -gt 1 ]; then
        echo "============================================================"
        echo "WARNING: Heterogeneous GPU setup detected!"
        echo "============================================================"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
        echo ""
        echo "GPUs have different memory sizes. Options:"
        echo "  1. Use single GPU: CUDA_VISIBLE_DEVICES=0 $0 $@"
        echo "  2. Use heterogeneous config: --config code/configs/moe/heterogeneous_multi_gpu.yaml"
        echo ""
        echo "Continuing with current settings..."
        echo "============================================================"
        echo ""
    fi
fi

echo "Running nsys profile with full CUDA/NVTX/memory tracing..."
echo "(Note: CPU profiling may be disabled due to kernel settings - GPU profiling works fine)"
echo ""

# Default to 500 steps if not specified
MAX_STEPS="${MAX_STEPS:-500}"

echo "Max steps: ${MAX_STEPS}"
echo ""

# Record timestamp before training starts (for finding run directory)
PRE_TRAINING_TIME=$(date +%s)

# Cleanup function to move nsys files on exit/interrupt
cleanup_and_move_profiles() {
    local exit_code=$?
    echo ""
    echo "============================================================"
    echo "Cleaning up and moving profile files..."
    echo "============================================================"

    # Find the run directory
    local run_dir=""
    for dir in "${RUNS_DIR}"/ava_training_*; do
        if [ -d "$dir" ]; then
            DIR_MTIME=$(stat -c %Y "$dir" 2>/dev/null || stat -f %m "$dir" 2>/dev/null || echo "0")
            if [ "$DIR_MTIME" -ge "$PRE_TRAINING_TIME" ]; then
                if [ -z "$run_dir" ]; then
                    run_dir="$dir"
                else
                    EXISTING_MTIME=$(stat -c %Y "$run_dir" 2>/dev/null || stat -f %m "$run_dir" 2>/dev/null || echo "0")
                    if [ "$DIR_MTIME" -gt "$EXISTING_MTIME" ]; then
                        run_dir="$dir"
                    fi
                fi
            fi
        fi
    done

    # Fallback to newest directory
    if [ -z "$run_dir" ]; then
        run_dir=$(ls -td "${RUNS_DIR}"/ava_training_* 2>/dev/null | head -1)
    fi

    # Move files if we found a directory and temp files exist
    if [ -n "$run_dir" ] && [ -d "$run_dir" ] && [ -d "${TEMP_NSYS_DIR}" ]; then
        local profile_dir="${run_dir}/profiles"
        mkdir -p "${profile_dir}"

        for f in "${TEMP_NSYS_DIR}"/*; do
            if [ -f "$f" ]; then
                mv "$f" "${profile_dir}/"
                echo "Moved $(basename "$f") to: ${profile_dir}/"
            fi
        done

        rm -rf "${TEMP_NSYS_DIR}" 2>/dev/null || true
        echo "Profile files saved to: ${profile_dir}"
    elif [ -d "${TEMP_NSYS_DIR}" ]; then
        echo "WARNING: Could not find run directory. Profile files remain in: ${TEMP_NSYS_DIR}"
    fi

    exit $exit_code
}

# Set trap for cleanup on interrupt (but not on normal exit - we handle that below)
trap 'cleanup_and_move_profiles' INT TERM

# Run nsys with comprehensive tracing options
# Output to temp directory, will be moved to run folder after
#
# NOTE: We do NOT use --enable-profiling here because nsys already provides
# comprehensive GPU profiling. Using both causes CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED
# since only one tool can subscribe to CUPTI at a time.
#
# nsys captures: CUDA kernels, memory operations, NVTX markers, API calls
# PyTorch profiler would be redundant and causes conflicts
#
# CPU Profiling Options:
#   --sample=cpu         : Enable CPU sampling (may require root or perf_event_paranoid=1)
#   --backtrace=dwarf    : Use DWARF for accurate Python backtraces
#   --cpuctxsw=process-tree : Track CPU context switches for process tree
#
# To enable CPU sampling without root, run:
#   sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
#   sudo sh -c 'echo 0 > /proc/sys/kernel/kptr_restrict'

# Check if CPU sampling is likely to work
CPU_SAMPLE_ARGS=""
if [ -r /proc/sys/kernel/perf_event_paranoid ]; then
    PARANOID_LEVEL=$(cat /proc/sys/kernel/perf_event_paranoid)
    if [ "$PARANOID_LEVEL" -le 1 ] || [ "$(id -u)" -eq 0 ]; then
        CPU_SAMPLE_ARGS="--sample=cpu --backtrace=dwarf --cpuctxsw=process-tree"
        echo "CPU sampling enabled (perf_event_paranoid=$PARANOID_LEVEL)"
    else
        echo "NOTE: CPU sampling disabled (perf_event_paranoid=$PARANOID_LEVEL > 1)"
        echo "      To enable: sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'"
    fi
fi

nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    ${CPU_SAMPLE_ARGS} \
    --stats=true \
    --force-overwrite=true \
    --kill=sigterm \
    --output="${TEMP_NSYS_DIR}/${PROFILE_NAME}" \
    python "${SCRIPT_DIR}/train_pipeline.py" \
        --max-steps "${MAX_STEPS}" \
        --save-dir "${BASE_OUTPUT_DIR}" \
        "$@"

NSYS_EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Profiling Complete"
echo "============================================================"

# Find the run directory created during this training session
# Use multiple strategies to ensure we find it
RUN_DIR=""

# Strategy 1: Find the most recently modified directory created after PRE_TRAINING_TIME
for dir in "${RUNS_DIR}"/ava_training_*; do
    if [ -d "$dir" ]; then
        # Check if directory was created after we started
        DIR_MTIME=$(stat -c %Y "$dir" 2>/dev/null || stat -f %m "$dir" 2>/dev/null || echo "0")
        if [ "$DIR_MTIME" -ge "$PRE_TRAINING_TIME" ]; then
            # Prefer the most recent one
            if [ -z "$RUN_DIR" ]; then
                RUN_DIR="$dir"
            else
                EXISTING_MTIME=$(stat -c %Y "$RUN_DIR" 2>/dev/null || stat -f %m "$RUN_DIR" 2>/dev/null || echo "0")
                if [ "$DIR_MTIME" -gt "$EXISTING_MTIME" ]; then
                    RUN_DIR="$dir"
                fi
            fi
        fi
    fi
done

# Strategy 2: If strategy 1 failed, find directory with matching timestamp prefix
if [ -z "$RUN_DIR" ]; then
    DATE_PREFIX=$(echo "$TIMESTAMP" | cut -c1-8)  # YYYYMMDD
    for dir in "${RUNS_DIR}"/ava_training_${DATE_PREFIX}*; do
        if [ -d "$dir" ]; then
            RUN_DIR="$dir"
            # Keep searching for a more recent one
        fi
    done
fi

# Strategy 3: Fallback to newest directory overall
if [ -z "$RUN_DIR" ]; then
    RUN_DIR=$(ls -td "${RUNS_DIR}"/ava_training_* 2>/dev/null | head -1)
fi

# Function to move nsys files (used for both success and cleanup)
move_nsys_files() {
    local target_dir="$1"
    local moved=0

    mkdir -p "${target_dir}"

    # Move .nsys-rep file
    if [ -f "${TEMP_NSYS_DIR}/${PROFILE_NAME}.nsys-rep" ]; then
        mv "${TEMP_NSYS_DIR}/${PROFILE_NAME}.nsys-rep" "${target_dir}/"
        echo "Moved nsys profile to: ${target_dir}/${PROFILE_NAME}.nsys-rep"
        moved=1
    fi

    # Move .sqlite file
    if [ -f "${TEMP_NSYS_DIR}/${PROFILE_NAME}.sqlite" ]; then
        mv "${TEMP_NSYS_DIR}/${PROFILE_NAME}.sqlite" "${target_dir}/"
        echo "Moved nsys sqlite to: ${target_dir}/${PROFILE_NAME}.sqlite"
        moved=1
    fi

    # Move any other nsys output files (e.g., .qdstrm)
    for f in "${TEMP_NSYS_DIR}"/*; do
        if [ -f "$f" ]; then
            mv "$f" "${target_dir}/"
            echo "Moved $(basename "$f") to: ${target_dir}/"
            moved=1
        fi
    done

    # Clean up temp directory if we moved files
    if [ $moved -eq 1 ] && [ -d "${TEMP_NSYS_DIR}" ]; then
        rm -rf "${TEMP_NSYS_DIR}"
    fi

    # Return 0 (success) if files were moved, 1 (failure) if not
    [ $moved -eq 1 ]
}

# If we found a run directory, move nsys output there
if [ -n "${RUN_DIR}" ] && [ -d "${RUN_DIR}" ]; then
    PROFILE_DIR="${RUN_DIR}/profiles"
    echo "Found run directory: ${RUN_DIR}"

    if move_nsys_files "${PROFILE_DIR}"; then
        FINAL_PROFILE="${PROFILE_DIR}/${PROFILE_NAME}.nsys-rep"
    else
        echo "WARNING: No nsys output files found in ${TEMP_NSYS_DIR}"
        FINAL_PROFILE="${TEMP_NSYS_DIR}/${PROFILE_NAME}.nsys-rep"
    fi
else
    echo "WARNING: Could not find training run directory, profile remains in temp location"
    echo "Temp directory: ${TEMP_NSYS_DIR}"
    echo "You can manually move files with:"
    echo "  mv ${TEMP_NSYS_DIR}/* <your_run_dir>/profiles/"
    FINAL_PROFILE="${TEMP_NSYS_DIR}/${PROFILE_NAME}.nsys-rep"
fi

if [ ${NSYS_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "SUCCESS: Nsys profile saved to ${FINAL_PROFILE}"
    echo ""
    echo "To view the profile:"
    echo "  1. GUI (requires display): nsys-ui ${FINAL_PROFILE}"
    echo "  2. Terminal stats: nsys stats ${FINAL_PROFILE}"
    echo "  3. Export to SQLite: nsys export -t sqlite ${FINAL_PROFILE}"
    echo ""
    if [ -n "${RUN_DIR}" ]; then
        echo "Training run directory: ${RUN_DIR}"
        echo "PyTorch profiler traces: ${PROFILE_DIR}/"
    fi
    echo ""

    # Print summary statistics
    echo "============================================================"
    echo "Quick Statistics Summary"
    echo "============================================================"
    nsys stats "${FINAL_PROFILE}" 2>/dev/null || true

    echo ""
    echo "============================================================"
    echo "PROFILE OUTPUT PATHS"
    echo "============================================================"
    echo "NSys Report: ${FINAL_PROFILE}"
    echo "SQLite DB:   ${FINAL_PROFILE%.nsys-rep}.sqlite"
    if [ -n "${RUN_DIR}" ]; then
        echo "Run Dir:     ${RUN_DIR}"
        echo "Profiles:    ${PROFILE_DIR}/"
    fi
    echo "============================================================"
else
    echo "ERROR: Profiling failed with exit code ${NSYS_EXIT_CODE}"
    exit ${NSYS_EXIT_CODE}
fi
