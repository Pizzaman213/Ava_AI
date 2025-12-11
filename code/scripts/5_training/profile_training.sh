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
# Output:
#   - Nsight Systems report saved to the training run's profiles/ folder
#   - PyTorch profiler traces also in the run's profiles/ folder
#   - Summary statistics printed to console
#
# Note: Profiling is DISABLED by default in normal training.
#       Use this script or --enable-profiling to enable it.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROFILE_NAME="nsys_profile_${TIMESTAMP}"

# Output directories
# RunManager creates runs in <base>/pretraining/<run_id>
BASE_OUTPUT_DIR="${SCRIPT_DIR}/outputs"
RUNS_DIR="${BASE_OUTPUT_DIR}/pretraining"
TEMP_NSYS_DIR="/tmp/nsys_profile_${TIMESTAMP}"
mkdir -p "${TEMP_NSYS_DIR}"
mkdir -p "${RUNS_DIR}"

echo "============================================================"
echo "Ava Training Profiler"
echo "============================================================"
echo "Nsys output will be saved to the training run's profiles/ folder"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "ERROR: nsys (Nsight Systems) not found in PATH"
    echo "Please install NVIDIA Nsight Systems or add it to your PATH"
    exit 1
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
nsys profile \
    --trace=cuda,nvtx \
    --cuda-memory-usage=true \
    --stats=true \
    --force-overwrite=true \
    --kill=sigterm \
    --output="${TEMP_NSYS_DIR}/${PROFILE_NAME}" \
    python "${SCRIPT_DIR}/train_pipeline.py" \
        --enable-profiling \
        --profile-start-step 0 \
        --profile-end-step "${MAX_STEPS}" \
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

    return $moved
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
else
    echo "ERROR: Profiling failed with exit code ${NSYS_EXIT_CODE}"
    exit ${NSYS_EXIT_CODE}
fi
