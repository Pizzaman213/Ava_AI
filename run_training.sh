#!/bin/bash

# Correct way to run training with memory management

# Navigate to project root
cd /project

# First, set the environment variable
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.6'

# Then run the training command (correct path)
python code/scripts/5_training/train.py --config configs/gpu/small.yaml