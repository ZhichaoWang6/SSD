#!/bin/bash
# Step 1: Generate training data for the Kangaroo adapter
# This collects hidden states from the full model on MMDuet2 multimodal data.

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # or your fine-tuned checkpoint
DATA_PATH=./data/annotations/ego-frame_input_format.json
OUTPUT_DIR=./training_data/
EXIT_LAYERS=2,3  # Save hidden states for both layer 2 and 3

cd "$(dirname "$0")/.."

python generate_training_data.py \
    --model_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --exit_layers $EXIT_LAYERS \
    --max_seq_len 4096 \
    --frame_interval 2.0
