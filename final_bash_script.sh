#!/bin/bash

# Base directory for storing runs
BASE_DIR="training_runs"

# Get current date and time
CURRENT_DATE=$(date +"%Y-%m-%d")
CURRENT_TIME=$(date +"%H-%M-%S")

# Define directories
DAY_FOLDER="$BASE_DIR/$CURRENT_DATE"
RUN_FOLDER="$DAY_FOLDER/$CURRENT_TIME"
DATA_FOLDER="$RUN_FOLDER/data"
CHECKPOINT_FOLDER="$RUN_FOLDER/checkpoints"
LOG_FOLDER="$RUN_FOLDER/logging"

# Create necessary directories
mkdir -p "$DATA_FOLDER" "$CHECKPOINT_FOLDER" "$LOG_FOLDER"

# Number of iterations for training loop
NUM_ITERATIONS=2  # Change as needed

# List of environments (passed as args)
ENV_IDS=("SpellingBee-v0")

# Maximum sequence length
MAX_SEQ_LEN=128 #8192
EPISODES_PER_ITER=2
current_checkpoint="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


echo "Starting RL training loop for $NUM_ITERATIONS iterations..."
echo "with environments: $ENV_IDS"

for ((i=1; i<=NUM_ITERATIONS; i++)); do
    echo "=== Iteration $i ==="

    # Run data collection
    python3 collect_data.py \
        --checkpoint $current_checkpoint\
        --episodes $EPISODES_PER_ITER \
        --max-seq-len $MAX_SEQ_LEN \
        --env-ids "${ENV_IDS[@]}" \
        --output-dir "$RUN_FOLDER" \
        --iter $i

    echo "[Training] Running training script..."
    
    # Run training script (modify with actual script path & arguments)
    deepspeed --num_gpus 0 train_lora_model.py \
        --max-seq-len $MAX_SEQ_LEN \
        --output-dir "$RUN_FOLDER" \
        --iter $i

    current_checkpoint="$CHECKPOINT_FOLDER/$i/model"

    echo "=== Completed Iteration $i ==="
done

echo "Training loop finished!"



