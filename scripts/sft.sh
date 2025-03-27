#!/bin/bash
set -e

# Base directory for storing runs
BASE_DIR="training_runs"
LOG_TYPE="wandb"  # either "wandb" or "offline"

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

# Check if wandb is selected and logged in
if [ "$LOG_TYPE" = "wandb" ]; then
    echo "Checking wandb login status..."
    if ! wandb status &>/dev/null; then
        echo "Error: wandb is not logged in. Please run 'wandb login' first."
        exit 1
    else
        echo "wandb is logged in and will be used for logging."
    fi
fi

# Number of iterations for training loop
NUM_ITERATIONS=4  # Change as needed

# List of environments (passed as args)
ENV_IDS=("TicTacToe-v0")
EVAL_ENV_IDS=("ConnectFour-v0")
EVAL_EPISODES=128

# Maximum sequence length
MAX_SEQ_LEN=16384
EPISODES_PER_ITER=128 #100 #100
current_checkpoint="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# current_checkpoint="/scratch/simon/AcornRL/training_runs/qwen_32b/checkpoints/1/model"

# Number of GPUs to use (set dynamically)
NUM_GPUS_REQUESTED=4  # Change this to set how many GPUs to use

# Detect number of GPUs
NUM_GPUS_AVAILABLE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS_AVAILABLE GPUs."

# Ensure we don't request more GPUs than available
if [ "$NUM_GPUS_REQUESTED" -gt "$NUM_GPUS_AVAILABLE" ]; then
    echo "Warning: Requested $NUM_GPUS_REQUESTED GPUs, but only $NUM_GPUS_AVAILABLE available."
    NUM_GPUS=$NUM_GPUS_AVAILABLE
else
    NUM_GPUS=$NUM_GPUS_REQUESTED
fi
GPU_IDS=$(seq -s, 0 $((NUM_GPUS - 1)))  # Generates "0,1" for 2 GPUs, etc.
GPU_IDS="4,5,6,7"

echo "Starting RL training loop for $NUM_ITERATIONS iterations..."
echo "with environments: $ENV_IDS"
echo "Using GPUs: $GPU_IDS"

for ((i=1; i<=NUM_ITERATIONS; i++)); do
    echo "=== Iteration $i ==="

    # Run data collection
    CUDA_VISIBLE_DEVICES=$GPU_IDS python3 collect_data.py \
        --checkpoint $current_checkpoint \
        --episodes $EPISODES_PER_ITER \
        --max-seq-len $MAX_SEQ_LEN \
        --env-ids "${ENV_IDS[@]}" \
        --output-dir "$RUN_FOLDER" \
        --iter $i \
        --run-eval \
        --eval-env-ids "${EVAL_ENV_IDS[@]}" \
        --eval-episodes $EVAL_EPISODES

    echo "[Training] Running training script..."
    
    # Run training script (modify with actual script path & arguments)
    deepspeed --include="localhost:4,5,6,7" train_model.py \
        --max-seq-len $MAX_SEQ_LEN \
        --output-dir /scratch/simon/AcornRL/training_runs/qwen_32b \
        --train-method sft \
        --iter $i \
        --use-wandb \
        --wandb-project "acornrl-sft-training" \
        --gamma 0.9

    current_checkpoint="$CHECKPOINT_FOLDER/$i/model"

    echo "=== Completed Iteration $i ==="
done

echo "Training loop finished!"



