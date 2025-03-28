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


# List of environments (passed as args)
ENV_IDS=("TicTacToe-v0")
EVAL_ENV_IDS=("ConnectFour-v0")

# Number of iterations for training loop
NUM_ITERATIONS=25
EPISODES_PER_ITER=4096 #512 #100 #100
EVAL_EPISODES=512 #0 #20

# Maximum sequence length
MAX_SEQ_LEN=8192
current_checkpoint="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TRAIN_METHOD="spag"

# Number of GPUs to use (set dynamically)
NUM_GPUS_REQUESTED=4  # Change this to set how many GPUs to use
MAX_WORKERS=512
VLLM_MAX_NUM_SEQ=64

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


# Define WandB run info
LOG_WANDB="TRUE"
WANDB_PROJECT="SuperHumanSofty"
WANDB_RUN_NAME="$current_checkpoint-$TRAIN_METHOD-$(date +"%Y%m%d-%H%M%S")"
WANDB_RUN_ID="master-run-$(date +"%Y%m%d-%H%M%S")" 

export WANDB_RUN_NAME
export WANDB_RUN_ID
export WANDB_PROJECT


echo "Starting RL training loop for $NUM_ITERATIONS iterations..."
echo "with environments: $ENV_IDS"
echo "Using GPUs: $GPU_IDS"

for ((i=1; i<=NUM_ITERATIONS; i++)); do
    echo "=== Iteration $i ==="

    # Run data collection
    CUDA_VISIBLE_DEVICES=$GPU_IDS python3 collect_data.py \
        --checkpoint $current_checkpoint\
        --episodes $EPISODES_PER_ITER \
        --max-seq-len $MAX_SEQ_LEN \
        --env-ids "${ENV_IDS[@]}" \
        --output-dir "$RUN_FOLDER" \
        --max-workers $MAX_WORKERS \
        --vllm-max-num-seq $VLLM_MAX_NUM_SEQ \
        --iter $i \
        --run-eval \
        --eval-env-ids "${EVAL_ENV_IDS[@]}" \
        --eval-episodes $EVAL_EPISODES

    echo "[Training] Running training script..."
    
    # Run training script (modify with actual script path & arguments)
    deepspeed --num_gpus $NUM_GPUS train_model.py \
        --max-seq-len $MAX_SEQ_LEN \
        --output-dir "$RUN_FOLDER" \
        --iter $i \
        --train-method $TRAIN_METHOD


    # Log relevant to wandb
    if [ "$LOG_WANDB" = "TRUE" ]; then
        echo "=== Updating WandB for iter  $i ==="
        python3 wandb_summary.py \
            --iter $i \
            --data-folder $RUN_FOLDER \
            --wandb-project $WANDB_PROJECT \
            --wandb-run-name $WANDB_RUN_NAME
    fi


    current_checkpoint="$CHECKPOINT_FOLDER/$i/model"

    echo "=== Completed Iteration $i ==="
done

echo "Training loop finished!"



