#!/bin/bash
set -e  # Exit on error

# Configuration
CUDA_VISIBLE_DEVICES="4,5,6,7"
MODEL_NAME=$1
MODEL_BASE_NAME=$(basename $MODEL_NAME)
EVAL_ENGINE="sglang"  # either vllm or sglang, recommend sglang for native data parallel
SERVER_PORT=8050
SERVER_HOST="127.0.0.1"
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}/v1"

# Validate input
if [ -z "$MODEL_NAME" ]; then
    echo "Error: Model name not provided"
    echo "Usage: $0 <model_path>"
    exit 1
fi

echo "Starting evaluation for model: $MODEL_NAME"
echo "Using evaluation engine: $EVAL_ENGINE"

# Function to start the inference server
start_server() {
    echo "Starting $EVAL_ENGINE server..."
    
    if [ "$EVAL_ENGINE" == "vllm" ]; then
        TENSOR_PARALLEL_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
        TENSOR_PARALLEL_SIZE=$((TENSOR_PARALLEL_SIZE + 1))
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES vllm serve $MODEL_NAME \
            --port $SERVER_PORT --tensor-parallel-size $TENSOR_PARALLEL_SIZE > vllm.log 2>&1 &
        SERVER_PID=$!
    elif [ "$EVAL_ENGINE" == "sglang" ]; then
        DATA_PARALLEL_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
        DATA_PARALLEL_SIZE=$((DATA_PARALLEL_SIZE + 1))
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m sglang.launch_server --model-path $MODEL_NAME \
            --host $SERVER_HOST --dp $DATA_PARALLEL_SIZE --port $SERVER_PORT > sglang.log 2>&1 &
        SERVER_PID=$!
    else
        echo "Error: Unknown evaluation engine '$EVAL_ENGINE'"
        exit 1
    fi
    
    # Wait for server to start up
    echo "Waiting for server to start up..."
    while ! curl -s "$SERVER_URL/models" > /dev/null; do
        sleep 2
        echo "Still waiting for server..."
        # Check if server process is still running
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Error: Server process died unexpectedly. Check logs."
            exit 1
        fi
    done
    echo "Server is up and running!"
}

# Function to stop the server
stop_server() {
    echo "Stopping $EVAL_ENGINE server..."
    if [ "$EVAL_ENGINE" == "vllm" ]; then
        pkill -f "vllm serve $MODEL_NAME --port $SERVER_PORT" || true
    elif [ "$EVAL_ENGINE" == "sglang" ]; then
        pkill -f "python -m sglang.launch_server --model-path $MODEL_NAME --host $SERVER_HOST --dp $DATA_PARALLEL_SIZE --port $SERVER_PORT" || true
    fi
    sleep 2
}

# Cleanup on script exit
cleanup() {
    stop_server
    echo "Cleanup complete"
}
trap cleanup EXIT

# Math Evaluation Harness
echo "Running Math Evaluation Harness..."
cd math-evaluation-harness || { echo "Error: math-evaluation-harness directory not found"; exit 1; }
OUTPUT_DIR="data/eval/$MODEL_BASE_NAME"
PROMPT_TYPE="qwen-self-play"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME $OUTPUT_DIR
cd ..

# Start inference server for remaining evaluations
start_server

# Run LiveCodeBench and IF-Eval
echo "Running LiveCodeBench and IF-Eval..."
cd qwq-eval || { echo "Error: qwq-eval directory not found"; exit 1; }
bash run.sh $MODEL_NAME $SERVER_URL
cd ..

# Run GPQA, MMLU, DROP, SimpleQA
echo "Running SimpleEvals (MMLU Pro)..."
python -m simple-evals.simple_evals \
    --model_name_or_path $MODEL_NAME \
    --base_url $SERVER_URL \
    --max_tokens 32768 \
    --tasks mmlu_pro

echo "Finished all evaluations successfully!"

