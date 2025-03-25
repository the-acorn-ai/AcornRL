CUDA_VISIBLE_DEVICES="4,5,6,7"
MODEL_NAME=$1
MODEL_BASE_NAME=$(basename $MODEL_NAME)
EVAL_ENGINE="sglang" # either vllm or sglang

# Math Evaluation Harness: Math-500, AIME 2024, AIME 2025, OlympiadBench
# cd math-evaluation-harness
# OUTPUT_DIR="data/eval/$MODEL_BASE_NAME"
# PROMPT_TYPE="qwen-self-play"
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME $OUTPUT_DIR

# SimpleEvals: gpqa, mmlu, drop, simpleqa
# cd ..
if [ "$EVAL_ENGINE" == "vllm" ]; then
    TENSOR_PARALLEL_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
    TENSOR_PARALLEL_SIZE=$((TENSOR_PARALLEL_SIZE + 1))
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES vllm serve $MODEL_NAME \
        --port 8050 --tensor-parallel-size $TENSOR_PARALLEL_SIZE > vllm.log 2>&1 &
elif [ "$EVAL_ENGINE" == "sglang" ]; then
    DATA_PARALLEL_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
    DATA_PARALLEL_SIZE=$((DATA_PARALLEL_SIZE + 1))
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m sglang.launch_server --model-path $MODEL_NAME \
        --host 127.0.0.1 --dp $DATA_PARALLEL_SIZE --port 8050 > sglang.log 2>&1 &
fi

VLLM_BASE_URL="http://127.0.0.1:8050/v1"
# Wait for vLLM server to start up
echo "Waiting for vLLM server to start up..."
while ! curl -s "$VLLM_BASE_URL/models" > /dev/null; do
    sleep 2
    echo "Still waiting for vLLM server..."
done
echo "vLLM server is up and running!"

# cd qwq-eval

# Run LiveCodeBench and IF-Eval
# bash run.sh $MODEL_NAME $VLLM_BASE_URL

# cd ..
# Run GPQA, MMLU, DROP, SimpleQA
python -m simple-evals.simple_evals \
    --model_name_or_path $MODEL_NAME \
    --base_url $VLLM_BASE_URL \
    --max_tokens 32768 \
    --tasks gpqa mmlu drop simpleqa

pkill -f "vllm serve $MODEL_NAME --port 8050"

echo "Finished all evaluations!"

