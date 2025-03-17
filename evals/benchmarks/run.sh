MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

export CUDA_VISIBLE_DEVICES="4,5,6,7"
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
vllm serve $MODEL_NAME --port 8030 --tensor-parallel-size 4 > vllm.log 2>&1 &

# Wait for vLLM server to start up
echo "Waiting for vLLM server to start up..."
while ! curl -s "http://127.0.0.1:8030/v1/models" > /dev/null; do
    sleep 2
    echo "Still waiting for vLLM server..."
done
echo "vLLM server is up and running!"

python -m simple-evals.simple_evals \
    --model_name_or_path $MODEL_NAME \
    --base_url http://127.0.0.1:8030/v1 \
    --max_tokens 32768 \
    --tasks gpqa mmlu drop simpleqa

pkill -f "vllm serve $MODEL_NAME --port 8030"

