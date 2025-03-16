MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

mkdir -p output

vllm serve $MODEL_NAME --port 8030 > vllm.log 2>&1 &
     
# Wait for vLLM server to start up
echo "Waiting for vLLM server to start up..."
while ! curl -s "http://127.0.0.1:8030/v1/models" > /dev/null; do
    sleep 2
    echo "Still waiting for vLLM server..."
done
echo "vLLM server is up and running!"

python ./generate_api_answers/infer_multithread.py \
    --input_file "./data/ifeval.jsonl" \
    --output_file "./output/ifeval_bz1.jsonl" \
    --base_url "http://127.0.0.1:8030/v1" \
    --model_name $MODEL_NAME \
    --n_samples 1

pkill -f "vllm serve $MODEL_NAME --port 8030"
