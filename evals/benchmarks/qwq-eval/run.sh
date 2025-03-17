MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

mkdir -p output

export CUDA_VISIBLE_DEVICES="0,1,2,3"
vllm serve $MODEL_NAME --port 8030 --tensor-parallel-size 4 > vllm.log 2>&1 &
     
# # Wait for vLLM server to start up
echo "Waiting for vLLM server to start up..."
while ! curl -s "http://127.0.0.1:8030/v1/models" > /dev/null; do
    sleep 2
    echo "Still waiting for vLLM server..."
done
echo "vLLM server is up and running!"

mkdir -p output
python ./generate_api_answers/infer_multithread.py \
    --input_file "./data/aime24.jsonl" \
    --output_file "./output/aime24_bz1.jsonl" \
    --base_url "http://127.0.0.1:8030/v1" \
    --model_name $MODEL_NAME \
    --n_samples 1

python ./generate_api_answers/infer_multithread.py \
    --input_file "./data/aime25.jsonl" \
    --output_file "./output/aime25_bz1.jsonl" \
    --base_url "http://127.0.0.1:8030/v1" \
    --model_name $MODEL_NAME \
    --n_samples 1

python ./generate_api_answers/infer_multithread.py \
    --input_file "./data/livecodebench_v5.jsonl" \
    --output_file "./output/livecodebench_v5_bz1.jsonl" \
    --base_url "http://127.0.0.1:8030/v1" \
    --model_name $MODEL_NAME \
    --n_samples 1

# pkill -f "vllm serve $MODEL_NAME --port 8030"

mkdir -p eval_res

python ./eval/eval.py \
    --input_path ./output/aime24_bz1.jsonl \
    --cache_path ./eval_res/aime24_bz1.jsonl \
    --task_name "math_opensource/aime24" > ./eval_res/aime24_bz1_res_result.txt

python ./eval/eval.py \
    --input_path ./output/aime25_bz1.jsonl \
    --cache_path ./eval_res/aime25_bz1.jsonl \
    --task_name "math_opensource/aime25" > ./eval_res/aime25_bz1_res_result.txt

python ./data/process_data.py

python  ./eval/eval.py \
    --input_path ./output/livecodebench_v5_bz1.jsonl \
    --cache_path ./eval_res/livecodebench_v5_bz1.jsonl \
    --task_name "livecodebench" > ./eval_res/livecodebench_v5_bz1_res_result.txt