MODEL_NAME=$1
BASE_URL=$2

# Check if BASE_URL is provided and running
if [ -z "$BASE_URL" ]; then
    # If BASE_URL is not provided, we'll start our own vLLM server
    PORT_NUMBER=8050
    echo "No BASE_URL provided, will start vLLM server on port $PORT_NUMBER"
    vllm serve $MODEL_NAME --port $PORT_NUMBER --tensor-parallel-size 4 > vllm.log 2>&1 &
    BASE_URL="http://127.0.0.1:$PORT_NUMBER/v1"

    while ! curl -s "$BASE_URL/models" > /dev/null; do
        sleep 2
        echo "Still waiting for vLLM server..."
    done
    echo "vLLM server is up and running!"
else
    # Check if the provided BASE_URL is accessible
    echo "Checking if provided BASE_URL $BASE_URL is accessible..."
    if ! curl -s "$BASE_URL/models" > /dev/null; then
        echo "Error: Cannot connect to $BASE_URL. Please check if the server is running."
        exit 1
    fi
    echo "BASE_URL is accessible, using external vLLM server."
    # We won't start our own server in this case
fi

mkdir -p output
python ./generate_api_answers/infer_multithread.py \
    --input_file "./data/aime24.jsonl" \
    --output_file "./output/aime24_bz64.jsonl" \
    --base_url $BASE_URL \
    --model_name $MODEL_NAME \
    --n_samples 64

python ./generate_api_answers/infer_multithread.py \
    --input_file "./data/aime25.jsonl" \
    --output_file "./output/aime25_bz64.jsonl" \
    --base_url $BASE_URL \
    --model_name $MODEL_NAME \
    --n_samples 64

python ./generate_api_answers/infer_multithread.py \
    --input_file "./data/livecodebench_v5.jsonl" \
    --output_file "./output/livecodebench_v5_bz1.jsonl" \
    --base_url $BASE_URL \
    --model_name $MODEL_NAME \
    --n_samples 1

python ./generate_api_answers/infer_multithread.py \
    --input_file "./data/ifeval.jsonl" \
    --output_file "./output/ifeval_bz1.jsonl" \
    --base_url $BASE_URL \
    --model_name $MODEL_NAME \
    --n_samples 1

mkdir -p eval_res

python ./eval/eval.py \
    --input_path ./output/aime24_bz64.jsonl \
    --cache_path ./eval_res/aime24_bz64.jsonl \
    --task_name "math_opensource/aime24" \
    --consensus \
    > ./eval_res/aime24_bz64_res_result.txt

python ./eval/eval.py \
    --input_path ./output/aime25_bz64.jsonl \
    --cache_path ./eval_res/aime25_bz64.jsonl \
    --task_name "math_opensource/aime25" \
    --consensus \
    > ./eval_res/aime25_bz64_res_result.txt

python ./data/process_data.py

python  ./eval/eval.py \
    --input_path ./output/livecodebench_v5_bz1.jsonl \
    --cache_path ./eval_res/livecodebench_v5_bz1.jsonl \
    --task_name "livecodebench" > ./eval_res/livecodebench_v5_bz1_res_result.txt

python  ./eval/eval.py \
    --input_path ./output/ifeval_bz1.jsonl \
    --cache_path ./eval_res/ifeval_bz1.jsonl \
    --task_name "ifeval" > ./eval_res/ifeval_bz1_res_result.txt