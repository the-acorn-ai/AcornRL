MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
CUDA_VISIBLE_DEVICES=0 python -m lcb_runner.runner.main \
    --model $MODEL_NAME \
    --scenario codegeneration \
    --evaluate \
    --release_version v4_v5 \
    --tensor_parallel_size 1 \
    --num_process_evaluate 32 \
    --max_tokens 32768