MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# Main GPQA tasks
# lm_eval --model vllm \
#   --model_args pretrained=$MODEL_NAME,trust_remote_code=true,data_parallel_size=4 \
#   --tasks gpqa_main_zeroshot,gpqa_main_n_shot,gpqa_main_generative_n_shot,gpqa_main_cot_zeroshot,gpqa_main_cot_n_shot \
#   --device cuda \
#   --output_path ./results/gpqa_main_${MODEL_NAME//\//_}.json

# Diamond GPQA tasks
CUDA_VISIBLE_DEVICES=1 lm_eval --model vllm \
  --model_args pretrained=$MODEL_NAME,trust_remote_code=true \
  --tasks gpqa_diamond_generative_n_shot \
  --apply_chat_template \
  --max_tokens 8192 \
  --output_path ./results/gpqa_diamond_${MODEL_NAME//\//_}.json

# CUDA_VISIBLE_DEVICES=1,2 lm_eval --model vllm \
#   --model_args pretrained=$MODEL_NAME,trust_remote_code=true,data_parallel_size=2 \
#   --tasks gpqa_diamond_generative_n_shot,gpqa_diamond_cot_zeroshot,gpqa_diamond_cot_n_shot \
#   --apply_chat_template \
#   --output_path ./results/gpqa_diamond_${MODEL_NAME//\//_}.json