# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen-self-play"

# Qwen2.5-Math-1.5B-Instruct
export CUDA_VISIBLE_DEVICES="0"
MODEL_NAME_OR_PATH="/root/work/projects/arc-alphazero/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_DIR="/root/work/projects/arc-alphazero/SPAG/data/eval/DeepSeek-R1-Distill-Qwen-1.5B"
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR
