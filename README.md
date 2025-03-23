# TODOS:

### Training
- add wandb tracker
- add "watch" tracker
- add non-lora training back in

### Data Collection
- make it easy to load data from previous N data collection rounds
- reduce the episode_to_details_csv to just one row per episode maybe?
- add force-answer function if cot is too long (and there is a max-seq argument)

### Misc
- Add pyproject.toml

---

### Eval
- Eval Benchmarks Implementation (Check [evals](evals/README.md) for benchmarks and guidelines)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
 --host 0.0.0.0 --dp 4 --port 8030
```

```bash

```