# TODOS:

### Training
- add wandb tracker
- add "watch" tracker
- add non-lora training back in

### Eval
- Eval Benchmarks Implementation (Check [evals](evals/README.md)) For now just vs 4o-mini

### Data Collection
- make it easy to load data from previous N data collection rounds
- reduce the episode_to_details_csv to just one row per episode maybe?
- add force-answer function if cot is too long (and there is a max-seq argument)

### Misc
- Add pyproject.toml