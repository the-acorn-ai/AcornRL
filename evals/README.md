# AcornRL Evals
We evaluate the performance of our models in the following three categories:
1. In-domain game performance (winrate vs untrained baseline)
2. Out-of-domain game performance (winrate vs SFT baseline)
3. OOD generalization (reasoning and code benchmark)
   
## Efficiency
- [ ] Unify the interface as vllm generate and data parallelism
- [ ] Speed up with long context parallelism (context parallelism)

## In-domain Game Performance
TODO
## Out-of-domain Game Performance
TODO
- [ ] [Mastermind](https://openreview.net/forum?id=H4donosutm)

## OOD Generalization
The following benchmarks covered in Deepseek R1 paper are used:
## English Language Benchmarks
- [X] **MMLU** (Pass@1) - Massive Multitask Language Understanding
- [X] **MMLU-Pro** (EM) - Professional/advanced version with Exact Match evaluation
- [X] **DROP** (3-shot F1) - Discrete Reasoning Over Paragraphs
- [X] **IF-Eval** (Prompt Strict) - Instruction Following Evaluation
- [X] **GPQA Diamond** (Pass@1) - Graduate-level Professional Question Answering
- [X] **SimpleQA** (Correct) - Basic question answering benchmark
- [ ] **FRAMES** (Acc.) - Measures accuracy on frame-based reasoning
- [ ] **AlpacaEval2.0** (LC-winrate) - Evaluates using length-controlled win rate
- [ ] **ArenaHard** (GPT-4-1106) - Challenging evaluation against GPT-4

## Code Benchmarks (Code generation✅, TODO: Self-repair)
- [X] **LiveCodeBench** (Pass@1-COT) - Real-time code generation evaluation
- [ ] **Codeforces** (Percentile) - Competitive programming benchmark (percentile)
- [ ] **Codeforces** (Rating) - Competitive programming benchmark (rating)
- [ ] **SWE Verified** (Resolved) - Software Engineering verification

## Math Benchmark (pass@1 and con@64)
- [X] **AIME 2024** (Pass@1) - American Invitational Mathematics Examination
- [X] **MATH-500** (Pass@1) - Collection of 500 challenging math problems
- [X] **AMC 2023** (Pass@1) - American Mathematics Competitions
- [X] **AIME 2025** (Pass@1) - American Invitational Mathematics Examination (New problems)
- [X] **OlympiadBench** (Pass@1) - Olympiad problems (used in [SimpleRL-Math](https://github.com/hkust-nlp/simpleRL-reason/tree/main))

## Chinese Language Benchmarks
- [ ] **CLUEWSC** (EM) - Chinese Language Understanding Evaluation
- [ ] **C-Eval** (EM) - Comprehensive Chinese evaluation
- [ ] **C-SimpleQA** (Correct) - Chinese version of SimpleQA
- [ ] **CNMO 2024** (Pass@1) - Chinese National Mathematics Olympiad

### Acknowledgements
- GPQA, MMLU, DROP, SimpleQA: [OpenAI/simple-evals](https://github.com/openai/simple-evals)
- MATH Related benchmarks: [hkust-nlp/simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)
- LiveCodeBench, IF-Eval: [QwenLM/QwQ](https://github.com/QwenLM/QwQ/tree/main/eval)
- SWE Verified: [SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench)
- Codeforces: [QwenLM/CodeElo](https://github.com/QwenLM/CodeElo)

### Run evals
```bash
bash evals/benchmarks/run.sh <model_name>
```

### Settings
We follow the Deepseek R1 paper settings for all benchmarks, where we use `temperature=0.6`, `top_p=0.95`, `max_tokens=32768`. `n_sampling=4-64` depending on the benchmark. (Math related benchmarks use `n_sampling=64`)

### Results
| Benchmark | dpsk-r1-distill-1.5b | dpsk-r1-distill-7b |
|-----------|----------------------|---------------------|
| MMLU | - | 67.42 |
| MMLU-Pro | - | - |
| DROP | - | 73.98 |
| IF-Eval | - | - |
| GPQA Diamond | 34.9 | 50.1 |
| SimpleQA | - | 1.73 |
| LiveCodeBench | - | 34.0 |
| AIME 2024 | - | - |
| MATH-500 | - | - |
| AMC 2023 | - | - |
| AIME 2025 | - | - |
| OlympiadBench | - | - |
