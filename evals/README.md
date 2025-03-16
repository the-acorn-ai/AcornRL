# AcornRL Evals
We evaluate the performance of our models in the following three categories:
1. In-domain game performance (winrate vs untrained baseline)
2. Out-of-domain game performance (winrate vs SFT baseline)
3. OOD generalization (reasoning and code benchmark)
   
## In-domain Game Performance
TODO
## Out-of-domain Game Performance
TODO
- [ ] [Mastermind](https://openreview.net/forum?id=H4donosutm)

## OOD Generalization
The following benchmarks covered in Deepseek R1 paper are used:
## English Language Benchmarks
- [ ] **MMLU** (Pass@1) - Massive Multitask Language Understanding
- [ ] **MMLU-Redux** (EM) - Enhanced version with Exact Match evaluation
- [ ] **MMLU-Pro** (EM) - Professional/advanced version with Exact Match evaluation
- [ ] **DROP** (3-shot F1) - Discrete Reasoning Over Paragraphs
- [ ] **IF-Eval** (Prompt Strict) - Instruction Following Evaluation
- [ ] **GPQA Diamond** (Pass@1) - Graduate-level Professional Question Answering
- [ ] **SimpleQA** (Correct) - Basic question answering benchmark
- [ ] **FRAMES** (Acc.) - Measures accuracy on frame-based reasoning
- [ ] **AlpacaEval2.0** (LC-winrate) - Evaluates using length-controlled win rate
- [ ] **ArenaHard** (GPT-4-1106) - Challenging evaluation against GPT-4

## Code Benchmarks
- [ ] **LiveCodeBench** (Pass@1-COT) - Real-time code generation evaluation
- [ ] **Codeforces** (Percentile) - Competitive programming benchmark (percentile)
- [ ] **Codeforces** (Rating) - Competitive programming benchmark (rating)
- [ ] **SWE Verified** (Resolved) - Software Engineering verification
- [ ] **Aider-Polyglot** (Acc.) - Multi-language code assistance accuracy

## Math Benchmarks
- [ ] **AIME 2024** (Pass@1) - American Invitational Mathematics Examination
- [ ] **MATH-500** (Pass@1) - Collection of 500 challenging math problems
- [ ] **CNMO 2024** (Pass@1) - Chinese National Mathematics Olympiad
- [ ] **AIME 2025** (Pass@1) - American Invitational Mathematics Examination (New problems)
- [ ] **OlympiadBench** (Pass@1) - Olympiad problems (used in [SimpleRL-Math](https://github.com/hkust-nlp/simpleRL-reason/tree/main))

## Chinese Language Benchmarks
- [ ] **CLUEWSC** (EM) - Chinese Language Understanding Evaluation
- [ ] **C-Eval** (EM) - Comprehensive Chinese evaluation
- [ ] **C-SimpleQA** (Correct) - Chinese version of SimpleQA
