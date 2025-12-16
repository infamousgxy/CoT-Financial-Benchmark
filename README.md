# CoT-Financial-Benchmark

An empirical comparison of direct and explicit reasoning paradigms in Large Language Models on financial tasks using the PIXIU benchmark.

## Overview

This repository contains the experimental data, evaluation scripts, and results from a study comparing instruction-tuned models with thinking-type (Chain-of-Thought fine-tuned) models on financial NLP tasks.

### Research Questions

**H1 (Performance Hypothesis):**
- (a) On quantitative tasks (FinQA, ConvFinQA), explicit reasoning models show significantly higher accuracy.
- (b) On qualitative tasks (sentiment analysis, news classification), generalist models perform equally well or better.

**H2 (Reasoning Fidelity Hypothesis):**
On quantitative tasks, reasoning traces from explicit reasoning models show greater logical coherence and computational accuracy compared to traces from direct reasoning models via zero-shot CoT prompting.

## Models

We compare three model configurations based on Qwen-30B:

| Model | Description |
|-------|-------------|
| **Instruct** | Qwen3-30B-A3B-Instruct-2507 (baseline, direct output) |
| **Instruct-CoT** | Qwen3-30B-A3B-Instruct-2507 with CoT prompting |
| **Thinking** | Qwen3-30B-A3B-Thinking-2507 (native CoT architecture) |

## Dataset

We use five subtasks from the [PIXIU benchmark](https://github.com/chancefocus/PIXIU):

| Task | Type | Samples | Category |
|------|------|---------|----------|
| FinQA | Numerical Reasoning | 1,147 | Quantitative |
| ConvFinQA | Multi-turn Reasoning | 1,490 | Quantitative |
| FPB | Sentiment Classification | 970 | Qualitative |
| FiQA-SA | Sentiment Analysis | 235 | Qualitative |
| Headlines | News Classification | 20,547 | Qualitative |
| **Total** | | **24,389** | |

## Repository Structure

```
CoT-Financial-Benchmark/
├── README.md
├── LICENSE
├── data/
│   ├── raw_results/              # Raw model outputs
│   │   ├── instruct/             # Qwen-30B-Instruct results
│   │   └── thinking/             # Qwen-30B-Thinking results
│   ├── instruct_cot_results/     # Instruct with CoT prompting results
│   └── parsed_results/           # Parsed results using v2 extractor
│       ├── instruct/
│       └── thinking/
├── evaluation/
│   ├── h1_results/               # H1 accuracy summaries
│   │   └── overall_summary.json
│   └── h2_results/               # H2 qualitative scoring results
│       └── *_scores.json
├── h2_samples/                   # Stratified samples for H2 evaluation
│   ├── finqa_samples_v6.json
│   ├── convfinqa_samples_v6.json
│   ├── fpb_samples_v6.json
│   ├── fiqasa_samples_v6.json
│   ├── headlines_samples_v6.json
│   └── sampling_summary_v6.json
├── scripts/
│   ├── custom_eval_v5.py         # Main evaluation script
│   ├── improved_extractor_v2.py  # Answer extraction with dual-parser
│   ├── h2_stratified_sampling.py # Stratified sampling for H2
│   └── h2_scorer.py              # H2 qualitative scoring script
├── figures/
│   ├── h1_results.pdf            # H1 performance comparison chart
│   └── h2_results.pdf            # H2 reasoning quality radar chart
└── paper/
    ├── JC3007_Project_Report.tex # LaTeX source
    └── myref.bib                 # Bibliography
```

## Key Results

### H1: Performance Comparison

| Task | Instruct | Instruct-CoT | Thinking |
|------|----------|--------------|----------|
| FinQA | 56.58% | **71.58%** | 53.88% |
| ConvFinQA | 70.40% | **75.64%** | 66.91% |
| FPB | **79.07%** | 72.27% | 72.68% |
| FiQA-SA | 75.32% | 79.57% | **82.55%** |
| Headlines | **71.46%** | 70.96% | 69.44% |

### H2: Reasoning Quality (Scale 1-5)

| Dimension | Instruct-CoT | Thinking |
|-----------|--------------|----------|
| Logical Coherence | **3.96** | 3.28 |
| Factual Fidelity | **3.67** | 3.55 |
| Execution Precision | 3.86 | **4.07** |
| **Overall** | **3.83** | 3.65 |

## Key Findings

1. **CoT improves quantitative tasks**: Chain-of-Thought reasoning significantly improves performance on numerical reasoning tasks (FinQA +15%, ConvFinQA +5.24%).

2. **Over-reasoning hurts simple tasks**: On simple classification tasks (FPB), CoT introduces unnecessary complexity, reducing accuracy from 79% to 72%.

3. **Thinking model format issues**: The native Thinking model underperforms due to output format failures (71.5% of FinQA errors from missing `</think>` tags), not reasoning capability deficits.

4. **Financial professionals prefer clarity**: Despite higher accuracy, Thinking model scored lower on reasoning quality due to verbose, redundant outputs lacking clarity.

## Dual-Parser Strategy

We implemented a dual-parser approach for answer extraction:

- **Parser 1 (Strict)**: Only extracts answers matching exact format patterns (measures production usability)
- **Parser 2 (Heuristic)**: Uses flexible matching to extract correct answers from verbose outputs (measures true reasoning capability)

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@article{guo2025cot,
  title={An Empirical Comparison of Direct and Explicit Reasoning Paradigms in Large Language Models on Financial Tasks},
  author={Guo, XinYi},
  journal={JC3007 Project Report, University of Aberdeen},
  year={2025}
}
```

## References

- [PIXIU Benchmark](https://github.com/chancefocus/PIXIU) - Xie et al., NeurIPS 2023
- [FinQA](https://github.com/czyssrs/FinQA) - Chen et al., EMNLP 2021
- [ConvFinQA](https://github.com/czyssrs/ConvFinQA) - Chen et al., EMNLP 2022
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) - Wei et al., NeurIPS 2022

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
