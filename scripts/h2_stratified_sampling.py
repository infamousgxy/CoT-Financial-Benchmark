#!/usr/bin/env python3
"""
H2 Qualitative Evaluation - Stratified Sampling Script
Extract 200 samples (40 per task) from test results for H2 qualitative analysis

Usage:
    python h2_stratified_sampling.py --task finqa --output-dir ../h2_samples
    python h2_stratified_sampling.py --all --output-dir ../h2_samples
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Fixed random seed
RANDOM_SEED = 42

# Data source paths
BASE_DIR = Path("../data")  # Adjust to your data directory
INSTRUCT_DIR = BASE_DIR / "Qwen3-30B-A3B-Instruct-2507/5tasks_full_round2_31176"
THINKING_DIR = BASE_DIR / "Qwen3-30B-A3B-Thinking-2507/5tasks_full_round2_32617"

# Sampling quota configuration
SAMPLING_CONFIG = {
    "finqa": {
        "instruct_wins": 16,
        "thinking_wins": 12,
        "both_correct": 12,
        "total": 40
    },
    "convfinqa": {
        "instruct_wins": 12,
        "thinking_wins": 16,
        "both_correct": 12,
        "total": 40
    },
    "fpb": {
        "thinking_wins": 16,
        "both_correct": 12,
        "random": 12,
        "total": 40
    },
    "fiqasa": {
        "thinking_wins": 16,
        "both_correct": 12,
        "random": 12,
        "total": 40
    },
    "headlines": {
        "thinking_wins": 16,
        "both_correct": 12,
        "random": 12,
        "total": 40
    }
}

# Analysis hints
ANALYSIS_HINTS = {
    "instruct_wins": "Check if Thinking has calculation errors, over-reasoning, or output format issues",
    "thinking_wins": "Analyze how Thinking's reasoning chain helps arrive at correct answer",
    "both_correct": "Compare reasoning process clarity and efficiency between two models",
    "random": "Random samples for coverage"
}


def load_results(task: str) -> Tuple[Dict, Dict]:
    """Load result files for both models"""
    instruct_file = INSTRUCT_DIR / f"{task}_results.json"
    thinking_file = THINKING_DIR / f"{task}_results.json"

    with open(instruct_file, 'r', encoding='utf-8') as f:
        instruct_data = json.load(f)

    with open(thinking_file, 'r', encoding='utf-8') as f:
        thinking_data = json.load(f)

    return instruct_data, thinking_data


def align_and_classify(instruct_data: Dict, thinking_data: Dict) -> Dict[str, List[Dict]]:
    """Align data by sample_id and classify"""
    # Create sample_id to data mapping
    instruct_map = {d['sample_id']: d for d in instruct_data['details']}
    thinking_map = {d['sample_id']: d for d in thinking_data['details']}

    # Classification
    classified = {
        "instruct_wins": [],
        "thinking_wins": [],
        "both_correct": [],
        "both_wrong": []
    }

    for sample_id in instruct_map:
        if sample_id not in thinking_map:
            continue

        instruct_item = instruct_map[sample_id]
        thinking_item = thinking_map[sample_id]

        instruct_correct = instruct_item['evaluation'].get('correct_v2', instruct_item['evaluation'].get('correct', False))
        thinking_correct = thinking_item['evaluation'].get('correct_v2', thinking_item['evaluation'].get('correct', False))

        # Build unified sample structure
        sample = {
            "sample_id": sample_id,
            "input": {
                "prompt": instruct_item['input'].get('prompt', ''),
                "original_query": instruct_item['input'].get('original_query', '')
            },
            "gold_answer": instruct_item['evaluation'].get('gold', ''),
            "instruct": {
                "raw_output": instruct_item['output'].get('raw', ''),
                "extracted_answer": instruct_item['output'].get('extracted_v2', instruct_item['output'].get('extracted', '')),
                "correct": instruct_correct,
                "output_length": len(instruct_item['output'].get('raw', ''))
            },
            "thinking": {
                "raw_output": thinking_item['output'].get('raw', ''),
                "extracted_answer": thinking_item['output'].get('extracted_v2', thinking_item['output'].get('extracted', '')),
                "correct": thinking_correct,
                "output_length": len(thinking_item['output'].get('raw', ''))
            }
        }

        # Classify
        if instruct_correct and not thinking_correct:
            classified["instruct_wins"].append(sample)
        elif thinking_correct and not instruct_correct:
            classified["thinking_wins"].append(sample)
        elif instruct_correct and thinking_correct:
            classified["both_correct"].append(sample)
        else:
            classified["both_wrong"].append(sample)

    return classified


def stratified_sample(classified: Dict[str, List[Dict]], config: Dict, seed: int = RANDOM_SEED) -> Dict[str, List[Dict]]:
    """Perform stratified sampling according to quotas"""
    random.seed(seed)

    sampled = {}

    for category, quota in config.items():
        if category == "total":
            continue

        if category == "random":
            # Random sampling from all samples
            all_samples = []
            for cat_samples in classified.values():
                all_samples.extend(cat_samples)

            if len(all_samples) >= quota:
                sampled["random"] = random.sample(all_samples, quota)
            else:
                sampled["random"] = all_samples
        else:
            pool = classified.get(category, [])
            if len(pool) >= quota:
                sampled[category] = random.sample(pool, quota)
            else:
                # If insufficient samples, use all
                sampled[category] = pool
                print(f"Warning: {category} pool ({len(pool)}) < quota ({quota})")

    # Add analysis hints
    for category, samples in sampled.items():
        for sample in samples:
            sample["analysis_hints"] = ANALYSIS_HINTS.get(category, "")
            sample["category"] = category

    return sampled


def save_task_samples(task: str, sampled: Dict[str, List[Dict]], config: Dict, pool_sizes: Dict, output_dir: Path):
    """Save sampling results for a single task"""
    output = {
        "task": task,
        "timestamp": datetime.now().isoformat(),
        "sampling_config": config,
        "statistics": {
            "pool_size": pool_sizes,
            "actual_sampled": {k: len(v) for k, v in sampled.items()}
        },
        "samples": sampled
    }

    output_file = output_dir / f"{task}_samples.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {task} samples to {output_file}")
    return output


def process_task(task: str, output_dir: Path) -> Dict:
    """Process sampling for a single task"""
    print(f"\n{'='*50}")
    print(f"Processing {task}...")
    print(f"{'='*50}")

    # Load data
    instruct_data, thinking_data = load_results(task)
    print(f"Loaded {len(instruct_data['details'])} samples")

    # Align and classify
    classified = align_and_classify(instruct_data, thinking_data)
    pool_sizes = {k: len(v) for k, v in classified.items()}
    print(f"Classification: {pool_sizes}")

    # Sample
    config = SAMPLING_CONFIG[task]
    sampled = stratified_sample(classified, config)

    # Statistics
    actual_sampled = {k: len(v) for k, v in sampled.items()}
    print(f"Sampled: {actual_sampled}")

    # Save
    result = save_task_samples(task, sampled, config, pool_sizes, output_dir)

    return result


def generate_summary(results: List[Dict], output_dir: Path):
    """Generate summary statistics"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": sum(r["statistics"]["actual_sampled"].get("instruct_wins", 0) +
                           r["statistics"]["actual_sampled"].get("thinking_wins", 0) +
                           r["statistics"]["actual_sampled"].get("both_correct", 0) +
                           r["statistics"]["actual_sampled"].get("random", 0)
                           for r in results),
        "tasks": {}
    }

    for result in results:
        task = result["task"]
        summary["tasks"][task] = {
            "pool_size": result["statistics"]["pool_size"],
            "actual_sampled": result["statistics"]["actual_sampled"],
            "sampling_config": result["sampling_config"]
        }

    output_file = output_dir / "sampling_summary.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary to {output_file}")
    return summary


def generate_readme(summary: Dict, output_dir: Path):
    """Generate README documentation"""
    readme = f"""# H2 Qualitative Evaluation - Stratified Sampling Results

## Sampling Overview

- **Generated at**: {summary['timestamp']}
- **Total samples**: {summary['total_samples']}
- **Sampling method**: Stratified Purposive Sampling
- **Random seed**: {RANDOM_SEED}

## Sampling Quotas

| Task | Instruct Wins | Thinking Wins | Both Correct | Random | Total |
|------|---------------|---------------|--------------|--------|-------|
"""

    for task, data in summary["tasks"].items():
        sampled = data["actual_sampled"]
        readme += f"| {task} | {sampled.get('instruct_wins', '-')} | {sampled.get('thinking_wins', '-')} | {sampled.get('both_correct', '-')} | {sampled.get('random', '-')} | {sum(sampled.values())} |\n"

    readme += f"""
## Sample Pool Statistics

| Task | Instruct Wins | Thinking Wins | Both Correct | Both Wrong |
|------|---------------|---------------|--------------|------------|
"""

    for task, data in summary["tasks"].items():
        pool = data["pool_size"]
        readme += f"| {task} | {pool.get('instruct_wins', 0)} | {pool.get('thinking_wins', 0)} | {pool.get('both_correct', 0)} | {pool.get('both_wrong', 0)} |\n"

    readme += """
## File Description

| File | Content |
|------|---------|
| `sampling_summary.json` | Sampling summary statistics |
| `finqa_samples.json` | FinQA task samples |
| `convfinqa_samples.json` | ConvFinQA task samples |
| `fpb_samples.json` | FPB task samples |
| `fiqasa_samples.json` | FiQASA task samples |
| `headlines_samples.json` | Headlines task samples |

## Sample Structure

Each sample contains:
- `sample_id`: Sample ID (for tracing back to original data)
- `input`: Input prompt
- `gold_answer`: Ground truth answer
- `instruct`: Instruct model output and evaluation result
- `thinking`: Thinking model output and evaluation result
- `category`: Classification (instruct_wins/thinking_wins/both_correct/random)
- `analysis_hints`: Analysis hints

## Usage

```python
import json

with open('finqa_samples.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Iterate through samples where Instruct wins
for sample in data['samples']['instruct_wins']:
    print(f"Sample {sample['sample_id']}:")
    print(f"  Gold: {sample['gold_answer']}")
    print(f"  Instruct: {sample['instruct']['extracted_answer']} (correct)")
    print(f"  Thinking: {sample['thinking']['extracted_answer']} (wrong)")
```
"""

    output_file = output_dir / "README.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(readme)

    print(f"Saved README to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='H2 Qualitative Evaluation Stratified Sampling')
    parser.add_argument('--task', type=str, choices=['finqa', 'convfinqa', 'fpb', 'fiqasa', 'headlines'],
                        help='Task name')
    parser.add_argument('--all', action='store_true', help='Process all tasks')
    parser.add_argument('--output-dir', type=str, default='../h2_samples',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help='Random seed')

    args = parser.parse_args()

    seed = args.seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        tasks = ['finqa', 'convfinqa', 'fpb', 'fiqasa', 'headlines']
    elif args.task:
        tasks = [args.task]
    else:
        print("Please specify --task or --all")
        return

    results = []
    for task in tasks:
        result = process_task(task, output_dir)
        results.append(result)

    if len(results) == 5:
        summary = generate_summary(results, output_dir)
        generate_readme(summary, output_dir)

    print(f"\n{'='*50}")
    print("Done!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
