#!/usr/bin/env python3
"""
H2 Reasoning Quality Scorer v2 - Strict Scoring System
Evaluates reasoning fidelity of Thinking and Instruct-CoT models

v2 improvements:
1. Score 5 is rare - only flawless reasoning gets 5
2. Score 3 is baseline - average reasoning should get 3
3. Wrong answers penalized - execution precision max 3 if answer is wrong
4. Strict scoring - any flaw results in point deduction
"""

import json
import os
import time
import argparse
from datetime import datetime
from openai import OpenAI

# ========== API Configuration ==========
API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
MODEL = os.environ.get("SCORER_MODEL", "your-model-here")

# ========== Path Configuration ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(os.path.dirname(BASE_DIR), "h2_samples")
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "evaluation", "h2_results")

# ========== Task Configuration ==========
TASKS = ["fpb", "fiqasa", "finqa", "convfinqa", "headlines"]

TASK_TYPES = {
    "fpb": {"type": "sentiment", "category": "Qualitative - Sentiment Classification"},
    "fiqasa": {"type": "sentiment", "category": "Qualitative - Sentiment Classification"},
    "finqa": {"type": "qa", "category": "Quantitative - Numerical Calculation"},
    "convfinqa": {"type": "qa", "category": "Quantitative - Multi-turn Calculation"},
    "headlines": {"type": "classification", "category": "Qualitative - Headline Classification"}
}

# ========== Strict System Prompt ==========
SYSTEM_PROMPT = """You are a strict financial NLP reasoning quality evaluator. Your scoring must be strict and discriminative.

Key principles:
1. Score 5 is extremely rare - only flawless reasoning gets 5, expected <5% of samples
2. Score 3 is baseline - average reasoning should get 3, most samples should be around 3
3. Wrong answers penalized - if final answer is wrong, execution precision max 3
4. Strict scoring - any flaw results in point deduction

Scoring dimensions:
1. Logical Coherence (1-5): Is the reasoning process smooth?
2. Factual Fidelity (1-5): Is it faithful to the source text?
3. Execution Precision (1-5): Is the final reasoning correct?

Please output scoring results in strict JSON format."""

# ========== Strict Scoring Criteria ==========
CRITERIA_TEMPLATE = """Please score strictly according to the following criteria. Note: 5 is extremely rare, 3 is baseline, most should be around 3.

### Dimension 1: Logical Coherence (1-5)
- 1: No reasoning/direct answer/incoherent
- 2: Has reasoning but logic is confused or contradictory
- 3: Logic is basically smooth but has redundancy or minor gaps (most here)
- 4: Logic is clear, steps are reasonable, no obvious redundancy
- 5: Steps are rigorous, each step has clear causality (extremely rare, <5%)

Deduction rules: Repeated arguments→max 4, irrelevant content→max 3, logical gaps→max 3

### Dimension 2: Factual Fidelity (1-5)
- 1: Fabricates numbers/events not in source
- 2: Obvious factual errors or serious misinterpretation
- 3: Basically correct but with minor deviations (most here)
- 4: Accurate citations, no factual errors
- 5: Every citation precisely traceable to source (extremely rare, <5%)

Deduction rules: Number citation error→max 2, misinterpretation→max 3, missing key info→max 3

### Dimension 3: Execution Precision (1-5)
{dimension3_criteria}

**Important rule: If final answer is wrong, execution precision max 3, even if reasoning looks reasonable.**

Please output in JSON format (only JSON, no other content):
```json
{{
  "logical_coherence": <1-5>,
  "logical_coherence_reason": "<brief reason>",
  "factual_fidelity": <1-5>,
  "factual_fidelity_reason": "<brief reason>",
  "execution_precision": <1-5>,
  "execution_precision_reason": "<brief reason>"
}}
```"""

DIMENSION3_QA = """Quantitative task criteria (max 3 if wrong):
- 1: Formula completely wrong/random calculation
- 2: Formula idea correct but major errors/missing key variables
- 3: Formula basically correct but calculation error, or answer wrong but approach correct (max if wrong)
- 4: Formula correct, calculation correct, answer correct
- 5: Formula+calculation+explanation all perfect, considers edge cases (extremely rare)"""

DIMENSION3_QUAL = """Qualitative task criteria (max 3 if wrong):
- 1: Reason is nonsense/doesn't support choice/completely off-topic
- 2: Reason is vague/cliches/misses the point
- 3: Reason is relevant but not deep enough, or answer wrong but has some merit (max if wrong)
- 4: Reason is sufficient, captures main point, answer correct
- 5: Deep insight, precise extraction of keywords, unique analysis (extremely rare)"""


def get_client():
    """Get API client"""
    return OpenAI(api_key=API_KEY, base_url=API_BASE)


def load_samples(task_name):
    """Load sample data"""
    file_path = os.path.join(SAMPLES_DIR, f"{task_name}_samples_v6.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", data)


def build_user_prompt(sample, model_type, task_name):
    """Build user prompt for scoring (strict version)"""
    task_info = TASK_TYPES[task_name]

    # Get model output
    if model_type == "thinking":
        model_output = sample.get("thinking_output", "")
        is_correct = sample.get("thinking_correct", False)
    else:  # instruct_v6
        model_output = sample.get("instruct_v6_output", "")
        is_correct = sample.get("instruct_v6_correct", False)

    # Get dimension 3 criteria
    if task_info["type"] == "qa":
        dimension3 = DIMENSION3_QA
    else:
        dimension3 = DIMENSION3_QUAL

    # Build complete prompt
    prompt = f"""## Task Type
{task_name.upper()} ({task_info['category']})

## Original Question (with full context)
{sample['prompt']}

## Correct Answer
{sample['gold_answer']}

## Model Output
{model_output}

## Is Model Answer Correct
{'Yes' if is_correct else 'No'}

---

{CRITERIA_TEMPLATE.format(dimension3_criteria=dimension3)}"""

    return prompt


def call_api(client, user_prompt, max_retries=3):
    """Call API for scoring"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            content = response.choices[0].message.content
            return content, None
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    [Retry {attempt+1}/{max_retries}] API error: {str(e)[:50]}")
                time.sleep(2 ** attempt)
            else:
                return None, str(e)
    return None, "Max retries exceeded"


def parse_response(response_text):
    """Parse JSON scoring result from response"""
    try:
        text = response_text.strip()

        # Remove markdown code block markers
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        scores = json.loads(text.strip())

        # Validate required fields
        required_fields = [
            "logical_coherence", "logical_coherence_reason",
            "factual_fidelity", "factual_fidelity_reason",
            "execution_precision", "execution_precision_reason"
        ]
        for field in required_fields:
            if field not in scores:
                return None, f"Missing field: {field}"

        # Validate score range
        for score_field in ["logical_coherence", "factual_fidelity", "execution_precision"]:
            if not (1 <= scores[score_field] <= 5):
                return None, f"Score out of range: {score_field}={scores[score_field]}"

        return scores, None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {str(e)}"
    except Exception as e:
        return None, f"Parse error: {str(e)}"


def load_existing_results(task_name):
    """Load existing scoring results (for resuming)"""
    result_file = os.path.join(RESULTS_DIR, f"{task_name}_scores.json")
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"task": task_name, "version": "v2_strict", "scores": [], "metadata": {}}


def save_results(task_name, results):
    """Save scoring results"""
    result_file = os.path.join(RESULTS_DIR, f"{task_name}_scores.json")
    results["metadata"]["last_updated"] = datetime.now().isoformat()
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def score_sample(client, sample, task_name, model_type):
    """Score a single sample"""
    user_prompt = build_user_prompt(sample, model_type, task_name)

    response, error = call_api(client, user_prompt)
    if error:
        return None, error

    scores, parse_error = parse_response(response)
    if parse_error:
        return None, parse_error

    return scores, None


def run_evaluation(tasks, resume=True, test_mode=False, models=None):
    """Run scoring"""
    if models is None:
        models = ["thinking", "instruct_v6"]

    client = get_client()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"H2 Reasoning Quality Scorer v2 - Strict Scoring")
    print(f"{'='*60}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Models: {', '.join(models)}")
    print(f"Output: {RESULTS_DIR}")
    print(f"{'='*60}\n")

    total_scored = 0
    total_errors = 0

    for task_name in tasks:
        print(f"\n=== {task_name.upper()} ===")

        samples = load_samples(task_name)
        print(f"Loaded samples: {len(samples)}")

        results = load_existing_results(task_name) if resume else {
            "task": task_name,
            "version": "v2_strict",
            "scores": [],
            "metadata": {"created": datetime.now().isoformat()}
        }

        # Build completed index
        completed = set()
        for r in results.get("scores", []):
            key = (r["sample_id"], r.get("model"))
            completed.add(key)

        for i, sample in enumerate(samples):
            if test_mode and i >= 1:
                print("  [Test mode] Only scoring 1 sample")
                break

            sample_id = sample.get("sample_id", i)

            for model_type in models:
                key = (sample_id, model_type)
                if key in completed:
                    print(f"  [{i+1}/{len(samples)}] {model_type}: Already done, skipping")
                    continue

                print(f"  [{i+1}/{len(samples)}] {model_type}: Scoring...", end=" ")

                scores, error = score_sample(client, sample, task_name, model_type)

                if error:
                    print(f"Error: {error[:30]}")
                    total_errors += 1
                    results["scores"].append({
                        "sample_id": sample_id,
                        "model": model_type,
                        "gold_answer": sample.get("gold_answer"),
                        "is_correct": sample.get(f"{model_type}_correct", sample.get("thinking_correct") if model_type == "thinking" else sample.get("instruct_v6_correct")),
                        "error": error
                    })
                else:
                    total_scored += 1
                    lc = scores["logical_coherence"]
                    ff = scores["factual_fidelity"]
                    ep = scores["execution_precision"]
                    print(f"LC={lc} FF={ff} EP={ep}")

                    results["scores"].append({
                        "sample_id": sample_id,
                        "model": model_type,
                        "gold_answer": sample.get("gold_answer"),
                        "is_correct": sample.get(f"{model_type}_correct", sample.get("thinking_correct") if model_type == "thinking" else sample.get("instruct_v6_correct")),
                        "scores": scores
                    })

                save_results(task_name, results)
                time.sleep(0.5)

        task_scores = [r for r in results["scores"] if "scores" in r]
        print(f"\n  {task_name} complete: {len(task_scores)} scored")

    print(f"\n{'='*60}")
    print(f"Scoring complete!")
    print(f"Success: {total_scored}, Errors: {total_errors}")
    print(f"{'='*60}")

    generate_summary(tasks)


def generate_summary(tasks):
    """Generate summary report"""
    print("\nGenerating summary report...")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "version": "v2_strict",
        "tasks": {},
        "models": {}
    }

    for task_name in tasks:
        result_file = os.path.join(RESULTS_DIR, f"{task_name}_scores.json")
        if not os.path.exists(result_file):
            continue

        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        task_scores = [r for r in data.get("scores", []) if "scores" in r]

        for model in ["thinking", "instruct_v6"]:
            model_scores = [r for r in task_scores if r.get("model") == model]
            if not model_scores:
                continue

            avg_lc = sum(r["scores"]["logical_coherence"] for r in model_scores) / len(model_scores)
            avg_ff = sum(r["scores"]["factual_fidelity"] for r in model_scores) / len(model_scores)
            avg_ep = sum(r["scores"]["execution_precision"] for r in model_scores) / len(model_scores)

            key = f"{task_name}_{model}"
            summary["tasks"][key] = {
                "task": task_name,
                "model": model,
                "count": len(model_scores),
                "avg_logical_coherence": round(avg_lc, 2),
                "avg_factual_fidelity": round(avg_ff, 2),
                "avg_execution_precision": round(avg_ep, 2),
                "avg_total": round((avg_lc + avg_ff + avg_ep) / 3, 2)
            }

            if model not in summary["models"]:
                summary["models"][model] = {
                    "total_samples": 0,
                    "sum_lc": 0, "sum_ff": 0, "sum_ep": 0
                }
            summary["models"][model]["total_samples"] += len(model_scores)
            summary["models"][model]["sum_lc"] += sum(r["scores"]["logical_coherence"] for r in model_scores)
            summary["models"][model]["sum_ff"] += sum(r["scores"]["factual_fidelity"] for r in model_scores)
            summary["models"][model]["sum_ep"] += sum(r["scores"]["execution_precision"] for r in model_scores)

    for model, data in summary["models"].items():
        n = data["total_samples"]
        if n > 0:
            data["avg_logical_coherence"] = round(data["sum_lc"] / n, 2)
            data["avg_factual_fidelity"] = round(data["sum_ff"] / n, 2)
            data["avg_execution_precision"] = round(data["sum_ep"] / n, 2)
            data["avg_total"] = round((data["sum_lc"] + data["sum_ff"] + data["sum_ep"]) / (3 * n), 2)
        del data["sum_lc"], data["sum_ff"], data["sum_ep"]

    summary_file = os.path.join(RESULTS_DIR, "all_scores_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Summary saved: {summary_file}")

    # Print comparison
    print("\n=== Thinking vs Instruct-CoT Comparison (v2 strict) ===")
    print(f"{'Dimension':<25} {'Thinking':>10} {'Instruct':>10} {'Diff':>10}")
    print("-" * 55)

    if "thinking" in summary["models"] and "instruct_v6" in summary["models"]:
        t = summary["models"]["thinking"]
        i = summary["models"]["instruct_v6"]

        for dim, name in [("avg_logical_coherence", "Logical Coherence"),
                          ("avg_factual_fidelity", "Factual Fidelity"),
                          ("avg_execution_precision", "Execution Precision"),
                          ("avg_total", "Total Average")]:
            diff = t.get(dim, 0) - i.get(dim, 0)
            sign = "+" if diff > 0 else ""
            print(f"{name:<25} {t.get(dim, 'N/A'):>10} {i.get(dim, 'N/A'):>10} {sign}{diff:>9.2f}")


def main():
    parser = argparse.ArgumentParser(description="H2 Reasoning Quality Scorer v2 - Strict Scoring")
    parser.add_argument("--tasks", default="all", help="Task list (comma-separated) or 'all'")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start from scratch")
    parser.add_argument("--test", action="store_true", help="Test mode (only score 1 sample per task)")
    parser.add_argument("--model", choices=["thinking", "instruct_v6", "all"], default="all",
                        help="Only evaluate specified model")
    parser.add_argument("--summary-only", action="store_true", help="Only generate summary report")

    args = parser.parse_args()

    if args.tasks == "all":
        tasks = TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    if args.model == "all":
        models = None
    else:
        models = [args.model]

    if args.summary_only:
        generate_summary(tasks)
    else:
        run_evaluation(tasks, resume=args.resume, test_mode=args.test, models=models)


if __name__ == "__main__":
    main()
