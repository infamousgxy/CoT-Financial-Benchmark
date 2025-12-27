#!/usr/bin/env python3
"""
Custom Evaluation Script v5 - Final Version
Improvements:
1. Fix trailing punctuation bug (60.94. -> 60.94)
2. Auto-save results after each task completes
3. Real-time saving of evaluation details
4. Completed results not lost on interruption
"""

import json
import time
import re
import os
import requests
from datetime import datetime
from datasets import load_dataset
import argparse

# API Configuration
VLLM_API_URL = "http://localhost:9000/v1/chat/completions"
MODEL_NAME = "/share/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe/"

# Task Configuration
TASK_CONFIGS = {
    "fpb": {"dataset": "TheFinAI/flare-fpb", "task_type": "sentiment"},
    "fiqasa": {"dataset": "TheFinAI/flare-fiqasa", "task_type": "sentiment"},
    "finqa": {"dataset": "TheFinAI/flare-finqa", "task_type": "qa"},
    "convfinqa": {"dataset": "TheFinAI/flare-convfinqa", "task_type": "qa"},
    "headlines": {"dataset": "TheFinAI/flare-headlines", "task_type": "classification"}
}

# Format instruction for numerical tasks
QA_FORMAT_INSTRUCTION = """

IMPORTANT: Your final answer must be a single number only.
- If the answer is a percentage like 10.5%, output the decimal form: 0.105
- If the answer is a dollar amount like $500, output just: 500
- Do NOT include units, symbols, or explanations in your final answer
- Format: Answer: [number only]"""

def call_vllm_api(prompt, max_tokens=2048, temperature=0.0):
    """Call vLLM API"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    start_time = time.time()
    try:
        response = requests.post(VLLM_API_URL, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        result = response.json()
        latency_ms = (time.time() - start_time) * 1000
        content = result["choices"][0]["message"]["content"]
        return content, latency_ms
    except Exception as e:
        return f"ERROR: {str(e)}", (time.time() - start_time) * 1000

def extract_answer_v5(raw_output, task_type, gold=None):
    """[v5] Extract answer from model output - fix trailing punctuation"""

    # [Key fix] Prioritize extracting content after </think>
    think_end_match = re.search(r"</think>\s*(.+)", raw_output, re.DOTALL)
    if think_end_match:
        # Get content after </think> as main text
        text = think_end_match.group(1).strip()
    else:
        # If no </think> tag, use original output
        text = raw_output

    # Remove residual think tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    # Remove markdown formatting
    text = re.sub(r"\*+", "", text)

    # 1. Try to match "Answer: xxx" format
    patterns = [
        r"Answer:\s*(-?[\d,.]+%?)",
        r"Answer:\s*(.+?)(?:\n|$)",
        r"answer:\s*(.+?)(?:\n|$)",
        r"The answer is:?\s*(.+?)(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r"^\[|\]$", "", answer).strip()
            answer = answer.replace("$", "").replace(",", "").strip()
            # [v5 fix] Remove trailing punctuation
            answer = answer.rstrip(".,!?;:")
            if answer:
                return answer

    # 2. Sentiment classification task
    if task_type == "sentiment":
        labels = ["positive", "negative", "neutral"]
        last_pos, last_label = -1, None
        for label in labels:
            for m in re.finditer(r"\b" + label + r"\b", text, re.IGNORECASE):
                if m.start() > last_pos:
                    last_pos, last_label = m.start(), label
        if last_label:
            return last_label

    # 3. Yes/No classification task
    if task_type == "classification":
        matches = list(re.finditer(r"\b(yes|no)\b", text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).capitalize()

    # 4. Numerical QA task
    if task_type == "qa":
        lines = text.strip().split("\n")
        for line in reversed(lines[-5:]):
            line_clean = re.sub(r"\*+", "", line).replace("$", "").replace(",", "")
            numbers = re.findall(r"-?[\d]+\.?\d*", line_clean)
            if numbers:
                valid_numbers = [n for n in numbers if not (1900 <= abs(float(n)) <= 2100 and "." not in n)]
                result = valid_numbers[-1] if valid_numbers else numbers[-1]
                # [v5 fix] Remove trailing punctuation
                return result.rstrip(".,")

        all_numbers = re.findall(r"-?[\d]+\.?\d*", text.replace(",", ""))
        if all_numbers:
            return all_numbers[-1].rstrip(".,")

    return text.strip().split("\n")[-1] if text.strip() else ""

def evaluate_answer_v5(extracted, gold, task_type):
    """Evaluate if answer is correct"""
    extracted = str(extracted).lower().strip()
    gold = str(gold).lower().strip()

    if task_type in ["sentiment", "classification"]:
        return extracted == gold

    if task_type == "qa":
        try:
            ext_clean = re.sub(r"[^\d.\-]", "", extracted)
            gold_clean = re.sub(r"[^\d.\-]", "", gold)

            if not ext_clean or not gold_clean:
                return extracted == gold

            ext_num = float(ext_clean)
            gold_num = float(gold_clean)

            if abs(ext_num - gold_num) < 0.001:
                return True

            if abs(gold_num) < 1 and abs(ext_num) > 1:
                if abs(ext_num / 100 - gold_num) < 0.01:
                    return True

            if abs(gold_num) > 1 and abs(ext_num) < 1:
                if abs(ext_num * 100 - gold_num) < 0.01:
                    return True

            if gold_num != 0 and abs((ext_num - gold_num) / gold_num) < 0.01:
                return True

        except Exception:
            pass

    return extracted == gold

def save_task_result(output_dir, task_name, task_results, summary):
    """[v5 new] Save single task results"""
    task_file = os.path.join(output_dir, f"{task_name}_results.json")
    task_data = {
        "task": task_name,
        "model": MODEL_NAME,
        "version": "v5",
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "details": task_results
    }
    with open(task_file, "w") as f:
        json.dump(task_data, f, indent=2, ensure_ascii=False)
    print(f"  [Saved] {task_name} results saved to {task_file}")

def save_sample_detail(output_dir, task_name, sample_id, detail):
    """[v5 new] Save single evaluation detail"""
    detail_dir = os.path.join(output_dir, "details", task_name)
    os.makedirs(detail_dir, exist_ok=True)
    detail_file = os.path.join(detail_dir, f"sample_{sample_id:04d}.json")
    with open(detail_file, "w") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)

def run_evaluation(tasks, limit=50, output_dir=None):
    """Run evaluation - v5 version supports incremental saving"""

    # Create output directory
    if output_dir is None:
        output_dir = f"./results/v5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "details"), exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    all_summary = {}

    for task_name in tasks:
        config = TASK_CONFIGS[task_name]
        print(f"\n=== Evaluating {task_name} ===")

        ds = load_dataset(config["dataset"], split="test")
        samples = list(ds)[:limit]

        correct, total = 0, 0
        task_results = []

        for i, sample in enumerate(samples):
            base_prompt = sample.get("query", sample.get("text", ""))

            if config["task_type"] == "qa":
                prompt = base_prompt + QA_FORMAT_INSTRUCTION
            else:
                prompt = base_prompt

            gold = sample.get("answer", "")

            raw_output, latency = call_vllm_api(prompt)
            extracted = extract_answer_v5(raw_output, config["task_type"], gold)

            is_correct = evaluate_answer_v5(extracted, gold, config["task_type"])
            if is_correct:
                correct += 1
            total += 1

            # Build detail record
            detail = {
                "task": task_name,
                "sample_id": i,
                "input": {
                    "prompt": prompt,
                    "original_query": base_prompt[:500] if len(base_prompt) > 500 else base_prompt
                },
                "output": {
                    "raw": raw_output,
                    "extracted": extracted
                },
                "evaluation": {
                    "gold": gold,
                    "correct": is_correct
                },
                "metadata": {
                    "latency_ms": latency,
                    "timestamp": datetime.now().isoformat()
                }
            }

            task_results.append(detail)

            # [v5 new] Real-time save each detail
            save_sample_detail(output_dir, task_name, i, detail)

            status = "+" if is_correct else "-"
            print(f"  [{i+1}/{limit}] Gold: {str(gold)[:15]:15} | Ext: {str(extracted)[:20]:20} | {status}")

        # Calculate task accuracy
        accuracy = correct / total * 100 if total > 0 else 0
        task_summary = {"correct": correct, "total": total, "accuracy": accuracy}
        all_summary[task_name] = task_summary

        print(f"  {task_name} Accuracy: {correct}/{total} = {accuracy:.1f}%")

        # [v5 new] Save immediately after each task completes
        save_task_result(output_dir, task_name, task_results, task_summary)

    # Calculate overall accuracy
    total_correct = sum(s["correct"] for s in all_summary.values())
    total_samples = sum(s["total"] for s in all_summary.values())
    overall_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
    all_summary["overall"] = {"correct": total_correct, "total": total_samples, "accuracy": overall_acc}

    print(f"\n=== Overall Accuracy: {total_correct}/{total_samples} = {overall_acc:.1f}% ===")

    # Save overall summary
    summary_file = os.path.join(output_dir, "summary.json")
    summary_data = {
        "model": MODEL_NAME,
        "version": "v5",
        "timestamp": datetime.now().isoformat(),
        "tasks": list(tasks),
        "limit_per_task": limit,
        "summary": all_summary
    }
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\nOverall summary saved to {summary_file}")

    return all_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="fpb,fiqasa,finqa,convfinqa,headlines")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output_dir", default=None, help="Output directory, auto-generated by default")
    args = parser.parse_args()

    tasks = args.tasks.split(",")
    run_evaluation(tasks, args.limit, args.output_dir)
