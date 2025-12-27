#!/usr/bin/env python3
"""
Improved Answer Extractor v2 - Fair Version (no gold answer usage)

Improvements:
1. Better handling of cases without </think> tag
2. Extract conclusive patterns like "answer is X", "therefore X" from reasoning
3. Filter out prompt format example numbers
4. [Important] Does not use gold answer for matching, ensuring fairness
"""

import re
import json
import os
from datetime import datetime


def extract_answer_v2(raw_output, task_type):
    """
    Improved answer extractor v2 (fair version, no gold answer usage)
    """

    text = raw_output

    # 1. Prioritize extracting content after </think>
    think_end_match = re.search(r"</think>\s*(.+)", raw_output, re.DOTALL)
    if think_end_match:
        text = think_end_match.group(1).strip()

    # Remove residual think tags and markdown formatting
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    text = re.sub(r"\*+", "", text)

    # 2. Try to match explicit answer formats
    answer_patterns = [
        r"Answer:\s*(-?[\d,.]+%?)",
        r"Answer:\s*([A-Za-z]+)",  # For classification tasks
        r"answer:\s*(-?[\d,.]+)",
        r"The answer is:?\s*(-?[\d,.]+)",
        r"the answer is:?\s*(-?[\d,.]+)",
        r"The answer is:?\s*([A-Za-z]+)",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r"^\[|\]$", "", answer).strip()
            answer = answer.replace("$", "").replace(",", "").strip()
            answer = answer.rstrip(".,!?;:")
            if answer:
                return answer

    # 3. Sentiment classification task
    if task_type == "sentiment":
        labels = ["positive", "negative", "neutral"]
        last_pos, last_label = -1, None
        for label in labels:
            for m in re.finditer(r"\b" + label + r"\b", text, re.IGNORECASE):
                if m.start() > last_pos:
                    last_pos, last_label = m.start(), label
        if last_label:
            return last_label

    # 4. Yes/No classification task
    if task_type == "classification":
        matches = list(re.finditer(r"\b(yes|no)\b", text, re.IGNORECASE))
        if matches:
            return matches[-1].group(1).capitalize()

    # 5. Numerical QA task - improved extraction logic (no gold answer usage)
    if task_type == "qa":
        # 5a. Extract from conclusive statements (most reliable)
        conclusion_patterns = [
            # "the answer is/would be/should be X"
            r"(?:the\s+)?(?:final\s+)?answer\s+(?:is|would be|should be)\s*:?\s*(-?[\d,.]+)",
            # "therefore/so/thus X"
            r"(?:therefore|so|thus|hence)[,\s]+(?:the\s+)?(?:answer\s+)?(?:is\s+|would be\s+|=\s*)?(-?[\d,.]+)",
            # "result is/= X"
            r"result\s*(?:is|=|:)\s*(-?[\d,.]+)",
            # "approximately X"
            r"(?:approximately|about)\s*(-?[\d,.]+)",
            # "= X" at end of line or sentence
            r"=\s*(-?[\d,.]+)\s*(?:\.|\n|$)",
            # "X%" conversion
            r"(\d+\.?\d*)\s*%\s*(?:would be|is|=)\s*(-?[\d,.]+)",
        ]

        for pattern in conclusion_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Take the last match (usually the final conclusion)
                if isinstance(matches[-1], tuple):
                    answer = matches[-1][-1]  # Take last element of tuple
                else:
                    answer = matches[-1]
                answer = answer.replace(",", "")
                if re.match(r'^-?[\d.]+$', answer):
                    return answer

        # 5b. Extract numbers from last few lines
        lines = text.strip().split("\n")
        # Define prompt example numbers to filter
        prompt_examples = {"10.5", "0.105", "500", "10.5%"}

        for line in reversed(lines[-10:]):
            line_clean = re.sub(r"\*+", "", line).replace("$", "").replace(",", "")

            # Skip lines containing prompt format instructions
            if "percentage like" in line.lower() or "dollar amount like" in line.lower():
                continue
            if "output the decimal form" in line.lower() or "output just" in line.lower():
                continue

            numbers = re.findall(r"-?[\d]+\.?\d*", line_clean)
            if numbers:
                # Filter out years and prompt examples
                valid_numbers = []
                for n in numbers:
                    try:
                        num_val = abs(float(n))
                        # Filter years
                        if 1900 <= num_val <= 2100 and "." not in n:
                            continue
                        # Filter prompt examples
                        if n in prompt_examples:
                            continue
                        valid_numbers.append(n)
                    except:
                        pass

                if valid_numbers:
                    return valid_numbers[-1].rstrip(".,")

        # 5c. Last resort: extract last valid number from all numbers
        all_numbers = re.findall(r"-?[\d]+\.?\d*", text.replace(",", ""))
        if all_numbers:
            # Filter years and format example numbers
            valid = []
            for n in all_numbers:
                try:
                    num_val = abs(float(n))
                    if 1900 <= num_val <= 2100 and "." not in n:
                        continue
                    if n in prompt_examples:
                        continue
                    valid.append(n)
                except:
                    pass
            if valid:
                return valid[-1].rstrip(".,")

    return text.strip().split("\n")[-1] if text.strip() else ""


def evaluate_answer(extracted, gold, task_type):
    """Evaluate if answer is correct (same as original version)"""
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

            # Exact match
            if abs(ext_num - gold_num) < 0.001:
                return True

            # Percentage conversion
            if abs(gold_num) < 1 and abs(ext_num) > 1:
                if abs(ext_num / 100 - gold_num) < 0.01:
                    return True

            if abs(gold_num) > 1 and abs(ext_num) < 1:
                if abs(ext_num * 100 - gold_num) < 0.01:
                    return True

            # Relative error < 1%
            if gold_num != 0 and abs((ext_num - gold_num) / gold_num) < 0.01:
                return True

        except Exception:
            pass

    return extracted == gold


def reeval_results_file(input_file, output_file, task_type):
    """Re-evaluate and save results using improved extractor"""
    with open(input_file, 'r') as f:
        data = json.load(f)

    original_correct = 0
    new_correct = 0
    new_details = []

    for d in data['details']:
        raw = d['output']['raw']
        gold = d['evaluation']['gold']

        # Original results
        if d['evaluation']['correct']:
            original_correct += 1

        # Use improved extractor (without gold)
        new_extracted = extract_answer_v2(raw, task_type)
        is_correct = evaluate_answer(new_extracted, gold, task_type)

        if is_correct:
            new_correct += 1

        # Update details
        new_detail = d.copy()
        new_detail['output'] = d['output'].copy()
        new_detail['output']['extracted_v2'] = new_extracted
        new_detail['output']['extracted_original'] = d['output']['extracted']
        new_detail['evaluation'] = d['evaluation'].copy()
        new_detail['evaluation']['correct_v2'] = is_correct
        new_detail['evaluation']['correct_original'] = d['evaluation']['correct']
        new_details.append(new_detail)

    # Build new results
    new_data = data.copy()
    new_data['details'] = new_details
    new_data['version'] = 'v2_improved_extractor'
    new_data['original_version'] = data.get('version', 'v5')
    new_data['timestamp'] = datetime.now().isoformat()

    # Update summary
    total = len(new_details)
    new_data['summary_v2'] = {
        'total': total,
        'correct_original': original_correct,
        'correct_v2': new_correct,
        'accuracy_original': original_correct / total * 100,
        'accuracy_v2': new_correct / total * 100,
        'improvement': new_correct - original_correct,
        'improvement_pct': (new_correct - original_correct) / total * 100
    }

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    return new_data['summary_v2']


def reeval_all_tasks(base_dir, output_base_dir):
    """Re-evaluate all tasks (including all data rounds)"""
    tasks = {
        'finqa': 'qa',
        'convfinqa': 'qa',
        'fpb': 'sentiment',
        'fiqasa': 'sentiment',
        'headlines': 'classification'
    }

    # All data round configurations
    models = [
        # Instruct model
        ('Qwen3-30B-A3B-Instruct-2507', '5tasks_5842samples'),
        ('Qwen3-30B-A3B-Instruct-2507', 'headlines_full_20547samples'),
        ('Qwen3-30B-A3B-Instruct-2507', '5tasks_full_round2_30295'),
        ('Qwen3-30B-A3B-Instruct-2507', '5tasks_full_round2_31176'),
        # Thinking model
        ('Qwen3-30B-A3B-Thinking-2507', '5tasks_5842samples'),
        ('Qwen3-30B-A3B-Thinking-2507', 'headlines_full_20547samples'),
        ('Qwen3-30B-A3B-Thinking-2507', '5tasks_full_round2_31856'),
        ('Qwen3-30B-A3B-Thinking-2507', '5tasks_full_round2_32617'),
    ]

    results = {}

    for model_name, run_name in models:
        print(f"\n=== {model_name} / {run_name} ===")
        results[f"{model_name}/{run_name}"] = {}

        for task, task_type in tasks.items():
            input_file = os.path.join(base_dir, model_name, run_name, f"{task}_results.json")
            output_file = os.path.join(output_base_dir, model_name, run_name, f"{task}_results.json")

            if os.path.exists(input_file):
                summary = reeval_results_file(input_file, output_file, task_type)
                results[f"{model_name}/{run_name}"][task] = summary
                print(f"  {task}: {summary['accuracy_original']:.2f}% -> {summary['accuracy_v2']:.2f}% ({summary['improvement']:+d})")
            else:
                print(f"  {task}: File not found")

    return results


def save_overall_summary(results, output_base_dir):
    """Save overall summary results"""
    summary_file = os.path.join(output_base_dir, "overall_summary.json")

    # Group statistics by model
    model_summaries = {}
    for key, tasks in results.items():
        model_name = key.split('/')[0]
        if model_name not in model_summaries:
            model_summaries[model_name] = {'runs': {}, 'total_original': 0, 'total_v2': 0, 'total_samples': 0}

        model_summaries[model_name]['runs'][key] = tasks
        for task, summary in tasks.items():
            model_summaries[model_name]['total_original'] += summary['correct_original']
            model_summaries[model_name]['total_v2'] += summary['correct_v2']
            model_summaries[model_name]['total_samples'] += summary['total']

    # Calculate accuracy
    for model_name, data in model_summaries.items():
        if data['total_samples'] > 0:
            data['accuracy_original'] = data['total_original'] / data['total_samples'] * 100
            data['accuracy_v2'] = data['total_v2'] / data['total_samples'] * 100
            data['improvement'] = data['total_v2'] - data['total_original']
            data['improvement_pct'] = data['improvement'] / data['total_samples'] * 100

    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'version': 'v2_improved_extractor',
            'model_summaries': model_summaries,
            'detailed_results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nOverall summary saved to: {summary_file}")
    return model_summaries


if __name__ == "__main__":
    base_dir = "./data/raw_results"  # Adjust to your input directory
    output_base_dir = "./data/parsed_results"  # Adjust to your output directory

    print("=" * 60)
    print("Re-evaluating all results with improved extractor v2")
    print("(Fair version: no gold answer used for matching)")
    print("=" * 60)

    results = reeval_all_tasks(base_dir, output_base_dir)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    # Save overall summary
    model_summaries = save_overall_summary(results, output_base_dir)

    # Print summary by model
    print("\n=== Instruct Model Summary ===")
    for key in results:
        if 'Instruct' in key:
            print(f"\n{key}:")
            for task, summary in results[key].items():
                print(f"  {task}: {summary['accuracy_original']:.2f}% -> {summary['accuracy_v2']:.2f}% ({summary['improvement']:+d})")

    print("\n=== Thinking Model Summary ===")
    for key in results:
        if 'Thinking' in key:
            print(f"\n{key}:")
            for task, summary in results[key].items():
                print(f"  {task}: {summary['accuracy_original']:.2f}% -> {summary['accuracy_v2']:.2f}% ({summary['improvement']:+d})")

    # Print overall model comparison
    print("\n" + "=" * 60)
    print("Overall Model Comparison (all rounds combined)")
    print("=" * 60)
    for model_name, data in model_summaries.items():
        print(f"\n{model_name}:")
        print(f"  Total samples: {data['total_samples']}")
        print(f"  Original accuracy: {data['accuracy_original']:.2f}%")
        print(f"  v2 accuracy: {data['accuracy_v2']:.2f}%")
        print(f"  Improvement: {data['improvement']:+d} ({data['improvement_pct']:+.2f}%)")
