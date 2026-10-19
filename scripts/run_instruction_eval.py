#!/usr/bin/env python3
"""Run a fixed TinyDL instruction set and save reproducible JSONL results."""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


def extract_answer(generated, prompt):
    text = generated.strip()
    if text.startswith(prompt):
        return text[len(prompt):].strip()
    markers = list(re.finditer(r"回答\s*[:：]", text))
    if markers and re.search(r"指令\s*[:：]", text[:markers[-1].start()]):
        return text[markers[-1].end():].strip()
    return text


def evaluate_answer(case, answer):
    checks = {}
    expected = case.get("expected_keywords", [])
    matched = [keyword for keyword in expected if keyword.lower() in answer.lower()]
    if expected:
        checks["all_keywords"] = len(matched) == len(expected)

    rule = case.get("rule", {})
    rule_type = rule.get("type")
    if rule_type == "exact_number":
        numbers = re.findall(r"(?<![\d.])-?\d+(?:\.\d+)?", answer)
        checks["exact_number"] = numbers == [str(rule["value"])]
    elif rule_type == "delimited_list":
        items = [item.strip() for item in re.split(rule.get("separator", r"[,，]"), answer)
                 if item.strip()]
        allowed = rule.get("allowed_items")
        checks["delimited_list"] = (
            len(items) == int(rule["count"])
            and len(set(items)) == len(items)
            and (allowed is None or all(item in allowed for item in items))
        )
    elif rule_type == "contains_any":
        checks["contains_any"] = any(
            value.lower() in answer.lower() for value in rule.get("values", []))
    elif rule_type == "regex":
        patterns = rule.get("patterns")
        if patterns is None:
            patterns = [rule["pattern"]]
        checks["regex"] = all(
            re.search(pattern, answer, re.IGNORECASE) is not None
            for pattern in patterns)
    elif rule_type:
        raise ValueError(f"unknown evaluation rule type: {rule_type}")

    return matched, checks, bool(checks) and all(checks.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--evaluation-file", default="data/instruction_eval_zh.jsonl")
    parser.add_argument("--include-category", action="append",
                        help="only evaluate these categories; may be repeated")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", help="HIP_VISIBLE_DEVICES value")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dcu", action="store_true")
    args = parser.parse_args()

    environment = os.environ.copy()
    if args.device is not None:
        environment["HIP_VISIBLE_DEVICES"] = args.device
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = keyword_passes = 0
    category_totals = Counter()
    category_passes = Counter()
    with Path(args.evaluation_file).open("r", encoding="utf-8") as source, \
            output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for line_number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            case = json.loads(raw)
            category = case.get("category", "uncategorized")
            if (args.include_category
                    and category not in set(args.include_category)):
                continue
            prompt = f"指令:{case['instruction']} 回答:"
            command = [
                args.binary, args.model, args.tokenizer, prompt,
                str(args.max_new_tokens), "--temperature", str(args.temperature),
                "--top-k", str(args.top_k), "--seed", str(args.seed),
            ]
            if args.dcu:
                command.append("--dcu")
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True,
                encoding="utf-8", env=environment)
            generated = completed.stdout.strip()
            answer = extract_answer(generated, prompt)
            matched, checks, passed = evaluate_answer(case, answer)
            result = dict(case)
            result.update({
                "prompt": prompt,
                "answer": answer,
                "matched_keywords": matched,
                "keyword_pass": checks.get("all_keywords", False),
                "checks": checks,
                "pass": passed,
            })
            destination.write(json.dumps(result, ensure_ascii=False) + "\n")
            total += 1
            keyword_passes += int(result["pass"])
            category_totals[category] += 1
            category_passes[category] += int(result["pass"])
            print(f"[{line_number}] {case['id']}: {answer}")

    print(f"wrote {total} results to {output_path}; "
          f"strict rule pass {keyword_passes}/{total}")
    for category in sorted(category_totals):
        print(f"  {category}: {category_passes[category]}/{category_totals[category]}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as error:
        print(f"instruction evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
