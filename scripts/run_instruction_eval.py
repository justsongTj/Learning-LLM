#!/usr/bin/env python3
"""Run a fixed TinyDL instruction set and save reproducible JSONL results."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


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
        checks["delimited_list"] = len(items) == int(rule["count"])
    elif rule_type == "contains_any":
        checks["contains_any"] = any(
            value.lower() in answer.lower() for value in rule.get("values", []))
    elif rule_type == "regex":
        checks["regex"] = re.search(rule["pattern"], answer, re.IGNORECASE) is not None
    elif rule_type:
        raise ValueError(f"unknown evaluation rule type: {rule_type}")

    return matched, checks, bool(checks) and all(checks.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--evaluation-file", default="data/instruction_eval_zh.jsonl")
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
    with Path(args.evaluation_file).open("r", encoding="utf-8") as source, \
            output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for line_number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            case = json.loads(raw)
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
            answer = generated[len(prompt):].strip() if generated.startswith(prompt) else generated
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
            print(f"[{line_number}] {case['id']}: {answer}")

    print(f"wrote {total} results to {output_path}; "
          f"strict rule pass {keyword_passes}/{total}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as error:
        print(f"instruction evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
