#!/usr/bin/env python3
"""Build a balanced, deterministic TinyDL SFT corpus from verified task families."""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

from prepare_instruction_corpus import classify_task, normalize, parse_instruction, stable_bucket


def stable_rank(text: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).digest()


def load_translation_records(paths, limit: int, seed: int):
    records = set()
    for filename in paths:
        with Path(filename).open(encoding="utf-8") as stream:
            for raw in stream:
                parsed = parse_instruction(raw)
                if not parsed:
                    continue
                prompt, answer = parsed
                if classify_task(prompt) == "translation":
                    records.add(f"指令:{prompt} 回答:{answer}")
    ordered = sorted(records, key=lambda value: stable_rank(value, seed))
    return ordered[:limit]


def generate_math(count: int, seed: int):
    randomizer = random.Random(seed + 101)
    records = set()
    while len(records) < count:
        operation = randomizer.choice(("+", "-", "×"))
        if operation == "+":
            left, right = randomizer.randint(2, 99), randomizer.randint(2, 99)
            result = left + right
        elif operation == "-":
            left, right = randomizer.randint(10, 120), randomizer.randint(1, 99)
            left, right = max(left, right), min(left, right)
            result = left - right
        else:
            left, right = randomizer.randint(2, 20), randomizer.randint(2, 20)
            result = left * right
        prompt = randomizer.choice((
            f"计算{left}{operation}{right}。",
            f"请给出算式{left}{operation}{right}的结果。",
        ))
        records.add(f"指令:{prompt} 回答:{result}")
    return sorted(records, key=lambda value: stable_rank(value, seed))


def generate_uncertainty(count: int, seed: int):
    randomizer = random.Random(seed + 202)
    names = ("小王", "小李", "小张", "小陈", "小林", "小周", "小赵", "小孙")
    owned = ("一本书", "一辆自行车", "一台电脑", "一只背包", "一部手机", "一把雨伞")
    unknowns = ("价格", "品牌", "颜色", "购买日期", "生产地点", "重量")
    answers = (
        "根据现有信息无法确定。",
        "题目提供的信息不足，不能确定答案。",
        "已知条件中没有相关信息，因此无法判断。",
    )
    records = set()
    while len(records) < count:
        name = randomizer.choice(names)
        item = randomizer.choice(owned)
        attribute = randomizer.choice(unknowns)
        prompt = f"根据已知信息回答：{name}有{item}。它的{attribute}是什么？"
        records.add(f"指令:{prompt} 回答:{randomizer.choice(answers)}")
    return sorted(records, key=lambda value: stable_rank(value, seed))


def generate_format(count: int, seed: int):
    randomizer = random.Random(seed + 303)
    pools = {
        "编程语言": ("Python", "C++", "Rust", "Java", "Go", "JavaScript"),
        "水果": ("苹果", "香蕉", "橙子", "葡萄", "梨", "草莓"),
        "颜色": ("红色", "蓝色", "绿色", "黄色", "紫色", "橙色"),
        "运动": ("跑步", "游泳", "骑行", "篮球", "羽毛球", "足球"),
        "乐器": ("钢琴", "吉他", "小提琴", "长笛", "架子鼓", "古筝"),
    }
    records = set()
    while len(records) < count:
        category = randomizer.choice(tuple(pools))
        values = randomizer.sample(pools[category], 3)
        prompt = f"仅列出三种常见的{category}，使用逗号分隔，不要添加解释。"
        records.add(f"指令:{prompt} 回答:{','.join(values)}")
    return sorted(records, key=lambda value: stable_rank(value, seed))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True,
                        help="strict cleaned corpus; may be repeated")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--translation-count", type=int, default=500)
    parser.add_argument("--math-count", type=int, default=500)
    parser.add_argument("--uncertainty-count", type=int, default=500)
    parser.add_argument("--format-count", type=int, default=500)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = (args.translation_count, args.math_count,
              args.uncertainty_count, args.format_count)
    if any(value <= 0 for value in counts):
        parser.error("all task counts must be positive")
    if not 0.0 < args.validation_ratio < 1.0:
        parser.error("validation-ratio must be between zero and one")

    groups = {
        "translation": load_translation_records(args.input, args.translation_count, args.seed),
        "math": generate_math(args.math_count, args.seed),
        "uncertainty": generate_uncertainty(args.uncertainty_count, args.seed),
        "format": generate_format(args.format_count, args.seed),
    }
    if len(groups["translation"]) < args.translation_count:
        raise RuntimeError(
            f"requested {args.translation_count} translations but found {len(groups['translation'])}")

    evaluation_prompts = set()
    evaluation_path = Path(__file__).resolve().parents[1] / "data" / "instruction_eval_narrow_v3.jsonl"
    with evaluation_path.open(encoding="utf-8") as stream:
        for raw in stream:
            evaluation_prompts.add(normalize(json.loads(raw)["instruction"]))

    train, validation = [], []
    output_counts = Counter()
    cutoff = int(args.validation_ratio * 10_000)
    for category, records in groups.items():
        for record in records:
            prompt, _ = parse_instruction(record)
            if normalize(prompt) in evaluation_prompts:
                continue
            destination = validation if stable_bucket(record, args.seed) < cutoff else train
            destination.append(record)
            output_counts[category] += 1

    train.sort(key=lambda value: stable_rank(value, args.seed + 1))
    validation.sort(key=lambda value: stable_rank(value, args.seed + 2))
    if not train or not validation:
        raise RuntimeError("balanced corpus produced an empty split")

    outputs = ((Path(args.train_output), train), (Path(args.validation_output), validation))
    for path, records in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        part.write_text("\n".join(records) + "\n", encoding="utf-8", newline="\n")
        part.replace(path)

    report = {
        "train_records": len(train),
        "validation_records": len(validation),
        "total_records": len(train) + len(validation),
        "category_counts": dict(sorted(output_counts.items())),
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
    }
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"balanced corpus generation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
