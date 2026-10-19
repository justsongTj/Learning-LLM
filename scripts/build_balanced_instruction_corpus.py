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


def load_translation_records(paths, limit: int, seed: int, excluded_prompts=None):
    excluded_prompts = excluded_prompts or set()
    records = set()
    for filename in paths:
        with Path(filename).open(encoding="utf-8") as stream:
            for raw in stream:
                parsed = parse_instruction(raw)
                if not parsed:
                    continue
                prompt, answer = parsed
                if (classify_task(prompt) == "translation"
                        and normalize(prompt) not in excluded_prompts):
                    records.add(f"指令:{prompt} 回答:{answer}")
    ordered = sorted(records, key=lambda value: stable_rank(value, seed))
    return ordered[:limit]


def generate_math(count: int, seed: int, excluded_prompts=None):
    excluded_prompts = excluded_prompts or set()
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
        expression = f"{left}{operation}{right}"
        prompt = randomizer.choice((
            f"计算{expression}。",
            f"请给出算式{expression}的结果。",
            f"{expression}等于多少？只写数字。",
            f"求{expression}的值，不要写计算过程。",
            f"完成下面的计算题：{expression}。答案仅保留整数。",
        ))
        if normalize(prompt) not in excluded_prompts:
            records.add(f"指令:{prompt} 回答:{result}")
    return sorted(records, key=lambda value: stable_rank(value, seed))


def generate_uncertainty(count: int, seed: int, excluded_prompts=None):
    excluded_prompts = excluded_prompts or set()
    randomizer = random.Random(seed + 202)
    names = ("小王", "小李", "小张", "小陈", "小林", "小周", "小赵", "小孙")
    owned = ("一本书", "一辆自行车", "一台电脑", "一只背包", "一部手机", "一把雨伞")
    unknowns = ("价格", "品牌", "颜色", "购买日期", "生产地点", "重量")
    knowns = (
        "这件物品目前由本人使用",
        "它被妥善保管着",
        "今天确认它仍然存在",
        "物品的主人没有发生变化",
        "记录中只说明了所有权",
    )
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
        fact = randomizer.choice(knowns)
        prompt = randomizer.choice((
            f"根据已知信息回答：{name}有{item}。它的{attribute}是什么？",
            f"已知{name}有{item}，并且{fact}。请问它的{attribute}是什么？",
            f"阅读事实后作答：{name}拥有{item}。资料没有给出其他属性。能否确定它的{attribute}？",
            f"现有记录仅表明{name}有{item}，还提到{fact}。请判断该物品的{attribute}。",
        ))
        if normalize(prompt) not in excluded_prompts:
            records.add(f"指令:{prompt} 回答:{randomizer.choice(answers)}")
    return sorted(records, key=lambda value: stable_rank(value, seed))


def generate_format(count: int, seed: int, excluded_prompts=None):
    excluded_prompts = excluded_prompts or set()
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
        item_count = randomizer.choice((2, 3, 4))
        values = randomizer.sample(pools[category], item_count)
        separator_name, separator = randomizer.choice((
            ("英文逗号", ","), ("中文逗号", "，"), ("顿号", "、"),
        ))
        chinese_count = {2: "两", 3: "三", 4: "四"}[item_count]
        prompt = randomizer.choice((
            f"仅列出{chinese_count}种常见的{category}，使用{separator_name}分隔，不要添加解释。",
            f"请写出{item_count}种{category}，各项之间用{separator_name}分隔，只输出列表。",
            f"按要求输出{category}：数量为{item_count}，分隔符为{separator_name}，不要说明原因。",
        ))
        if normalize(prompt) not in excluded_prompts:
            records.add(f"指令:{prompt} 回答:{separator.join(values)}")
    return sorted(records, key=lambda value: stable_rank(value, seed))


def generate_long_context(count: int, seed: int, excluded_prompts=None,
                          record_filter=None):
    """Generate grounded prompts whose answer depends on facts near the beginning."""
    excluded_prompts = excluded_prompts or set()
    randomizer = random.Random(seed + 404)
    names = ("林晨", "周宁", "陈远", "赵清", "孙悦", "吴桐")
    cities = ("杭州", "成都", "青岛", "昆明", "苏州", "厦门")
    activities = ("阅读", "游泳", "摄影", "跑步", "绘画", "下棋")
    distractors = (
        "会议室在二楼，窗边摆着三盆绿植。",
        "当天上午天气晴朗，下午有短时阵雨。",
        "记录员使用蓝色文件夹整理了全部材料。",
        "活动结束后，工作人员检查了门窗和照明。",
        "公告栏上还张贴着下周的值班安排。",
        "参与者依次签名，并在入口领取了号码牌。",
    )
    templates = (
        "请阅读记录并回答最后的问题。第一条关键信息：{name}居住在{city}，最喜欢的活动是{activity}。{details}根据记录，{name}居住在哪座城市？",
        "根据以下资料作答。资料开头说明{name}来自{city}，业余爱好是{activity}。{details}问题：{name}的业余爱好是什么？",
    )
    records = set()
    attempts = 0
    maximum_attempts = max(10_000, count * 1_000)
    while len(records) < count:
        attempts += 1
        if attempts > maximum_attempts:
            raise RuntimeError(
                f"could not generate {count} unique long-context records "
                "within the requested token range")
        name = randomizer.choice(names)
        city = randomizer.choice(cities)
        activity = randomizer.choice(activities)
        details = "".join(randomizer.sample(distractors, 4))
        template_index = randomizer.randrange(len(templates))
        prompt = templates[template_index].format(
            name=name, city=city, activity=activity, details=details)
        answer = city if template_index == 0 else activity
        record = f"指令:{prompt} 回答:{answer}"
        if (normalize(prompt) not in excluded_prompts
                and (record_filter is None or record_filter(record))):
            records.add(record)
    return sorted(records, key=lambda value: stable_rank(value, seed))


def load_evaluation_prompts(paths):
    prompts = set()
    for filename in paths:
        with Path(filename).open(encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                item = json.loads(raw)
                instruction = item.get("instruction")
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError(
                        f"missing instruction at {filename}:{line_number}")
                prompts.add(normalize(instruction))
    return prompts


def make_token_length_filter(tokenizer_model, minimum: int, maximum: int):
    try:
        import sentencepiece as spm
    except ImportError as error:
        raise RuntimeError(
            "sentencepiece is required when --tokenizer-model is used") from error

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_model))

    def accepts(record):
        record_tokens = (
            len(tokenizer.encode(record, out_type=int))
            + int(tokenizer.bos_id() >= 0)
            + int(tokenizer.eos_id() >= 0)
        )
        input_tokens = record_tokens - 1
        return minimum <= input_tokens <= maximum

    return accepts


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
    parser.add_argument("--long-context-count", type=int, default=500)
    parser.add_argument("--evaluation-file", action="append",
                        help="evaluation JSONL to exclude; may be repeated")
    parser.add_argument("--tokenizer-model",
                        help="SentencePiece model used to enforce long-context length")
    parser.add_argument("--min-long-context-input-tokens", type=int, default=129)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = (args.translation_count, args.math_count,
              args.uncertainty_count, args.format_count,
              args.long_context_count)
    if any(value <= 0 for value in counts):
        parser.error("all task counts must be positive")
    if not 0.0 < args.validation_ratio < 1.0:
        parser.error("validation-ratio must be between zero and one")
    if args.min_long_context_input_tokens <= 0:
        parser.error("min-long-context-input-tokens must be positive")
    if args.max_input_tokens < args.min_long_context_input_tokens:
        parser.error("max-input-tokens must be at least min-long-context-input-tokens")

    evaluation_paths = args.evaluation_file or [
        Path(__file__).resolve().parents[1] / "data" / "instruction_eval_narrow_v3.jsonl"
    ]
    evaluation_prompts = load_evaluation_prompts(evaluation_paths)
    long_context_filter = None
    if args.tokenizer_model:
        long_context_filter = make_token_length_filter(
            args.tokenizer_model,
            args.min_long_context_input_tokens,
            args.max_input_tokens)

    groups = {
        "translation": load_translation_records(
            args.input, args.translation_count, args.seed, evaluation_prompts),
        "math": generate_math(args.math_count, args.seed, evaluation_prompts),
        "uncertainty": generate_uncertainty(
            args.uncertainty_count, args.seed, evaluation_prompts),
        "format": generate_format(args.format_count, args.seed, evaluation_prompts),
        "long_context": generate_long_context(
            args.long_context_count, args.seed, evaluation_prompts,
            long_context_filter),
    }
    if len(groups["translation"]) < args.translation_count:
        raise RuntimeError(
            f"requested {args.translation_count} translations but found {len(groups['translation'])}")

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
        "tokenizer_model": args.tokenizer_model,
        "min_long_context_input_tokens": (
            args.min_long_context_input_tokens if args.tokenizer_model else None),
        "max_input_tokens": args.max_input_tokens if args.tokenizer_model else None,
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
