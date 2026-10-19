#!/usr/bin/env python3
"""Normalize and clean an existing mixed corpus for instruction fine-tuning."""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from contextlib import ExitStack
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional, Pattern, Set, Tuple


SPACE_RE = re.compile(r"[\t\r\f\v ]+")
REQUESTED_LENGTH_RE = re.compile(
    r"(?:写|生成|撰写|一篇|文章|短文|报告|新闻|长度|篇幅|至少|不少于)"
    r"[^\d]{0,24}(\d{2,5})\s*个?字")
LEADING_LABEL_RE = re.compile(r"^(?:问题|指令)\s*[:：]\s*")
ANSWER_MARKER_RE = re.compile(r"回答\s*[:：]")
EXPLICIT_TASK_RE = re.compile(
    r"[?？]|(?:请|解释|描述|生成|写|翻译|总结|概括|改写|分析|计算|列出|"
    r"给出|推荐|判断|选择|创建|制定|回答|说明|比较|提取|分类|纠正|检查|"
    r"预测|什么|为什么|如何|怎么|哪些|谁|何时|何地)")
COPY_NORMALIZE_RE = re.compile(r"[^0-9A-Za-z\u3400-\u4dbf\u4e00-\u9fff]+")
TRAILING_INCOMPLETE = (",", "，", ":", "：", ";", "；")
TEMPLATE_CONTAMINATION_RE = re.compile(r"(?:问题|指令|回答|预测|选择)\s*[:：]")
QUOTED_SOURCE_RE = re.compile(r"[\"“](.{2,}?)[\"”]")
SOURCE_AFTER_COLON_RE = re.compile(r"[:：]\s*(.{4,})$")
REQUESTED_ITEMS_RE = re.compile(
    r"(?:列出|给出|提供|写出|包含|包括)\s*([一二两三四五六七八九十\d]+)\s*(?:种|个|项|条|句)")
CHINESE_NUMBERS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
CATEGORY_PATTERNS = (
    ("translation", re.compile(r"翻译|译成")),
    ("summarization", re.compile(r"总结|概括|摘要|提炼主旨")),
    ("rewriting", re.compile(r"改写|润色|书面语|纠正|拼写检查|语法是否正确")),
    ("format", re.compile(r"仅列|JSON|表格|使用.{0,8}分隔|列出\s*[一二两三四五六七八九十\d]+")),
    ("uncertainty", re.compile(r"根据已知信息|信息不足|无法确定|能否确定|是否能够确定")),
    ("math", re.compile(r"(?:计算|求值|算式|阶乘|方程|等于多少)|\d+\s*[+×÷*/]\s*\d+")),
    ("health", re.compile(r"健康|饮食|睡眠|运动|疾病|症状")),
    ("ai_ml", re.compile(r"人工智能|机器学习|深度学习|神经网络|计算机视觉|语言模型")),
)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return SPACE_RE.sub(" ", value).strip()


def stable_bucket(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10_000


def load_reject_patterns(filename: Optional[str]) -> List[Pattern[str]]:
    if not filename:
        return []
    patterns = []
    with Path(filename).open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            try:
                patterns.append(re.compile(text))
            except re.error as error:
                raise ValueError(
                    f"invalid reject pattern at {filename}:{line_number}: {error}") from error
    return patterns


def parse_instruction(line: str) -> Optional[Tuple[str, str]]:
    text = normalize(line)
    marker = ANSWER_MARKER_RE.search(text)
    if not marker:
        return None
    prompt = LEADING_LABEL_RE.sub("", text[:marker.start()].strip(), count=1)
    answer = text[marker.end():].strip()
    return prompt, answer


def copy_similarity(prompt: str, answer: str) -> float:
    prompt_key = COPY_NORMALIZE_RE.sub("", prompt).lower()
    answer_key = COPY_NORMALIZE_RE.sub("", answer).lower()
    if min(len(prompt_key), len(answer_key)) < 20:
        return 0.0
    return SequenceMatcher(None, prompt_key, answer_key, autojunk=False).ratio()


def classify_task(prompt: str) -> str:
    for category, pattern in CATEGORY_PATTERNS:
        if pattern.search(prompt):
            return category
    if re.search(r"什么是|为什么|谁|何时|何地|哪些", prompt):
        return "knowledge"
    return "other"


def requested_item_count(prompt: str) -> Optional[int]:
    match = REQUESTED_ITEMS_RE.search(prompt)
    if not match:
        return None
    value = match.group(1)
    return int(value) if value.isdigit() else CHINESE_NUMBERS.get(value)


def answer_item_count(answer: str) -> int:
    numbered = re.findall(
        r"(?:^|[\s;；。])(?:\d+|[一二三四五六七八九十])[.、)]\s*", answer)
    if numbered:
        return len(numbered)
    return len([part for part in re.split(r"[,，;；、]", answer) if part.strip()])


def has_task_source(prompt: str) -> bool:
    quoted = QUOTED_SOURCE_RE.search(prompt)
    if quoted and len(quoted.group(1).strip()) >= 3:
        return True
    after_colon = SOURCE_AFTER_COLON_RE.search(prompt)
    return bool(after_colon and len(after_colon.group(1).strip()) >= 4)


def rejection_reason(prompt: str, answer: str, args, reject_patterns) -> Optional[str]:
    if len(prompt) < args.min_prompt_chars:
        return "empty_or_short_prompt"
    if len(answer) < args.min_answer_chars:
        return "empty_or_short_answer"
    if len(prompt) + len(answer) > args.max_chars:
        return "too_long"
    if prompt == answer:
        return "copied_answer"
    if copy_similarity(prompt, answer) >= args.max_copy_similarity:
        return "near_copied_answer"
    requested = REQUESTED_LENGTH_RE.search(prompt)
    if requested and len(answer) < int(requested.group(1)) * args.requested_length_ratio:
        return "requested_length_not_met"
    combined = f"{prompt}\n{answer}"
    if any(pattern.search(combined) for pattern in reject_patterns):
        return "manual_reject_pattern"
    if args.require_explicit_task and not EXPLICIT_TASK_RE.search(prompt):
        return "ambiguous_prompt"
    if args.reject_incomplete_ending and answer.endswith(TRAILING_INCOMPLETE):
        return "incomplete_ending"
    if args.strict_task_quality:
        if TEMPLATE_CONTAMINATION_RE.search(answer):
            return "template_contamination"
        category = classify_task(prompt)
        if category in {"translation", "summarization", "rewriting"} and not has_task_source(prompt):
            return "missing_task_source"
        requested_items = requested_item_count(prompt)
        if requested_items is not None and answer_item_count(answer) < requested_items:
            return "requested_items_not_met"
    return None


def load_evaluation_prompts(filename: Optional[str]) -> Set[str]:
    prompts = set()
    if not filename:
        return prompts
    with Path(filename).open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {filename}:{line_number}: {error}") from error
            prompt = item.get("instruction")
            if not isinstance(prompt, str) or not normalize(prompt):
                raise ValueError(
                    f"missing non-empty instruction at {filename}:{line_number}")
            prompts.add(normalize(LEADING_LABEL_RE.sub("", prompt, count=1)))
    return prompts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True,
                        help="mixed UTF-8 corpus; may be repeated")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--stats-output", help="optional JSON cleaning report")
    parser.add_argument("--rejected-output",
                        help="optional JSONL audit file for rejected instruction records")
    parser.add_argument("--reject-pattern-file",
                        help="optional UTF-8 file containing one rejection regex per line")
    parser.add_argument("--evaluation-file",
                        help="JSONL fixed evaluation set whose exact prompts are excluded")
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--min-prompt-chars", type=int, default=2)
    parser.add_argument("--min-answer-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--tokenizer-model",
                        help="SentencePiece model used to reject records exceeding model context")
    parser.add_argument("--max-tokens", type=int,
                        help="maximum input tokens; includes BOS and predicts through EOS")
    parser.add_argument("--requested-length-ratio", type=float, default=0.5,
                        help="minimum answer/requested character ratio")
    parser.add_argument("--max-copy-similarity", type=float, default=0.96,
                        help="reject near-copies at or above this SequenceMatcher ratio")
    parser.add_argument("--require-explicit-task", action="store_true",
                        help="reject prompts without a question or recognized task verb")
    parser.add_argument("--reject-incomplete-ending", action="store_true",
                        help="reject answers ending in a comma, colon, or semicolon")
    parser.add_argument("--strict-task-quality", action="store_true",
                        help="reject template leakage, missing task sources, and incomplete lists")
    parser.add_argument("--include-category", action="append",
                        choices=[name for name, _ in CATEGORY_PATTERNS] + ["knowledge", "other"],
                        help="retain only selected task categories; may be repeated")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.validation_ratio < 1.0:
        parser.error("validation-ratio must be between 0 and 1")
    if args.min_prompt_chars <= 0 or args.min_answer_chars <= 0:
        parser.error("minimum lengths must be positive")
    if args.max_chars <= args.min_prompt_chars + args.min_answer_chars:
        parser.error("max-chars is too small")
    if (args.tokenizer_model is None) != (args.max_tokens is None):
        parser.error("--tokenizer-model and --max-tokens must be used together")
    if args.max_tokens is not None and args.max_tokens <= 1:
        parser.error("max-tokens must be greater than one")
    if not 0.0 <= args.requested_length_ratio <= 1.0:
        parser.error("requested-length-ratio must be between 0 and 1")
    if not 0.0 < args.max_copy_similarity <= 1.0:
        parser.error("max-copy-similarity must be between 0 and 1")

    train_path = Path(args.train_output)
    validation_path = Path(args.validation_output)
    if train_path.resolve() == validation_path.resolve():
        parser.error("train and validation outputs must be different")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    train_part = train_path.with_name(train_path.name + ".part")
    validation_part = validation_path.with_name(validation_path.name + ".part")
    rejected_path = Path(args.rejected_output) if args.rejected_output else None
    rejected_part = (rejected_path.with_name(rejected_path.name + ".part")
                     if rejected_path else None)
    if rejected_path:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)

    reject_patterns = load_reject_patterns(args.reject_pattern_file)
    evaluation_prompts = load_evaluation_prompts(args.evaluation_file)
    cutoff = int(args.validation_ratio * 10_000)
    reasons = Counter()
    category_counts = Counter()
    seen = set()
    train_count = validation_count = 0
    tokenizer = None
    if args.tokenizer_model:
        try:
            import sentencepiece as spm
        except ImportError as error:
            raise RuntimeError(
                "sentencepiece Python package is required with --tokenizer-model") from error
        tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer_model)

    try:
        with ExitStack() as stack:
            train_stream = stack.enter_context(
                train_part.open("w", encoding="utf-8", newline="\n"))
            validation_stream = stack.enter_context(
                validation_part.open("w", encoding="utf-8", newline="\n"))
            rejected_stream = (stack.enter_context(
                rejected_part.open("w", encoding="utf-8", newline="\n"))
                if rejected_part else None)
            for filename in args.input:
                with Path(filename).open("r", encoding="utf-8") as source:
                    for line_number, raw in enumerate(source, 1):
                        reasons["input_lines"] += 1
                        parsed = parse_instruction(raw)
                        if parsed is None:
                            reasons["plain_text_excluded"] += 1
                            continue
                        prompt, answer = parsed
                        reason = rejection_reason(prompt, answer, args, reject_patterns)
                        if reason:
                            reasons[reason] += 1
                            if rejected_stream:
                                rejected_stream.write(json.dumps({
                                    "reason": reason,
                                    "source": filename,
                                    "line": line_number,
                                    "prompt": prompt,
                                    "answer": answer,
                                }, ensure_ascii=False) + "\n")
                            continue
                        category = classify_task(prompt)
                        if args.include_category and category not in args.include_category:
                            reasons["category_not_selected"] += 1
                            if rejected_stream:
                                rejected_stream.write(json.dumps({
                                    "reason": "category_not_selected",
                                    "source": filename,
                                    "line": line_number,
                                    "category": category,
                                    "prompt": prompt,
                                    "answer": answer,
                                }, ensure_ascii=False) + "\n")
                            continue
                        normalized = f"指令:{prompt} 回答:{answer}"
                        if tokenizer is not None:
                            special_tokens = int(tokenizer.bos_id() >= 0) + int(tokenizer.eos_id() >= 0)
                            token_count = len(tokenizer.encode(normalized, out_type=int)) + special_tokens
                            if token_count > args.max_tokens + 1:
                                reasons["too_many_tokens"] += 1
                                if rejected_stream:
                                    rejected_stream.write(json.dumps({
                                        "reason": "too_many_tokens",
                                        "source": filename,
                                        "line": line_number,
                                        "token_count": token_count,
                                        "prompt": prompt,
                                        "answer": answer,
                                    }, ensure_ascii=False) + "\n")
                                continue
                        fingerprint = hashlib.sha256(normalized.encode("utf-8")).digest()
                        if fingerprint in seen:
                            reasons["exact_duplicate"] += 1
                            if rejected_stream:
                                rejected_stream.write(json.dumps({
                                    "reason": "exact_duplicate",
                                    "source": filename,
                                    "line": line_number,
                                    "prompt": prompt,
                                    "answer": answer,
                                }, ensure_ascii=False) + "\n")
                            continue
                        seen.add(fingerprint)
                        if normalize(prompt) in evaluation_prompts:
                            reasons["evaluation_prompt_excluded"] += 1
                            if rejected_stream:
                                rejected_stream.write(json.dumps({
                                    "reason": "evaluation_prompt_excluded",
                                    "source": filename,
                                    "line": line_number,
                                    "prompt": prompt,
                                    "answer": answer,
                                }, ensure_ascii=False) + "\n")
                            continue
                        category_counts[category] += 1
                        if stable_bucket(normalized, args.seed) < cutoff:
                            validation_stream.write(normalized + "\n")
                            validation_count += 1
                        else:
                            train_stream.write(normalized + "\n")
                            train_count += 1

        if train_count == 0 or validation_count == 0:
            raise RuntimeError("cleaning produced an empty train or validation file")
        train_part.replace(train_path)
        validation_part.replace(validation_path)
        if rejected_part and rejected_path:
            rejected_part.replace(rejected_path)
    except Exception:
        train_part.unlink(missing_ok=True)
        validation_part.unlink(missing_ok=True)
        if rejected_part:
            rejected_part.unlink(missing_ok=True)
        raise

    report = {
        "train_records": train_count,
        "validation_records": validation_count,
        "accepted_records": train_count + validation_count,
        "category_counts": dict(sorted(category_counts.items())),
        "counts": dict(sorted(reasons.items())),
        "configuration": {
            "validation_ratio": args.validation_ratio,
            "min_prompt_chars": args.min_prompt_chars,
            "min_answer_chars": args.min_answer_chars,
            "max_chars": args.max_chars,
            "tokenizer_model": args.tokenizer_model,
            "max_tokens": args.max_tokens,
            "requested_length_ratio": args.requested_length_ratio,
            "max_copy_similarity": args.max_copy_similarity,
            "require_explicit_task": args.require_explicit_task,
            "reject_incomplete_ending": args.reject_incomplete_ending,
            "strict_task_quality": args.strict_task_quality,
            "include_categories": args.include_category,
            "seed": args.seed,
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"instruction corpus preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
