#!/usr/bin/env python3
"""Normalize and clean an existing mixed corpus for instruction fine-tuning."""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List, Optional, Pattern, Set, Tuple


SPACE_RE = re.compile(r"[\t\r\f\v ]+")
REQUESTED_LENGTH_RE = re.compile(r"(?:写|生成|撰写)[^\d]{0,24}(\d{2,5})\s*字")
LEADING_LABEL_RE = re.compile(r"^(?:问题|指令)\s*[:：]\s*")
ANSWER_MARKER_RE = re.compile(r"回答\s*[:：]")
TRAILING_INCOMPLETE = (",", "，", ":", "：", ";", "；")


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


def rejection_reason(prompt: str, answer: str, args, reject_patterns) -> Optional[str]:
    if len(prompt) < args.min_prompt_chars:
        return "empty_or_short_prompt"
    if len(answer) < args.min_answer_chars:
        return "empty_or_short_answer"
    if len(prompt) + len(answer) > args.max_chars:
        return "too_long"
    if prompt == answer:
        return "copied_answer"
    requested = REQUESTED_LENGTH_RE.search(prompt)
    if requested and len(answer) < int(requested.group(1)) * args.requested_length_ratio:
        return "requested_length_not_met"
    if args.reject_incomplete_ending and answer.endswith(TRAILING_INCOMPLETE):
        return "incomplete_ending"
    combined = f"{prompt}\n{answer}"
    if any(pattern.search(combined) for pattern in reject_patterns):
        return "manual_reject_pattern"
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
    parser.add_argument("--reject-pattern-file",
                        help="optional UTF-8 file containing one rejection regex per line")
    parser.add_argument("--evaluation-file",
                        help="JSONL fixed evaluation set whose exact prompts are excluded")
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--min-prompt-chars", type=int, default=2)
    parser.add_argument("--min-answer-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--requested-length-ratio", type=float, default=0.5,
                        help="minimum answer/requested character ratio")
    parser.add_argument("--reject-incomplete-ending", action="store_true",
                        help="reject answers ending in a comma, colon, or semicolon")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.validation_ratio < 1.0:
        parser.error("validation-ratio must be between 0 and 1")
    if args.min_prompt_chars <= 0 or args.min_answer_chars <= 0:
        parser.error("minimum lengths must be positive")
    if args.max_chars <= args.min_prompt_chars + args.min_answer_chars:
        parser.error("max-chars is too small")
    if not 0.0 <= args.requested_length_ratio <= 1.0:
        parser.error("requested-length-ratio must be between 0 and 1")

    train_path = Path(args.train_output)
    validation_path = Path(args.validation_output)
    if train_path.resolve() == validation_path.resolve():
        parser.error("train and validation outputs must be different")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    train_part = train_path.with_name(train_path.name + ".part")
    validation_part = validation_path.with_name(validation_path.name + ".part")

    reject_patterns = load_reject_patterns(args.reject_pattern_file)
    evaluation_prompts = load_evaluation_prompts(args.evaluation_file)
    cutoff = int(args.validation_ratio * 10_000)
    reasons = Counter()
    seen = set()
    train_count = validation_count = 0

    try:
        with train_part.open("w", encoding="utf-8", newline="\n") as train_stream, \
                validation_part.open("w", encoding="utf-8", newline="\n") as validation_stream:
            for filename in args.input:
                with Path(filename).open("r", encoding="utf-8") as source:
                    for raw in source:
                        reasons["input_lines"] += 1
                        parsed = parse_instruction(raw)
                        if parsed is None:
                            reasons["plain_text_excluded"] += 1
                            continue
                        prompt, answer = parsed
                        reason = rejection_reason(prompt, answer, args, reject_patterns)
                        if reason:
                            reasons[reason] += 1
                            continue
                        normalized = f"指令:{prompt} 回答:{answer}"
                        fingerprint = hashlib.sha256(normalized.encode("utf-8")).digest()
                        if fingerprint in seen:
                            reasons["exact_duplicate"] += 1
                            continue
                        seen.add(fingerprint)
                        if normalize(prompt) in evaluation_prompts:
                            reasons["evaluation_prompt_excluded"] += 1
                            continue
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
    except Exception:
        train_part.unlink(missing_ok=True)
        validation_part.unlink(missing_ok=True)
        raise

    report = {
        "train_records": train_count,
        "validation_records": validation_count,
        "accepted_records": train_count + validation_count,
        "counts": dict(sorted(reasons.items())),
        "configuration": {
            "validation_ratio": args.validation_ratio,
            "min_prompt_chars": args.min_prompt_chars,
            "min_answer_chars": args.min_answer_chars,
            "max_chars": args.max_chars,
            "requested_length_ratio": args.requested_length_ratio,
            "reject_incomplete_ending": args.reject_incomplete_ending,
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
