#!/usr/bin/env python3
"""Stream, clean, deduplicate, and split text for TinyDL training."""

import argparse
import hashlib
import re
import sys
import unicodedata
from pathlib import Path


SPACE_RE = re.compile(r"[\t\r\f\v ]+")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def clean_paragraph(value: object, min_chars: int, max_chars: int, min_cjk_ratio: float):
    if not isinstance(value, str):
        return
    for raw in value.splitlines():
        text = unicodedata.normalize("NFKC", raw)
        text = SPACE_RE.sub(" ", text).strip()
        if len(text) < min_chars or len(text) > max_chars:
            continue
        visible = sum(not character.isspace() for character in text)
        if visible == 0 or len(CJK_RE.findall(text)) / visible < min_cjk_ratio:
            continue
        yield text


def stable_bucket(text: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 10_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Hugging Face dataset repository name")
    parser.add_argument("--config", help="optional dataset configuration name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--local-text", action="append", default=[],
                        help="local UTF-8 text file to mix in; may be repeated")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--max-output-mb", type=float, default=200.0,
                        help="maximum combined UTF-8 output size")
    parser.add_argument("--validation-ratio", type=float, default=0.02)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dataset and not args.local_text:
        parser.error("provide --dataset, --local-text, or both")
    if args.max_output_mb <= 0 or not 0.0 < args.validation_ratio < 1.0:
        parser.error("max-output-mb must be positive and validation-ratio must be between 0 and 1")
    if args.min_chars <= 0 or args.max_chars < args.min_chars:
        parser.error("invalid paragraph length limits")
    if not 0.0 <= args.min_cjk_ratio <= 1.0:
        parser.error("min-cjk-ratio must be between 0 and 1")

    def source_texts():
        for filename in args.local_text:
            with Path(filename).open("r", encoding="utf-8") as stream:
                yield from stream
        if args.dataset:
            try:
                from datasets import load_dataset
            except ImportError as error:
                raise RuntimeError(
                    "Hugging Face support requires: python3 -m pip install datasets") from error
            positional = [args.dataset]
            if args.config:
                positional.append(args.config)
            dataset = load_dataset(*positional, split=args.split, streaming=True)
            for row in dataset:
                if args.text_column not in row:
                    raise RuntimeError(
                        f"dataset row has no {args.text_column!r} column; available: {sorted(row)}")
                yield row[args.text_column]

    train_path = Path(args.train_output)
    validation_path = Path(args.validation_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    limit = int(args.max_output_mb * 1024 * 1024)
    validation_cutoff = int(args.validation_ratio * 10_000)
    seen = set()
    total_bytes = train_count = validation_count = 0

    with train_path.open("w", encoding="utf-8", newline="\n") as train_stream, \
            validation_path.open("w", encoding="utf-8", newline="\n") as validation_stream:
        for document in source_texts():
            for text in clean_paragraph(document, args.min_chars, args.max_chars,
                                        args.min_cjk_ratio):
                fingerprint = hashlib.sha256(text.encode("utf-8")).digest()
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                encoded_size = len(text.encode("utf-8")) + 1
                if total_bytes + encoded_size > limit:
                    break
                if stable_bucket(text, args.seed) < validation_cutoff:
                    validation_stream.write(text + "\n")
                    validation_count += 1
                else:
                    train_stream.write(text + "\n")
                    train_count += 1
                total_bytes += encoded_size
            if total_bytes >= limit:
                break

    if train_count == 0 or validation_count == 0:
        raise RuntimeError("filtering produced an empty train or validation file")
    print(f"wrote {train_count} training and {validation_count} validation paragraphs "
          f"({total_bytes / 1024 / 1024:.1f} MiB total)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"corpus preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
