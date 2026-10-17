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
    parser.add_argument("--modelscope-dataset", help="ModelScope dataset repository name")
    parser.add_argument("--config", help="optional dataset configuration name")
    parser.add_argument("--modelscope-subset", help="optional ModelScope subset name")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--text-columns",
                        help="comma-separated fields to join, for example instruction,input,output")
    parser.add_argument("--instruction-format", action="store_true",
                        help="format instruction/input/output fields as a supervised QA paragraph")
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

    if args.dataset and args.modelscope_dataset:
        parser.error("choose only one of --dataset and --modelscope-dataset")
    if not args.dataset and not args.modelscope_dataset and not args.local_text:
        parser.error("provide --dataset, --modelscope-dataset, --local-text, or a mixture")
    if args.max_output_mb <= 0 or not 0.0 < args.validation_ratio < 1.0:
        parser.error("max-output-mb must be positive and validation-ratio must be between 0 and 1")
    if args.min_chars <= 0 or args.max_chars < args.min_chars:
        parser.error("invalid paragraph length limits")
    if not 0.0 <= args.min_cjk_ratio <= 1.0:
        parser.error("min-cjk-ratio must be between 0 and 1")

    text_columns = ([field.strip() for field in args.text_columns.split(",") if field.strip()]
                    if args.text_columns else [args.text_column])
    if args.instruction_format and not {"instruction", "output"}.issubset(text_columns):
        parser.error("--instruction-format requires instruction and output in --text-columns")

    def row_text(row):
        missing = [field for field in text_columns if field not in row]
        if missing:
            raise RuntimeError(
                f"dataset row has no {missing!r} column(s); available: {sorted(row)}")
        if args.instruction_format:
            instruction = str(row.get("instruction") or "").strip()
            input_text = str(row.get("input") or "").strip()
            output = str(row.get("output") or "").strip()
            if not instruction or not output:
                return ""
            parts = [f"指令:{instruction}"]
            if input_text:
                parts.append(f"输入:{input_text}")
            parts.append(f"回答:{output}")
            return " ".join(parts)
        values = [row[field].strip() for field in text_columns
                  if isinstance(row[field], str) and row[field].strip()]
        return " ".join(values)

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
                yield row_text(row)
        if args.modelscope_dataset:
            try:
                from modelscope.msdatasets import MsDataset
            except ImportError as error:
                raise RuntimeError(
                    "ModelScope support requires: python3 -m pip install modelscope addict") from error
            options = {
                "split": args.split,
                "use_streaming": True,
            }
            if args.modelscope_subset:
                options["subset_name"] = args.modelscope_subset
            dataset = MsDataset.load(args.modelscope_dataset, **options)
            for row in dataset:
                yield row_text(row)

    train_path = Path(args.train_output)
    validation_path = Path(args.validation_output)
    if train_path.resolve() == validation_path.resolve():
        parser.error("train and validation outputs must be different")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    train_temporary = train_path.with_name(train_path.name + ".part")
    validation_temporary = validation_path.with_name(validation_path.name + ".part")
    limit = int(args.max_output_mb * 1024 * 1024)
    validation_cutoff = int(args.validation_ratio * 10_000)
    seen = set()
    total_bytes = train_count = validation_count = 0

    try:
        with train_temporary.open("w", encoding="utf-8", newline="\n") as train_stream, \
                validation_temporary.open("w", encoding="utf-8", newline="\n") as validation_stream:
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
        train_temporary.replace(train_path)
        validation_temporary.replace(validation_path)
    except Exception:
        train_temporary.unlink(missing_ok=True)
        validation_temporary.unlink(missing_ok=True)
        raise
    print(f"wrote {train_count} training and {validation_count} validation paragraphs "
          f"({total_bytes / 1024 / 1024:.1f} MiB total)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"corpus preparation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
