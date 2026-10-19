#!/usr/bin/env python3

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "build_balanced_instruction_corpus.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("build_balanced_instruction_corpus", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildBalancedInstructionCorpusTest(unittest.TestCase):
    def test_long_context_filter_refills_to_requested_count(self):
        records = MODULE.generate_long_context(
            20, 42, record_filter=lambda record: "杭州" in record)
        self.assertEqual(len(records), 20)
        self.assertTrue(all("杭州" in record for record in records))

    def test_builds_deterministic_balanced_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "strict.txt"
            train = root / "train.txt"
            validation = root / "validation.txt"
            stats = root / "stats.json"
            source.write_text("\n".join(
                f"指令:将“测试句子{i}”翻译成英语。 回答:Test sentence {i}."
                for i in range(20)) + "\n", encoding="utf-8")

            subprocess.run([
                sys.executable, str(SCRIPT), "--input", str(source),
                "--train-output", str(train), "--validation-output", str(validation),
                "--stats-output", str(stats), "--translation-count", "10",
                "--math-count", "10", "--uncertainty-count", "10",
                "--format-count", "10", "--long-context-count", "10",
                "--validation-ratio", "0.5",
            ], check=True, cwd=REPOSITORY, capture_output=True, text=True,
               encoding="utf-8")

            report = json.loads(stats.read_text(encoding="utf-8"))
            records = train.read_text(encoding="utf-8").splitlines() + \
                validation.read_text(encoding="utf-8").splitlines()
            self.assertEqual(report["total_records"], 50)
            self.assertEqual(set(report["category_counts"]),
                             {"translation", "math", "uncertainty", "format",
                              "long_context"})
            self.assertTrue(all(record.startswith("指令:") for record in records))
            self.assertTrue(any("无法" in record or "信息不足" in record for record in records))
            self.assertTrue(any("第一条关键信息" in record or "资料开头" in record
                                for record in records))

    def test_excludes_evaluation_prompts_without_reducing_target_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "strict.txt"
            evaluation = root / "eval.jsonl"
            train = root / "train.txt"
            validation = root / "validation.txt"
            stats = root / "stats.json"
            source.write_text("\n".join(
                f"指令:将“样例{i}”翻译成英语。 回答:Example {i}."
                for i in range(20)) + "\n", encoding="utf-8")
            held_out, _ = MODULE.parse_instruction(MODULE.generate_math(10, 42)[0])
            evaluation.write_text(
                json.dumps({"id": "held_out", "instruction": held_out},
                           ensure_ascii=False) + "\n", encoding="utf-8")

            subprocess.run([
                sys.executable, str(SCRIPT), "--input", str(source),
                "--train-output", str(train), "--validation-output", str(validation),
                "--stats-output", str(stats), "--translation-count", "10",
                "--math-count", "10", "--uncertainty-count", "10",
                "--format-count", "10", "--long-context-count", "10",
                "--evaluation-file", str(evaluation), "--validation-ratio", "0.5",
            ], check=True, cwd=REPOSITORY, capture_output=True, text=True, encoding="utf-8")

            report = json.loads(stats.read_text(encoding="utf-8"))
            records = train.read_text(encoding="utf-8").splitlines() + \
                validation.read_text(encoding="utf-8").splitlines()
            self.assertEqual(report["total_records"], 50)
            self.assertFalse(any(MODULE.parse_instruction(record)[0] == held_out
                                 for record in records))

    def test_allows_disabling_unwanted_task_families(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "strict.txt"
            train = root / "train.txt"
            validation = root / "validation.txt"
            stats = root / "stats.json"
            source.write_text(
                "指令:将“测试”翻译成英语。 回答:Test.\n", encoding="utf-8")

            subprocess.run([
                sys.executable, str(SCRIPT), "--input", str(source),
                "--train-output", str(train), "--validation-output", str(validation),
                "--stats-output", str(stats), "--translation-count", "0",
                "--math-count", "0", "--uncertainty-count", "20",
                "--format-count", "20", "--long-context-count", "0",
                "--validation-ratio", "0.5",
            ], check=True, cwd=REPOSITORY, capture_output=True, text=True,
               encoding="utf-8")

            report = json.loads(stats.read_text(encoding="utf-8"))
            self.assertEqual(report["total_records"], 40)
            self.assertEqual(set(report["category_counts"]),
                             {"uncertainty", "format"})


if __name__ == "__main__":
    unittest.main()
