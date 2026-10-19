#!/usr/bin/env python3

import importlib.util
import json
import re
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
    def test_copy2_curriculum_has_deterministic_order_and_separator(self):
        records = MODULE.generate_format(100, 42, curriculum="copy2")
        self.assertEqual(len(records), 100)
        for record in records:
            prompt, answer = MODULE.parse_instruction(record)
            items = answer.split(",")
            self.assertEqual(len(items), 2)
            first = prompt.find(items[0])
            second = prompt.find(items[1])
            self.assertGreaterEqual(first, 0)
            self.assertGreater(second, first)

    def test_copy_count_curriculum_covers_two_three_and_four_items(self):
        records = MODULE.generate_format(300, 42, curriculum="copy_count")
        self.assertEqual(len(records), 300)
        observed_counts = set()
        for record in records:
            prompt, answer = MODULE.parse_instruction(record)
            items = answer.split(",")
            observed_counts.add(len(items))
            self.assertIn(len(items), (2, 3, 4))
            self.assertIn("、".join(items), prompt)
        self.assertEqual(observed_counts, {2, 3, 4})

    def test_open_copy_curriculum_uses_mixed_items_and_replays_legacy(self):
        records = MODULE.generate_format(1000, 42, curriculum="copy_count_open")
        self.assertEqual(len(records), 1000)
        open_records = 0
        legacy_records = 0
        observed_counts = set()
        for record in records:
            prompt, answer = MODULE.parse_instruction(record)
            items = answer.split(",")
            observed_counts.add(len(items))
            self.assertIn(len(items), (2, 3, 4))
            self.assertIn("、".join(items), prompt)
            if all(item in MODULE.OPEN_COPY_TRAIN_ITEMS for item in items):
                open_records += 1
            else:
                legacy_records += 1
        self.assertEqual(observed_counts, {2, 3, 4})
        self.assertGreater(open_records, 700)
        self.assertGreater(legacy_records, 100)

    def test_open_copy_ood_candidates_are_held_out_from_training_pool(self):
        evaluation = REPOSITORY / "data" / \
            "instruction_eval_format_copy_count_ood_v1.jsonl"
        cases = [json.loads(line) for line in evaluation.read_text(
            encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(cases), 30)
        self.assertEqual({case["rule"]["type"] for case in cases}, {"exact_text"})
        self.assertEqual({len(case["rule"]["value"].split(",")) for case in cases},
                         {2, 3, 4})
        training_items = set(MODULE.OPEN_COPY_TRAIN_ITEMS)
        training_items.update(
            item for pool in MODULE.FORMAT_POOLS.values() for item in pool)
        for case in cases:
            candidates = re.search(r"(?:为：|列表|排列：|清单：)(.+?)(?:。请|中按|。输出|。保留)",
                                   case["instruction"])
            self.assertIsNotNone(candidates)
            candidate_items = candidates.group(1).split("、")
            self.assertEqual(len(candidate_items), 6)
            self.assertFalse(training_items.intersection(candidate_items))
            self.assertEqual(case["rule"]["value"].split(","),
                             candidate_items[:len(case["rule"]["value"].split(","))])

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
            prompts = [MODULE.normalize(MODULE.parse_instruction(record)[0])
                       for record in records]
            self.assertEqual(len(prompts), len(set(prompts)))

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
            train_records = train.read_text(encoding="utf-8").splitlines()
            validation_records = validation.read_text(encoding="utf-8").splitlines()
            records = train_records + validation_records
            self.assertEqual(report["total_records"], 50)
            self.assertFalse(any(MODULE.parse_instruction(record)[0] == held_out
                                 for record in records))
            train_prompts = {
                MODULE.normalize(MODULE.parse_instruction(record)[0])
                for record in train_records}
            validation_prompts = {
                MODULE.normalize(MODULE.parse_instruction(record)[0])
                for record in validation_records}
            self.assertFalse(train_prompts & validation_prompts)

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
