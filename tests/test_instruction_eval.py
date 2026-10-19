#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_instruction_eval.py"
SPEC = importlib.util.spec_from_file_location("run_instruction_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class InstructionEvaluationTest(unittest.TestCase):
    def test_extracts_normalized_prompt_echo(self):
        prompt = "指令:根据已知信息回答：小吴有一台电脑。 回答:"
        generated = "指令:根据已知信息回答:小吴有一台电脑。 回答: 无法确定。"
        self.assertEqual(MODULE.extract_answer(generated, prompt), "无法确定。")

    def test_exact_number_rejects_keyword_collision(self):
        case = {"rule": {"type": "exact_number", "value": 35}}
        self.assertTrue(MODULE.evaluate_answer(case, "35")[2])
        self.assertFalse(MODULE.evaluate_answer(case, "35和40都可能")[2])

    def test_delimited_list_requires_exact_item_count(self):
        case = {"rule": {"type": "delimited_list", "count": 3,
                         "separator": "[,，]"}}
        self.assertTrue(MODULE.evaluate_answer(case, "C,Python,Rust")[2])
        self.assertFalse(MODULE.evaluate_answer(case, "我知道，很多，语言，类型")[2])

    def test_contains_any_accepts_uncertainty_variants(self):
        case = {"rule": {"type": "contains_any",
                         "values": ["无法确定", "信息不足"]}}
        self.assertTrue(MODULE.evaluate_answer(case, "根据现有信息无法确定作者。")[2])


if __name__ == "__main__":
    unittest.main()
