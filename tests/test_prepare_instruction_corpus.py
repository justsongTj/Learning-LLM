#!/usr/bin/env python3

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "prepare_instruction_corpus.py"


class PrepareInstructionCorpusTest(unittest.TestCase):
    def test_normalizes_filters_deduplicates_and_excludes_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mixed.txt"
            evaluation = root / "evaluation.jsonl"
            reject = root / "reject.txt"
            train = root / "train.txt"
            validation = root / "validation.txt"
            stats = root / "stats.json"

            records = [
                "这是不应进入指令微调数据的百科正文。",
                "问题:什么是太阳系? 回答:太阳系由太阳和围绕它运行的天体组成。",
                "什么是太阳系? 回答:太阳系由太阳和围绕它运行的天体组成。",
                "问题:空答案 回答:",
                "请写500字文章。 回答:这是一个过短的答案。",
                "问题:敏感错误样本 回答:这条应该由人工规则删除。",
                "问题:为什么天空是蓝色的? 回答:因为大气对蓝光的瑞利散射更强。",
            ]
            records.extend(
                f"任务{i} 回答:这是编号{i}的完整有效答案。" for i in range(20))
            source.write_text("\n".join(records) + "\n", encoding="utf-8")
            evaluation.write_text(
                json.dumps({"instruction": "为什么天空是蓝色的?"}, ensure_ascii=False) + "\n",
                encoding="utf-8")
            reject.write_text("敏感错误样本\n", encoding="utf-8")

            subprocess.run([
                sys.executable, str(SCRIPT), "--input", str(source),
                "--train-output", str(train), "--validation-output", str(validation),
                "--stats-output", str(stats), "--evaluation-file", str(evaluation),
                "--reject-pattern-file", str(reject), "--validation-ratio", "0.5",
            ], check=True, capture_output=True, text=True, encoding="utf-8")

            output = train.read_text(encoding="utf-8") + validation.read_text(encoding="utf-8")
            report = json.loads(stats.read_text(encoding="utf-8"))
            self.assertTrue(output)
            self.assertTrue(all(line.startswith("指令:") for line in output.splitlines()))
            self.assertEqual(output.count("什么是太阳系?"), 1)
            self.assertNotIn("百科正文", output)
            self.assertNotIn("天空是蓝色", output)
            self.assertNotIn("敏感错误样本", output)
            self.assertEqual(report["counts"]["plain_text_excluded"], 1)
            self.assertEqual(report["counts"]["exact_duplicate"], 1)
            self.assertEqual(report["counts"]["requested_length_not_met"], 1)
            self.assertEqual(report["counts"]["evaluation_prompt_excluded"], 1)
            self.assertGreater(report["train_records"], 0)
            self.assertGreater(report["validation_records"], 0)


if __name__ == "__main__":
    unittest.main()
