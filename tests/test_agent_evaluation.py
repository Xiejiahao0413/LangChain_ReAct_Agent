import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.agent_evaluator import (  # noqa: E402
    DEFAULT_SAMPLE_PATH,
    evaluate_predictions,
    load_samples,
)


class AgentEvaluationTest(unittest.TestCase):
    def test_eval_samples_are_valid(self):
        samples = load_samples(DEFAULT_SAMPLE_PATH)
        self.assertGreaterEqual(len(samples), 30)
        self.assertLessEqual(len(samples), 50)
        self.assertEqual(len({sample.id for sample in samples}), len(samples))

    def test_metrics_can_be_computed_from_predictions(self):
        samples = load_samples(DEFAULT_SAMPLE_PATH)[:3]
        predictions = [
            {
                "id": sample.id,
                "query": sample.query,
                "answer": "根据知识库参考资料，漏扫问题可以通过排查传感器和清理主刷处理。",
                "tools_used": sample.expected_tools,
                "route": "normal",
                "error": None,
            }
            for sample in samples
        ]

        report = evaluate_predictions(samples, predictions)
        self.assertIn("summary", report)
        self.assertIn("per_sample", report)
        self.assertEqual(report["summary"]["sample_count"], 3)
        self.assertGreaterEqual(report["summary"]["tool_call_accuracy"], 0)


if __name__ == "__main__":
    unittest.main()
