import unittest

from financial_qa.agent.gigachat_agent_loop import GigaChatAgentLoop


class TestGigaChatAgentLoopArithmetic(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = GigaChatAgentLoop.__new__(GigaChatAgentLoop)

    def test_safe_eval_expression_basic_ops(self) -> None:
        self.assertAlmostEqual(self.loop._safe_eval_expression("1 + 2"), 3.0)
        self.assertAlmostEqual(self.loop._safe_eval_expression("10 - 4"), 6.0)
        self.assertAlmostEqual(self.loop._safe_eval_expression("3 * 5"), 15.0)
        self.assertAlmostEqual(self.loop._safe_eval_expression("8 / 4"), 2.0)

    def test_safe_eval_expression_unary(self) -> None:
        self.assertAlmostEqual(self.loop._safe_eval_expression("-7"), -7.0)
        self.assertAlmostEqual(self.loop._safe_eval_expression("+9"), 9.0)

    def test_safe_eval_expression_rejects_unsupported(self) -> None:
        with self.assertRaises(ValueError):
            self.loop._safe_eval_expression("2 ** 3")
        with self.assertRaises(ValueError):
            self.loop._safe_eval_expression("__import__('os').system('echo nope')")
        with self.assertRaises(ValueError):
            self.loop._safe_eval_expression("")

    def test_parse_text_function_call_calculate(self) -> None:
        call = self.loop._parse_text_function_call("calculate(\"1 + 2\")", "fallback")
        self.assertIsNotNone(call)
        self.assertEqual(call["name"], "calculate")
        self.assertEqual(call["arguments"]["expression"], "1 + 2")


if __name__ == "__main__":
    unittest.main()
