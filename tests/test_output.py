"""How timing is shown in the terminal."""

import unittest

from moa.core import Answer
from moa.output import describe_timing


class TimingOutputTests(unittest.TestCase):
    def test_timing_line_names_each_stage(self):
        line = describe_timing({"setup_s": 1.24, "model_s": [70.1, 80.34], "tools_s": 1.9})
        self.assertEqual(line, "MCP setup 1.2s | model calls (2): 70.1s, 80.3s | tools 1.9s")

    def test_answer_without_timing_is_still_valid(self):
        self.assertIsNone(Answer("local", "A", True).timing)


if __name__ == "__main__":
    unittest.main()
