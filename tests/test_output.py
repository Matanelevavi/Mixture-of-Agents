"""How timing is shown in the terminal."""

import contextlib
import io
import unittest
from unittest import mock

from moa.core import Answer
from moa.output import confirm_in_terminal, describe_timing


class TimingOutputTests(unittest.TestCase):
    def test_timing_line_names_each_stage(self):
        line = describe_timing({"setup_s": 1.24, "model_s": [70.1, 80.34], "tools_s": 1.9})
        self.assertEqual(line, "MCP setup 1.2s | model calls (2): 70.1s, 80.3s | tools 1.9s")

    def test_approval_wait_is_shown_only_when_there_was_one(self):
        timing = {"setup_s": 1, "model_s": [2.0], "tools_s": 1, "approval_s": 98.8}
        self.assertTrue(describe_timing(timing).endswith("waiting for your approval 98.8s"))
        self.assertNotIn("approval", describe_timing({**timing, "approval_s": 0}))

    def test_approval_question_shows_the_request_and_accepts_only_yes(self):
        for typed, expected in [("y", True), ("yes", True), ("n", False), ("", False), ("maybe", False)]:
            err = io.StringIO()
            with contextlib.redirect_stderr(err), mock.patch("builtins.input", return_value=typed):
                allowed = confirm_in_terminal("gmail__trash_email", {"message_id": "abc1234"},
                                              ["id=abc1234 | subject: To delete"], "Summarize my emails")
            self.assertEqual(allowed, expected, typed)
        self.assertIn("your request was: Summarize my emails", err.getvalue())
        self.assertIn("seen earlier: id=abc1234", err.getvalue())

    def test_answer_without_timing_is_still_valid(self):
        self.assertIsNone(Answer("local", "A", True).timing)


if __name__ == "__main__":
    unittest.main()
