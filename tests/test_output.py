"""How timing is shown in the terminal."""

import contextlib
import io
import unittest
from unittest import mock

from moa.core import Answer
from moa.ollama_client import _stats
from moa.output import confirm_in_terminal, describe_timing


class TimingOutputTests(unittest.TestCase):
    def test_timing_line_names_each_stage(self):
        line = describe_timing({"setup_s": 1.24, "model_s": [70.1, 80.34], "tools_s": 1.9})
        self.assertEqual(line, "MCP setup 1.2s | model calls (2): 70.1s, 80.3s | tools 1.9s")

    def test_ollama_numbers_are_shown_for_a_plain_model_call(self):
        line = describe_timing({"load_s": 0.1, "prompt_tokens": 700, "prompt_s": 40.2,
                                "output_tokens": 190, "output_s": 250.5})
        self.assertEqual(line, "Ollama reports: model load 0.1s | prompt 700 tokens read in 40.2s | "
                               "answer 190 tokens written in 250.5s")

    def test_ollama_reply_is_turned_into_seconds_and_token_counts(self):
        reply = {"load_duration": 1_500_000_000, "prompt_eval_count": 700, "prompt_eval_duration": 40_000_000_000,
                 "eval_count": 190, "eval_duration": 250_000_000_000}
        self.assertEqual(_stats(reply), {"load_s": 1.5, "prompt_s": 40.0, "output_s": 250.0,
                                         "prompt_tokens": 700, "output_tokens": 190})
        self.assertIsNone(_stats({"response": "no numbers here"}))

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

    def test_control_characters_from_tool_results_cannot_reach_the_terminal(self):
        hostile = "id=abc1234 | subject: \x1b[2K\x1b[1Ahello\rfake line\x07"
        err = io.StringIO()
        with contextlib.redirect_stderr(err), mock.patch("builtins.input", return_value="n"):
            confirm_in_terminal("gmail__trash_email", {"message_id": "abc1234"}, [hostile], "request\x1b[0m")
        shown = err.getvalue()
        for forbidden in ("\x1b", "\r", "\x07"):
            self.assertNotIn(forbidden, shown)
        self.assertIn("hello", shown)  # the text stays readable

    def test_long_lines_in_the_question_are_cut(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), mock.patch("builtins.input", return_value="n"):
            confirm_in_terminal("t", {"message_id": "abc1234"}, ["x" * 5000], "")
        self.assertLess(len(err.getvalue()), 1000)

    def test_answer_without_timing_is_still_valid(self):
        self.assertIsNone(Answer("local", "A", True).timing)


if __name__ == "__main__":
    unittest.main()
