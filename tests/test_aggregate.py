"""The aggregator: one more model call that combines both answers."""

import contextlib
import io
import json
import unittest

from moa.aggregate import aggregate, build_prompt
from moa.cli import main
from moa.core import Answer
from moa.ollama_client import OllamaClient
from moa.peer_server import PeerServer

from .fake_ollama import FakeOllama

FILE_TOOL = [{"name": "files__read_file", "arguments": {"path": "todo.txt"}, "ok": True, "error": None}]


class PromptTests(unittest.TestCase):
    def test_prompt_has_the_question_both_answers_and_their_tool_records(self):
        a = Answer("local", "A", True, "qwen2.5:7b", "The file says buy milk.", tools=FILE_TOOL)
        b = Answer("peer", "B", True, "llama3.2:3b", "The file says buy bread.", tools=[])
        prompt = build_prompt("What is in todo.txt?", [a, b])
        self.assertIn("What is in todo.txt?", prompt)
        self.assertIn("buy milk", prompt)
        self.assertIn("buy bread", prompt)
        self.assertIn("files__read_file(path='todo.txt') ok", prompt)  # answer 1 really read the file
        self.assertIn("Tool record: none used", prompt)                # answer 2 did not
        self.assertIn("(unverified)", prompt)

    def test_plain_answers_are_described_as_such(self):
        a = Answer("local", "A", True, "m", "x")
        self.assertIn("no tools were available", build_prompt("q", [a, a]))

    def test_unusable_servers_are_part_of_the_record(self):
        a = Answer("local", "A", True, "m", "x", tools=[], warnings=["MCP server 'pc_b' unavailable: HTTP 401"])
        self.assertIn("HTTP 401", build_prompt("q", [a, a]))


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeOllama("A", ["qwen2.5:7b", "llama3.2:3b"])
        self.ollama = OllamaClient(self.fake.url)
        self.a = Answer("local", "A", True, "qwen2.5:7b", "answer one")
        self.b = Answer("peer", "B", True, "llama3.2:3b", "answer two")

    def tearDown(self):
        self.fake.close()

    def test_combines_with_the_chosen_model(self):
        combined = aggregate("q", [self.a, self.b], self.ollama, model="llama3.2:3b")
        self.assertTrue(combined.ok)
        self.assertEqual(combined.model, "llama3.2:3b")
        self.assertEqual(combined.source, "aggregate")
        self.assertIn("answer one", combined.text)  # the fake model echoes its prompt
        self.assertIn("answer two", combined.text)

    def test_skipped_when_one_answer_failed(self):
        failed = Answer("peer", "B", False, error="connection refused")
        self.assertIsNone(aggregate("q", [self.a, failed], self.ollama))

    def test_aggregator_failure_is_a_failed_answer(self):
        combined = aggregate("q", [self.a, self.b], self.ollama, model="missing-model")
        self.assertFalse(combined.ok)
        self.assertIn("missing-model", combined.error)


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.fake_a = FakeOllama("A", ["qwen2.5:7b"])
        self.fake_b = FakeOllama("B", ["llama3.2:3b"])
        self.server_b = PeerServer(OllamaClient(self.fake_b.url), host="127.0.0.1", port=0,
                                   node_name="machine-B", log=lambda _: None)
        self.server_b.start_in_background()

    def tearDown(self):
        self.server_b.shutdown()
        self.fake_a.close()
        self.fake_b.close()

    def _run(self, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["ask", "--ollama", self.fake_a.url, "--name", "machine-A",
                         "--peer", f"127.0.0.1:{self.server_b.address[1]}", *args])
        return code, out.getvalue()

    def test_json_has_both_answers_and_the_combined_one(self):
        code, out = self._run("--aggregate", "--json", "What is TCP?")
        result = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(result["aggregate"]["model"], "qwen2.5:7b")  # default: this machine's model
        self.assertIn("[A/qwen2.5:7b]", result["aggregate"]["text"])
        self.assertIn("[B/llama3.2:3b] What is TCP?", result["aggregate"]["text"])

    def test_without_the_flag_there_is_no_aggregate(self):
        _, out = self._run("--json", "What is TCP?")
        self.assertNotIn("aggregate", json.loads(out))

    def test_aggregate_without_peer_is_refused(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["ask", "--aggregate", "hello"])


if __name__ == "__main__":
    unittest.main()
