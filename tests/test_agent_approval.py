"""Tools that change something run only after a person approves the exact call.

The files server (read-only tools) and tests/notes_server.py (one tool that
writes a file, not marked read-only) run as real MCP servers over stdio.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from moa.agent import AgentConfig, run_agent
from moa.ollama_client import OllamaClient

from .fake_ollama import FakeOllama

ADD = 'CALL notes__add_note {"text": "pay rent"}'


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        shared = self.tmp / "shared"
        shared.mkdir()
        (shared / "todo.txt").write_text("pay rent", encoding="utf-8")
        self.notes = self.tmp / "notes.txt"
        config = self.tmp / "moa.json"
        config.write_text(json.dumps({"mcpServers": {
            "files": {"command": sys.executable, "args": ["-m", "moa.servers.files", "--root", str(shared)]},
            "notes": {"command": sys.executable, "args": ["-m", "tests.notes_server", "--file", str(self.notes)]},
        }}), encoding="utf-8")
        self.config = AgentConfig.load(str(config))
        self.fake = FakeOllama("A", ["qwen2.5:7b"])
        self.ollama = OllamaClient(self.fake.url)
        self.asked = []

    def tearDown(self):
        self.fake.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _confirm(self, answer):
        def confirm(name, arguments, mentions):
            self.asked.append((name, arguments, mentions))
            return answer
        return confirm

    def test_approved_call_runs(self):
        result = run_agent(ADD, self.ollama, self.config, confirm=self._confirm(True))
        self.assertTrue(result.tools_used[0].ok)
        self.assertEqual(self.notes.read_text(encoding="utf-8"), "pay rent\n")
        self.assertEqual(self.asked[0][:2], ("notes__add_note", {"text": "pay rent"}))

    def test_declined_call_is_not_run_and_the_model_is_told(self):
        result = run_agent(ADD, self.ollama, self.config, confirm=self._confirm(False))
        self.assertFalse(self.notes.exists())
        self.assertFalse(result.tools_used[0].ok)
        self.assertIn("did not approve", result.tools_used[0].error)
        self.assertIn("did not approve", result.text)  # the fake model repeats what the tool said

    def test_without_anyone_to_ask_the_call_is_refused(self):
        result = run_agent(ADD, self.ollama, self.config)  # e.g. a prompt from the peer
        self.assertFalse(self.notes.exists())
        self.assertIn("no one is here to approve", result.tools_used[0].error)

    def test_read_only_tools_run_without_asking(self):
        result = run_agent('CALL files__read_file {"path": "todo.txt"}', self.ollama, self.config,
                           confirm=self._confirm(False))
        self.assertTrue(result.tools_used[0].ok)
        self.assertEqual(self.asked, [])

    def test_question_shows_earlier_results_that_mention_the_arguments(self):
        prompt = 'CALL files__read_file {"path": "todo.txt"}\n' + ADD  # both in one turn
        run_agent(prompt, self.ollama, self.config, confirm=self._confirm(False))
        self.assertEqual(self.asked[0][2], ["pay rent"])  # the line of todo.txt that matched


if __name__ == "__main__":
    unittest.main()
