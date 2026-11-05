"""Agent mode: a model using the real files MCP server, through the agent loop.

The model is the scripted fake (see FakeOllama.chat); the MCP server is the
real moa.servers.files process, started by the agent over stdio.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from moa.agent import AgentConfig, AgentError, run_agent
from moa.core import ask_both
from moa.ollama_client import OllamaClient
from moa.peer import PeerServer

from .fake_ollama import FakeOllama

ROOT = Path(__file__).resolve().parent.parent


def _write_config(folder: Path, shared: Path, remote_allowed=(), max_calls=8, command="python") -> str:
    path = folder / "moa.json"
    path.write_text(json.dumps({
        "mcpServers": {"files": {"command": command,
                                 "args": ["-m", "moa.servers.files", "--root", str(shared)]}},
        "remoteAllowedTools": list(remote_allowed),
        "maxToolCalls": max_calls,
    }), encoding="utf-8")
    return str(path)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.shared = self.tmp / "shared"
        self.shared.mkdir()
        (self.shared / "todo.txt").write_text("buy milk", encoding="utf-8")
        (self.tmp / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
        self.fake = FakeOllama("A", ["llama3.2:3b"])
        self.ollama = OllamaClient(self.fake.url)
        self.config = AgentConfig.load(_write_config(self.tmp, self.shared, max_calls=3))

    def tearDown(self):
        self.fake.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_model_uses_a_tool_and_answers_from_its_result(self):
        result = run_agent("CALL files__read_file {\"path\": \"todo.txt\"}", self.ollama, self.config)
        self.assertIn("buy milk", result.text)
        self.assertEqual([(t.name, t.arguments, t.ok) for t in result.tools_used],
                         [("files__read_file", {"path": "todo.txt"}, True)])

    def test_tools_are_offered_to_the_model_with_their_schema(self):
        run_agent("hello", self.ollama, self.config)
        offered = {t["function"]["name"]: t["function"] for t in self.fake.chat_requests[0]["tools"]}
        self.assertEqual(set(offered), {"files__list_files", "files__read_file"})
        self.assertIn("path", offered["files__read_file"]["parameters"]["properties"])

    def test_plain_answer_when_the_model_needs_no_tool(self):
        result = run_agent("hello", self.ollama, self.config)
        self.assertEqual(result.tools_used, [])
        self.assertIn("hello", result.text)

    def test_path_outside_the_folder_is_refused(self):
        result = run_agent("CALL files__read_file {\"path\": \"../secret.txt\"}", self.ollama, self.config)
        self.assertFalse(result.tools_used[0].ok)
        self.assertIn("outside the shared folder", result.tools_used[0].error)
        self.assertNotIn("TOP SECRET", result.text)

    def test_absolute_path_is_refused(self):
        secret = json.dumps({"path": str(self.tmp / "secret.txt")})
        result = run_agent(f"CALL files__read_file {secret}", self.ollama, self.config)
        self.assertFalse(result.tools_used[0].ok)
        self.assertNotIn("TOP SECRET", result.text)

    def test_symlink_leading_out_of_the_folder_is_refused_and_hidden(self):
        try:
            (self.shared / "link.txt").symlink_to(self.tmp / "secret.txt")
        except (OSError, NotImplementedError):
            self.skipTest("cannot create symlinks here (Windows without the right privilege)")
        result = run_agent("CALL files__list_files {}\nCALL files__read_file {\"path\": \"link.txt\"}",
                           self.ollama, self.config)
        self.assertNotIn("link.txt", result.text.split("files__read_file:")[0])  # not listed
        self.assertFalse(result.tools_used[1].ok)                          # not readable
        self.assertNotIn("TOP SECRET", result.text)

    def test_tool_not_allowed_is_neither_offered_nor_run(self):
        result = run_agent("CALL files__read_file {\"path\": \"todo.txt\"}", self.ollama, self.config,
                           allowed_tools=frozenset({"files__list_files"}))
        offered = [t["function"]["name"] for t in self.fake.chat_requests[0]["tools"]]
        self.assertEqual(offered, ["files__list_files"])
        self.assertFalse(result.tools_used[0].ok)
        self.assertIn("not available", result.tools_used[0].error)
        self.assertNotIn("buy milk", result.text)

    def test_no_allowed_tools_means_a_plain_model(self):
        result = run_agent("CALL files__read_file {\"path\": \"todo.txt\"}", self.ollama, self.config,
                           allowed_tools=frozenset())
        self.assertNotIn("tools", self.fake.chat_requests[0])
        self.assertEqual(result.tools_used, [])
        self.assertNotIn("buy milk", result.text)

    def test_tool_call_limit_stops_a_looping_model(self):
        result = run_agent("LOOP", self.ollama, self.config)
        self.assertTrue(result.stopped_early)
        self.assertEqual(len(result.tools_used), self.config.max_tool_calls)
        self.assertIn("stopped", result.text)

    def test_server_that_cannot_start_is_a_warning_not_a_crash(self):
        bad = AgentConfig.load(_write_config(self.tmp, self.shared, command="no-such-program-xyz"))
        result = run_agent("hello", self.ollama, bad)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("'files' unavailable", result.warnings[0])
        self.assertIn("hello", result.text)  # the agent still answered

    def test_config_needs_exactly_one_of_command_and_url(self):
        path = self.tmp / "bad.json"
        for spec in ({"command": "python", "url": "http://x:1/mcp"}, {}):
            path.write_text(json.dumps({"mcpServers": {"files": spec}}), encoding="utf-8")
            with self.assertRaises(AgentError):
                AgentConfig.load(str(path))

    def test_config_rejects_server_names_that_would_break_tool_names(self):
        path = self.tmp / "bad.json"
        for name in ("my files", "a__b", ""):
            path.write_text(json.dumps({"mcpServers": {name: {"command": "python"}}}), encoding="utf-8")
            with self.assertRaises(AgentError):
                AgentConfig.load(str(path))


class TwoMachineAgentTests(unittest.TestCase):
    """Both machines are agents, each with its own folder."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.servers, self.fakes = [], []
        for side, text in (("a", "notes of machine A"), ("b", "notes of machine B")):
            folder = self.tmp / side
            (folder / "shared").mkdir(parents=True)
            (folder / "shared" / f"{side}.txt").write_text(text, encoding="utf-8")
        self.config_a = AgentConfig.load(_write_config(self.tmp / "a", self.tmp / "a" / "shared"))
        # B lets remote prompts list and read its files, nothing else.
        self.config_b = AgentConfig.load(_write_config(self.tmp / "b", self.tmp / "b" / "shared",
                                                       remote_allowed=["files__list_files", "files__read_file"]))
        self.ollama_a = self._fake("A")
        ollama_b = self._fake("B")
        server = PeerServer(OllamaClient(ollama_b.url), host="127.0.0.1", port=0,
                            node_name="machine-B", log=lambda _: None, agent=self.config_b)
        server.start_in_background()
        self.servers.append(server)
        self.port = server.address[1]

    def tearDown(self):
        for s in self.servers:
            s.shutdown()
        for f in self.fakes:
            f.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake(self, name):
        fake = FakeOllama(name, ["llama3.2:3b"])
        self.fakes.append(fake)
        return fake

    def test_each_side_answers_from_its_own_folder(self):
        local, remote = ask_both("CALL files__list_files {}", OllamaClient(self.ollama_a.url),
                                 "127.0.0.1", self.port, agent=self.config_a)
        self.assertTrue(local.ok, local.error)
        self.assertTrue(remote.ok, remote.error)
        self.assertIn("a.txt", local.text)
        self.assertNotIn("b.txt", local.text)
        self.assertIn("b.txt", remote.text)
        self.assertNotIn("a.txt", remote.text)
        self.assertEqual(remote.tools[0]["name"], "files__list_files")  # carried over the TCP protocol
        self.assertTrue(remote.tools[0]["ok"])

    def test_remote_prompt_cannot_use_a_tool_outside_the_allowlist(self):
        self.config_b.remote_allowed_tools = frozenset({"files__list_files"})
        _, remote = ask_both("CALL files__read_file {\"path\": \"b.txt\"}", OllamaClient(self.ollama_a.url),
                             "127.0.0.1", self.port, agent=self.config_a)
        self.assertTrue(remote.ok, remote.error)
        self.assertFalse(remote.tools[0]["ok"])
        self.assertNotIn("notes of machine B", remote.text)

    def test_cli_json_includes_the_tool_calls(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        config = str(self.tmp / "a" / "moa.json")
        cmd = [sys.executable, "-m", "moa", "ask", "--ollama", self.ollama_a.url, "--config", config,
               "--peer", f"127.0.0.1:{self.port}", "--json", "CALL files__list_files {}"]
        result = subprocess.run(cmd, capture_output=True, env=env, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        data = json.loads(result.stdout.decode("utf-8"))
        self.assertEqual(data["local"]["tools"][0]["name"], "files__list_files")
        self.assertEqual(data["peer"]["tools"][0]["name"], "files__list_files")


if __name__ == "__main__":
    unittest.main()
