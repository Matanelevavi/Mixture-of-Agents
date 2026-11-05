"""Agent with a local MCP server (stdio) and a remote one (Streamable HTTP).

The remote files server runs as a real HTTP server with token auth; the
model is the scripted fake. This is the "connector over the network" mode:
machine A's model calls tools that machine B exposes, with no model on B.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from moa.agent import AgentConfig, run_agent
from moa.ollama_client import OllamaClient

from .fake_ollama import FakeOllama
from .test_files_http import TOKEN, _free_port, _run_server, _wait_for_port

ROOT = Path(__file__).resolve().parent.parent
TOKEN_ENV = "MOA_TEST_REMOTE_TOKEN"


class RemoteMcpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        (cls.tmp / "a_shared").mkdir()
        (cls.tmp / "a_shared" / "a.txt").write_text("notes of machine A", encoding="utf-8")
        (cls.tmp / "b_public").mkdir()
        (cls.tmp / "b_public" / "b.txt").write_text("hello from B", encoding="utf-8")
        (cls.tmp / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
        cls.port = _free_port()
        cls.server = _run_server(cls.tmp / "b_public", "--http", f"127.0.0.1:{cls.port}")
        _wait_for_port(cls.port, cls.server)

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(timeout=10)
        cls.server.stderr.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.fake = FakeOllama("A", ["llama3.2:3b"])
        self.ollama = OllamaClient(self.fake.url)
        os.environ[TOKEN_ENV] = TOKEN

    def tearDown(self):
        os.environ.pop(TOKEN_ENV, None)
        self.fake.close()

    def _config(self, url=None, token_env=TOKEN_ENV) -> str:
        path = self.tmp / "moa.json"
        remote = {"url": url or f"http://127.0.0.1:{self.port}/mcp"}
        if token_env:
            remote["tokenEnv"] = token_env
        path.write_text(json.dumps({"mcpServers": {
            "my_files": {"command": "python", "args": ["-m", "moa.servers.files", "--root", "a_shared"]},
            "pc_b_files": remote,
        }}), encoding="utf-8")
        return str(path)

    def _run(self, prompt, **config_kwargs):
        return run_agent(prompt, self.ollama, AgentConfig.load(self._config(**config_kwargs)))

    def _offered(self):
        return {t["function"]["name"]: t["function"]["description"]
                for t in self.fake.chat_requests[0].get("tools", [])}

    def test_both_servers_are_offered_under_distinct_names(self):
        self._run("hello")
        offered = self._offered()
        self.assertEqual(set(offered), {"my_files__list_files", "my_files__read_file",
                                        "pc_b_files__list_files", "pc_b_files__read_file"})
        self.assertIn("remote server pc_b_files", offered["pc_b_files__read_file"])
        self.assertIn("local server my_files", offered["my_files__read_file"])

    def test_one_answer_uses_a_local_and_a_remote_tool(self):
        result = self._run('CALL my_files__read_file {"path": "a.txt"}\n'
                           'CALL pc_b_files__read_file {"path": "b.txt"}')
        self.assertEqual([(t.name, t.ok) for t in result.tools_used],
                         [("my_files__read_file", True), ("pc_b_files__read_file", True)])
        self.assertIn("notes of machine A", result.text)
        self.assertIn("hello from B", result.text)
        self.assertEqual(result.warnings, [])

    def test_remote_server_still_refuses_paths_outside_its_folder(self):
        result = self._run('CALL pc_b_files__read_file {"path": "../secret.txt"}')
        self.assertFalse(result.tools_used[0].ok)
        self.assertIn("outside the shared folder", result.tools_used[0].error)
        self.assertNotIn("TOP SECRET", result.text)

    def test_wrong_token_is_a_clear_warning_and_local_tools_still_work(self):
        os.environ[TOKEN_ENV] = "wrong-token-0123456789"
        result = self._run('CALL my_files__read_file {"path": "a.txt"}')
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("refused the token (HTTP 401)", result.warnings[0])
        self.assertEqual(set(self._offered()), {"my_files__list_files", "my_files__read_file"})
        self.assertIn("notes of machine A", result.text)
        # The model is told that the server is missing, not that the file is.
        system = " ".join(m["content"] for m in self.fake.chat_requests[0]["messages"] if m["role"] == "system")
        self.assertIn("'pc_b_files' unavailable", system)
        self.assertIn("refused the token (HTTP 401)", system)
        self.assertIn("do not conclude that a file does not exist", system)

    def test_missing_token_variable_means_no_connection(self):
        os.environ.pop(TOKEN_ENV)
        result = self._run("hello")
        self.assertIn(f"{TOKEN_ENV} is not set", result.warnings[0])
        self.assertNotIn("pc_b_files__read_file", self._offered())

    def test_unreachable_server_is_a_warning(self):
        result = self._run("hello", url=f"http://127.0.0.1:{_free_port()}/mcp")
        self.assertIn("cannot reach", result.warnings[0])
        self.assertIn("hello", result.text)  # still answered, with the local tools only

    def test_remote_tools_are_offered_only_if_allowed(self):
        config = AgentConfig.load(self._config())
        result = run_agent('CALL pc_b_files__read_file {"path": "b.txt"}', self.ollama, config,
                           allowed_tools=frozenset({"my_files__list_files"}))
        self.assertEqual(set(self._offered()), {"my_files__list_files"})
        self.assertFalse(result.tools_used[0].ok)
        self.assertNotIn("hello from B", result.text)

    def test_cli_without_peer_uses_the_remote_server_directly(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        cmd = [sys.executable, "-m", "moa", "ask", "--ollama", self.fake.url, "--config", self._config(),
               "--json", 'CALL pc_b_files__read_file {"path": "b.txt"}']
        result = subprocess.run(cmd, capture_output=True, env=env, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        data = json.loads(result.stdout.decode("utf-8"))
        self.assertNotIn("peer", data)
        self.assertEqual(data["local"]["tools"][0]["name"], "pc_b_files__read_file")
        self.assertTrue(data["local"]["tools"][0]["ok"])
        self.assertIn("hello from B", data["local"]["text"])


if __name__ == "__main__":
    unittest.main()
