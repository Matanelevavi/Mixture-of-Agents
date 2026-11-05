"""Both machines are agents: agent to agent over the TCP protocol (channel 3).

Each machine runs the real files MCP server over stdio on its own folder;
the models are scripted fakes.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from moa.agent import AgentConfig
from moa.core import ask_both
from moa.ollama_client import OllamaClient
from moa.peer_server import PeerServer

from .fake_ollama import FakeOllama
from .test_agent import _write_config

ROOT = Path(__file__).resolve().parent.parent


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
