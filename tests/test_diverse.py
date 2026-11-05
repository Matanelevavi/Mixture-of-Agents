"""--diverse: the other machine answers with a different model when it can."""

import json
import os
import subprocess
import sys
import unittest

from moa.core import ask_both
from moa.ollama_client import OllamaClient, same_model
from moa.peer_server import PeerServer

from .fake_ollama import FakeOllama

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class SameModelTests(unittest.TestCase):
    def test_missing_tag_means_latest(self):
        self.assertTrue(same_model("llama3.2", "llama3.2:latest"))
        self.assertTrue(same_model("Llama3.2:latest", "llama3.2"))

    def test_different_tags_are_different_models(self):
        self.assertFalse(same_model("llama3.2:1b", "llama3.2:3b"))
        self.assertFalse(same_model("llama3.2", "qwen2.5"))


class DiverseTests(unittest.TestCase):
    def setUp(self):
        self.servers, self.fakes = [], []
        # Machine A has only llama; it is also machine B's FIRST model.
        self.ollama_a = self._fake("A", ["llama3.2:latest"])

    def tearDown(self):
        for server in self.servers:
            server.shutdown()
        for fake in self.fakes:
            fake.close()

    def _fake(self, name, models):
        fake = FakeOllama(name, models)
        self.fakes.append(fake)
        return fake

    def _peer(self, models, model=None):
        fake = self._fake("B", models)
        server = PeerServer(OllamaClient(fake.url), host="127.0.0.1", port=0,
                            model=model, node_name="machine-B", log=lambda _: None)
        server.start_in_background()
        self.servers.append(server)
        return server.address[1]

    def _ask(self, port, **kwargs):
        return ask_both("hi", OllamaClient(self.ollama_a.url), "127.0.0.1", port, **kwargs)

    def test_without_diverse_both_use_the_first_model(self):
        port = self._peer(["llama3.2:latest", "qwen2.5:latest"])
        local, remote = self._ask(port)
        self.assertEqual(local.model, "llama3.2:latest")
        self.assertEqual(remote.model, "llama3.2:latest")

    def test_diverse_picks_a_different_model_on_the_peer(self):
        port = self._peer(["llama3.2:latest", "qwen2.5:latest"])
        local, remote = self._ask(port, diverse=True)
        self.assertEqual(local.model, "llama3.2:latest")
        self.assertEqual(remote.model, "qwen2.5:latest")

    def test_diverse_matches_names_without_a_tag(self):
        port = self._peer(["llama3.2:latest", "qwen2.5:latest"])
        local, remote = self._ask(port, diverse=True, local_model="llama3.2")
        self.assertEqual(remote.model, "qwen2.5:latest")

    def test_diverse_falls_back_to_the_same_model_when_no_other_exists(self):
        port = self._peer(["llama3.2:latest"])
        local, remote = self._ask(port, diverse=True)
        self.assertTrue(remote.ok, remote.error)
        self.assertEqual(remote.model, "llama3.2:latest")

    def test_explicit_peer_model_wins_over_diverse(self):
        port = self._peer(["llama3.2:latest", "qwen2.5:latest"])
        _, remote = self._ask(port, diverse=True, peer_model="llama3.2:latest")
        self.assertEqual(remote.model, "llama3.2:latest")

    def test_server_pinned_model_wins_over_diverse(self):
        port = self._peer(["llama3.2:latest", "qwen2.5:latest"], model="llama3.2:latest")
        _, remote = self._ask(port, diverse=True)
        self.assertEqual(remote.model, "llama3.2:latest")

    def test_diverse_with_local_ollama_down_still_gets_peer_answer(self):
        port = self._peer(["llama3.2:latest", "qwen2.5:latest"])
        self.ollama_a.close()
        self.fakes.remove(self.ollama_a)
        local, remote = self._ask(port, diverse=True)
        self.assertFalse(local.ok)
        self.assertTrue(remote.ok, remote.error)

    def test_cli_reports_when_both_used_the_same_model(self):
        port = self._peer(["llama3.2:latest"])
        env = dict(os.environ, PYTHONPATH=ROOT)
        cmd = [sys.executable, "-m", "moa", "ask", "--ollama", self.ollama_a.url,
               "--peer", f"127.0.0.1:{port}", "--diverse", "--json", "hi"]
        result = subprocess.run(cmd, capture_output=True, env=env, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        data = json.loads(result.stdout.decode("utf-8"))
        self.assertIn("no model other than", data["note"])


if __name__ == "__main__":
    unittest.main()
