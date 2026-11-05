"""Full flow on one computer: two fake Ollamas stand in for the two machines."""

import json
import os
import socket
import subprocess
import sys
import time
import unittest

from moa.core import ask_both
from moa.ollama_client import OllamaClient
from moa.peer_server import PeerServer

from .fake_ollama import FakeOllama

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.ollama_a = FakeOllama("A", ["llama3.2"])
        self.ollama_b = FakeOllama("B", ["qwen2.5"])
        self.server_b = self._start_peer(self.ollama_b, token=None)

    def tearDown(self):
        self.server_b.shutdown()
        self.ollama_a.close()
        self.ollama_b.close()

    def _start_peer(self, fake, token):
        server = PeerServer(OllamaClient(fake.url), host="127.0.0.1", port=0,
                            token=token, node_name="machine-B", log=lambda _: None)
        server.start_in_background()
        return server

    def _ask(self, port=None, **kwargs):
        port = port or self.server_b.address[1]
        return ask_both("What is TCP?", OllamaClient(self.ollama_a.url),
                        "127.0.0.1", port, node_name="machine-A", **kwargs)

    def test_returns_both_answers(self):
        local, remote = self._ask()
        self.assertTrue(local.ok, local.error)
        self.assertTrue(remote.ok, remote.error)
        self.assertEqual(local.text, "[A/llama3.2] What is TCP?")
        self.assertEqual(remote.text, "[B/qwen2.5] What is TCP?")
        self.assertEqual(remote.node, "machine-B")
        self.assertEqual(remote.model, "qwen2.5")

    def test_both_sides_run_in_parallel(self):
        self.ollama_a.delay_s = self.ollama_b.delay_s = 0.6
        started = time.perf_counter()
        local, remote = self._ask()
        elapsed = time.perf_counter() - started
        self.assertTrue(local.ok and remote.ok)
        self.assertLess(elapsed, 1.1, "answers were fetched one after the other")

    def test_peer_down_still_returns_local_answer(self):
        local, remote = self._ask(port=_free_port())
        self.assertTrue(local.ok)
        self.assertFalse(remote.ok)
        self.assertIn("refused", remote.error)

    def test_local_ollama_down_still_returns_peer_answer(self):
        self.ollama_a.close()
        local, remote = self._ask()
        self.assertFalse(local.ok)
        self.assertIn("cannot reach Ollama", local.error)
        self.assertTrue(remote.ok)

    def test_unknown_model_on_peer_is_reported(self):
        local, remote = self._ask(peer_model="does-not-exist")
        self.assertTrue(local.ok)
        self.assertFalse(remote.ok)
        self.assertIn("not found", remote.error)

    def test_token_is_enforced(self):
        secured = self._start_peer(self.ollama_b, token="s3cret")
        try:
            port = secured.address[1]
            _, wrong = self._ask(port=port, token="guess")
            _, missing = self._ask(port=port)
            _, right = self._ask(port=port, token="s3cret")
        finally:
            secured.shutdown()
        self.assertFalse(wrong.ok)
        self.assertIn("token", wrong.error)
        self.assertFalse(missing.ok)
        self.assertTrue(right.ok, right.error)

    def test_cli_json_output_is_utf8_even_with_a_legacy_default_encoding(self):
        # On Hebrew Windows the default encoding of a pipe is cp1255, not UTF-8.
        # PYTHONIOENCODING reproduces that on any system.
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="cp1255")
        cmd = [sys.executable, "-m", "moa", "ask", "--ollama", self.ollama_a.url,
               "--peer", f"127.0.0.1:{self.server_b.address[1]}", "--json", "שלום"]
        result = subprocess.run(cmd, capture_output=True, env=env, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        data = json.loads(result.stdout.decode("utf-8"))  # must be valid UTF-8
        self.assertEqual(data["local"]["text"], "[A/llama3.2] שלום")

    def test_cli_json_output(self):
        env = dict(os.environ, PYTHONPATH=ROOT)
        cmd = [sys.executable, "-m", "moa", "ask", "--ollama", self.ollama_a.url,
               "--peer", f"127.0.0.1:{self.server_b.address[1]}", "--json", "שלום", "עולם"]
        result = subprocess.run(cmd, capture_output=True, env=env, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        data = json.loads(result.stdout.decode("utf-8"))
        self.assertEqual(data["prompt"], "שלום עולם")
        self.assertEqual(data["local"]["text"], "[A/llama3.2] שלום עולם")
        self.assertEqual(data["peer"]["text"], "[B/qwen2.5] שלום עולם")


if __name__ == "__main__":
    unittest.main()
