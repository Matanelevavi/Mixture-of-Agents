"""The files MCP server in HTTP mode, tested against a real running server.

Authentication lives in the HTTP layer, so it is tested with real HTTP
requests (raw and through the MCP client), never with an in-memory
connection that would bypass it.
"""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKEN = "test-token-0123456789abcdef"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
INITIALIZE = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {},
    "clientInfo": {"name": "test", "version": "0"}}}).encode()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_server(root, *args, token=TOKEN):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    env.pop("MOA_MCP_TOKEN", None)
    if token is not None:
        env["MOA_MCP_TOKEN"] = token
    return subprocess.Popen([sys.executable, "-m", "moa.servers.files", "--root", str(root), *args],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _wait_for_port(port, proc, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited: {proc.stderr.read().decode(errors='replace')}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("server did not start")


def _post(port, headers):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/mcp", data=INITIALIZE, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Accept": "application/json, text/event-stream", **headers})
    try:
        with OPENER.open(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


class HttpServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.public = cls.tmp / "public"
        cls.public.mkdir()
        (cls.public / "b.txt").write_text("hello from B", encoding="utf-8")
        (cls.tmp / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
        cls.port = _free_port()
        cls.proc = _run_server(cls.public, "--http", f"127.0.0.1:{cls.port}")
        _wait_for_port(cls.port, cls.proc)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.proc.stderr.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_request_without_token_is_401(self):
        self.assertEqual(_post(self.port, {}), 401)

    def test_request_with_wrong_token_is_401(self):
        self.assertEqual(_post(self.port, {"Authorization": "Bearer wrong-token-0123456789"}), 401)

    def test_request_with_correct_token_succeeds(self):
        self.assertEqual(_post(self.port, {"Authorization": f"Bearer {TOKEN}"}), 200)

    def test_unknown_host_header_is_rejected(self):  # DNS rebinding protection
        status = _post(self.port, {"Authorization": f"Bearer {TOKEN}", "Host": f"evil.example:{self.port}"})
        self.assertEqual(status, 421)

    def test_request_from_a_browser_page_is_rejected(self):
        status = _post(self.port, {"Authorization": f"Bearer {TOKEN}", "Origin": "http://evil.example"})
        self.assertEqual(status, 403)

    def test_full_mcp_session_with_token(self):
        import httpx2
        from mcp import Client
        from mcp.client.streamable_http import streamable_http_client

        async def session():
            async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {TOKEN}"}) as http:
                url = f"http://127.0.0.1:{self.port}/mcp"
                async with Client(streamable_http_client(url, http_client=http)) as client:
                    names = sorted(t.name for t in (await client.list_tools()).tools)
                    ok = await client.call_tool("read_file", {"path": "b.txt"})
                    escape = await client.call_tool("read_file", {"path": "../secret.txt"})
                    return names, ok.content[0].text, escape.is_error, escape.content[0].text

        names, text, escape_refused, escape_text = asyncio.run(session())
        self.assertEqual(names, ["list_files", "read_file"])
        self.assertEqual(text, "hello from B")
        self.assertTrue(escape_refused)
        self.assertNotIn("TOP SECRET", escape_text)

    def test_mcp_session_without_token_is_refused(self):
        from mcp import Client

        async def session():
            async with Client(f"http://127.0.0.1:{self.port}/mcp") as client:
                await client.list_tools()

        with self.assertRaises(BaseException):
            asyncio.run(session())


class HttpStartupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _refused(self, *args, token=TOKEN):
        proc = _run_server(self.tmp, *args, token=token)
        try:
            _, err = proc.communicate(timeout=20)
        finally:
            if proc.poll() is None:
                proc.kill()
        self.assertNotEqual(proc.returncode, 0)
        return err.decode(errors="replace")

    def test_refuses_to_start_without_a_token(self):
        self.assertIn("needs a token", self._refused("--http", f"127.0.0.1:{_free_port()}", token=None))

    def test_refuses_a_short_token(self):
        self.assertIn("at least", self._refused("--http", f"127.0.0.1:{_free_port()}", token="short"))

    def test_wildcard_address_needs_allow_host(self):
        self.assertIn("--allow-host", self._refused("--http", f"0.0.0.0:{_free_port()}"))

    def test_wildcard_address_accepts_only_the_allowed_host(self):
        port = _free_port()
        proc = _run_server(self.tmp, "--http", f"0.0.0.0:{port}", "--allow-host", "pc-b.test")
        try:
            _wait_for_port(port, proc)
            auth = {"Authorization": f"Bearer {TOKEN}"}
            self.assertEqual(_post(port, {**auth, "Host": f"pc-b.test:{port}"}), 200)
            self.assertEqual(_post(port, {**auth, "Host": f"other.test:{port}"}), 421)
        finally:
            proc.terminate()
            proc.wait(timeout=10)
            proc.stderr.close()


if __name__ == "__main__":
    unittest.main()
