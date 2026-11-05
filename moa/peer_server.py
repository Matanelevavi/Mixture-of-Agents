"""The answerer: waits for prompts from the other machine and answers them.

Each prompt runs on THIS machine's Ollama and the answer goes back over the
same TCP connection. With an MCP configuration it answers as an agent,
limited to the tools that the configuration allows for remote requests.

Ollama itself stays bound to localhost; only this small TCP service is
reachable over the network.
"""

from __future__ import annotations

import hmac
import socket
import socketserver
import threading
import time
from typing import Callable, Optional, Tuple

from .agent import AgentConfig, AgentError, AgentResult, run_agent
from .ollama_client import OllamaClient, OllamaError
from .protocol import (DEFAULT_PEER_PORT, PROTOCOL_VERSION, Message, ProtocolError,
                       recv_message, send_message)

REQUEST_READ_TIMEOUT_S = 30.0  # how long to wait for the prompt frame


class PeerServer:
    def __init__(
        self,
        ollama: OllamaClient,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PEER_PORT,
        model: Optional[str] = None,
        token: Optional[str] = None,
        node_name: Optional[str] = None,
        log: Callable[[str], None] = print,
        agent: Optional[AgentConfig] = None,
    ):
        self.ollama = ollama
        self.agent = agent
        self.model = model
        self.token = token
        self.node_name = node_name or socket.gethostname()
        self.log = log
        self._tcp = _ThreadingServer((host, port), self)
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> Tuple[str, int]:
        return self._tcp.server_address[:2]

    def serve_forever(self) -> None:
        self._tcp.serve_forever()

    def start_in_background(self) -> None:
        self._thread = threading.Thread(target=self._tcp.serve_forever, daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._tcp.shutdown()
        self._tcp.server_close()

    def handle_connection(self, sock: socket.socket, client: Tuple[str, int]) -> None:
        """One call per incoming TCP connection, each in its own thread."""
        who = f"{client[0]}:{client[1]}"
        request_id = None
        try:
            sock.settimeout(REQUEST_READ_TIMEOUT_S)
            request = recv_message(sock)
            request_id = request.get("id")
            prompt = self._validate(request)
            self.log(f"[serve] prompt from {who} ({len(prompt)} chars), running on local Ollama...")
            answer = self._answer(request, prompt)
            send_message(sock, answer)
            self.log(f"[serve] answered {who} with {answer['model']} in {answer['elapsed_ms']} ms")
        except (ProtocolError, OllamaError, AgentError, PermissionError) as exc:
            self.log(f"[serve] request from {who} failed: {exc}")
            self._try_send_error(sock, request_id, str(exc))
        except OSError as exc:  # client vanished, timeout while reading, etc.
            self.log(f"[serve] connection with {who} dropped: {exc}")

    def _answer(self, request: Message, prompt: str) -> Message:
        """Run the prompt and build the answer message.

        Model choice: the asker's explicit request, else this server's own
        --model, else the first installed model that differs from the asker's
        (avoid_model, sent with --diverse).
        """
        started = time.perf_counter()
        model = request.get("model") or self.model
        avoid = request.get("avoid_model")
        answer: Message = {"v": PROTOCOL_VERSION, "type": "answer", "id": request.get("id"),
                           "ok": True, "node": self.node_name}

        if self.agent is None:
            result = self.ollama.generate(prompt, model, avoid=avoid)
        else:
            # A prompt from the network may only use the tools listed in
            # remoteAllowedTools; nothing else is even offered to the model.
            result = run_agent(prompt, self.ollama, self.agent, model=model, avoid=avoid,
                               allowed_tools=self.agent.remote_allowed_tools)
            answer["tools"] = [t.to_dict() for t in result.tools_used]
            if result.warnings:
                answer["warnings"] = result.warnings
            self._log_agent_run(result)

        answer.update(model=result.model, answer=result.text,
                      elapsed_ms=int((time.perf_counter() - started) * 1000))
        return answer

    def _log_agent_run(self, result: AgentResult) -> None:
        for warning in result.warnings:
            self.log(f"[serve] warning: {warning}")
        for t in result.tools_used:
            outcome = "ok" if t.ok else f"refused/failed: {t.error}"
            self.log(f"[serve] tool {t.name}({t.arguments}) -> {outcome}")

    def _validate(self, request: Message) -> str:
        if request.get("v") != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol version {request.get('v')!r}")
        if request.get("type") != "prompt":
            raise ProtocolError(f"unexpected message type {request.get('type')!r}")
        if self.token is not None:
            # compare_digest takes the same time whether the first or last
            # character differs, so the token cannot be guessed by timing.
            given = str(request.get("token") or "")
            if not hmac.compare_digest(given.encode(), self.token.encode()):
                raise PermissionError("invalid or missing token")
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ProtocolError("prompt must be a non-empty string")
        for name in ("model", "avoid_model"):
            if request.get(name) is not None and not isinstance(request.get(name), str):
                raise ProtocolError(f"{name} must be a string or null")
        return prompt

    def _try_send_error(self, sock: socket.socket, request_id, error: str) -> None:
        try:
            send_message(sock, {"v": PROTOCOL_VERSION, "type": "answer", "id": request_id,
                                "ok": False, "node": self.node_name, "error": error})
        except OSError:
            pass  # the client is already gone; nothing more to do


class _ThreadingServer(socketserver.ThreadingTCPServer):
    """One thread per connection; each connection is handed to the PeerServer."""
    allow_reuse_address = True  # restart quickly on the same port
    daemon_threads = True       # do not block exit on open connections

    def __init__(self, address: Tuple[str, int], peer: PeerServer):
        self.peer = peer
        super().__init__(address, _Handler)


class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.server.peer.handle_connection(self.request, self.client_address)
