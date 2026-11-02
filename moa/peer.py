"""The two roles a peer can play.

* PeerServer - "answerer": waits for prompts from the other machine, runs
  them on THIS machine's Ollama and sends the answer back.
* ask_peer   - "asker": sends a prompt to the other machine and waits for
  its answer.

Both machines run the same program; which role is active depends on the
command line (see cli.py). Ollama itself stays bound to localhost on both
machines - only this small TCP service is reachable over the network.
"""

from __future__ import annotations

import hmac
import socket
import socketserver
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .ollama_client import OllamaClient, OllamaError
from .protocol import PROTOCOL_VERSION, ProtocolError, recv_message, send_message

DEFAULT_PEER_PORT = 5005
CONNECT_TIMEOUT_S = 10.0
REQUEST_READ_TIMEOUT_S = 30.0  # how long the answerer waits for the prompt frame


class PeerError(Exception):
    """The peer could not be reached or reported a failure."""


# --------------------------------------------------------------------------
# Answerer side
# --------------------------------------------------------------------------

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
    ):
        self.ollama = ollama
        self.model = model
        self.token = token
        self.node_name = node_name or socket.gethostname()
        self.log = log

        server = self  # captured by the handler class below

        class _Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                server._handle_connection(self.request, self.client_address)

        class _TCPServer(socketserver.ThreadingTCPServer):
            allow_reuse_address = True  # restart quickly on the same port
            daemon_threads = True       # do not block exit on open connections

        self._tcp = _TCPServer((host, port), _Handler)
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

    # One call per incoming TCP connection (each in its own thread).
    def _handle_connection(self, sock: socket.socket, client: Tuple[str, int]) -> None:
        who = f"{client[0]}:{client[1]}"
        request_id = None
        try:
            sock.settimeout(REQUEST_READ_TIMEOUT_S)
            request = recv_message(sock)
            request_id = request.get("id")
            prompt = self._validate(request)
            self.log(f"[serve] prompt from {who} ({len(prompt)} chars), running on local Ollama...")

            # Model choice on this side: the asker's explicit request, else this
            # server's own --model, else the first installed model that differs
            # from the asker's model (avoid_model, sent with --diverse).
            started = time.perf_counter()
            result = self.ollama.generate(
                prompt,
                request.get("model") or self.model,
                avoid=request.get("avoid_model"),
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            send_message(sock, {
                "v": PROTOCOL_VERSION, "type": "answer", "id": request_id, "ok": True,
                "node": self.node_name, "model": result.model,
                "answer": result.text, "elapsed_ms": elapsed_ms,
            })
            self.log(f"[serve] answered {who} with {result.model} in {elapsed_ms} ms")
        except (ProtocolError, OllamaError, PermissionError) as exc:
            self.log(f"[serve] request from {who} failed: {exc}")
            self._try_send_error(sock, request_id, str(exc))
        except OSError as exc:  # client vanished, timeout while reading, etc.
            self.log(f"[serve] connection with {who} dropped: {exc}")

    def _validate(self, request: dict) -> str:
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
        for field in ("model", "avoid_model"):
            if request.get(field) is not None and not isinstance(request.get(field), str):
                raise ProtocolError(f"{field} must be a string or null")
        return prompt

    def _try_send_error(self, sock: socket.socket, request_id, error: str) -> None:
        try:
            send_message(sock, {
                "v": PROTOCOL_VERSION, "type": "answer", "id": request_id,
                "ok": False, "node": self.node_name, "error": error,
            })
        except OSError:
            pass  # the client is already gone; nothing more to do


# --------------------------------------------------------------------------
# Asker side
# --------------------------------------------------------------------------

@dataclass
class PeerAnswer:
    text: str
    model: str
    node: str
    elapsed_ms: int


def ask_peer(
    host: str,
    port: int,
    prompt: str,
    model: Optional[str] = None,
    token: Optional[str] = None,
    timeout: float = 300.0,
    avoid_model: Optional[str] = None,
) -> PeerAnswer:
    request_id = str(uuid.uuid4())
    request = {
        "v": PROTOCOL_VERSION, "type": "prompt", "id": request_id,
        "prompt": prompt, "model": model, "avoid_model": avoid_model, "token": token,
    }
    try:
        with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT_S) as sock:
            sock.settimeout(timeout)  # generation on the peer may take a while
            send_message(sock, request)
            reply = recv_message(sock)
    except ConnectionRefusedError as exc:
        raise PeerError(f"{host}:{port} refused the connection. Is `python -m moa serve` running there?") from exc
    except (socket.timeout, TimeoutError) as exc:
        raise PeerError(f"no answer from {host}:{port} within the timeout") from exc
    except ProtocolError as exc:
        raise PeerError(f"bad reply from {host}:{port}: {exc}") from exc
    except OSError as exc:
        raise PeerError(f"cannot reach {host}:{port}: {exc}") from exc

    if reply.get("id") != request_id:
        raise PeerError("reply id does not match the request id")
    if not reply.get("ok"):
        raise PeerError(f"peer {reply.get('node', host)} reported: {reply.get('error')}")
    return PeerAnswer(
        text=str(reply.get("answer", "")),
        model=str(reply.get("model", "?")),
        node=str(reply.get("node", host)),
        elapsed_ms=int(reply.get("elapsed_ms", 0)),
    )
