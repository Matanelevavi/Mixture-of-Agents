"""The asker: sends a prompt to the other machine and waits for its answer."""

from __future__ import annotations

import socket
import uuid
from dataclasses import dataclass
from typing import List, Optional

from .protocol import PROTOCOL_VERSION, ProtocolError, recv_message, send_message

CONNECT_TIMEOUT_S = 10.0


class PeerError(Exception):
    """The peer could not be reached or reported a failure."""


@dataclass
class PeerAnswer:
    text: str
    model: str
    node: str
    elapsed_ms: int
    tools: Optional[List[dict]] = None  # present when the peer answered as an agent
    warnings: Optional[List[str]] = None


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
        tools=_tool_list(reply.get("tools")),
        warnings=[w for w in reply.get("warnings") or [] if isinstance(w, str)] or None,
    )


def _tool_list(value) -> Optional[List[dict]]:
    """Keep the peer's tool record only if it has the expected shape."""
    if not isinstance(value, list):
        return None
    return [t for t in value if isinstance(t, dict) and isinstance(t.get("name"), str)]
