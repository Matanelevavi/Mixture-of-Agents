"""Wire protocol between the two peers: length-prefixed JSON over TCP.

Every message on the wire is one frame:

    +--------------------+------------------------------+
    | length (4 bytes,   | payload: UTF-8 encoded JSON  |
    | unsigned, big-end.)| object, exactly `length` B   |
    +--------------------+------------------------------+

TCP is a byte stream and has no notion of "messages": one send() may arrive
split across several recv() calls, and two sends may arrive in one recv().
The 4-byte length prefix tells the receiver exactly where a message ends.

Message types (all carry "v": PROTOCOL_VERSION and "type"):

    prompt  (asker -> answerer)
        {"v": 1, "type": "prompt", "id": "<uuid>", "prompt": "...",
         "model": "<name>" | null, "avoid_model": "<name>" | null,
         "token": "<shared secret>" | null}

        avoid_model (optional): when no model is requested, the answerer
        prefers an installed model other than this one, so the two answers
        come from different models. Older peers simply ignore the field.

    answer  (answerer -> asker), success
        {"v": 1, "type": "answer", "id": "<same uuid>", "ok": true,
         "node": "<answerer name>", "model": "<model used>",
         "answer": "...", "elapsed_ms": 1234}

        agent mode adds "tools": [{"name": "files__read_file", "arguments": {...},
        "ok": true, "error": null}, ...] - the tool calls behind the answer.
        It may also carry "warnings": ["MCP server ... unavailable: ..."].
        Older askers simply ignore both fields.

    answer  (answerer -> asker), failure
        {"v": 1, "type": "answer", "id": "<same uuid>", "ok": false,
         "node": "<answerer name>", "error": "<human readable reason>"}
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict

PROTOCOL_VERSION = 1

# Upper bound for one frame. Protects the receiver from allocating an absurd
# amount of memory because of a corrupt or malicious length prefix.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024  # 16 MiB

# "!I" = network byte order (big-endian), unsigned 32-bit integer.
_HEADER = struct.Struct("!I")

Message = Dict[str, Any]


class ProtocolError(Exception):
    """The peer sent something that does not follow the protocol."""


def send_message(sock: socket.socket, message: Message) -> None:
    """Serialize `message` to JSON and send it as a single frame."""
    payload = json.dumps(message, ensure_ascii=False).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ProtocolError(
            f"message is {len(payload)} bytes, limit is {MAX_MESSAGE_BYTES}"
        )
    # sendall() keeps calling send() until every byte is written.
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    """Read exactly `size` bytes, looping over partial recv() results."""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(min(remaining, 64 * 1024))
        if not chunk:  # recv() returns b"" when the other side closed.
            raise ProtocolError("connection closed before the full message arrived")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> Message:
    """Receive one frame and return the decoded JSON object."""
    (length,) = _HEADER.unpack(_recv_exact(sock, _HEADER.size))
    if length > MAX_MESSAGE_BYTES:
        raise ProtocolError(f"incoming message of {length} bytes exceeds the limit")
    payload = _recv_exact(sock, length)
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"payload is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(message, dict):
        raise ProtocolError("payload must be a JSON object")
    return message
