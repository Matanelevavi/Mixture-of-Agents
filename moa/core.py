"""Run one prompt on both machines at the same time and collect both answers."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

from .ollama_client import OllamaClient, OllamaError
from .peer import PeerError, ask_peer


@dataclass
class Answer:
    source: str              # "local" or "peer"
    node: str                # machine name (or address) that produced it
    ok: bool
    model: Optional[str] = None
    text: Optional[str] = None
    error: Optional[str] = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _local(ollama: OllamaClient, prompt: str, model: Optional[str], node: str) -> Answer:
    started = time.perf_counter()
    try:
        result = ollama.generate(prompt, model)
        return Answer("local", node, True, result.model, result.text, elapsed_s=result.elapsed_s)
    except OllamaError as exc:
        return Answer("local", node, False, error=str(exc), elapsed_s=time.perf_counter() - started)


def _remote(host: str, port: int, prompt: str, model: Optional[str],
            token: Optional[str], timeout: float, avoid: Optional[str]) -> Answer:
    started = time.perf_counter()
    try:
        result = ask_peer(host, port, prompt, model=model, token=token,
                          timeout=timeout, avoid_model=avoid)
        return Answer("peer", result.node, True, result.model, result.text,
                      elapsed_s=time.perf_counter() - started)
    except PeerError as exc:
        return Answer("peer", f"{host}:{port}", False, error=str(exc),
                      elapsed_s=time.perf_counter() - started)


def ask_both(
    prompt: str,
    ollama: OllamaClient,
    peer_host: str,
    peer_port: int,
    local_model: Optional[str] = None,
    peer_model: Optional[str] = None,
    token: Optional[str] = None,
    timeout: float = 300.0,
    node_name: str = "local",
    diverse: bool = False,
) -> Tuple[Answer, Answer]:
    """Send `prompt` to the local Ollama and to the peer in parallel.

    Total wait time is max(local, peer) rather than local + peer. A failure
    on one side never hides the answer from the other side.

    diverse=True: if no peer model was requested, tell the peer which model
    answers locally, so it can pick a different one. This needs the local
    model name BEFORE the parallel step: one quick local /api/tags call.
    """
    avoid = None
    if diverse and not peer_model:
        try:
            local_model = ollama.resolve_model(local_model)
            avoid = local_model
        except OllamaError:
            pass  # local Ollama is down; _local below reports it as usual

    with ThreadPoolExecutor(max_workers=2) as pool:
        local_future = pool.submit(_local, ollama, prompt, local_model, node_name)
        peer_future = pool.submit(_remote, peer_host, peer_port, prompt,
                                  peer_model, token, timeout, avoid)
        return local_future.result(), peer_future.result()
