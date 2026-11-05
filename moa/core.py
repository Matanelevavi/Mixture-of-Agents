"""Run one prompt on both machines at the same time and collect both answers."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

from .agent import AgentConfig, AgentError, run_agent
from .ollama_client import OllamaClient, OllamaError
from .peer_client import PeerError, ask_peer


@dataclass
class Answer:
    source: str              # "local" or "peer"
    node: str                # machine name (or address) that produced it
    ok: bool
    model: Optional[str] = None
    text: Optional[str] = None
    error: Optional[str] = None
    elapsed_s: float = 0.0
    tools: Optional[List[dict]] = None  # agent mode only: the tool calls that produced the answer
    warnings: Optional[List[str]] = None  # agent mode only: MCP servers that could not be used

    def to_dict(self) -> dict:
        return asdict(self)


def _local(ollama: OllamaClient, prompt: str, model: Optional[str], node: str,
           agent: Optional[AgentConfig]) -> Answer:
    started = time.perf_counter()
    try:
        if agent is None:
            result = ollama.generate(prompt, model)
            return Answer("local", node, True, result.model, result.text, elapsed_s=result.elapsed_s)
        # The local user may use every configured tool (allowed_tools=None).
        result = run_agent(prompt, ollama, agent, model=model)
        return Answer("local", node, True, result.model, result.text,
                      elapsed_s=time.perf_counter() - started,
                      tools=[t.to_dict() for t in result.tools_used],
                      warnings=result.warnings or None)
    except (OllamaError, AgentError) as exc:
        return Answer("local", node, False, error=str(exc), elapsed_s=time.perf_counter() - started)


def _remote(host: str, port: int, prompt: str, model: Optional[str],
            token: Optional[str], timeout: float, avoid: Optional[str]) -> Answer:
    started = time.perf_counter()
    try:
        result = ask_peer(host, port, prompt, model=model, token=token,
                          timeout=timeout, avoid_model=avoid)
        return Answer("peer", result.node, True, result.model, result.text,
                      elapsed_s=time.perf_counter() - started, tools=result.tools,
                      warnings=result.warnings)
    except PeerError as exc:
        return Answer("peer", f"{host}:{port}", False, error=str(exc),
                      elapsed_s=time.perf_counter() - started)


def ask_local(prompt: str, ollama: OllamaClient, model: Optional[str] = None,
              node_name: str = "local", agent: Optional[AgentConfig] = None) -> Answer:
    """Only this machine answers (no peer). With an agent config it may still
    use remote MCP servers, so another machine's tools can be used directly."""
    return _local(ollama, prompt, model, node_name, agent)


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
    agent: Optional[AgentConfig] = None,
) -> Tuple[Answer, Answer]:
    """Send `prompt` to the local Ollama and to the peer in parallel.

    Total wait time is max(local, peer) rather than local + peer. A failure
    on one side never hides the answer from the other side.

    diverse=True: if no peer model was requested, tell the peer which model
    answers locally, so it can pick a different one. This needs the local
    model name BEFORE the parallel step: one quick local /api/tags call.

    agent: with an MCP configuration, the local answer comes from an agent
    that can use the configured tools. Whether the peer uses tools is decided
    by the peer's own configuration.
    """
    avoid = None
    if diverse and not peer_model:
        try:
            local_model = ollama.resolve_model(local_model)
            avoid = local_model
        except OllamaError:
            pass  # local Ollama is down; _local below reports it as usual

    with ThreadPoolExecutor(max_workers=2) as pool:
        local_future = pool.submit(_local, ollama, prompt, local_model, node_name, agent)
        peer_future = pool.submit(_remote, peer_host, peer_port, prompt,
                                  peer_model, token, timeout, avoid)
        return local_future.result(), peer_future.result()
