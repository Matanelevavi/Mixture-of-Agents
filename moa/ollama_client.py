"""Minimal client for the local Ollama REST API (standard library only).

Ollama listens on http://127.0.0.1:11434 by default. We use three endpoints:
    GET  /api/tags      -> list of installed models
    POST /api/generate  -> run one prompt ("stream": false = one JSON reply)
    POST /api/chat      -> a conversation, optionally with tools (agent mode)
"""

from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

# Ollama is a local/LAN service. An empty ProxyHandler makes sure an
# HTTP(S)_PROXY environment variable never routes these requests elsewhere.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class OllamaError(Exception):
    """Ollama could not be reached or returned an error."""


def same_model(a: str, b: str) -> bool:
    """Compare model names the way Ollama does: no tag means ':latest'.

    /api/tags reports 'llama3.2:latest', while a user may type 'llama3.2'.
    Both refer to the same model.
    """
    def normalize(name: str) -> str:
        name = name.strip().lower()
        return name if ":" in name else name + ":latest"

    return normalize(a) == normalize(b)


@dataclass
class Generation:
    text: str
    model: str
    elapsed_s: float
    stats: Optional[Dict[str, float]] = None  # Ollama's own numbers (see _stats), when it reports them


@dataclass
class ToolCall:
    """One tool the model asked to run."""
    name: str
    arguments: Dict[str, Any]


@dataclass
class ChatReply:
    text: str                  # the model's text (may be empty when it only asks for tools)
    tool_calls: List[ToolCall]  # empty = this is the final answer
    model: str


def _stats(reply: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Where Ollama says the time went: loading the model, reading the prompt, writing the answer.

    Ollama reports durations in nanoseconds and token counts. None if it reports none of them.
    """
    keys = {"load_s": "load_duration", "prompt_s": "prompt_eval_duration", "output_s": "eval_duration"}
    counts = {"prompt_tokens": "prompt_eval_count", "output_tokens": "eval_count"}
    stats: Dict[str, float] = {}
    for name, key in keys.items():
        if isinstance(reply.get(key), (int, float)):
            stats[name] = round(reply[key] / 1e9, 2)
    for name, key in counts.items():
        if isinstance(reply.get(key), (int, float)):
            stats[name] = int(reply[key])
    return stats or None


_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with _OPENER.open(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Ollama reports errors as {"error": "..."} with a 4xx/5xx status.
            detail = exc.read().decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except (json.JSONDecodeError, AttributeError):
                pass
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"cannot reach Ollama at {self.base_url} ({exc.reason}). Is `ollama serve` running?"
            ) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise OllamaError(f"Ollama did not answer within {self.timeout:.0f}s") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama sent invalid JSON: {exc}") from exc

    def list_models(self) -> List[str]:
        reply = self._request("GET", "/api/tags")
        return [m["name"] for m in reply.get("models", []) if "name" in m]

    def resolve_model(self, preferred: Optional[str], avoid: Optional[str] = None) -> str:
        """Pick the model to run.

        1. An explicitly requested model always wins.
        2. Otherwise the first installed model that is NOT `avoid`
           (used so the two machines answer with different models).
        3. If every installed model equals `avoid`, use it anyway: a
           second answer from the same model is better than no answer.
        """
        if preferred:
            return preferred
        models = self.list_models()
        if not models:
            raise OllamaError("no models are installed. Run e.g. `ollama pull llama3.2` first")
        if avoid:
            for name in models:
                if not same_model(name, avoid):
                    return name
        return models[0]

    def generate(self, prompt: str, model: Optional[str] = None,
                 avoid: Optional[str] = None) -> Generation:
        name = self.resolve_model(model, avoid)
        started = time.perf_counter()
        reply = self._request(
            "POST",
            "/api/generate",
            {"model": name, "prompt": prompt, "stream": False},
        )
        if "response" not in reply:
            raise OllamaError(f"unexpected reply from Ollama: {reply!r}")
        return Generation(
            text=reply["response"],
            model=reply.get("model", name),
            elapsed_s=time.perf_counter() - started,
            stats=_stats(reply),
        )

    def chat(self, messages: List[Dict[str, Any]], model: str,
             tools: Optional[List[Dict[str, Any]]] = None) -> ChatReply:
        """One round of /api/chat. With `tools`, the model may answer with tool calls.

        Ollama's format: tools = [{"type": "function", "function": {name, description,
        parameters}}]; a tool call comes back as message.tool_calls[i].function with
        `name` and `arguments` (a JSON object).
        """
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if tools:
            body["tools"] = tools
        reply = self._request("POST", "/api/chat", body)
        message = reply.get("message")
        if not isinstance(message, dict):
            raise OllamaError(f"unexpected reply from Ollama: {reply!r}")

        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):  # tolerate arguments sent as JSON text
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            calls.append(ToolCall(name=str(function.get("name", "")),
                                  arguments=arguments if isinstance(arguments, dict) else {}))

        # Some models print their reasoning between <think> tags; it is not the answer.
        text = _THINKING.sub("", message.get("content") or "").strip()
        return ChatReply(text=text, tool_calls=calls, model=reply.get("model", model))
