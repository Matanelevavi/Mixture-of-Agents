"""Minimal client for the local Ollama REST API (standard library only).

Ollama listens on http://127.0.0.1:11434 by default. We use two endpoints:
    GET  /api/tags      -> list of installed models
    POST /api/generate  -> run one prompt ("stream": false = one JSON reply)
"""

from __future__ import annotations

import json
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


@dataclass
class Generation:
    text: str
    model: str
    elapsed_s: float


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

    def resolve_model(self, preferred: Optional[str]) -> str:
        """Use the requested model, or fall back to the first installed one."""
        if preferred:
            return preferred
        models = self.list_models()
        if not models:
            raise OllamaError("no models are installed. Run e.g. `ollama pull llama3.2` first")
        return models[0]

    def generate(self, prompt: str, model: Optional[str] = None) -> Generation:
        name = self.resolve_model(model)
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
        )
