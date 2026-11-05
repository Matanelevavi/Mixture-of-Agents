"""The agent configuration file (moa.json): which MCP servers to use.

Same "mcpServers" layout as Claude Desktop's config:

    {
      "mcpServers": {
        "my_files":   {"command": "python", "args": ["-m", "moa.servers.files", "--root", "shared"]},
        "pc_b_files": {"url": "http://192.168.10.7:8000/mcp", "tokenEnv": "MOA_MCP_TOKEN"}
      },
      "remoteAllowedTools": ["my_files__list_files", "my_files__read_file"],
      "maxToolCalls": 8
    }

A server is local ("command", started over stdio) or remote ("url", reached
over Streamable HTTP). The token itself is never written in the file, only
the name of the environment variable that holds it.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

SEPARATOR = "__"  # tools are shown to the model as "<server>__<tool>"
SERVER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,30}$")  # keeps tool names valid for the model API


class AgentError(Exception):
    """The agent could not run (bad config, a server failed, the model failed, ...)."""


@dataclass
class ServerConfig:
    name: str
    # local server (stdio)
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    # remote server (Streamable HTTP)
    url: Optional[str] = None
    token_env: Optional[str] = None  # name of the environment variable that holds the token

    @property
    def remote(self) -> bool:
        return self.url is not None

    @property
    def label(self) -> str:
        """How the model is told where a tool runs."""
        return f"remote server {self.name} ({self.url})" if self.remote else f"local server {self.name}"


@dataclass
class AgentConfig:
    servers: List[ServerConfig]
    remote_allowed_tools: FrozenSet[str] = frozenset()  # tools a prompt from the peer may use
    max_tool_calls: int = 8

    @classmethod
    def load(cls, path: str) -> AgentConfig:
        config_path = Path(path).resolve()
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentError(f"cannot read config {path!r}: {exc}") from exc

        servers = [_server(name, spec, config_path.parent)
                   for name, spec in (data.get("mcpServers") or {}).items()]
        return cls(
            servers=servers,
            remote_allowed_tools=frozenset(data.get("remoteAllowedTools") or []),
            max_tool_calls=int(data.get("maxToolCalls", 8)),
        )


def _server(name: str, spec: Any, folder: Path) -> ServerConfig:
    """One entry of "mcpServers". Relative paths are relative to moa.json's folder."""
    if not SERVER_NAME.match(name) or SEPARATOR in name:
        raise AgentError(f"server name {name!r}: use 1-30 letters, digits, '_' or '-', without '{SEPARATOR}'")
    if not isinstance(spec, dict) or bool(spec.get("command")) == bool(spec.get("url")):
        raise AgentError(f"mcpServers.{name} needs exactly one of 'command' (local) or 'url' (remote)")

    if spec.get("url"):
        url = str(spec["url"])
        if not url.startswith(("http://", "https://")):
            raise AgentError(f"mcpServers.{name}.url must start with http:// or https://")
        return ServerConfig(name=name, url=url, token_env=spec.get("tokenEnv"))

    command = spec["command"]
    if command in ("python", "python3"):
        command = sys.executable  # the same interpreter (and venv) that runs moa
    return ServerConfig(
        name=name,
        command=command,
        args=[str(a) for a in spec.get("args", [])],
        env={str(k): str(v) for k, v in (spec.get("env") or {}).items()},
        cwd=str((folder / spec["cwd"]).resolve()) if spec.get("cwd") else str(folder),
    )
