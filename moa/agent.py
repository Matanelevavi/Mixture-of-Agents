"""Agent mode: a model that can use tools from MCP servers ("connectors").

The model never talks to an MCP server itself. This loop sits in between:

    1. connect to every configured MCP server and ask it for its tools (tools/list)
    2. send the prompt and the tool descriptions to the model
    3. if the model asks for a tool, run it through MCP (tools/call) and send
       the result back to the model
    4. repeat until the model answers without asking for a tool

An MCP server is either local (a "command" started over stdio) or remote (a
"url" reached over Streamable HTTP, with a bearer token). Every tool is shown
to the model as "<server>__<tool>", so two servers with the same tools (for
example this machine's files and the other machine's files) never collide,
and the model can tell which machine a tool belongs to.

Only tools in `allowed_tools` are offered to the model and executed. The
server side uses this to restrict what a prompt from the network may do.
A server that cannot be reached is reported as a warning and its tools are
left out; the agent still answers with the others.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from .ollama_client import OllamaClient, ToolCall

MAX_TOOL_RESULT_CHARS = 8000   # what the model sees from one tool result
TOOL_TIMEOUT_S = 60.0
CONNECT_TIMEOUT_S = 10.0       # HTTP connect timeout for remote servers
SEPARATOR = "__"               # "<server>__<tool>"
SERVER_NAME = re.compile(r"^[A-Za-z0-9_-]{1,30}$")  # keeps tool names valid for the model API

SYSTEM_PROMPT = (
    "You are an assistant running on one computer. You have tools from MCP servers; each tool "
    "name starts with the name of its server, and its description says which server it is on. "
    "When the question is about those resources, use the tools and base your answer only on "
    "what they return. Do not invent file names or contents."
)

PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)  # so servers can run `-m moa.servers...`


class AgentError(Exception):
    """The agent could not run (bad config, the model failed, ...)."""


# --------------------------------------------------------------------------
# Configuration (moa.json)
# --------------------------------------------------------------------------

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


@dataclass
class AgentConfig:
    servers: List[ServerConfig]
    remote_allowed_tools: FrozenSet[str] = frozenset()  # "<server>__<tool>" names a peer's prompt may use
    max_tool_calls: int = 8

    @classmethod
    def load(cls, path: str) -> "AgentConfig":
        """Read moa.json. Same "mcpServers" layout as Claude Desktop's config.

        {
          "mcpServers": {
            "my_files":   {"command": "python", "args": ["-m", "moa.servers.files", "--root", "shared"]},
            "pc_b_files": {"url": "http://192.168.10.7:8000/mcp", "tokenEnv": "MOA_MCP_TOKEN"}
          },
          "remoteAllowedTools": ["my_files__list_files", "my_files__read_file"],
          "maxToolCalls": 8
        }

        The token itself is never written in the file, only the name of the
        environment variable that holds it.
        """
        config_path = Path(path).resolve()
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentError(f"cannot read config {path!r}: {exc}") from exc

        servers = []
        for name, spec in (data.get("mcpServers") or {}).items():
            if not SERVER_NAME.match(name) or SEPARATOR in name:
                raise AgentError(f"server name {name!r}: use 1-30 letters, digits, '_' or '-', without '{SEPARATOR}'")
            if not isinstance(spec, dict) or bool(spec.get("command")) == bool(spec.get("url")):
                raise AgentError(f"mcpServers.{name} needs exactly one of 'command' (local) or 'url' (remote)")
            if spec.get("url"):
                if not str(spec["url"]).startswith(("http://", "https://")):
                    raise AgentError(f"mcpServers.{name}.url must start with http:// or https://")
                servers.append(ServerConfig(name=name, url=str(spec["url"]),
                                            token_env=spec.get("tokenEnv")))
                continue
            command = spec["command"]
            if command in ("python", "python3"):
                command = sys.executable  # the same interpreter (and venv) that runs moa
            servers.append(ServerConfig(
                name=name,
                command=command,
                args=[str(a) for a in spec.get("args", [])],
                env={str(k): str(v) for k, v in (spec.get("env") or {}).items()},
                cwd=str((config_path.parent / spec["cwd"]).resolve()) if spec.get("cwd") else str(config_path.parent),
            ))
        return cls(
            servers=servers,
            remote_allowed_tools=frozenset(data.get("remoteAllowedTools") or []),
            max_tool_calls=int(data.get("maxToolCalls", 8)),
        )


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------

@dataclass
class ToolUse:
    """One tool call the model made, for the record shown to the user."""
    name: str                 # "<server>__<tool>"
    arguments: Dict[str, Any]
    ok: bool
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentResult:
    text: str
    model: str
    tools_used: List[ToolUse]
    stopped_early: bool = False                         # hit max_tool_calls before a final answer
    warnings: List[str] = field(default_factory=list)   # servers that could not be used


# --------------------------------------------------------------------------
# Connecting to servers
# --------------------------------------------------------------------------

async def _connect(stack: AsyncExitStack, server: ServerConfig):
    """Open an MCP client for one server. Raises AgentError with a clear reason."""
    from mcp import Client, StdioServerParameters  # imported here: plain mode needs no MCP SDK

    if not server.remote:
        params = StdioServerParameters(command=server.command, args=server.args, cwd=server.cwd,
                                       env={"PYTHONPATH": _pythonpath(), **server.env})
        try:
            client = await stack.enter_async_context(Client(params, read_timeout_seconds=TOOL_TIMEOUT_S))
            return client, await client.list_tools()
        except Exception as exc:  # the SDK raises several types here
            raise AgentError(f"could not start: {_reason(exc)}") from exc

    import httpx2
    from mcp.client.streamable_http import streamable_http_client

    headers = {}
    if server.token_env:
        token = os.environ.get(server.token_env)
        if not token:  # never connect without the token the config asks for
            raise AgentError(f"environment variable {server.token_env} is not set")
        headers["Authorization"] = f"Bearer {token}"

    statuses: List[int] = []

    async def remember_status(response):  # the SDK hides the HTTP status; keep it for the error message
        statuses.append(response.status_code)

    http = await stack.enter_async_context(httpx2.AsyncClient(
        headers=headers,
        timeout=httpx2.Timeout(TOOL_TIMEOUT_S, connect=CONNECT_TIMEOUT_S),
        trust_env=False,  # a LAN or VPN address must never go through an HTTP proxy
        event_hooks={"response": [remember_status]},
    ))
    try:
        client = await stack.enter_async_context(Client(
            streamable_http_client(server.url, http_client=http), read_timeout_seconds=TOOL_TIMEOUT_S))
        return client, await client.list_tools()
    except Exception as exc:
        if 401 in statuses:
            raise AgentError(f"{server.url} refused the token (HTTP 401)") from exc
        if 421 in statuses:
            raise AgentError(f"{server.url} does not accept this address (HTTP 421); "
                             f"start it with --allow-host for the address used here") from exc
        raise AgentError(f"cannot reach {server.url}: {_reason(exc)}") from exc


def _reason(exc: BaseException) -> str:
    """The innermost useful message (the SDK wraps errors in exception groups)."""
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def run_agent(prompt: str, ollama: OllamaClient, config: AgentConfig,
              model: Optional[str] = None, avoid: Optional[str] = None,
              allowed_tools: Optional[FrozenSet[str]] = None) -> AgentResult:
    """Answer `prompt` with the model, letting it use the configured MCP tools.

    allowed_tools=None means every configured tool (a local user on this
    machine); a set limits the agent to those "<server>__<tool>" names (a
    prompt from the peer). Blocking call: runs its own event loop, so it is
    safe from any thread.
    """
    return asyncio.run(_run(prompt, ollama, config, model, avoid, allowed_tools))


async def _run(prompt, ollama, config, model, avoid, allowed_tools) -> AgentResult:
    async with AsyncExitStack() as stack:
        tools: Dict[str, Tuple[Any, str]] = {}   # "<server>__<tool>" -> (MCP client, tool name on the server)
        tool_specs: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for server in config.servers:
            try:
                client, listed = await _connect(stack, server)
            except AgentError as exc:
                warnings.append(f"MCP server {server.name!r} unavailable: {exc}")
                continue
            where = f"remote server {server.name} ({server.url})" if server.remote else f"local server {server.name}"
            for tool in listed.tools:
                qualified = f"{server.name}{SEPARATOR}{tool.name}"
                if allowed_tools is not None and qualified not in allowed_tools:
                    continue  # not offered to the model at all
                tools[qualified] = (client, tool.name)
                tool_specs.append({"type": "function", "function": {
                    "name": qualified,
                    "description": f"[{where}] {tool.description or ''}".strip(),
                    "parameters": tool.input_schema,
                }})

        model_name = await asyncio.to_thread(ollama.resolve_model, model, avoid)
        messages: List[Dict[str, Any]] = []
        if tool_specs:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        if warnings:  # tell the model, so it does not pretend it could look
            messages.append({"role": "system", "content": "Unavailable right now: " + "; ".join(warnings)})
        messages.append({"role": "user", "content": prompt})
        used: List[ToolUse] = []

        while True:
            reply = await asyncio.to_thread(ollama.chat, messages, model_name, tool_specs or None)
            if not reply.tool_calls:
                return AgentResult(reply.text, reply.model, used, warnings=warnings)
            if len(used) + len(reply.tool_calls) > config.max_tool_calls:
                note = f"(stopped: the model asked for more than {config.max_tool_calls} tool calls)"
                return AgentResult((reply.text + "\n" + note).strip(), reply.model, used,
                                   stopped_early=True, warnings=warnings)

            messages.append({"role": "assistant", "content": reply.text, "tool_calls": [
                {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in reply.tool_calls]})
            for call in reply.tool_calls:
                result_text, use = await _call_tool(tools, call)
                used.append(use)
                messages.append({"role": "tool", "tool_name": call.name,
                                 "content": result_text[:MAX_TOOL_RESULT_CHARS]})


async def _call_tool(tools: Dict[str, Tuple[Any, str]], call: ToolCall):
    """Run one tool call. Never raises: failures go back to the model as text."""
    entry = tools.get(call.name)
    if entry is None:  # unknown, or not allowed for this request: never executed
        error = f"tool {call.name!r} is not available"
        return f"Error: {error}", ToolUse(call.name, call.arguments, False, error)
    client, tool_name = entry
    try:
        result = await client.call_tool(tool_name, call.arguments)
    except Exception as exc:
        return f"Error: {_reason(exc)}", ToolUse(call.name, call.arguments, False, _reason(exc))
    text = "\n".join(block.text for block in result.content if getattr(block, "type", None) == "text")
    if result.is_error:
        return text or "Error", ToolUse(call.name, call.arguments, False, text or "tool error")
    return text, ToolUse(call.name, call.arguments, True)


def _pythonpath() -> str:
    existing = os.environ.get("PYTHONPATH")
    return PACKAGE_ROOT + (os.pathsep + existing if existing else "")
