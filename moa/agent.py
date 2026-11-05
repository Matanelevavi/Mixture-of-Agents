"""Agent mode: a model that can use tools from MCP servers ("connectors").

The model never talks to an MCP server itself. This loop sits in between:

    1. start every configured MCP server and ask it for its tools (tools/list)
    2. send the prompt and the tool descriptions to the model
    3. if the model asks for a tool, run it through MCP (tools/call) and send
       the result back to the model
    4. repeat until the model answers without asking for a tool

Only tools in `allowed_tools` are offered to the model and executed. The
server side uses this to restrict what a prompt from the network may do.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional

from .ollama_client import OllamaClient, ToolCall

MAX_TOOL_RESULT_CHARS = 8000   # what the model sees from one tool result
TOOL_TIMEOUT_S = 60.0

SYSTEM_PROMPT = (
    "You are an assistant running on one computer. You have tools that give access "
    "to resources on this computer. When the question is about those resources, use the "
    "tools and base your answer only on what they return. Do not invent file names or contents."
)

PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)  # so servers can run `-m moa.servers...`


class AgentError(Exception):
    """The agent could not run (bad config, an MCP server failed to start, ...)."""


# --------------------------------------------------------------------------
# Configuration (moa.json)
# --------------------------------------------------------------------------

@dataclass
class ServerConfig:
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None


@dataclass
class AgentConfig:
    servers: List[ServerConfig]
    remote_allowed_tools: FrozenSet[str] = frozenset()  # tools a prompt from the network may use
    max_tool_calls: int = 8

    @classmethod
    def load(cls, path: str) -> "AgentConfig":
        """Read moa.json. Same "mcpServers" layout as Claude Desktop's config.

        {
          "mcpServers": {"files": {"command": "python",
                                   "args": ["-m", "moa.servers.files", "--root", "shared"]}},
          "remoteAllowedTools": ["list_files", "read_file"],
          "maxToolCalls": 8
        }
        """
        config_path = Path(path).resolve()
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentError(f"cannot read config {path!r}: {exc}") from exc

        servers = []
        for name, spec in (data.get("mcpServers") or {}).items():
            if not isinstance(spec, dict) or not spec.get("command"):
                raise AgentError(f"mcpServers.{name} needs a 'command'")
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
    name: str
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
    stopped_early: bool = False  # hit max_tool_calls before a final answer


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def run_agent(prompt: str, ollama: OllamaClient, config: AgentConfig,
              model: Optional[str] = None, avoid: Optional[str] = None,
              allowed_tools: Optional[FrozenSet[str]] = None) -> AgentResult:
    """Answer `prompt` with the model, letting it use the configured MCP tools.

    allowed_tools=None means every configured tool (a local user on this
    machine); a set limits the agent to those names (a prompt from the peer).
    Blocking call: runs its own event loop, so it is safe from any thread.
    """
    return asyncio.run(_run(prompt, ollama, config, model, avoid, allowed_tools))


async def _run(prompt, ollama, config, model, avoid, allowed_tools) -> AgentResult:
    from mcp import Client, StdioServerParameters  # imported here: plain mode needs no MCP SDK

    async with AsyncExitStack() as stack:
        clients: Dict[str, Any] = {}       # tool name -> MCP client that provides it
        tool_specs: List[Dict[str, Any]] = []

        for server in config.servers:
            params = StdioServerParameters(
                command=server.command, args=server.args, cwd=server.cwd,
                env={"PYTHONPATH": _pythonpath(), **server.env},
            )
            try:
                client = await stack.enter_async_context(
                    Client(params, read_timeout_seconds=TOOL_TIMEOUT_S))
                listed = await client.list_tools()
            except Exception as exc:  # the SDK raises several types here
                raise AgentError(f"MCP server {server.name!r} failed to start: {exc}") from exc

            for tool in listed.tools:
                if allowed_tools is not None and tool.name not in allowed_tools:
                    continue  # not offered to the model at all
                if tool.name in clients:
                    raise AgentError(f"tool {tool.name!r} is provided by two MCP servers")
                clients[tool.name] = client
                tool_specs.append({"type": "function", "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                }})

        model_name = await asyncio.to_thread(ollama.resolve_model, model, avoid)
        messages: List[Dict[str, Any]] = []
        if tool_specs:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": prompt})
        used: List[ToolUse] = []

        while True:
            reply = await asyncio.to_thread(ollama.chat, messages, model_name, tool_specs or None)
            if not reply.tool_calls:
                return AgentResult(reply.text, reply.model, used)
            if len(used) + len(reply.tool_calls) > config.max_tool_calls:
                note = f"(stopped: the model asked for more than {config.max_tool_calls} tool calls)"
                return AgentResult((reply.text + "\n" + note).strip(), reply.model, used, stopped_early=True)

            messages.append({"role": "assistant", "content": reply.text, "tool_calls": [
                {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in reply.tool_calls]})
            for call in reply.tool_calls:
                result_text, use = await _call_tool(clients, call)
                used.append(use)
                messages.append({"role": "tool", "tool_name": call.name,
                                 "content": result_text[:MAX_TOOL_RESULT_CHARS]})


async def _call_tool(clients: Dict[str, Any], call: ToolCall):
    """Run one tool call. Never raises: failures go back to the model as text."""
    client = clients.get(call.name)
    if client is None:  # unknown, or not allowed for this request: never executed
        error = f"tool {call.name!r} is not available"
        return f"Error: {error}", ToolUse(call.name, call.arguments, False, error)
    try:
        result = await client.call_tool(call.name, call.arguments)
    except Exception as exc:
        return f"Error: {exc}", ToolUse(call.name, call.arguments, False, str(exc))
    text = "\n".join(block.text for block in result.content if getattr(block, "type", None) == "text")
    if result.is_error:
        return text or "Error", ToolUse(call.name, call.arguments, False, text or "tool error")
    return text, ToolUse(call.name, call.arguments, True)


def _pythonpath() -> str:
    existing = os.environ.get("PYTHONPATH")
    return PACKAGE_ROOT + (os.pathsep + existing if existing else "")
