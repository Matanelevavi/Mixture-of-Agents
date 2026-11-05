"""The agent loop: the model asks for tools, the agent runs them through MCP.

The model never talks to an MCP server itself. This loop sits in between:

    1. connect to every configured MCP server and ask it for its tools (tools/list)
    2. send the prompt and the tool descriptions to the model
    3. if the model asks for a tool, run it through MCP (tools/call) and send
       the result back to the model
    4. repeat until the model answers without asking for a tool

Every tool is shown to the model as "<server>__<tool>", so two servers with
the same tools (this machine's files and the other machine's files) never
collide, and the model can tell which machine a tool belongs to.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from ..ollama_client import ChatReply, OllamaClient, ToolCall
from .config import SEPARATOR, AgentConfig, AgentError, ServerConfig
from .connect import connect, reason

MAX_TOOL_RESULT_CHARS = 8000  # what the model sees from one tool result

SYSTEM_PROMPT = (
    "You are an assistant running on one computer. You have tools from MCP servers; each tool "
    "name starts with the name of its server, and its description says which server it is on. "
    "When the question is about those resources, use the tools and base your answer only on "
    "what they return. Do not invent file names or contents."
)


@dataclass
class ToolUse:
    """One tool call the model made, for the record shown to the user."""
    name: str  # "<server>__<tool>"
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
    stopped_early: bool = False                        # hit max_tool_calls before a final answer
    warnings: List[str] = field(default_factory=list)  # servers that could not be used


@dataclass
class Toolbox:
    """The tools offered to the model for one prompt."""
    routes: Dict[str, Tuple[Any, str]] = field(default_factory=dict)  # name -> (MCP client, name on the server)
    specs: List[Dict[str, Any]] = field(default_factory=list)          # descriptions sent to the model
    warnings: List[str] = field(default_factory=list)

    def add(self, server: ServerConfig, client: Any, tool: Any) -> None:
        name = f"{server.name}{SEPARATOR}{tool.name}"
        self.routes[name] = (client, tool.name)
        self.specs.append({"type": "function", "function": {
            "name": name,
            "description": f"[{server.label}] {tool.description or ''}".strip(),
            "parameters": tool.input_schema,
        }})


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


async def _run(prompt: str, ollama: OllamaClient, config: AgentConfig, model: Optional[str],
               avoid: Optional[str], allowed_tools: Optional[FrozenSet[str]]) -> AgentResult:
    async with AsyncExitStack() as stack:  # closes every MCP connection at the end
        toolbox = await _open_toolbox(stack, config, allowed_tools)
        model_name = await asyncio.to_thread(ollama.resolve_model, model, avoid)
        messages = _first_messages(prompt, toolbox)
        used: List[ToolUse] = []

        while True:
            reply = await asyncio.to_thread(ollama.chat, messages, model_name, toolbox.specs or None)
            if not reply.tool_calls:
                return AgentResult(reply.text, reply.model, used, warnings=toolbox.warnings)
            if len(used) + len(reply.tool_calls) > config.max_tool_calls:
                note = f"(stopped: the model asked for more than {config.max_tool_calls} tool calls)"
                return AgentResult(f"{reply.text}\n{note}".strip(), reply.model, used,
                                   stopped_early=True, warnings=toolbox.warnings)

            messages.append(_assistant_message(reply))
            for call in reply.tool_calls:
                result_text, use = await _call_tool(toolbox, call)
                used.append(use)
                messages.append({"role": "tool", "tool_name": call.name,
                                 "content": result_text[:MAX_TOOL_RESULT_CHARS]})


async def _open_toolbox(stack: AsyncExitStack, config: AgentConfig,
                        allowed_tools: Optional[FrozenSet[str]]) -> Toolbox:
    """Connect to every server; offer only the allowed tools. A server that
    cannot be used becomes a warning, and the agent continues without it."""
    toolbox = Toolbox()
    for server in config.servers:
        try:
            client, listed = await connect(stack, server)
        except AgentError as exc:
            toolbox.warnings.append(f"MCP server {server.name!r} unavailable: {exc}")
            continue
        for tool in listed.tools:
            if allowed_tools is None or f"{server.name}{SEPARATOR}{tool.name}" in allowed_tools:
                toolbox.add(server, client, tool)  # anything else is not even offered
    return toolbox


def _first_messages(prompt: str, toolbox: Toolbox) -> List[Dict[str, Any]]:
    messages = []
    if toolbox.specs:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    if toolbox.warnings:  # tell the model, so it does not answer as if it had looked
        messages.append({"role": "system", "content": _unavailable_note(toolbox.warnings)})
    messages.append({"role": "user", "content": prompt})
    return messages


def _unavailable_note(warnings: List[str]) -> str:
    """System message for servers that could not be used.

    Found in a real run: with the remote server refused (HTTP 401), the model
    tried the local files tool instead and answered that the file "does not
    exist", although it was on the other machine. This note is a hint to the
    model, not a guarantee; the warnings also reach the user separately.
    """
    return ("These MCP servers could not be used for this question: " + "; ".join(warnings) + ". "
            "Their tools are missing, and the tools of other servers cannot see their files. "
            "If the question is about one of them, say that it could not be reached and why; "
            "do not conclude that a file does not exist.")


def _assistant_message(reply: ChatReply) -> Dict[str, Any]:
    """The model's turn, with its tool calls, as Ollama expects it in the history."""
    return {"role": "assistant", "content": reply.text, "tool_calls": [
        {"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
        for c in reply.tool_calls]}


async def _call_tool(toolbox: Toolbox, call: ToolCall) -> Tuple[str, ToolUse]:
    """Run one tool call. Never raises: failures go back to the model as text."""
    route = toolbox.routes.get(call.name)
    if route is None:  # unknown, or not allowed for this request: never executed
        error = f"tool {call.name!r} is not available"
        return f"Error: {error}", ToolUse(call.name, call.arguments, False, error)

    client, tool_name = route
    try:
        result = await client.call_tool(tool_name, call.arguments)
    except Exception as exc:
        return f"Error: {reason(exc)}", ToolUse(call.name, call.arguments, False, reason(exc))

    text = "\n".join(block.text for block in result.content if getattr(block, "type", None) == "text")
    if result.is_error:
        return text or "Error", ToolUse(call.name, call.arguments, False, text or "tool error")
    return text, ToolUse(call.name, call.arguments, True)
