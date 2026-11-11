"""The agent loop: the model asks for tools, the agent runs them through MCP.

The model never talks to an MCP server itself. This loop sits in between:

    1. connect to every configured MCP server and ask it for its tools (tools/list)
    2. send the prompt and the tool descriptions to the model
    3. if the model asks for a tool, run it through MCP (tools/call) and send
       the result back to the model
    4. repeat until the model answers without asking for a tool

A tool that its server does not mark as read-only (e.g. gmail trash_email) is
run only after a person approves that exact call (the `confirm` callback). With
no one to ask (a prompt from the peer), such calls are refused.

Every tool is shown to the model as "<server>__<tool>", so two servers with
the same tools (this machine's files and the other machine's files) never
collide, and the model can tell which machine a tool belongs to.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from ..ollama_client import ChatReply, OllamaClient, ToolCall
from .config import SEPARATOR, AgentConfig, AgentError, ServerConfig
from .connect import connect, reason

MAX_TOOL_RESULT_CHARS = 8000  # what the model sees from one tool result

# Asked before a tool that changes something runs: (tool name, arguments,
# lines of earlier tool results that mention those arguments) -> approved?
Confirm = Callable[[str, Dict[str, Any], List[str]], bool]

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
    seconds: float = 0.0  # how long the tool call took, through MCP

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentResult:
    text: str
    model: str
    tools_used: List[ToolUse]
    stopped_early: bool = False                        # hit max_tool_calls before a final answer
    warnings: List[str] = field(default_factory=list)  # servers that could not be used
    setup_s: float = 0.0                               # connecting to the MCP servers and listing their tools
    model_s: List[float] = field(default_factory=list)  # one entry per call to the model


@dataclass
class Toolbox:
    """The tools offered to the model for one prompt."""
    routes: Dict[str, Tuple[Any, str]] = field(default_factory=dict)  # name -> (MCP client, name on the server)
    specs: List[Dict[str, Any]] = field(default_factory=list)          # descriptions sent to the model
    warnings: List[str] = field(default_factory=list)
    needs_approval: Set[str] = field(default_factory=set)  # tools not marked read-only by their server

    def add(self, server: ServerConfig, client: Any, tool: Any) -> None:
        name = f"{server.name}{SEPARATOR}{tool.name}"
        self.routes[name] = (client, tool.name)
        # The mark comes from the server itself: trusted here because we
        # configured these servers. Unmarked tools are treated as changing things.
        if not (tool.annotations and tool.annotations.read_only_hint):
            self.needs_approval.add(name)
        self.specs.append({"type": "function", "function": {
            "name": name,
            "description": f"[{server.label}] {tool.description or ''}".strip(),
            "parameters": tool.input_schema,
        }})


def run_agent(prompt: str, ollama: OllamaClient, config: AgentConfig,
              model: Optional[str] = None, avoid: Optional[str] = None,
              allowed_tools: Optional[FrozenSet[str]] = None,
              confirm: Optional[Confirm] = None) -> AgentResult:
    """Answer `prompt` with the model, letting it use the configured MCP tools.

    allowed_tools=None means every configured tool (a local user on this
    machine); a set limits the agent to those "<server>__<tool>" names (a
    prompt from the peer). confirm=None means no one can approve changes, so
    tools that are not read-only are refused. Blocking call: runs its own
    event loop, so it is safe from any thread.
    """
    return asyncio.run(_run(prompt, ollama, config, model, avoid, allowed_tools, confirm))


async def _run(prompt: str, ollama: OllamaClient, config: AgentConfig, model: Optional[str],
               avoid: Optional[str], allowed_tools: Optional[FrozenSet[str]],
               confirm: Optional[Confirm]) -> AgentResult:
    async with AsyncExitStack() as stack:  # closes every MCP connection at the end
        started = time.perf_counter()
        toolbox = await _open_toolbox(stack, config, allowed_tools)
        setup_s = time.perf_counter() - started
        model_name = await asyncio.to_thread(ollama.resolve_model, model, avoid)
        messages = _first_messages(prompt, toolbox)
        used: List[ToolUse] = []
        model_s: List[float] = []

        while True:
            started = time.perf_counter()
            reply = await asyncio.to_thread(ollama.chat, messages, model_name, toolbox.specs or None)
            model_s.append(time.perf_counter() - started)
            if not reply.tool_calls:
                return AgentResult(reply.text, reply.model, used, warnings=toolbox.warnings,
                                   setup_s=setup_s, model_s=model_s)
            if len(used) + len(reply.tool_calls) > config.max_tool_calls:
                note = f"(stopped: the model asked for more than {config.max_tool_calls} tool calls)"
                return AgentResult(f"{reply.text}\n{note}".strip(), reply.model, used,
                                   stopped_early=True, warnings=toolbox.warnings,
                                   setup_s=setup_s, model_s=model_s)

            messages.append(_assistant_message(reply))
            for call in reply.tool_calls:
                result_text, use = await _call_tool(toolbox, call, confirm, messages)
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


def _today(now: Optional[datetime] = None) -> str:
    """The date line for the model, read from this machine's clock at the start of every question.

    A model does not know today's date (found in a real run: asked for "today's
    emails" it invented the Gmail query `after:yesterday before:now`, and then
    answered that no email had arrived). Taken fresh for each question, so a
    long interactive session never works with an old date.
    """
    now = now or datetime.now().astimezone()
    return f"Today is {now:%A}, {now:%Y-%m-%d}, {now:%H:%M} (this machine's local time, UTC{now:%z})."


def _first_messages(prompt: str, toolbox: Toolbox, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    messages = []
    if toolbox.specs:
        messages.append({"role": "system", "content": f"{SYSTEM_PROMPT} {_today(now)}"})
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


async def _call_tool(toolbox: Toolbox, call: ToolCall, confirm: Optional[Confirm],
                     history: List[Dict[str, Any]]) -> Tuple[str, ToolUse]:
    """Run one tool call. Never raises: failures go back to the model as text."""
    route = toolbox.routes.get(call.name)
    if route is None:  # unknown, or not allowed for this request: never executed
        error = f"tool {call.name!r} is not available"
        return f"Error: {error}", ToolUse(call.name, call.arguments, False, error)

    if call.name in toolbox.needs_approval:
        if confirm is None:
            error = "this tool changes something and no one is here to approve it"
        elif not await asyncio.to_thread(confirm, call.name, call.arguments,
                                         _mentions(history, call.arguments)):
            error = "the user did not approve this action"
        else:
            error = None
        if error:  # never executed
            return f"Error: {error}; it was not done", ToolUse(call.name, call.arguments, False, error)

    client, tool_name = route
    started = time.perf_counter()
    try:
        result = await client.call_tool(tool_name, call.arguments)
    except Exception as exc:
        return (f"Error: {reason(exc)}",
                ToolUse(call.name, call.arguments, False, reason(exc), time.perf_counter() - started))
    seconds = time.perf_counter() - started

    text = "\n".join(block.text for block in result.content if getattr(block, "type", None) == "text")
    if result.is_error:
        return text or "Error", ToolUse(call.name, call.arguments, False, text or "tool error", seconds)
    return text, ToolUse(call.name, call.arguments, True, None, seconds)


def _mentions(history: List[Dict[str, Any]], arguments: Dict[str, Any]) -> List[str]:
    """Lines of earlier tool results that contain an argument value, shown with
    the approval question, so the user sees e.g. which email an id belongs to."""
    values = [str(v) for v in arguments.values() if isinstance(v, (str, int)) and len(str(v)) >= 4]
    lines: List[str] = []
    for message in history:
        if message["role"] != "tool":
            continue
        for line in message["content"].splitlines():
            line = line.strip()
            if any(v in line for v in values) and line not in lines:
                lines.append(line)
    return lines[:5]
