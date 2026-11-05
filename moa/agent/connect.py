"""Open an MCP client for one configured server and list its tools.

Local servers are started as a child process and spoken to over stdio.
Remote servers are reached over Streamable HTTP with a bearer token.
Every failure becomes an AgentError with a reason a person can act on.
"""

from __future__ import annotations

import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from .config import AgentError, ServerConfig

TOOL_TIMEOUT_S = 60.0
CONNECT_TIMEOUT_S = 10.0  # HTTP connect timeout for remote servers
PACKAGE_ROOT = str(Path(__file__).resolve().parents[2])  # so local servers can run `-m moa.servers...`


async def connect(stack: AsyncExitStack, server: ServerConfig) -> Tuple[Client, Any]:
    """Return (client, result of tools/list). The stack closes the client later."""
    if server.remote:
        return await _connect_http(stack, server)
    return await _connect_stdio(stack, server)


async def _connect_stdio(stack: AsyncExitStack, server: ServerConfig) -> Tuple[Client, Any]:
    params = StdioServerParameters(command=server.command, args=server.args, cwd=server.cwd,
                                   env={"PYTHONPATH": _pythonpath(), **server.env})
    try:
        client = await stack.enter_async_context(Client(params, read_timeout_seconds=TOOL_TIMEOUT_S))
        return client, await client.list_tools()
    except Exception as exc:  # the SDK raises several types here
        raise AgentError(f"could not start: {reason(exc)}") from exc


async def _connect_http(stack: AsyncExitStack, server: ServerConfig) -> Tuple[Client, Any]:
    statuses: List[int] = []

    async def remember_status(response) -> None:  # the SDK hides the HTTP status; keep it
        statuses.append(response.status_code)

    http = await stack.enter_async_context(httpx2.AsyncClient(
        headers=_auth_headers(server),
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
        raise AgentError(f"cannot reach {server.url}: {reason(exc)}") from exc


def _auth_headers(server: ServerConfig) -> Dict[str, str]:
    """The bearer token, read from the environment variable the config names."""
    if not server.token_env:
        return {}
    token = os.environ.get(server.token_env)
    if not token:  # never connect without the token the config asks for
        raise AgentError(f"environment variable {server.token_env} is not set")
    return {"Authorization": f"Bearer {token}"}


def reason(exc: BaseException) -> str:
    """The innermost useful message (the SDK wraps errors in exception groups)."""
    while getattr(exc, "exceptions", None):
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


def _pythonpath() -> str:
    existing = os.environ.get("PYTHONPATH")
    return PACKAGE_ROOT + (os.pathsep + existing if existing else "")
