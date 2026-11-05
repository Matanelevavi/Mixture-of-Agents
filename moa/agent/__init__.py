"""Agent mode: a model that can use tools from MCP servers ("connectors").

    config.py   reads moa.json: the MCP servers, and which tools a peer may use
    connect.py  opens an MCP client for one server (local over stdio, remote over HTTP)
    loop.py     the agent loop: list the tools, ask the model, run the tools it asks for
"""

from .config import AgentConfig, AgentError, ServerConfig
from .loop import AgentResult, ToolUse, run_agent

__all__ = ["AgentConfig", "AgentError", "AgentResult", "ServerConfig", "ToolUse", "run_agent"]
