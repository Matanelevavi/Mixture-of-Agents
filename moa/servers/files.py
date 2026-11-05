"""A small read-only MCP server that exposes one folder to a model.

Local (stdio, started by the agent on the same machine):
    python -m moa.servers.files --root ./shared

Remote (Streamable HTTP, for agents on other machines):
    MOA_MCP_TOKEN=<secret> python -m moa.servers.files --root ./public --http 192.168.10.7:8000

Tools:
    list_files()          -> names and sizes of the files under the root
    read_file(path)       -> text of one file under the root

Everything is confined to the root folder: paths that resolve outside it
("../secret.txt", an absolute path, a symlink pointing out) are refused.
Nothing is ever written or deleted.

In HTTP mode every request must carry "Authorization: Bearer <token>", and
the server refuses to start without a token. Only the Host values given by
--http / --allow-host are accepted (DNS rebinding protection). Plain HTTP is
not encrypted: use it on a network you control, or through a VPN such as
Tailscale; never expose the port to the internet.
"""

from __future__ import annotations

import argparse
import hmac
import os
import sys
from pathlib import Path
from typing import List, Optional

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings

MAX_FILES = 200             # list_files stops after this many entries
MAX_READ_BYTES = 64 * 1024  # read_file refuses larger files
DEFAULT_TOKEN_ENV = "MOA_MCP_TOKEN"
MIN_TOKEN_LENGTH = 16

ROOT = Path(os.environ.get("MOA_FILES_ROOT", ".")).resolve()


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

def _inside_root(relative: str) -> Path:
    """Resolve `relative` under ROOT and refuse anything that escapes it.

    resolve() follows symlinks and removes "..", so the check below also
    catches a symlink inside the folder that points somewhere else.
    """
    candidate = (ROOT / relative).resolve()
    if candidate != ROOT and ROOT not in candidate.parents:
        raise ToolError(f"access denied: {relative!r} is outside the shared folder")
    return candidate


def list_files() -> str:
    """List the files in the shared folder (relative path and size in bytes)."""
    lines = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames.sort()
        for name in sorted(filenames):
            full = Path(dirpath, name)
            try:
                _inside_root(str(full.relative_to(ROOT)))  # skips symlinks that leave the root
            except ToolError:
                continue
            lines.append(f"{full.relative_to(ROOT).as_posix()}  ({full.stat().st_size} bytes)")
            if len(lines) >= MAX_FILES:
                lines.append(f"... stopped after {MAX_FILES} files")
                return "\n".join(lines)
    return "\n".join(lines) if lines else "(the shared folder is empty)"


def read_file(path: str) -> str:
    """Return the text of one file from the shared folder. `path` is relative, as shown by list_files."""
    target = _inside_root(path)
    if not target.is_file():
        raise ToolError(f"no such file: {path!r}")
    size = target.stat().st_size
    if size > MAX_READ_BYTES:
        raise ToolError(f"{path!r} is {size} bytes; the limit is {MAX_READ_BYTES}")
    # utf-8-sig drops a leading byte order mark (Windows PowerShell 5 writes one
    # with -Encoding utf8) and reads files without one exactly like utf-8.
    return target.read_text(encoding="utf-8-sig", errors="replace")


# --------------------------------------------------------------------------
# Authentication for HTTP mode
# --------------------------------------------------------------------------

class StaticTokenVerifier:
    """The SDK's TokenVerifier interface, for one shared secret.

    The SDK calls verify_token() with the bearer token of every HTTP request;
    returning None makes it answer 401. compare_digest takes the same time
    wherever the strings differ, so the token cannot be guessed by timing.
    """

    def __init__(self, token: str):
        self._token = token.encode()

    async def verify_token(self, token: str) -> Optional[AccessToken]:
        if hmac.compare_digest(token.encode(), self._token):
            return AccessToken(token=token, client_id="moa-peer", scopes=[])
        return None


def build_server(token: Optional[str] = None, public_url: Optional[str] = None) -> MCPServer:
    """The MCP server with both tools. With a token, HTTP requests need it."""
    auth_kwargs = {}
    if token is not None:
        auth_kwargs = dict(
            token_verifier=StaticTokenVerifier(token),
            # AuthSettings comes from OAuth, where tokens are issued by a separate
            # authorization server. Here the token is a shared secret set by hand,
            # so there is no such server and both URLs name this server itself
            # (they only appear in the /.well-known metadata it publishes).
            auth=AuthSettings(issuer_url=public_url, resource_server_url=public_url,
                              validate_token_resource=False),
        )
    server = MCPServer(
        "files",
        version="1.0",
        instructions="Read-only access to the files in one shared folder.",
        log_level="ERROR",  # stderr is shared with the moa console; refused calls are reported to the caller anyway
        **auth_kwargs,
    )
    server.tool()(list_files)
    server.tool()(read_file)
    return server


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def _host_port(value: str):
    host, sep, port = value.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise argparse.ArgumentTypeError(f"expected HOST:PORT, got {value!r}")
    return host, int(port)


def main(argv: Optional[List[str]] = None) -> None:
    global ROOT
    parser = argparse.ArgumentParser(prog="python -m moa.servers.files")
    parser.add_argument("--root", default=str(ROOT), help="folder to expose (read-only)")
    parser.add_argument("--http", type=_host_port, metavar="HOST:PORT",
                        help="serve over Streamable HTTP instead of stdio, e.g. 192.168.10.7:8000")
    parser.add_argument("--allow-host", action="append", default=[], metavar="HOST",
                        help="another address clients use to reach this server (repeatable); "
                             "required when --http listens on 0.0.0.0")
    parser.add_argument("--token-env", default=DEFAULT_TOKEN_ENV,
                        help=f"environment variable holding the shared token (default: {DEFAULT_TOKEN_ENV})")
    args = parser.parse_args(argv)

    ROOT = Path(args.root).resolve()
    if not ROOT.is_dir():
        parser.error(f"--root {args.root!r} is not a folder")

    if args.http is None:
        build_server().run()  # stdio: the MCP client starts this process and talks over stdin/stdout
        return

    host, port = args.http
    token = os.environ.get(args.token_env, "")
    if len(token) < MIN_TOKEN_LENGTH:
        parser.error(f"HTTP mode needs a token of at least {MIN_TOKEN_LENGTH} characters "
                     f"in the environment variable {args.token_env}")

    wildcard = host in ("0.0.0.0", "::")
    if wildcard and not args.allow_host:
        parser.error("--http on 0.0.0.0 needs --allow-host with the address clients will use")
    names = ([] if wildcard else [host]) + args.allow_host + ["127.0.0.1", "localhost"]
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[f"{name}:{port}" for name in names],
        allowed_origins=[],  # no browser page may call this server
    )
    public_url = f"http://{names[0]}:{port}"
    print(f"[files] serving {ROOT} read-only at {public_url}/mcp (token required)", file=sys.stderr)
    build_server(token, public_url).run(
        transport="streamable-http", host=host, port=port, transport_security=security)


if __name__ == "__main__":
    main()
