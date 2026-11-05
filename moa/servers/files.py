"""A small read-only MCP server that exposes one folder to a model.

    python -m moa.servers.files --root ./shared

Tools:
    list_files()          -> names and sizes of the files under the root
    read_file(path)       -> text of one file under the root

Everything is confined to the root folder: paths that resolve outside it
("../secret.txt", an absolute path, a symlink pointing out) are refused.
Nothing is ever written or deleted.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

MAX_FILES = 200            # list_files stops after this many entries
MAX_READ_BYTES = 64 * 1024  # read_file refuses larger files

ROOT = Path(os.environ.get("MOA_FILES_ROOT", ".")).resolve()

server = MCPServer(
    "files",
    version="1.0",
    instructions="Read-only access to the files in one shared folder.",
    log_level="ERROR",  # stderr is shared with the moa console; refused calls are reported to the caller anyway
)


def _inside_root(relative: str) -> Path:
    """Resolve `relative` under ROOT and refuse anything that escapes it.

    resolve() follows symlinks and removes "..", so the check below also
    catches a symlink inside the folder that points somewhere else.
    """
    candidate = (ROOT / relative).resolve()
    if candidate != ROOT and ROOT not in candidate.parents:
        raise ToolError(f"access denied: {relative!r} is outside the shared folder")
    return candidate


@server.tool()
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


@server.tool()
def read_file(path: str) -> str:
    """Return the text of one file from the shared folder. `path` is relative, as shown by list_files."""
    target = _inside_root(path)
    if not target.is_file():
        raise ToolError(f"no such file: {path!r}")
    size = target.stat().st_size
    if size > MAX_READ_BYTES:
        raise ToolError(f"{path!r} is {size} bytes; the limit is {MAX_READ_BYTES}")
    return target.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    global ROOT
    parser = argparse.ArgumentParser(prog="python -m moa.servers.files")
    parser.add_argument("--root", default=str(ROOT), help="folder to expose (read-only)")
    args = parser.parse_args()
    ROOT = Path(args.root).resolve()
    if not ROOT.is_dir():
        parser.error(f"--root {args.root!r} is not a folder")
    server.run()  # stdio: the MCP client starts this process and talks over stdin/stdout


if __name__ == "__main__":
    main()
