"""One folder, exposed read-only: the two tools of the files MCP server.

Everything is confined to the root folder: paths that resolve outside it
("../secret.txt", an absolute path, a symlink pointing out) are refused.
Nothing is ever written or deleted.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver.exceptions import ToolError

MAX_FILES = 200             # list_files stops after this many entries
MAX_READ_BYTES = 64 * 1024  # read_file refuses larger files


class SharedFolder:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def inside_root(self, relative: str) -> Path:
        """Resolve `relative` under the root and refuse anything that escapes it.

        resolve() follows symlinks and removes "..", so the check below also
        catches a symlink inside the folder that points somewhere else.
        """
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ToolError(f"access denied: {relative!r} is outside the shared folder")
        return candidate

    def list_files(self) -> str:
        """List the files in the shared folder (relative path and size in bytes)."""
        lines = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames.sort()
            for name in sorted(filenames):
                full = Path(dirpath, name)
                relative = full.relative_to(self.root)
                try:
                    self.inside_root(str(relative))  # skips symlinks that leave the root
                except ToolError:
                    continue
                lines.append(f"{relative.as_posix()}  ({full.stat().st_size} bytes)")
                if len(lines) >= MAX_FILES:
                    lines.append(f"... stopped after {MAX_FILES} files")
                    return "\n".join(lines)
        return "\n".join(lines) if lines else "(the shared folder is empty)"

    def read_file(self, path: str) -> str:
        """Return the text of one file from the shared folder. `path` is relative, as shown by list_files."""
        target = self.inside_root(path)
        if not target.is_file():
            raise ToolError(f"no such file: {path!r}")
        size = target.stat().st_size
        if size > MAX_READ_BYTES:
            raise ToolError(f"{path!r} is {size} bytes; the limit is {MAX_READ_BYTES}")
        # utf-8-sig drops a leading byte order mark (Windows PowerShell 5 writes one
        # with -Encoding utf8) and reads files without one exactly like utf-8.
        return target.read_text(encoding="utf-8-sig", errors="replace")
