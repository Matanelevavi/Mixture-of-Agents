"""A tiny MCP server with one tool that changes something, for the approval tests.

    python -m tests.notes_server --file notes.txt

add_note appends a line to the file. It is not marked read-only, so the agent
must ask before running it; the tests check the file to see whether it ran.
"""

import argparse
from pathlib import Path

from mcp.server.mcpserver import MCPServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    notes = Path(parser.parse_args().file)

    server = MCPServer("notes", log_level="ERROR")

    @server.tool()
    def add_note(text: str) -> str:
        """Append one line to the notes file."""
        with notes.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
        return f"added: {text}"

    server.run()


if __name__ == "__main__":
    main()
