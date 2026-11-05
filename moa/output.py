"""How answers are shown in the terminal, and console encoding setup."""

from __future__ import annotations

import json
import sys
from typing import List, Optional

from .core import Answer

RULE = "=" * 72


def configure_console() -> None:
    """Make printing model answers safe on any console.

    Model answers may contain characters the console encoding cannot show
    (e.g. Hebrew on a legacy Windows console); replace them instead of crashing.

    When output goes to a pipe or a file (`--json > out.json`), Python uses the
    system code page by default (cp1255 for Hebrew Windows, not UTF-8), so the
    same text would be written differently on different machines. JSON must be
    UTF-8, so in that case we force it. An interactive console keeps its own
    handling, which already supports Unicode.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")


def print_answer(title: str, answer: Answer) -> None:
    print(RULE)
    print(title)
    if answer.ok:
        print(f"model: {answer.model} | time: {answer.elapsed_s:.1f}s")
        if answer.tools is not None:
            print(f"tools: {describe_tools(answer.tools)}")
        for warning in answer.warnings or []:
            print(f"warning: {warning}")
        print(RULE)
        print(answer.text.strip() if answer.text else "")
    else:
        print(f"ERROR after {answer.elapsed_s:.1f}s: {answer.error}")
        print(RULE)
    print()


def print_json(prompt: str, local: Answer, peer: Optional[Answer] = None, note: Optional[str] = None) -> None:
    result = {"prompt": prompt, "local": local.to_dict()}
    if peer is not None:
        result["peer"] = peer.to_dict()
    if note:
        result["note"] = note
    print(json.dumps(result, ensure_ascii=False, indent=2))


def describe_tools(tools: List[dict]) -> str:
    """One line per answer: which tools ran, with which arguments, and whether they worked."""
    if not tools:
        return "none used"
    parts = []
    for t in tools:
        arguments = ", ".join(f"{k}={v!r}" for k, v in (t.get("arguments") or {}).items())
        status = "ok" if t.get("ok") else f"FAILED ({t.get('error')})"
        parts.append(f"{t.get('name')}({arguments}) {status}")
    return "; ".join(parts)
