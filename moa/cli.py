"""Command line interface.

    python -m moa serve                          # answerer (machine B)
    python -m moa ask --peer 192.168.1.20 "..."  # asker (machine A), one prompt
    python -m moa ask --peer 192.168.1.20        # asker, interactive session
    python -m moa ask --peer 192.168.1.20 --serve  # both roles at once (symmetric)
    python -m moa ask --config moa.json "..."      # this machine only (its agent may
                                                   # still use remote MCP servers)

Add --config moa.json to either command to answer as an agent that can use
the MCP servers (connectors) listed in that file.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from typing import List, Optional, Tuple

from .agent import AgentConfig, AgentError
from .core import Answer, ask_both, ask_local
from .ollama_client import DEFAULT_OLLAMA_URL, OllamaClient, same_model
from .peer import DEFAULT_PEER_PORT, PeerServer

RULE = "=" * 72


def _parse_peer(value: str) -> Tuple[str, int]:
    """'host' or 'host:port' -> (host, port)."""
    if value.count(":") == 1:  # IPv6 addresses contain several colons
        host, _, port = value.partition(":")
        try:
            return host, int(port)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid port in {value!r}")
    return value, DEFAULT_PEER_PORT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m moa",
        description="Send one prompt to your own Ollama and to a second machine's Ollama, and get both answers.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--ollama", default=os.environ.get("MOA_OLLAMA_URL", DEFAULT_OLLAMA_URL),
                        help=f"URL of this machine's Ollama (default: {DEFAULT_OLLAMA_URL})")
    common.add_argument("--model", default=None,
                        help="model on this machine (default: first installed model)")
    common.add_argument("--token", default=os.environ.get("MOA_TOKEN"),
                        help="shared secret both peers must use (or env MOA_TOKEN)")
    common.add_argument("--timeout", type=float, default=300.0,
                        help="seconds to wait for a model answer (default: 300)")
    common.add_argument("--name", default=socket.gethostname(),
                        help="name this machine reports to the other side")
    common.add_argument("--config", default=os.environ.get("MOA_CONFIG"),
                        help="moa.json with MCP servers: answer as an agent that can use their tools (or env MOA_CONFIG)")

    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", parents=[common], help="answer prompts sent by the other machine")
    serve.add_argument("--host", default="0.0.0.0", help="interface to listen on (default: all)")
    serve.add_argument("--port", type=int, default=DEFAULT_PEER_PORT,
                       help=f"TCP port to listen on (default: {DEFAULT_PEER_PORT})")

    ask = sub.add_parser("ask", parents=[common], help="send a prompt to both machines")
    ask.add_argument("--peer", type=_parse_peer, metavar="HOST[:PORT]",
                     help=f"address of the other machine (port default: {DEFAULT_PEER_PORT}); "
                          "without it only this machine answers")
    ask.add_argument("--peer-model", default=None,
                     help="model to request on the other machine (default: its own default)")
    ask.add_argument("--diverse", action="store_true",
                     help="ask the other machine to answer with a different model than this one, if it has one")
    ask.add_argument("--json", action="store_true", help="print both answers as JSON")
    ask.add_argument("--serve", action="store_true",
                     help="also answer prompts from the other machine while this one asks")
    ask.add_argument("--listen-port", type=int, default=DEFAULT_PEER_PORT,
                     help="port for --serve (default: %(default)s)")
    ask.add_argument("prompt", nargs="*", help="the prompt; omit it for an interactive session")
    return parser


def _print_answer(title: str, answer: Answer) -> None:
    print(RULE)
    print(title)
    if answer.ok:
        print(f"model: {answer.model} | time: {answer.elapsed_s:.1f}s")
        if answer.tools is not None:
            print(f"tools: {_describe_tools(answer.tools)}")
        for warning in answer.warnings or []:
            print(f"warning: {warning}")
        print(RULE)
        print(answer.text.strip() if answer.text else "")
    else:
        print(f"ERROR after {answer.elapsed_s:.1f}s: {answer.error}")
        print(RULE)
    print()


def _describe_tools(tools) -> str:
    if not tools:
        return "none used"
    parts = []
    for t in tools:
        arguments = ", ".join(f"{k}={v!r}" for k, v in (t.get("arguments") or {}).items())
        status = "ok" if t.get("ok") else f"FAILED ({t.get('error')})"
        parts.append(f"{t.get('name')}({arguments}) {status}")
    return "; ".join(parts)


def _run_local_only(args, ollama: OllamaClient, prompt: str) -> bool:
    local = ask_local(prompt, ollama, model=args.model, node_name=args.name, agent=args.agent)
    if args.json:
        print(json.dumps({"prompt": prompt, "local": local.to_dict()}, ensure_ascii=False, indent=2))
    else:
        _print_answer(f"Answer - this machine ({local.node})", local)
    return local.ok


def _run_once(args, ollama: OllamaClient, prompt: str) -> bool:
    if args.peer is None:
        return _run_local_only(args, ollama, prompt)
    host, port = args.peer
    if not args.json:
        print(f"Sending the prompt to this machine and to {host}:{port} ...\n")
    local, remote = ask_both(
        prompt, ollama, host, port,
        local_model=args.model, peer_model=args.peer_model,
        token=args.token, timeout=args.timeout, node_name=args.name,
        diverse=args.diverse, agent=args.agent,
    )
    note = None
    if args.diverse and local.ok and remote.ok and same_model(local.model, remote.model):
        note = f"the other machine has no model other than {remote.model}, so both answers used it"

    if args.json:
        result = {"prompt": prompt, "local": local.to_dict(), "peer": remote.to_dict()}
        if note:
            result["note"] = note
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_answer(f"Answer 1 - this machine ({local.node})", local)
        _print_answer(f"Answer 2 - other machine ({remote.node})", remote)
        if note:
            print(f"Note: {note}\n")
    return local.ok and remote.ok


def _cmd_serve(args, ollama: OllamaClient) -> int:
    server = PeerServer(ollama, host=args.host, port=args.port, model=args.model,
                        token=args.token, node_name=args.name, agent=args.agent)
    host, port = server.address
    print(f"[serve] '{args.name}' listening on {host}:{port}, using Ollama at {args.ollama}")
    print("[serve] token required" if args.token else "[serve] WARNING: no token set, anyone on the network can send prompts")
    if args.agent is not None:
        allowed = ", ".join(sorted(args.agent.remote_allowed_tools)) or "none"
        print(f"[serve] agent mode: MCP servers {', '.join(s.name for s in args.agent.servers) or 'none'}; "
              f"tools allowed for remote prompts: {allowed}")
    print("[serve] press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopping")
    finally:
        server.shutdown()
    return 0


def _cmd_ask(args, ollama: OllamaClient) -> int:
    server = None
    if args.serve:
        server = PeerServer(ollama, port=args.listen_port, model=args.model,
                            token=args.token, node_name=args.name, agent=args.agent)
        server.start_in_background()
        print(f"[serve] also answering the other machine on port {server.address[1]}")

    try:
        if args.prompt:
            return 0 if _run_once(args, ollama, " ".join(args.prompt)) else 1

        print("Interactive mode. Type a prompt and press Enter (empty line or 'exit' to quit).")
        while True:
            try:
                prompt = input("\nprompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt or prompt.lower() in {"exit", "quit"}:
                break
            _run_once(args, ollama, prompt)
        return 0
    finally:
        if server is not None:
            server.shutdown()


def main(argv: Optional[List[str]] = None) -> int:
    # Model answers may contain characters the console encoding cannot show
    # (e.g. Hebrew on a legacy Windows console); replace them instead of crashing.
    #
    # When output goes to a pipe or a file (`--json > out.json`), Python uses the
    # system code page by default (cp1255 for Hebrew Windows, not UTF-8), so the
    # same text would be written differently on different machines. JSON must be
    # UTF-8, so in that case we force it. An interactive console keeps its own
    # handling, which already supports Unicode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.agent = AgentConfig.load(args.config) if args.config else None
    except AgentError as exc:
        parser.error(str(exc))
    ollama = OllamaClient(args.ollama, timeout=args.timeout)
    if args.command == "serve":
        return _cmd_serve(args, ollama)
    return _cmd_ask(args, ollama)
