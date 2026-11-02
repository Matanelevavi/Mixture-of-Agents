"""Command line interface.

    python -m moa serve                          # answerer (machine B)
    python -m moa ask --peer 192.168.1.20 "..."  # asker (machine A), one prompt
    python -m moa ask --peer 192.168.1.20        # asker, interactive session
    python -m moa ask --peer 192.168.1.20 --serve  # both roles at once (symmetric)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from typing import List, Optional, Tuple

from .core import Answer, ask_both
from .ollama_client import DEFAULT_OLLAMA_URL, OllamaClient
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

    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", parents=[common], help="answer prompts sent by the other machine")
    serve.add_argument("--host", default="0.0.0.0", help="interface to listen on (default: all)")
    serve.add_argument("--port", type=int, default=DEFAULT_PEER_PORT,
                       help=f"TCP port to listen on (default: {DEFAULT_PEER_PORT})")

    ask = sub.add_parser("ask", parents=[common], help="send a prompt to both machines")
    ask.add_argument("--peer", required=True, type=_parse_peer, metavar="HOST[:PORT]",
                     help=f"address of the other machine (port default: {DEFAULT_PEER_PORT})")
    ask.add_argument("--peer-model", default=None,
                     help="model to request on the other machine (default: its own default)")
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
        print(RULE)
        print(answer.text.strip() if answer.text else "")
    else:
        print(f"ERROR after {answer.elapsed_s:.1f}s: {answer.error}")
        print(RULE)
    print()


def _run_once(args, ollama: OllamaClient, prompt: str) -> bool:
    host, port = args.peer
    if not args.json:
        print(f"Sending the prompt to this machine and to {host}:{port} ...\n")
    local, remote = ask_both(
        prompt, ollama, host, port,
        local_model=args.model, peer_model=args.peer_model,
        token=args.token, timeout=args.timeout, node_name=args.name,
    )
    if args.json:
        print(json.dumps({"prompt": prompt, "local": local.to_dict(), "peer": remote.to_dict()},
                         ensure_ascii=False, indent=2))
    else:
        _print_answer(f"Answer 1 - this machine ({local.node})", local)
        _print_answer(f"Answer 2 - other machine ({remote.node})", remote)
    return local.ok and remote.ok


def _cmd_serve(args, ollama: OllamaClient) -> int:
    server = PeerServer(ollama, host=args.host, port=args.port,
                        model=args.model, token=args.token, node_name=args.name)
    host, port = server.address
    print(f"[serve] '{args.name}' listening on {host}:{port}, using Ollama at {args.ollama}")
    print("[serve] token required" if args.token else "[serve] WARNING: no token set, anyone on the network can send prompts")
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
                            token=args.token, node_name=args.name)
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
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    args = _build_parser().parse_args(argv)
    ollama = OllamaClient(args.ollama, timeout=args.timeout)
    if args.command == "serve":
        return _cmd_serve(args, ollama)
    return _cmd_ask(args, ollama)
