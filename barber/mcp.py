#!/usr/bin/env python3
"""barber as an MCP server — one tool, ``trim``, over stdio.

The hook in ``contrib/`` only works inside Claude Code, because it needs a
PostToolUse point that rewrites tool output and Claude Code is currently the
only agent that honours one. MCP is the portable shape: Cursor, Claude Code and
anything else that speaks the protocol can call this.

Run it directly (``barber-mcp`` after ``pip install barber-llm``, or
``python -m barber.mcp``) and register it with your client:

    {"mcpServers": {"barber": {"command": "barber-mcp"}}}

ponytail: hand-rolled JSON-RPC rather than the `mcp` SDK, because that SDK
pulls pydantic/anyio/httpx and this package advertises zero dependencies. The
ceiling is protocol drift: this implements initialize / tools/list / tools/call
and echoes the client's protocolVersion back. If MCP grows a handshake this
cannot satisfy, take the SDK as an extra rather than widening the base install.
"""
from __future__ import annotations

import json
import sys

from . import trim
from .core import SelectionConfig

TOOL = {
    "name": "trim",
    "description": (
        "Drop the parts of a block of text that are irrelevant to a question, "
        "keeping the rest byte-for-byte. Nothing is rewritten or summarized: "
        "passages survive verbatim or vanish, so quoting the result is safe. "
        "Use it on retrieved documents, long file contents or command output "
        "before reasoning over them. Returns the text unchanged when it finds "
        "no structure to work with, when the text is under 800 characters, or "
        "when there are fewer than 4 chunks to choose between."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "text": {"type": "string",
                     "description": "The text to trim."},
            "query": {"type": "string",
                      "description": "The question the text should answer. "
                                     "Relevance is judged against this."},
            "keep": {"type": "number", "minimum": 0, "maximum": 1,
                     "description": "Fraction of chunks retained per block. "
                                    "Defaults to 0.6, the benchmarked value."},
        },
        "required": ["text", "query"],
    },
}


def _trim(args: dict) -> str:
    """The two-message shape is the one barber documents: context in its own
    message, question last. Anything else is a no-op, which is why the tool
    builds it here rather than trusting a caller to."""
    text, query = args.get("text") or "", args.get("query") or ""
    if not text or not query:
        return text
    keep = args.get("keep")
    kw = {"keep": float(keep)} if isinstance(keep, (int, float)) else {}
    try:
        result = trim([{"role": "user", "content": text},
                       {"role": "user", "content": query}], **kw)
        return result.messages[0]["content"]
    except Exception:
        return text          # fail open, exactly as trim() itself does


def handle(req: dict) -> dict | None:
    """A JSON-RPC response, or None for a notification (which takes no reply)."""
    method, rid = req.get("method"), req.get("id")
    if rid is None:
        return None

    def ok(result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    if method == "initialize":
        # Echo the client's protocol version rather than asserting one, so a
        # newer client is not rejected by a server that does nothing exotic.
        version = (req.get("params") or {}).get("protocolVersion", "2025-06-18")
        return ok({"protocolVersion": version,
                   "capabilities": {"tools": {}},
                   "serverInfo": {"name": "barber",
                                  "version": _version()}})
    if method == "tools/list":
        return ok({"tools": [TOOL]})
    if method == "tools/call":
        params = req.get("params") or {}
        if params.get("name") != TOOL["name"]:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32602,
                              "message": f"unknown tool: {params.get('name')}"}}
        out = _trim(params.get("arguments") or {})
        return ok({"content": [{"type": "text", "text": out}],
                   "isError": False})
    return {"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": f"unknown method: {method}"}}


def _version() -> str:
    from . import __version__
    return __version__


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue                      # a malformed line is not fatal
        resp = handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
