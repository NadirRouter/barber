"""Drive the MCP server the way a client does: real JSON-RPC over stdio."""
import json
import subprocess
import sys

from barber.mcp import TOOL, handle


def _session(*requests):
    """Spawn the server as a subprocess and read back one line per request."""
    stdin = "".join(json.dumps(r) + "\n" for r in requests)
    p = subprocess.run([sys.executable, "-m", "barber.mcp"], input=stdin,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return [json.loads(l) for l in p.stdout.splitlines() if l.strip()]


def test_handshake_then_list_then_call():
    body = "\n\n".join(
        [f"Passage {i}: facts about topic {i}." for i in range(40)])
    out = _session(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "trim",
                    "arguments": {"text": body,
                                  "query": "What about topic 7?"}}},
    )
    # The notification gets no reply, so three requests produce three responses.
    assert [r["id"] for r in out] == [1, 2, 3]
    assert out[0]["result"]["protocolVersion"] == "2025-06-18"
    assert out[1]["result"]["tools"][0]["name"] == "trim"
    trimmed = out[2]["result"]["content"][0]["text"]
    assert len(trimmed) < len(body)
    assert "topic 7." in trimmed          # the answer survived


def test_survivors_are_byte_exact():
    """The whole premise: kept text is a substring of the input, never reworded."""
    body = "\n\n".join([f"Passage {i}: facts about topic {i}." for i in range(40)])
    out = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                  "params": {"name": "trim",
                             "arguments": {"text": body,
                                           "query": "topic 7"}}})
    for line in out["result"]["content"][0]["text"].splitlines():
        line = line.strip()
        if line and not line.startswith("["):     # skip the drop marker
            assert line in body


def test_unknown_tool_and_method_are_errors_not_crashes():
    assert handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                   "params": {"name": "nope"}})["error"]["code"] == -32602
    assert handle({"jsonrpc": "2.0", "id": 2,
                   "method": "resources/list"})["error"]["code"] == -32601


def test_notifications_get_no_response():
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_missing_args_and_junk_fail_open():
    """A tool that mangles input is worse than one that declines it."""
    for args in ({}, {"text": "hi"}, {"query": "hi"}, {"text": "x", "query": "y"}):
        r = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": "trim", "arguments": args}})
        assert r["result"]["isError"] is False
        assert r["result"]["content"][0]["text"] == (args.get("text") or "")


def test_schema_declares_what_the_tool_reads():
    props = TOOL["inputSchema"]["properties"]
    assert set(TOOL["inputSchema"]["required"]) == {"text", "query"}
    assert set(props) == {"text", "query", "keep"}
