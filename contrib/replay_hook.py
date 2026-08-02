#!/usr/bin/env python3
"""Replay the barber PostToolUse hook over real Claude Code transcripts at
several `keep` settings, to answer one question: what does the hook actually
remove at its new default of 0.8, versus the 0.6 the published 21.5% was
measured at?

Faithfulness matters more than speed here, so this imports the shipped hook
module and reuses its own TRIMMABLE set, its `_output_text`, and its
`_question_and_args` argument extraction rather than reimplementing any of it.
The one thing it cannot reuse is the transcript read inside
`_question_and_args` (that walks a live file backwards from the end); the
replay supplies the equivalent by tracking the most recent user text message
as it walks the transcript forward, which is the same message the hook would
have found at that point in the session.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO)

spec = importlib.util.spec_from_file_location(
    "barber_hook", os.path.join(REPO, "contrib", "claude_code_hook.py"))
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)

from barber.core import SelectionConfig, SelectionStats, _select_block, _tokenize
from barber.embedders import lexical

import tiktoken
ENC = tiktoken.get_encoding("o200k_base")          # what barber._token_counter uses
def ntok(s: str) -> int:
    return len(ENC.encode(s, disallowed_special=()))

MIN_CHARS = 800
KEEPS = [0.6, 0.7, 0.8, 0.9]


def user_text(msg) -> str:
    """The text of a user turn, or '' for a tool-result-only turn. Mirrors the
    content walk in the hook's _question_and_args."""
    c = msg.get("content")
    if isinstance(c, str):
        return c if c.strip() else ""
    if isinstance(c, list):
        t = " ".join(p.get("text", "") for p in c
                     if isinstance(p, dict) and p.get("type") == "text")
        return t if t.strip() else ""
    return ""


def walk(path):
    """Yield (tool_name, tool_input, output_text, live_question) per tool result,
    in transcript order."""
    pending = {}          # tool_use_id -> (name, input)
    question = ""
    for line in open(path, errors="replace"):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")

        if msg.get("role") == "assistant" and isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("type") == "tool_use":
                    pending[p.get("id")] = (p.get("name"), p.get("input") or {})

        if msg.get("role") == "user":
            t = user_text(msg)
            if t:
                question = t
            if isinstance(content, list):
                for p in content:
                    if not (isinstance(p, dict) and p.get("type") == "tool_result"):
                        continue
                    name, tin = pending.get(p.get("tool_use_id"), (None, {}))
                    if not name:
                        continue
                    # `toolUseResult` is the structured value Claude Code hands a
                    # hook as tool_response; fall back to the rendered part.
                    resp = rec.get("toolUseResult", p.get("content"))
                    out = hook._output_text({"tool_response": resp})
                    if out is None and isinstance(p.get("content"), str):
                        out = p["content"]
                    if out:
                        yield name, tin, out, question


def eligible(name, text) -> bool:
    """The hook's own gate, in the hook's own order."""
    return (name in hook.TRIMMABLE
            and len(text) >= MIN_CHARS
            and text.lstrip()[:1] not in "{[")


def replay(paths):
    emb = lexical()
    cfg = SelectionConfig(min_message_chars=MIN_CHARS)
    totals = {"out_tok": 0, "results": 0, "elig_tok": 0, "elig": 0}
    removed = {k: 0 for k in KEEPS}
    fired = {k: 0 for k in KEEPS}
    per_session = {k: [] for k in KEEPS}

    for path in paths:
        s_out = s_removed = 0
        s_rem = {k: 0 for k in KEEPS}
        for name, tin, text, question in walk(path):
            tk = ntok(text)
            totals["out_tok"] += tk
            totals["results"] += 1
            if not eligible(name, text):
                continue
            q = hook._question_and_args({"tool_name": name, "tool_input": tin})
            if question:
                q = question + "\n" + q
            if not q.strip():
                continue
            totals["elig_tok"] += tk
            totals["elig"] += 1
            s_out += tk
            ents = {w for w in _tokenize(q) if len(w) >= 4}
            for k in KEEPS:
                try:
                    new, changed = _select_block(text, q, emb, k, cfg, ents,
                                                 SelectionStats())
                except Exception:
                    continue
                if changed and len(new) < len(text):
                    fired[k] += 1
                    d = tk - ntok(new)
                    if d > 0:
                        removed[k] += d
                        s_rem[k] += d
        if s_out:
            for k in KEEPS:
                per_session[k].append(100 * s_rem[k] / s_out)
    return totals, removed, fired, per_session


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    # usage: replay_hook.py [sessions] [transcript-root]
    root = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
        "~/.claude/projects")
    if os.path.isfile(root):                 # one named transcript: /barber:stats
        files = [root]
    else:
        files = (glob.glob(os.path.join(root, "*", "*.jsonl"))
                 or glob.glob(os.path.join(root, "*.jsonl")))
    if not files:
        sys.exit(f"no transcripts under {root}")
    files.sort(key=lambda p: -os.path.getsize(p))
    # One transcript per project directory, largest first: 30 sessions from 30
    # different codebases, not 30 slices of the same one.
    seen, picked = set(), []
    for f in files:
        d = os.path.dirname(f)
        if d in seen:
            continue
        seen.add(d)
        picked.append(f)
        if len(picked) >= n:
            break

    totals, removed, fired, per_session = replay(picked)
    print(f"corpus: {len(picked)} sessions from {len(seen)} projects")
    print(f"  {totals['results']:,} tool results, {totals['out_tok']:,} tokens of tool output")
    print(f"  {totals['elig']:,} eligible ({totals['elig_tok']:,} tokens, "
          f"{100*totals['elig_tok']/max(totals['out_tok'],1):.1f}% of output)\n")
    print(f"{'keep':>6} {'fires':>7} {'% all output':>14} {'% eligible':>12} "
          f"{'median session':>16}")
    for k in KEEPS:
        pct_all = 100 * removed[k] / max(totals["out_tok"], 1)
        pct_el = 100 * removed[k] / max(totals["elig_tok"], 1)
        med = statistics.median(per_session[k]) if per_session[k] else 0.0
        print(f"{k:>6} {fired[k]:>7,} {pct_all:>13.1f}% {pct_el:>11.1f}% {med:>15.1f}%")


if __name__ == "__main__":
    main()
