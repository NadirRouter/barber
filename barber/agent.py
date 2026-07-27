"""barber.agent — drop the provably dead weight in a coding-agent transcript.

This is not selection. ``trim()`` asks "is this chunk relevant to the question?"
and answers with an embedding, which is a judgement that can be wrong. The
sweep here asks "is this content still true?" and answers from the transcript
itself, which cannot be wrong:

* a file body written before the same file was fully rewritten is not the file
  any more,
* a read of a file that has been modified since is not the file any more,
* a byte-identical repeat of an earlier tool payload says nothing new.

Nothing is judged, nothing is summarized, and every drop names the file so the
agent can read it again. On six real Claude Code sessions (1.0M tool tokens)
this is 15.7% of all tool tokens — about twice what selection finds in the same
transcripts, and complementary to it.

WHY THIS IS A SEPARATE FUNCTION, NOT PART OF ``trim()``
``trim()`` never rewrites history: its freeze-on-first-sight cache exists
specifically to keep the prefix byte-stable so the provider prompt cache stays
warm (see core.py, design decision 1). This sweep does the opposite — it edits
messages in the middle of the conversation, so every cached token after the
earliest edit has to be re-primed once. That trade is worth making when
context is tight and there are many turns left to amortize it over, and it is
a bad trade on turn three. So it is an explicit call you make at a compaction
point, not something ``trim()`` does behind your back.

    from barber.agent import sweep

    result = sweep(messages)          # -> TrimResult, same shape as trim()
    send(result.messages)
"""
from __future__ import annotations

import hashlib
import json

# Tools whose payload is a whole-file body: a later one replaces the file
# outright, which is what makes every earlier body for that path dead.
_FULL_WRITE = ("Write", "NotebookEdit")
# Tools that change a file at all: enough to make an earlier read of it stale,
# not enough to make an earlier full body dead (an Edit keeps the rest).
_MUTATES = ("Write", "Edit", "MultiEdit", "NotebookEdit")

SUPERSEDED = ("[… body omitted: {path} was rewritten later in this conversation, "
              "so this is not its current content — read the file if you need it …]")
STALE = ("[… read omitted: {path} was modified after this read, so this is not its "
         "current content — read the file again if you need it …]")
DUPLICATE = "[… identical to an earlier tool result in this conversation …]"


def _body_of(inner) -> str:
    if isinstance(inner, str):
        return inner
    if isinstance(inner, list):
        return "\n".join(p.get("text", "") for p in inner
                         if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _parts(messages: list) -> list[tuple[int, int, dict]]:
    return [(mi, pi, p)
            for mi, m in enumerate(messages)
            if isinstance(m.get("content"), list)
            for pi, p in enumerate(m["content"])
            if isinstance(p, dict)]


def sweep(messages: list, *, min_chars: int = 400) -> "TrimResult":
    """Replace superseded, stale, and duplicate tool payloads with markers.

    Blocks are never removed, only emptied: an Anthropic request rejects a
    ``tool_use`` with no matching ``tool_result``, so the structure has to
    survive even when the content does not.

    `min_chars` skips small payloads. Editing history costs one prompt-cache
    re-prime of everything after the edit, so clawing back 40 tokens from an
    early message is a net loss however good the accounting looks.
    """
    from . import TrimResult, _count_tokens, _token_counter

    parts = _parts(messages)
    # tool_use id -> (tool name, file path), so a result can be attributed to
    # the file its call was about.
    meta = {p.get("id"): (p.get("name"), (p.get("input") or {}).get("file_path"))
            for _, _, p in parts if p.get("type") == "tool_use" and p.get("id")}

    last_full_write: dict[str, int] = {}
    last_touch: dict[str, int] = {}
    for order, (_, _, p) in enumerate(parts):
        if p.get("type") != "tool_use":
            continue
        path = (p.get("input") or {}).get("file_path")
        if not path:
            continue
        if p.get("name") in _FULL_WRITE:
            last_full_write[path] = order
        if p.get("name") in _MUTATES:
            last_touch[path] = order

    # (msg index, part index) -> replacement marker
    edits: dict[tuple[int, int], str] = {}
    seen: set[str] = set()
    for order, (mi, pi, p) in enumerate(parts):
        kind = p.get("type")
        if kind == "tool_use":
            inp = p.get("input") or {}
            body = inp.get("content")
            if not isinstance(body, str) or len(body) < min_chars:
                continue
            path = inp.get("file_path")
            if path and last_full_write.get(path, -1) > order:
                edits[(mi, pi)] = SUPERSEDED.format(path=path)
        elif kind == "tool_result":
            body = _body_of(p.get("content"))
            if len(body) < min_chars:
                continue
            digest = hashlib.md5(body.encode()).hexdigest()
            if digest in seen:
                edits[(mi, pi)] = DUPLICATE
                continue
            seen.add(digest)
            tool, path = meta.get(p.get("tool_use_id"), (None, None))
            if not path:
                continue
            if tool == "Read" and last_touch.get(path, -1) > order:
                edits[(mi, pi)] = STALE.format(path=path)
            elif tool in _MUTATES and last_full_write.get(path, -1) > order:
                edits[(mi, pi)] = SUPERSEDED.format(path=path)

    if not edits:
        return TrimResult(messages=list(messages), tokens_saved=0,
                          chunks_dropped=0, changed=False)

    out = []
    for mi, m in enumerate(messages):
        hits = [pi for (emi, pi) in edits if emi == mi]
        if not hits:
            out.append(m); continue
        new_parts = []
        for pi, p in enumerate(m["content"]):
            marker = edits.get((mi, pi))
            if marker is None:
                new_parts.append(p)
            elif p.get("type") == "tool_use":
                new_parts.append({**p, "input": {**p["input"], "content": marker}})
            else:
                new_parts.append({**p, "content": marker})
        out.append({**m, "content": new_parts})

    ntok = _token_counter()
    saved = _sweep_tokens(messages, ntok) - _sweep_tokens(out, ntok)
    return TrimResult(messages=out, tokens_saved=saved,
                      chunks_dropped=len(edits), changed=True)


def _sweep_tokens(messages: list, ntok) -> int:
    """Count tool payload tokens. Not `barber._count_tokens`: that one counts
    text and tool results, and the sweep also rewrites `tool_use` inputs, which
    would otherwise vanish from the accounting and report a smaller saving than
    it delivered."""
    total = 0
    for _, _, p in _parts(messages):
        if p.get("type") == "tool_use":
            total += ntok(json.dumps(p.get("input") or {}, sort_keys=True))
        elif p.get("type") == "tool_result":
            total += ntok(_body_of(p.get("content")))
    for m in messages:
        if isinstance(m.get("content"), str):
            total += ntok(m["content"])
    return total
