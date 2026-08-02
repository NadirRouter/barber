#!/usr/bin/env python3
"""barber as a Claude Code PostToolUse hook — trim tool output before it lands.

EXPERIMENTAL. Not part of the barber public API; not covered by the published
benchmark. See the caveats at the bottom before running it on real work.

WHY POSTTOOLUSE AND NOT PRETOOLUSE OR PRECOMPACT
Claude Code re-sends the whole conversation every turn, so total input over an
N-turn session is roughly ``N*B + g*N**2/2`` — the growth rate ``g`` is the only
term with a square in it. A token removed before it ever enters context is
therefore saved once per remaining turn, which is why admission-time filtering
beats every retrospective cleanup available to a plugin.

``PostToolUse`` is the one hook that can act there: ``updatedToolOutput``
"Replaces the tool output before it is sent to the model" (Claude Code 2.1.211;
https://code.claude.com/docs/en/hooks). Nothing has entered context yet, so
unlike ``barber.agent.sweep`` this costs no prompt-cache re-prime at all — it is
a strictly better injection point than the one barber was designed for.

Measured on 12 real sessions (3,995 tool results, 958K tokens of tool output):
this policy fires 464 times and removes 21.5% of tool-output tokens, versus
12.1% for token-optimizer's first-read structure map — and it keeps survivors
byte-exact, where a structure map discards function bodies irrecoverably.

WHAT `keep` ACTUALLY COSTS. That run used keep=0.6, and the obvious worry about
defaulting to 0.8 is that it buys safety by giving up most of the saving. A
second replay says it does not. Over 30 sessions from 30 different projects
(3,970 tool results, 1.57M tokens of tool output, 799 of those results eligible
after the TRIMMABLE/min-chars/JSON gates):

    keep   fires   of eligible tokens   of all tool output   median session
    0.6     476           18.3%                10.4%              14.7%
    0.8     467           18.1%                10.3%              14.6%

Two decimal points of difference, because `keep` is a budget cap and the
relative floor is what actually does the cutting — the cap rarely binds. 0.8 is
therefore close to free, which is why it is the default.

Note the two denominators. Only 56.6% of tool-output tokens are eligible at
all, so "18.1% of eligible" and "10.3% of everything the tools emitted" are the
same removal described two ways. Quote whichever you mean, and say which.

INSTALL (as a Claude Code plugin — one command, nothing to pip install)
    claude plugin marketplace add NadirRouter/barber
    claude plugin install barber@barber
The repo hosts its own single-plugin marketplace, so the install clone already
contains the ``barber`` package this file imports; the path shim below finds it.
``hooks/hooks.json`` carries the PostToolUse registration, and its matcher must
stay in sync with ``TRIMMABLE`` below — tests/test_plugin.py enforces that.

INSTALL (by hand, settings.json — for a source checkout or a pinned interpreter)
    {"hooks": {"PostToolUse": [{"matcher": "Read|Grep|Glob|Bash|WebFetch",
      "hooks": [{"type": "command",
                 "command": "python3 /path/to/contrib/claude_code_hook.py"}]}]}}

TUNING
    BARBER_HOOK_KEEP=0.8     fraction of chunks retained (caveat 4 below; the
                             library's own default is 0.6, deliberately not
                             inherited here because a dropped tool result is
                             not recoverable from the model's side)
    BARBER_HOOK_MIN_CHARS=800  don't touch anything smaller
    BARBER_HOOK_DISABLE=1    off without editing settings
"""
from __future__ import annotations

import json
import os
import sys

# Make ``import barber`` work with nothing installed. This file lives at
# <root>/contrib/, and <root> holds the barber package — true in a source
# checkout and equally true in a plugin install, which is a clone of this repo.
# Resolved from __file__ rather than ${CLAUDE_PLUGIN_ROOT} so both cases work
# and neither depends on the environment the hook is spawned with.
# APPENDED, not prepended: an explicitly installed barber keeps winning, so this
# is only ever the fallback, and a broken install still reports itself below
# instead of being silently masked by the copy sitting next to this file.
_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if os.path.isdir(os.path.join(_ROOT, "barber")) and _ROOT not in sys.path:
    sys.path.append(_ROOT)

# Tools whose output is prose/listing the model reads, not a value it parses.
# Deliberately excludes Edit/Write/TodoWrite/Task: their results are short
# status payloads or structured objects, and trimming them buys nothing while
# risking a shape mismatch.
# Duplicated as the `matcher` regex in hooks/hooks.json, which decides whether
# this process is spawned at all; tests/test_plugin.py fails if the two drift.
TRIMMABLE = {"Read", "Grep", "Glob", "Bash", "WebFetch", "WebSearch",
             "NotebookRead", "BashOutput"}


def _question_and_args(payload: dict) -> str:
    """The information need: the human's live question plus this tool call's own
    arguments. The prompt is the intent; the grep pattern or file path is the
    precise per-output ask, and it is usually the sharper signal."""
    parts = []
    tp = payload.get("transcript_path")
    if tp and os.path.exists(tp):
        try:
            with open(tp) as f:
                lines = f.readlines()
            for line in reversed(lines[-400:]):        # recent turns only
                rec = json.loads(line)
                m = rec.get("message")
                if not isinstance(m, dict) or m.get("role") != "user":
                    continue
                c = m.get("content")
                if isinstance(c, str) and c.strip():
                    parts.append(c)
                    break
                if isinstance(c, list):
                    t = " ".join(p.get("text", "") for p in c
                                 if isinstance(p, dict) and p.get("type") == "text")
                    if t.strip():
                        parts.append(t)
                        break
        except Exception:
            pass                                        # a query is optional

    inp = payload.get("tool_input") or {}
    bits = [str(payload.get("tool_name") or "")]
    for k in ("pattern", "query", "command", "file_path", "path", "glob", "url"):
        v = inp.get(k)
        if isinstance(v, str) and v:
            bits.append(v[:200])
    parts.append(" ".join(bits))
    return "\n".join(p for p in parts if p.strip())


def _output_text(payload: dict) -> str | None:
    """Only a plain-string tool output is safe to replace: Claude Code validates
    the replacement against the tool's declared output shape and discards a
    mismatch, so structured results are left alone."""
    resp = payload.get("tool_response")
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("stdout", "content", "output", "text"):
            v = resp.get(k)
            if isinstance(v, str) and v:
                return v
    return None


def main() -> int:
    if os.environ.get("BARBER_HOOK_DISABLE"):
        return 0
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        return 0

    if payload.get("tool_name") not in TRIMMABLE:
        return 0

    text = _output_text(payload)
    min_chars = int(os.environ.get("BARBER_HOOK_MIN_CHARS", "800"))
    if not text or len(text) < min_chars:
        return 0
    # A JSON body is one value; dropping records out of the middle yields
    # something that no longer parses. barber's line splitter already declines
    # these, but the cheap check belongs at the boundary too.
    if text.lstrip()[:1] in "{[":
        return 0

    try:
        from barber.core import (SelectionConfig, SelectionStats, _select_block,
                                 _tokenize)
        from barber.embedders import lexical
    except ImportError as e:
        # Distinct from "nothing to trim": a hook that silently does nothing
        # because the package is not on its path looks exactly like a hook that
        # is working and finding no savings. stderr goes to the hook log, not
        # into the model's context.
        print(f"barber hook: cannot import barber ({e}); "
              f"install it into the interpreter running this hook "
              f"(pip install barber-llm)", file=sys.stderr)
        return 0

    try:
        # 0.8, not the library's 0.6: caveat 4 below is this file's own advice
        # and the default should not contradict it. Tool output is acted on,
        # not read, and the agent cannot ask for the dropped lines back.
        keep = float(os.environ.get("BARBER_HOOK_KEEP", "0.8"))
        cfg = SelectionConfig(min_message_chars=min_chars)
        query = _question_and_args(payload)
        if not query.strip():
            return 0
        entities = {w for w in _tokenize(query) if len(w) >= 4}
        new, changed = _select_block(text, query, lexical(), keep, cfg,
                                     entities, SelectionStats())
    except Exception:
        return 0        # fail-open: never break the tool call being optimized

    if not changed or len(new) >= len(text):
        return 0

    json.dump({"hookSpecificOutput": {
        "hookEventName": "PostToolUse",
        "updatedToolOutput": new,
    }}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# CAVEATS, because this is the part that decides whether you should run it:
#
# 1. The published benchmark does not cover this. barber's retention numbers
#    were judged on RAG passages answering a question, not on tool output an
#    agent is about to ACT on. Dropping the one grep hit the agent needed is a
#    different and worse failure than dropping a passage a reader didn't need.
# 2. The 21.5% above is potential, not net. A replay cannot model the agent
#    re-running a command because the first result came back thinner.
# 3. The lexical embedder matches words, not meaning. On tool output that is
#    mostly fine (the query's identifiers literally appear in the results), but
#    it is the weakest link — barber.embedders.sentence_transformers is better
#    if you can afford the latency inside a hook.
# 4. Start with BARBER_HOOK_KEEP=0.8 on real work and tighten only if nothing
#    breaks. Trimming a tool result is not reversible from the model's side.
#    0.8 is what this hook now defaults to, and the replay above shows that
#    costs 0.2 points of removal against 0.6. Turning it down is not the lever
#    it looks like; if you want materially more removed, the floor is the knob,
#    and that is a benchmarked decision this hook does not get to make.
