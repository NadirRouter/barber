"""contrib/claude_code_hook.py: the envelope Claude Code accepts, and the
refusals that keep it from breaking a tool call it does not understand."""
import json
import os
import subprocess
import sys
from pathlib import Path

HOOK = Path(__file__).resolve().parent.parent / "contrib" / "claude_code_hook.py"
REPO = str(Path(__file__).resolve().parent.parent)

GREP_HITS = "\n\n".join(
    f"src/mod{i}.py:{i * 7}: def handler_{i}(request): return route(request, timeout={i})"
    for i in range(1, 25)
) + "\n\nsrc/auth/session.py:14: def refresh_token(session): return rotate(session.refresh_token)"


def run(payload: dict):
    env = {**os.environ, "PYTHONPATH": REPO}
    p = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(payload),
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    return p.stdout


def test_emits_the_envelope_claude_code_accepts():
    out = run({"tool_name": "Grep", "tool_input": {"pattern": "refresh_token"},
               "tool_response": GREP_HITS})
    hs = json.loads(out)["hookSpecificOutput"]
    # Field names are load-bearing: Claude Code ignores a payload it cannot
    # match, and an ignored hook is indistinguishable from one that saved nothing.
    assert hs["hookEventName"] == "PostToolUse"
    new = hs["updatedToolOutput"]
    assert len(new) < len(GREP_HITS)
    assert "refresh_token" in new, "dropped the line the pattern actually asked for"


def test_declines_what_it_could_break():
    """Each of these must pass through untouched, i.e. emit nothing at all."""
    big = GREP_HITS
    assert run({"tool_name": "Edit", "tool_input": {}, "tool_response": big}) == "", \
        "Edit results are status payloads; trimming them risks a shape mismatch"
    assert run({"tool_name": "Grep", "tool_input": {},
                "tool_response": '{"matches": [' + '"x",' * 400 + '"y"]}'}) == "", \
        "a JSON body is one value; dropping records from the middle stops it parsing"
    assert run({"tool_name": "Grep", "tool_input": {}, "tool_response": "short"}) == "", \
        "below the size floor there is nothing worth the risk"
    assert run({"tool_name": "Grep", "tool_input": {}, "tool_response": {"rows": [1, 2]}}) == "", \
        "structured output is not a string and must not be replaced with one"


def test_fail_open_on_garbage_input():
    env = {**os.environ, "PYTHONPATH": REPO}
    p = subprocess.run([sys.executable, str(HOOK)], input="not json at all",
                       capture_output=True, text=True, env=env)
    assert p.returncode == 0 and p.stdout == ""


def test_missing_barber_says_so_instead_of_going_quiet():
    """A hook that silently does nothing because the import failed looks exactly
    like one that is working and finding no savings."""
    env = {**os.environ, "PYTHONPATH": "/nonexistent"}
    p = subprocess.run([sys.executable, str(HOOK)], capture_output=True, text=True, env=env,
                       input=json.dumps({"tool_name": "Grep", "tool_input": {},
                                         "tool_response": GREP_HITS}), cwd="/tmp")
    assert p.returncode == 0 and p.stdout == ""
    assert "cannot import barber" in p.stderr
