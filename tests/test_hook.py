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


def test_works_with_barber_installed_nowhere_at_all(tmp_path):
    """The whole claim of the plugin install: no pip step. `-E -S` removes
    PYTHONPATH and site-packages, so nothing can supply barber except the hook's
    own sys.path shim resolving <this file>/../ — which in a plugin install is
    the clone Claude Code made, package included. cwd is elsewhere so a stray
    relative path cannot rescue it either.
    """
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    p = subprocess.run([sys.executable, "-E", "-S", str(HOOK)],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path),
                       input=json.dumps({"tool_name": "Grep",
                                         "tool_input": {"pattern": "refresh_token"},
                                         "tool_response": GREP_HITS}))
    assert p.returncode == 0, p.stderr
    assert "cannot import barber" not in p.stderr, p.stderr
    new = json.loads(p.stdout)["hookSpecificOutput"]["updatedToolOutput"]
    assert len(new) < len(GREP_HITS) and "refresh_token" in new


def test_missing_barber_says_so_instead_of_going_quiet(tmp_path):
    """A hook that silently does nothing because the import failed looks exactly
    like one that is working and finding no savings.

    The unimportable barber is SIMULATED with a stub package first on
    PYTHONPATH, not by pointing PYTHONPATH somewhere empty: CI installs barber
    into site-packages (`pip install .`), so an empty path proves nothing there
    and the test passed only on a machine where barber happened to be missing.

    It is also why the hook's sys.path shim APPENDS the repo root instead of
    prepending it: PYTHONPATH still wins, so a barber that is present and broken
    is still reported rather than masked by the copy sitting next to the hook.
    """
    stub = tmp_path / "barber"
    stub.mkdir()
    (stub / "__init__.py").write_text("raise ImportError('simulated: barber not installed')\n")

    env = {**os.environ, "PYTHONPATH": str(tmp_path)}
    p = subprocess.run([sys.executable, str(HOOK)], capture_output=True, text=True, env=env,
                       input=json.dumps({"tool_name": "Grep", "tool_input": {},
                                         "tool_response": GREP_HITS}), cwd=str(tmp_path))
    assert p.returncode == 0, "a broken install must not fail the tool call"
    assert p.stdout == "", "no envelope means the tool output passes through untouched"
    assert "cannot import barber" in p.stderr, "the failure left no trace in the hook log"
