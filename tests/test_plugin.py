"""The Claude Code plugin: the knowledge duplicated between hooks/hooks.json,
.claude-plugin/*.json and contrib/claude_code_hook.py, pinned so it cannot drift.

None of this is exercised by importing barber. The failure mode it guards is a
plugin that installs cleanly and silently does nothing: a matcher that no longer
lists a tool the hook would have trimmed, or a command pointing at a file that
moved. Both look exactly like a hook finding no savings.
"""
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOOKS_JSON = ROOT / "hooks" / "hooks.json"
PLUGIN_JSON = ROOT / ".claude-plugin" / "plugin.json"
MARKETPLACE_JSON = ROOT / ".claude-plugin" / "marketplace.json"
HOOK_PY = ROOT / "contrib" / "claude_code_hook.py"


def _load_hook_module():
    """Import the hook by path: contrib/ is not a package and never ships in the
    wheel, so there is no importable name for it."""
    spec = importlib.util.spec_from_file_location("_barber_hook", HOOK_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _post_tool_use_entry():
    cfg = json.loads(HOOKS_JSON.read_text())
    entries = cfg["hooks"]["PostToolUse"]
    assert len(entries) == 1, "one matcher, or the sync check below is ambiguous"
    return entries[0]


def test_matcher_lists_exactly_the_tools_the_hook_will_trim():
    """The matcher decides whether this process is spawned; TRIMMABLE decides
    whether it does anything once it is. A tool in TRIMMABLE but not the matcher
    is never trimmed at all. A tool in the matcher but not TRIMMABLE spawns a
    Python process per tool call to immediately return 0."""
    matcher = _post_tool_use_entry()["matcher"]
    assert re.fullmatch(r"\w+(\|\w+)*", matcher), \
        "kept a plain alternation so this comparison is a split, not a regex parse"
    assert set(matcher.split("|")) == _load_hook_module().TRIMMABLE
    assert matcher.split("|") == sorted(matcher.split("|")), \
        "sorted, so a diff to this line is readable"


def test_the_command_points_at_a_file_that_exists():
    hook, = _post_tool_use_entry()["hooks"]
    assert hook["type"] == "command"
    cmd = hook["command"]
    # Quoted this way per the plugin docs, so a plugin root containing a space
    # survives the shell. Unquoted ${CLAUDE_PLUGIN_ROOT} is the classic break.
    assert '"${CLAUDE_PLUGIN_ROOT}"/' in cmd, cmd
    rel = cmd.split('"${CLAUDE_PLUGIN_ROOT}"/', 1)[1].strip()
    assert (ROOT / rel).is_file(), f"{rel} does not exist in the plugin root"


def test_the_plugin_ships_the_package_the_hook_imports():
    """The whole install story: no pip step, because the clone Claude Code makes
    already contains barber/ next to the hook. If the marketplace ever stops
    pointing at the repo root, the hook's sys.path shim stops finding it."""
    assert json.loads(MARKETPLACE_JSON.read_text())["plugins"][0]["source"] == "./"
    assert (ROOT / "barber" / "__init__.py").is_file()


def test_manifest_and_marketplace_agree_on_the_plugin_name():
    plugin = json.loads(PLUGIN_JSON.read_text())
    market = json.loads(MARKETPLACE_JSON.read_text())
    assert len(market["plugins"]) == 1
    # A marketplace entry name that differs from plugin.json's silently wins for
    # `/plugin install`, which is a confusing way to find out they disagree.
    assert market["plugins"][0]["name"] == plugin["name"] == "barber"


def test_plugin_version_tracks_the_package_version():
    """The plugin ships this source tree, so its version is that tree's version.
    Claude Code pins to the string in plugin.json: leave it stale and installed
    users never receive an update, and `/plugin update` reports success."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    version = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject).group(1)
    assert json.loads(PLUGIN_JSON.read_text())["version"] == version


def test_the_hook_stays_out_of_the_wheel():
    """release.yml refuses to publish a wheel containing contrib/. That rests on
    packages = ["barber"], and the plugin is the reason someone might be tempted
    to widen it: the plugin is a separate channel, the quarantine still holds."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    wheel = pyproject.split("[tool.hatch.build.targets.wheel]", 1)[1]
    wheel = wheel.split("[tool.hatch.build.targets.sdist]", 1)[0]
    assert 'packages = ["barber"]' in wheel
