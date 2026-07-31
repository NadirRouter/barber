#!/usr/bin/env python3
"""Assert the four version strings in this repo agree.

There are four, and a release that bumps only some of them ships a package
that lies about itself:

    pyproject.toml      version =                 -> what PyPI serves
    barber/__init__.py  __version__               -> what `barber.__version__` says
    js/package.json     version                   -> what npm serves
    js/index.js         export const VERSION      -> what `barber --version` prints

0.4.0 went to PyPI with __version__ still reading "0.3.1" and the JS CLI still
printing "0.2.1", because only the two manifests were bumped. 0.4.1 exists for
no reason other than to correct that.

TWO PAIRS, NOT ONE SET. Python and JS version independently and are meant to
disagree with each other: the npm port has trim() but not sweep() or the eval
harness, so PyPI runs ahead. What must never disagree is a registry manifest
and the constant the code it ships reports. So this compares
pyproject <-> __init__ and package.json <-> index.js, and says nothing about
Python <-> JS.

WHY NOT gen_golden.py --check. That guard cannot catch this even in principle:
it stamps `barber.__version__` into the fixture it then compares against, so a
stale __version__ produces a stale-in-the-same-way fixture and the check stays
green. Leaving it as the only version-adjacent gate is how 0.4.0 shipped.

Run it directly (python tests/test_versions.py) or let pytest collect it.
Everything here is read off the source text, never off an import: the point is
what this tree will publish, not what happens to be installed in the
environment running the check.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (label, path, how to pull the version out of that file's text)
SOURCES = {
    "python": [
        ("pyproject.toml", "pyproject.toml",
         lambda t: re.search(r'(?m)^version\s*=\s*"([^"]+)"', t)),
        ("barber/__init__.py", "barber/__init__.py",
         lambda t: re.search(r'(?m)^__version__\s*=\s*"([^"]+)"', t)),
    ],
    "js": [
        ("js/package.json", "js/package.json",
         lambda t: re.search(r'"version"\s*:\s*"([^"]+)"', t)),
        ("js/index.js", "js/index.js",
         lambda t: re.search(r'(?m)^export\s+const\s+VERSION\s*=\s*"([^"]+)"', t)),
    ],
}


def read_versions() -> dict[str, list[tuple[str, str]]]:
    """{"python": [(label, version), ...], "js": [...]}.

    A pattern that stops matching is a failure, not a skip: the file was
    reformatted or renamed and this check silently stopped guarding anything.
    """
    out: dict[str, list[tuple[str, str]]] = {}
    for lang, entries in SOURCES.items():
        found = []
        for label, rel, extract in entries:
            path = ROOT / rel
            if not path.exists():
                raise AssertionError(f"{label}: missing (looked in {path})")
            m = extract(path.read_text(encoding="utf-8"))
            if not m:
                raise AssertionError(
                    f"{label}: no version literal matched. The declaration moved or "
                    f"changed shape; fix the pattern in tests/test_versions.py, "
                    f"because as written it is no longer checking anything."
                )
            found.append((label, m.group(1)))
        out[lang] = found
    return out


def _report(lang: str, found: list[tuple[str, str]]) -> str | None:
    versions = {v for _, v in found}
    if len(versions) == 1:
        return None
    width = max(len(label) for label, _ in found)
    lines = [f"{lang} version strings disagree:"]
    lines += [f"    {label:<{width}} = {v}" for label, v in found]
    lines.append("    all of these must carry the same number in one commit.")
    return "\n".join(lines)


def check() -> list[str]:
    return [msg for lang, found in read_versions().items()
            if (msg := _report(lang, found))]


def test_python_versions_agree():
    found = read_versions()["python"]
    assert _report("python", found) is None, _report("python", found)


def test_js_versions_agree():
    found = read_versions()["js"]
    assert _report("js", found) is None, _report("js", found)


def test_versions_are_pep440_and_semver_shaped():
    """A trailing space or a stray 'v' is a broken publish, not a typo."""
    for lang, found in read_versions().items():
        for label, v in found:
            assert re.fullmatch(r"\d+\.\d+\.\d+(?:[-.a-zA-Z0-9]*)", v), \
                f"{label}: {v!r} is not a plain X.Y.Z version string"


def test_package_json_parses():
    """The regex above reads package.json as text, so confirm separately that
    the file is still valid JSON and that its `version` is the one matched."""
    data = json.loads((ROOT / "js" / "package.json").read_text(encoding="utf-8"))
    assert data["version"] == dict(read_versions()["js"])["js/package.json"]


def main() -> int:
    problems = check()
    if problems:
        print("\n\n".join(problems), file=sys.stderr)
        print("\nPython and JS are allowed to differ from EACH OTHER; the files "
              "within one language are not.", file=sys.stderr)
        return 1
    for lang, found in read_versions().items():
        print(f"ok: {lang} at {found[0][1]} "
              f"({', '.join(label for label, _ in found)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
