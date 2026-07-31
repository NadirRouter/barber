# Releasing barber

Maintainer notes. Nothing here is needed to *use* the package.

Two registries, two independent version lines:

| | Package | Version lives in |
|---|---|---|
| PyPI | `barber-llm` | `pyproject.toml`, `barber/__init__.py` |
| npm | `barber-llm` | `js/package.json`, `js/index.js` (`VERSION`) |

**PyPI and npm are not meant to agree with each other** and no release is ever
cut to make them agree. They ship different surfaces — `sweep()`, the semantic
embedders and the eval harness are Python-only — so PyPI runs ahead. Within one
language the two files must always agree; that is what
`python tests/test_versions.py` enforces, in CI and in `release.yml`'s test job.

## The four-version trap

0.4.0 shipped to PyPI reporting `barber.__version__ == "0.3.1"`, and its CLI
printed `0.2.1`, because the bump touched only the two manifests. A PyPI upload
cannot be replaced or re-uploaded — yanking does not free the number — so the
only remedy was 0.4.1, a patch release that fixed nothing but its own version
string.

`python tests/gen_golden.py --check` does **not** cover this, despite sounding
like it might. It stamps `barber.__version__` into the golden fixture and then
compares against a fresh run, so a stale `__version__` regenerates a
stale-in-the-same-way fixture and the check passes. It gates selection
decisions, not versions.

Run this before every release, manual or automated:

```bash
python tests/test_versions.py
```

It names the disagreeing file and prints both values.

## Cutting a release

1. Bump **all four** locations in one commit. Python and JS bump independently:
   a Python-only change bumps the two Python files and leaves `js/` alone.
2. `python tests/test_versions.py && pytest tests/ -q && python tests/gen_golden.py --check`
3. `cd js && node --test`
4. Commit, push to `main`, wait for `ci.yml` to be green.
5. Tag `vX.Y.Z` and push the tag. That triggers `release.yml`, which re-runs
   the suite against the tagged tree and then publishes — see below for why the
   publish half currently does not work.
6. npm is separate and manual: `cd js && npm publish` (see
   [Publishing to npm](#publishing-to-npm)).

## PyPI Trusted Publishing: NOT REGISTERED

`release.yml` is written for Trusted Publishing (OIDC): GitHub proves the
workflow's identity to PyPI directly, so no API token exists anywhere to store,
paste, leak or rotate.

**The trusted publisher has never been registered on PyPI.** Until it is, the
publish step cannot work and every release goes up by hand.

Only the repo owner can register it — it is an account action on PyPI, not
something that can be done from this repository or by anyone without owner
rights on the project. At
<https://pypi.org/manage/project/barber-llm/settings/publishing/>, add a
trusted publisher to the **existing** project:

| Field | Value |
|---|---|
| Owner | `NadirRouter` |
| Repository | `barber` |
| Workflow filename | `release.yml` — the filename, not a path, not the workflow's `name:` |
| Environment | **leave blank** |

Three ways this is commonly filled in wrong:

- **"Pending publisher"** is the form for projects that do not exist yet. It
  will never match `barber-llm`, which is already on PyPI. Use the project's own
  publishing settings page.
- **Environment** must be empty. `release.yml` declares no `environment:`, so
  the OIDC token carries no environment claim, and any value typed in that box
  can never match.
- **Workflow filename** is `release.yml`. Not `.github/workflows/release.yml`,
  and not `release` (the `name:` at the top of the file).

If the publish job later fails at "Prove PyPI accepts this workflow's identity",
that step prints the OIDC claims PyPI matched against, field by field. Compare
them line by line with the form above; "no corresponding publisher" on its own
tells you nothing about which field disagrees.

## A green `release` run does not prove publishing works

This is the failure mode most likely to bite. `release.yml`'s publish job asks
"is there anything to do?" **first**, and a version already on PyPI is treated
as a no-op, not an error — correctly, since the upload cannot be repeated. But
that step returns green and skips everything after it, **including the auth
step**. So:

> A `release` run against an already-published version exits green without ever
> contacting PyPI's OIDC endpoint. It says nothing at all about whether the
> trusted publisher is registered.

The only run that tests the publisher is one on a version genuinely not yet on
PyPI. Re-running the workflow on an old tag, or dispatching it manually on a
tree whose version is already up, proves nothing. Do not read those green
checkmarks as "publishing is configured".

Corollary: the first real test of Trusted Publishing will be the first release
after it is registered. Cut that one with time to fall back to the manual path.

## Manual upload (the current reality)

Every release so far has gone up this way, and every release will until the
trusted publisher exists.

**A manual upload bypasses `release.yml` entirely**, and with it the guard that
`contrib/` never enters the wheel. That guard exists because
`contrib/claude_code_hook.py` is experimental and is not covered by the
published retention benchmark, so it must stay reachable in the source tree and
absent from every installed environment. The invariant rests on
`packages = ["barber"]` in `pyproject.toml`, one edit away from silently not
holding, and a PyPI upload cannot be undone. **Run the check yourself, locally,
before uploading.**

```bash
# 0. versions agree, tests pass, fixtures fresh
python tests/test_versions.py
pytest tests/ -q
python tests/gen_golden.py --check

# 1. clean build (a stale dist/ is how the wrong version gets uploaded)
rm -rf dist/ build/
python -m build
twine check dist/*

# 2. the guard release.yml would have run — contrib must not be in the WHEEL
python - <<'PY'
import glob, sys, zipfile
wheel = glob.glob("dist/*.whl")[0]
names = zipfile.ZipFile(wheel).namelist()
bad = [n for n in names if "contrib" in n]
if bad:
    sys.exit(f"contrib leaked into the wheel: {bad}")
print(f"ok: {wheel} has {len(names)} entries, no contrib")
PY

# 3. confirm the artifacts carry the version you think they do
ls dist/

# 4. upload
twine upload dist/*
```

The sdist **does** contain `contrib/`, deliberately — `tests/test_hook.py`
needs the hook it tests, and the wheel is governed separately by
`packages = ["barber"]`. The check above is wheel-only on purpose; do not
"fix" it to cover the sdist.

### Credentials

The repo-root `.env` is gitignored and holds the publish tokens. Never commit
it, never paste its contents anywhere, never echo it into a terminal that is
being recorded or shared, and never add a token to a workflow file or a repo
secret while Trusted Publishing is the intended end state — the whole point of
registering it is that no token needs to exist. Registering the trusted
publisher retires this section.

## Publishing to npm

npm has no equivalent of the PyPI workflow; it is manual by design and unrelated
to the Trusted Publishing question above.

```bash
python tests/test_versions.py     # js/package.json vs js/index.js VERSION
python tests/gen_golden.py --check
cd js
node --test
npm publish --dry-run             # check the file list against "files" in package.json
npm publish
```

`npm publish --dry-run` is worth the extra step: `package.json`'s `files` array
is the only thing keeping the test fixtures and this repo's Python side out of
the tarball.

## Post-release

- Confirm the published artifacts report the right version:
  `pip install --no-cache-dir barber-llm==X.Y.Z && python -c "import barber; print(barber.__version__)"`
  and `npx barber-llm@X.Y.Z --version`. This is the check that would have caught
  0.4.0 from the outside.
- The npm and PyPI badges in `README.md` will show different numbers. That is
  expected and documented in `js/README.md`; leave them alone.
