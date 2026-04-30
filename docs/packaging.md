# Packaging & Release

`drift_monitor` is a standard PEP 621 Python package. This document covers
how to build the distribution artefacts, smoke-test them, and publish a
release to PyPI.

## Install (consumers)

Once published:

```bash
pip install drift_monitor             # core
pip install "drift_monitor[logs]"     # + drain3 log template miner
pip install "drift_monitor[dev]"      # + pytest, build, twine (contributors)
```

From source (this repository):

```bash
git clone https://github.com/nrohilla-fibonacci/infodrift.git
cd infodrift
pip install -e ".[logs,dev]"
```

## Build artefacts (maintainers)

The build is fully declarative — no `setup.py`. Build sdist + wheel:

```bash
pip install --upgrade build "twine>=6"
rm -rf dist build *.egg-info
python -m build
```

Output:

```
dist/drift_monitor-<version>-py3-none-any.whl   # the wheel
dist/drift_monitor-<version>.tar.gz             # the sdist
```

Validate the metadata before pushing anywhere:

```bash
python -m twine check dist/*
```

Both artefacts must report `PASSED`. If you see
`unrecognized field 'license-file'` upgrade twine to ≥ 6.

## Smoke-test the wheel

Install the wheel into a clean venv and run the test suite against it
(the tests live outside the wheel and import the installed package):

```bash
python -m venv /tmp/dm_venv
/tmp/dm_venv/bin/pip install dist/drift_monitor-<version>-py3-none-any.whl pytest
/tmp/dm_venv/bin/python -c "import drift_monitor as dm; print(dm.__version__, dm.__all__)"
cp -r tests /tmp/dm_test_pkg && cd /tmp/dm_test_pkg && /tmp/dm_venv/bin/pytest -q
```

If the suite is green inside the venv, the wheel is shippable.

## Versioning

Single source of truth: `pyproject.toml` `[project] version`. Bump the
version in the same commit that introduces the user-visible change.
Follow [SemVer](https://semver.org/):

| Change | Bump |
|---|---|
| Bug fix, doc-only, internal refactor | PATCH (`0.1.0` → `0.1.1`) |
| New monitor, new metric, new optional kwarg | MINOR (`0.1.0` → `0.2.0`) |
| Removed/renamed public API, changed alert semantics, changed default thresholds | MAJOR (`0.1.0` → `1.0.0`) |

Tag the release commit: `git tag -a v0.2.0 -m "Release v0.2.0"`.

## Release process

The repo ships a GitHub Actions workflow that builds + publishes on tag
push (`.github/workflows/release.yml`).

### One-time setup (PyPI Trusted Publishing)

1. Create the project on PyPI (first release can be done manually with
   `twine upload` to claim the name).
2. In the project settings on PyPI, add a Trusted Publisher for this
   repo:
   - Owner: `nrohilla-fibonacci`
   - Repository: `infodrift`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. In GitHub repo settings → Environments, create an environment named
   `pypi`. No secrets required — Trusted Publishing uses OIDC.

### Per-release

1. Update version in `pyproject.toml`.
2. Update changelog in the GitHub Release notes (the workflow does
   not auto-generate them).
3. Commit and merge to `main`.
4. Tag and push:
   ```bash
   git checkout main && git pull
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```
5. The `release` workflow runs:
   - Builds sdist + wheel.
   - Verifies the tag matches the `pyproject.toml` version (fails the
     run if mismatched — protects against tagging the wrong commit).
   - Runs `twine check`.
   - Publishes to PyPI via Trusted Publishing.

### Manual fallback

If you need to publish without Trusted Publishing (e.g. first claim of
the name on PyPI):

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*       # uses ~/.pypirc or interactive token
```

Use a project-scoped API token, never your account password.

### TestPyPI dry run

For risky releases, push to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ drift_monitor==<version>
```

## What ships in the wheel

Only `drift_monitor/` and the `py.typed` marker. Tests, examples,
docs, the white paper PDF, and CI workflows are excluded — they are
useful for contributors but unwanted bloat for installs.

`MANIFEST.in` controls the sdist contents and mirrors the wheel
exclusions, so `pip install drift_monitor.tar.gz` produces the same
installed footprint as the wheel.
