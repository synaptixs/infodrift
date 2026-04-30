# Contributing to `drift_monitor`

Thanks for your interest in improving the Information Drift monitoring framework.
This guide covers contribution workflows for **both internal Fibonacci team
members and external open-source contributors**. The rules are the same; the
access paths differ.

---

## Table of contents

1. [Who this guide is for](#who-this-guide-is-for)
2. [Code of conduct](#code-of-conduct)
3. [Ground rules](#ground-rules)
4. [Development setup](#development-setup)
5. [Repository layout](#repository-layout)
6. [Branching & workflow](#branching--workflow)
7. [Coding standards](#coding-standards)
8. [Testing](#testing)
9. [Commits & pull requests](#commits--pull-requests)
10. [Review process](#review-process)
11. [Reporting bugs & requesting features](#reporting-bugs--requesting-features)
12. [Security disclosures](#security-disclosures)
13. [Releases & versioning](#releases--versioning)
14. [Internal-only notes](#internal-only-notes)

---

## Who this guide is for

| You are… | Read |
|---|---|
| An **external contributor** (community, partner, researcher) | Sections 1–13 |
| An **internal contributor** (Fibonacci ML / Platform / SRE) | All sections, including 14 |

The codebase is the reference implementation of the white paper
*"Monitoring Information Drift in Production Predictive AI Systems"* (Rohilla,
v2.0, March 2026). Every change should be traceable to either a section of the
white paper or a concrete production RCA need.

---

## Code of conduct

Be respectful, assume good intent, and keep discussion focused on the work.
Harassment, personal attacks, or discriminatory language will result in removal
from the project. Report concerns to the maintainers (see internal section for
internal channels; external contributors may open a confidential issue or email
the maintainer listed in `pyproject.toml`).

---

## Ground rules

- **Stay scoped.** Bug fixes don't bundle refactors. Refactors don't bundle new
  features. One concern per PR.
- **No silent behavior changes.** Any change to a metric, threshold default, or
  alert semantic must be called out in the PR description and reflected in tests.
- **Ontology-entity scoping is non-negotiable.** New monitors must accept and
  respect an `entity_key`; metrics that cannot be segmented per entity will be
  rejected.
- **Match the white paper.** If your change diverges from the white paper, say
  so explicitly and justify it. We track a target of 55 automated checks
  derived from the paper — additions count toward that target.
- **Don't add dependencies casually.** New runtime dependencies require
  maintainer approval. Prefer the standard scientific stack (numpy, pandas,
  scipy, scikit-learn) already in `pyproject.toml`.

---

## Development setup

Requires Python ≥ 3.10.

```bash
git clone <your-fork-or-internal-remote>
cd infodrift

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev,logs]"
pytest
```

If `pytest` passes on a clean checkout, you're set up correctly. If it doesn't,
that's a bug — open an issue before changing anything.

---

## Repository layout

```
drift_monitor/
  core/          # Monitor implementations (PSI, divergence, logs, performance)
  monitoring/    # Orchestrator + alerting
  reporting/     # Health reports, canary comparisons, JSON export
tests/           # Automated suite (white-paper-derived checks)
pyproject.toml   # Package metadata; source of truth for deps
```

When adding a new monitor, place it under `drift_monitor/core/`, register it
with the orchestrator, and add tests under `tests/` mirroring the module path.

---

## Branching & workflow

- **Default branch:** `main`. It is always releasable.
- **Feature branches:** `feat/<short-description>`, `fix/<short-description>`,
  `docs/<short-description>`. Keep them short-lived.
- **External contributors:** fork the repo, branch from `main`, open a PR
  against `main`.
- **Internal contributors:** branch directly in the repo if you have write
  access; otherwise fork. Either is fine.

Rebase on top of `main` before requesting review. We squash-merge by default.

---

## Coding standards

- **Style:** PEP 8. We don't enforce a formatter in CI yet, but match the
  surrounding code. If you run `black` / `ruff format`, do it in a separate
  commit so review diffs stay clean.
- **Type hints:** required on all new public functions and class methods.
- **Docstrings:** one short line for internal helpers; a brief NumPy-style
  docstring (Parameters / Returns) for anything in a module's public surface.
- **Comments:** explain *why*, not *what*. Don't narrate the code.
- **Errors:** raise specific exceptions (`ValueError`, `KeyError`) at boundaries.
  Don't swallow exceptions to "be safe" — silent failure in a drift monitor
  defeats its purpose.
- **Numerical code:** be explicit about edge cases (empty bins, zero variance,
  single-class predictions). Add a test for each edge case you handle.

---

## Testing

- Every new monitor, metric, or orchestrator behavior needs tests.
- Every bug fix needs a regression test that fails before the fix and passes
  after.
- Tests live under `tests/` and mirror the source path.
- Run the full suite locally before opening a PR:

  ```bash
  pytest -q
  ```

- For numerical assertions, use `numpy.testing.assert_allclose` with explicit
  tolerances. Don't compare floats with `==`.
- Don't mock the math. You can mock I/O and clocks; never mock the metric
  under test.

---

## Commits & pull requests

**Commits**

- Imperative mood, ~50 char subject line, optional body explaining *why*.
- Reference the white-paper section if relevant: `Add JS divergence (§3.2)`.
- Keep commits logically atomic. Don't squash unrelated work into one commit.

**Pull requests**

Open a PR with:

1. **What** changed and **why** — link to the issue, white-paper section, or
   incident.
2. **How it was tested** — commands run, new tests added.
3. **Behavior changes** — call out anything that changes outputs, thresholds,
   or public APIs.
4. **Checklist:**
   - [ ] Tests added / updated
   - [ ] `pytest` passes locally
   - [ ] Public APIs typed and documented
   - [ ] No new runtime dependencies (or approval secured)
   - [ ] Per-entity scoping preserved

Draft PRs are welcome for early feedback.

---

## Review process

- At least **one maintainer approval** is required to merge.
- Changes that touch alert semantics, thresholds, or the orchestrator's public
  API require **two approvals**, one of which must be an internal maintainer.
- Reviewers focus on: correctness vs. white paper, test coverage, scoping,
  numerical edge cases, and API stability.
- Expect first response within ~3 business days. Ping the PR if it goes quiet
  longer than that.

---

## Reporting bugs & requesting features

Open a GitHub issue. Good bug reports include:

- `drift_monitor` version (`pip show drift_monitor`)
- Python version and OS
- Minimal reproducer (≤ 30 lines)
- Expected vs. actual behavior
- Stack trace if applicable

Feature requests should explain the **operational RCA question** the feature
answers. "Add metric X" is not enough; "this metric localizes calibration
erosion to specific log templates after a deploy" is.

---

## Security disclosures

**Do not open public issues for security problems.**

- External: email the maintainer in `pyproject.toml` with subject
  `[infodrift security]`.
- Internal: file via the standard Fibonacci security intake; do not post in
  team channels.

We aim to acknowledge within 2 business days and patch within 30 days for
confirmed issues.

---

## Releases & versioning

- Semantic Versioning (`MAJOR.MINOR.PATCH`).
- `MAJOR` for breaking changes to the orchestrator or monitor public APIs.
- `MINOR` for new monitors, metrics, or non-breaking additions.
- `PATCH` for bug fixes and doc-only changes.
- Releases are cut from `main` by maintainers. External contributors do not
  need to bump the version in their PR — maintainers handle it at release time.

---

## Internal-only notes

This section is for interAl team members. External contributors can stop
here.

- **Production deployment context:** this package is consumed by the platform
  monitoring pipeline. Treat any change to `DriftOrchestrator.run()` output
  shape as a breaking platform change and coordinate with the platform team
  before merging.
- **Incident-driven changes:** if a change is motivated by a specific
  production incident, link the incident ticket in the PR description. Do not
  paste customer data, log lines, or entity keys from production into public
  PRs or issues — sanitize first.
- **Internal review channels:** use the team's review channel for fast-turn
  reviews; for changes touching alert thresholds, also tag the on-call lead.
- **White-paper alignment:** when adding checks toward the 55-check target,
  update the internal coverage tracker. The tracker is the source of truth for
  which paper sections are still uncovered.
- **Confidential dependencies:** do not introduce dependencies on internal-only
  packages in this repo. It is intended to remain externally buildable.

---

Questions about this guide? Open an issue or ping a maintainer. Improvements to
the guide itself are welcome — submit them the same way you would any other
change.
