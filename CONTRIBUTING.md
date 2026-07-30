# Contributing to JAX-BO

Thanks for your interest in contributing. This document covers everything a
contributor needs: where work happens, how to set up a dev environment, what
CI runs against your change, and the rules a pull request must satisfy.

## Where work happens: this fork only

This repository is a maintained fork of
[PredictiveIntelligenceLab/JAX-BO](https://github.com/PredictiveIntelligenceLab/JAX-BO).
All development happens here, on `ricardogr07/JAX-BO`. Please do not open
pull requests, issues, or comments on the upstream repository on behalf of
this project; upstream is credited (see the README acknowledgment section and
`CITATION.cff`) but is not contacted. Agent sessions have additional
mechanics for this rule in `AGENTS.md`.

## Dev setup

The project uses [uv](https://docs.astral.sh/uv/) with a committed lockfile
(`uv.lock`). Python 3.10 or newer is required.

```bash
git clone https://github.com/ricardogr07/JAX-BO.git
cd JAX-BO
uv sync          # creates .venv from uv.lock; installs the dev group
uv run pytest tests
```

`uv sync` installs the `dev` dependency group, which includes pytest,
pytest-cov, and the packages backing the optional extras (numpyro,
scikit-learn, KDEpy) so the whole test suite, including the extras and their
compatibility shims, runs locally.

The package itself installs with exactly 4 runtime dependencies (`jax`,
`jaxlib`, `numpy`, `scipy`); everything else lives behind the optional
extras `[mcmc]`, `[multifidelity]`, `[weighted]`, and their union `[all]`.
Keep it that way: no core module may import an extras dependency, even
transitively. `tests/test_import_guards.py` enforces this.

## Tox environments

`tox.ini` defines the local multi-python runner plus the pinned lint env:

| Env | What it runs |
|---|---|
| `py310` to `py314` | `pytest tests` against that interpreter, with `extras = all` |
| `lint` | `black --check` and `ruff check` over `jaxbo/` and `tests/` |

Run them with `uvx tox -e py312` or `uvx tox -e lint`. black and ruff are
pinned in `tox.ini` on purpose (unpinned formatters broke CI twice); bump
them deliberately, together with the `[tool.black]` and `[tool.ruff]`
sections in `pyproject.toml`.

## What CI runs against your change

CI (`.github/workflows/ci.yml`) gates work by changed paths, so a docs-only
PR does not pay for the full test matrix:

| You changed | Jobs that run |
|---|---|
| Only `**/*.md` or `docs/**` | `lint` in docs scope (the markdown dash scan) |
| `jaxbo/**`, `tests/**`, `pyproject.toml`, `tox.ini`, `uv.lock` | full `lint` + the `test` matrix |
| `benchmarks/**` | `bench-smoke` (collection only, no timing) |
| `.github/**` | everything |

The `test` matrix is {py3.10, 3.11, 3.12, 3.13, 3.14} x {jax floor, jax
latest} within the `jax>=0.6,<0.11` pin. The 3.13 and 3.14 lanes are
advisory (`continue-on-error`) until they hold a green streak. CI installs
frozen from `uv.lock`, then re-pins jax per lane. `ci-ok` aggregates all
jobs and is the only required check; the release workflow refuses to publish
unless `ci-ok` succeeded on the tagged commit.

Full lint scope also runs two greps the codebase must stay clean under: no
em or en dashes in markdown, and no `clip(a_min=, a_max=)` kwargs (removed
in JAX 0.10; use positional `min, max`).

## Pull request rules

- Branch from `main`; open the PR against `main` on this fork.
- **Conventional commit titles** (`fix:`, `feat:`, `docs:`, `refactor:`,
  `test:`, `ci:`). Releases are cut by release-please from these prefixes,
  so the title is load-bearing: a `feat:` bumps the minor version, a `fix:`
  the patch version.
- Reference the issue the PR closes: `Closes #<n>` in the body.
- Green `ci-ok` before merge. Do not merge your own PR unless you are the
  maintainer.
- Every removal or public signature change gets a changelog line before
  merge (release-please generates `CHANGELOG.md` from the commits, so say
  it in the commit message).
- Keep PRs sliced small; one concern per PR.

### House style

- No em or en dashes anywhere: code, docs, commits, PR bodies. Use a colon
  or comma instead, and write ranges as "X to Y". CI enforces this for
  markdown.
- black + ruff clean (`uvx tox -e lint`), with the pinned versions.
- Public functions and classes carry type hints and docstrings. This is a
  review expectation, not a CI gate: the 70% docstring coverage figure
  comes from the reposage audit the revamp is graded against (see
  `docs/audits/`), which is re-run at release candidate time, not on every
  PR.

## The benchmark delta rule

**No optimization lands without a before/after benchmark delta.** Any PR
that claims a performance change must include numbers from the
`benchmarks/` suite, produced on the same machine, same env, same seeds,
before and after the change. If the delta is inside the per-bench noise
band documented in the results file, the claim does not go in the PR
description. A perf change without numbers is rejected regardless of how
plausible it looks. See `benchmarks/README.md` for how to run the suite and
how the rule is enforced.

## Reporting bugs and requesting features

Use the issue templates (they ask for the jaxbo/jax/python versions and a
minimal repro). Security issues go through `SECURITY.md`, not the public
tracker.

## Code of conduct

Participation in this project is covered by `CODE_OF_CONDUCT.md`
(Contributor Covenant 2.1).
