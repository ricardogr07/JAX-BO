## What

<!-- One or two sentences: what changes and why. -->

Closes #

## Checklist

- [ ] Conventional commit title (`fix:`, `feat:`, `docs:`, `refactor:`, `test:`, `ci:`); release-please cuts releases from it
- [ ] `uv run pytest tests` green locally
- [ ] `uvx tox -e lint` green (pinned black + ruff)
- [ ] No em or en dashes anywhere (code, docs, this PR body); ranges written as "X to Y"
- [ ] Perf claim? Before/after numbers from the `benchmarks/` suite are in the description (see CONTRIBUTING.md, the benchmark delta rule)
- [ ] Removal or public signature change? The commit message carries the changelog line
