# Agent rules for JAX-BO

Contributor rules (dev setup, tox, the CI gate map, PR rules, the benchmark
delta rule, house style) live in **`CONTRIBUTING.md`**. Read it first; this
file only carries what is specific to agent sessions.

## HARD RULE: fork only, never upstream

NEVER create or modify ANYTHING on github.com/PredictiveIntelligenceLab/JAX-BO
(the fork parent): no PRs, no issues, no comments, no pushes. All work happens
on this fork, github.com/ricardogr07/JAX-BO.

Mechanics:
- Every `gh` command MUST pass `--repo ricardogr07/JAX-BO` explicitly.
  `gh pr create` defaults to the fork parent; that has caused accidental
  upstream PRs twice.
- The `upstream` remote's push URL is set to DISABLED on purpose. Do not
  restore it. Fetch from upstream is allowed for divergence checks only.

## Session rules

- Workers open PRs on the fork; they do not merge.
- No AI attribution ever: no Co-Authored-By trailers, no "Generated with"
  lines. Ricardo is the sole author.
- The no em/en dash rule and the benchmark delta rule live in
  `CONTRIBUTING.md`; they bind agents exactly as they bind humans.
