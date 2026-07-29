# Agent rules for JAX-BO

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

## House rules

- No em or en dashes anywhere: code, docs, commits, PR bodies. Use a colon or
  comma; write ranges as "X to Y".
- No AI attribution ever: no Co-Authored-By trailers, no "Generated with"
  lines. Ricardo is the sole author.
- Workers open PRs on the fork; they do not merge.
- Optimization PRs must include a before/after benchmark delta from the
  benchmarks suite; no perf change lands without numbers.
