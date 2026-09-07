# jaxbo benchmarks

Baseline performance harness for the refactor. It measures the paths the
refactor touches, shaped to match the real consumer (shelter-pulse
`jaxbo_optimizer.py`: 4D inputs, a fresh GP per BO iteration, per-candidate
EI through `predict` plus `acquisitions.EI`):

- GP training, both on a warmed instance (jit cache hot) and on a fresh
  instance constructed inside the timed call (the consumer pattern, and the
  direct gate for issue #30);
- one batched `predict` call (the reference a future `score_candidates`
  would use);
- per-candidate EI scoring in the exact consumer path, with the fused
  `gp.acquisition` loop kept as a secondary case.

Only the public API is imported (`from jaxbo.models import GP`,
`from jaxbo import acquisitions`), so the suite must keep running unchanged
against the refactored package.

## The rule

**No optimization without a delta.** Any PR that claims a performance change
must show before/after numbers from this suite on the same machine, same
env, same seeds. If the delta is inside the per-bench noise band documented
in the results file, the claim does not go in the PR description.

Two protocol requirements, learned the hard way (see the 2026-08-01
baseline's protocol section): **check the machine is quiet before
recording** (background load inflated medians 2 to 10x in the discarded
runs; clean suite wall time is about 30 seconds), and **record
before/after deltas as paired runs in one session**, because
cross-session absolutes on this hardware carry a 27 to 86 percent
machine-state noise band.

### Enforcement, honestly

- **Review-enforced today.** AGENTS.md carries the rule ("Optimization PRs
  must include a before/after benchmark delta from the benchmarks suite")
  and reviewers apply it on every P3 PR. There is no bot behind it.
- **CI smoke (planned).** Once the Phase 00 pipeline lands (issue #46), CI
  runs `pytest benchmarks --benchmark-disable` as a correctness smoke so the
  suite cannot silently rot. That only checks the benches still run; it
  compares no numbers.
- **Automated delta comparison** (for example `pytest-benchmark compare`
  against a stored JSON) is a possible later hardening. It is not claimed
  today.

## How to run

Since v0.2.0 the package runs on current jax with no pins and no shims
(the pre-refactor code needed `jax<0.10` and a `pyDOE.py` module shim;
those instructions live in git history with that code).

```
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -e ".[bench]"
```

Run the suite:

```
.venv/Scripts/python.exe -m pytest benchmarks -q
```

Save a comparable record: add
`--benchmark-json=benchmarks/results/<date>-<label>.json`.

Regenerate the training profile artifact:

```
.venv/Scripts/python.exe benchmarks/profile_train.py
```

## What is measured

All fixtures are 4D (`[0, 1]^4`, seeded), matching the consumer's 4 budget
shares.

| File | Case |
|---|---|
| `bench_train.py` | `train_warm_instance`: one GP instance, first call untimed, re-trained in the timed rounds (jit cache hot), n = 32 / 128 / 512, `num_restarts=3`. `train_fresh_instance`: a new `GP(...)` constructed inside every timed round and trained once (n=128), the consumer pattern; every round pays the instance-keyed recompilation |
| `bench_predict.py` | one batched `GP.predict` over 256 4D points after one train (n=128) |
| `bench_acquisition.py` | `ei_consumer_path`: per candidate, a `(1, 4)` `gp.predict` then `acquisitions.EI(mu, std, best)` then a host `float()`, over 256 candidates (the exact shelter-pulse loop). `ei_fused_acquisition`: the same loop through the fused `gp.acquisition` graph, kept for comparison. `ei_score_candidates`: all 256 candidates in one batched `acquisitions.score_candidates` call, the loop's replacement (issue #28) |

Every bench times its first call separately and reports it as
`first_call_latency_s` in `extra_info`. First-call latency conflates trace +
compile + execute (and, in a fresh process, jax init); it is not a pure
compile-time measure. The benchmarked path is synchronized with
`jax.block_until_ready` (or a host `float()` on the consumer path, which
forces the same sync).

## Current evidence

The public benchmark record is
`results/2026-08-15-rc-final.md`. It contains the `0.2.2` medians,
comparison ratios, environment, and noise-band interpretation. Raw timing
files and intermediate comparison reports are intentionally excluded from the
library tree.