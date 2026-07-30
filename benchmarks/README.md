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

The current (unrefactored) code needs an old jax: it still uses
`jnp.clip(..., a_min=...)`, removed in jax 0.10, and it imports the module
`pyDOE`, while the installed distribution (pydoe 1.3.0) only exposes the
lowercase module `pydoe`.

```
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -e ".[bench]" "jax[cpu]<0.10"
```

Then write a one-file shim `pyDOE.py` into `.venv/Lib/site-packages/`:

```python
from pydoe import *
from pydoe import lhs
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

## Baseline (2026-07-28, amended after adversarial review)

Machine: 11th Gen Intel Core i7-1185G7 @ 3.00GHz, 8 threads, Windows 10 Pro
build 19045, CPU-only jax. Env: Python 3.12.6, jax 0.9.2, jaxlib 0.9.2,
numpy 2.5.1, scipy 1.18.0. Full detail, per-bench noise bands, and IQRs:
`results/2026-07-28-baseline.md`.

| Bench | Median | First-call latency |
|---|---|---|
| train warm instance, n=32 | 88.1 ms | 0.72 to 0.76 s |
| train warm instance, n=128 | 179.9 ms | 0.60 to 0.68 s |
| train warm instance, n=512 | 2437 ms | 2.70 to 3.08 s |
| train fresh instance, n=128 | 654.0 ms | every round is a first call |
| predict, 256 points batched | 1.07 ms | 0.28 to 0.38 s |
| EI consumer path, 256 candidates | 137.0 ms (0.535 ms per candidate) | 0.39 to 0.48 s |
| EI fused acquisition, 256 candidates | 76.0 ms (0.297 ms per candidate) | 0.34 to 0.41 s |

Two numbers worth naming: the consumer EI path costs 1.8x the fused variant
(two dispatches plus a host sync per candidate instead of one fused graph),
and a fresh GP instance pays about 474 ms of instance-keyed recompilation
per train at n=128 (654 ms fresh vs 180 ms warm, in-session). The latter is
the number issue #30 gates on.
