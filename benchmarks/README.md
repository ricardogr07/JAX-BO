# jaxbo benchmarks

Baseline performance harness for the refactor. It measures the three paths the
refactor touches: GP training, batched prediction, and per-candidate EI
scoring (the loop that a batched `score_candidates` will replace).

Only the public API is imported (`from jaxbo.models import GP`), so the suite
must keep running unchanged against the refactored package.

## The rule

**No optimization without a delta.** Any PR that claims a performance change
must show before/after numbers from this suite on the same machine, same
env, same seeds. If the delta is inside the noise band documented in the
results file, the claim does not go in the PR description.

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

| File | Case |
|---|---|
| `bench_train.py` | `GP({"kernel": "Matern52", ...}).train`, 2D synthetic data, n = 32 / 128 / 512, `num_restarts=3`, seeded |
| `bench_predict.py` | `GP.predict` over a 16x16 = 256 point grid after one train (n=128) |
| `bench_acquisition.py` | EI via `gp.acquisition` called once per candidate in a Python loop over 256 candidates |

Every bench times the first (cold) call separately, it includes JIT
compilation, and reports it as `cold_first_call_s` in `extra_info`. The
benchmarked path is the warm one, synchronized with `jax.block_until_ready`.

## Baseline (2026-07-28)

Machine: 11th Gen Intel Core i7-1185G7 @ 3.00GHz, 8 threads, Windows 10 Pro
build 19045, CPU-only jax. Env: Python 3.12.6, jax 0.9.2, jaxlib 0.9.2,
numpy 2.5.1, scipy 1.18.0. Full detail and noise analysis:
`results/2026-07-28-baseline.md`.

| Bench | Warm median | Cold first call |
|---|---|---|
| train n=32 | 33.7 ms | 0.57 s |
| train n=128 | 92.8 ms | 0.47 s (1.95 s in a fresh process) |
| train n=512 | 1596 ms | 2.12 s |
| predict, 256-point grid | 0.85 ms | 0.25 s |
| EI Python loop, 256 candidates | 62.8 ms (0.245 ms per candidate) | 0.33 s |
