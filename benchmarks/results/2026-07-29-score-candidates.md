# score_candidates delta: batched EI scoring vs the consumer loop, 2026-07-29

Delta record for `acquisitions.score_candidates` (issue #28, slice 2c), per
the AGENTS.md rule that no perf change lands without before/after numbers
from this suite. Before and after are measured in the SAME environment and
session (`pytest benchmarks/bench_acquisition.py`, 3 runs, same seeds and
fixtures as the baseline suite); the 2026-07-28 baseline file is a
historical record and is not edited.

## Environment

Same machine as the baseline (11th Gen Intel Core i7-1185G7, 8 logical
CPUs, Windows 10 Pro build 19045, CPU-only jax). Python 3.12.6, jax /
jaxlib 0.10.2 (the baseline recorded jax 0.9.2; #52 moved the pin), numpy
2.5.1, scipy 1.18.0, pytest-benchmark 5.2.3.

## Numbers

Medians per run in ms; the headline number is the median of the three run
medians. All cases score the same 256 Dirichlet candidates against the
same trained GP (n=128, 4D).

| Case | Run 1 | Run 2 | Run 3 | Median | Per candidate |
|---|---|---|---|---|---|
| EI consumer path, 256 candidates (before) | 230.3 | 224.4 | 177.5 | 224.4 | 0.877 ms |
| EI fused `gp.acquisition` loop (reference) | 150.2 | 131.7 | 129.7 | 131.7 | 0.514 ms |
| `score_candidates`, 256 candidates (after) | 2.95 | 2.91 | 2.67 | 2.91 | 0.011 ms |

- **Delta: 224.4 ms to 2.91 ms, a 77x speedup (98.7 percent reduction).**
  The consumer-path noise band in `2026-07-28-baseline.md` is 9.9 percent;
  this clears it by two orders of magnitude.
- First-call latency (trace + compile + execute) of the batched case: 0.67
  to 0.90 s across the runs, vs 0.43 to 0.55 s for the loop's first
  candidate; the win is not smuggled into compile time.
- The 2026-07-28 baseline recorded the consumer path at 137.0 ms on jax
  0.9.2. These runs sat on a busier machine and a newer jax (both loop and
  batched case move together), which is why before AND after were
  re-measured here instead of comparing against the stored baseline
  number.

Raw pytest-benchmark JSON for the three runs was captured locally
(`bench-2c-final-{1,2,3}.json`); the medians above are reproducible with
`pytest benchmarks/bench_acquisition.py --benchmark-json=...` on any
machine, subject to its own noise band.
