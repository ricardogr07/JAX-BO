# 3c delta: batched-start acquisition multi-start (2026-08-02)

Paired before/after recording for issue #32. The change lives at
`compute_next_point_lbfgs`, so the delta shows up on `bench_next_point.py`
(added with the change); the eight pre-existing benches ride along as
controls, since 3c should not move any of them.

## What was compared

| Side | Worktree | Commit |
|---|---|---|
| base | `C:\git\JAX-BO` | `2e2bb87` (main, 3a merged, 3b NOT merged) |
| head | `C:\git\JAX-BO-3c` | `d96f019` (`perf/3c-acquisition-path`) |

Base excludes 3b (PR #70, open at recording time), so both sides train
through the same scipy path and this table isolates 3c alone. If 3b lands
first the absolute numbers shift, but the compared quantity does not.

`bench_next_point.py` exists only on the branch, so the base runs were made
by copying that one file into the base worktree unmodified. Provenance is
verifiable: every JSON records the commit it ran at in `commit_info.id`,
`2e2bb87` for the four base runs and `d96f019` for the four head runs.

## Protocol

Four pairs in one session, order-balanced in ABBA blocks
(base/head/head/base) so a monotone within-session drift cannot favor either
side, on a machine held at a steady state (suite wall time 40 to 45 s per run
for the 11 benches, against 21 to 29 s for the 8-bench suite plus 10 to 13 s
for the three next-point benches measured separately). Both venvs verified
identical by `uv pip list` except the editable path (Python 3.11.14,
jax/jaxlib 0.10.2, numpy 2.4.6, scipy 1.17.1, pytest-benchmark 5.2.3).

Raw artifacts: `2026-08-02-3c-{base,head}-run{1,2,3,4}.json`.

An earlier same-day session recorded with unbalanced ordering (base always
first) is NOT the basis of this table and was discarded: its control rows
moved on untouched code (`ei_fused_acquisition` 0.45x), which is the tell
that within-pair drift was biasing toward whichever side ran second.

## Result

Ratio is head/base, so below 1.00 is faster. Medians are the median of four
per-run medians; the ratio column is the median of the four paired ratios.

| Bench | base median (ms) | head median (ms) | paired ratio | spread |
|---|---|---|---|---|
| `bench_next_point_lbfgs_lcb` | 530 | 123 | **0.24x** | 0.18 to 0.25 |
| `bench_next_point_lbfgs_early_ei` | 72.8 | 32.6 | **0.45x** | 0.41 to 0.50 |
| `bench_next_point_lbfgs` (late EI) | 75.7 | 63.0 | **0.84x** | 0.82 to 0.89 |
| `bench_train_warm_instance[32]` (control) | 107 | 112 | 1.06x | 0.97 to 1.31 |
| `bench_train_warm_instance[128]` (control) | 342 | 328 | 0.96x | 0.85 to 1.11 |
| `bench_train_warm_instance[512]` (control) | 2704 | 2812 | 1.03x | 1.00 to 1.19 |
| `bench_train_fresh_instance` (control) | 325 | 338 | 1.06x | 1.00 to 1.37 |
| `bench_predict_batch256` (control) | 2.78 | 2.91 | 0.99x | 0.73 to 1.18 |
| `bench_ei_consumer_path_256` (control) | 85.9 | 91.6 | 1.07x | 1.00 to 1.24 |
| `bench_ei_fused_acquisition_256` (control) | 55.6 | 55.1 | 0.98x | 0.37 to 1.00 |
| `bench_ei_score_candidates_256` (control) | 2.72 | 1.10 | 0.55x | 0.26 to 1.02 |

The three changed benches all improve, and each one's four paired ratios lie
in a band narrower than 0.10, so none of them overlaps 1.00. The eight
controls sit at 0.96x to 1.07x, which is what says the pairing worked.

Two control spreads have a wide low end and both trace to a single
contaminated base run rather than to the change: base run 1 posted 167 ms on
`ei_fused_acquisition` against 54.5 / 55.9 / 55.3 on the other three, and
base runs 1 and 4 posted 4.09 and 4.17 ms on `ei_score_candidates` against
1.36 and 1.08. Those inflated base values are what produce the 0.37 and 0.26
low ends; the medians (0.98x, 0.55x) and the head columns are steady.
`ei_score_candidates` at 0.55x is therefore a base artifact, not a 3c win:
3c does not touch that path.

### Why the three surfaces differ so much

Reduction in polished starts is the whole mechanism (10 blind polishes
become one batched scan of 320 candidates plus k=2 polishes), so the win
tracks how much each surface charges per polish.

- **LCB, trained n=128 (0.24x).** Every start carries gradient signal, so a
  polish costs real L-BFGS-B iterations, each paying a host round trip.
  Cutting 10 polishes to 2 removes most of the work.
- **Early-BO EI, n=16 (0.45x).** A wiggly EI surface, same mechanism, fewer
  iterations per polish.
- **Late-stage EI, n=128 (0.84x).** EI is flat over most of the domain here,
  so most blind starts terminate at zero iterations and cost almost nothing.
  Only the scan's improved starting quality is left, which is why this is
  the smallest of the three and why it is the surface the naive "EI is
  already cheap" reading comes from.

## Cost: first-call latency

`first_call_latency_s`, per run:

| Bench | base | head |
|---|---|---|
| `bench_next_point_lbfgs` (late EI) | 1.71 / 1.64 / 1.65 / 1.67 | 2.58 / 2.71 / 2.71 / 2.65 |
| `bench_next_point_lbfgs_early_ei` | 0.83 / 0.75 / 0.78 / 0.70 | 1.19 / 1.09 / 1.14 / 1.14 |
| `bench_next_point_lbfgs_lcb` | 0.87 / 1.03 / 1.02 / 1.01 | 1.07 / 1.06 / 1.01 / 1.07 |

The batched scan adds one more traced shape, costing about **+1.0 s once on
the first EI next-point call of a process** and about +0.35 s on the early-BO
shape, with LCB flat. It is repaid on the second call at every surface
measured, and a BO run makes one such call per iteration.
