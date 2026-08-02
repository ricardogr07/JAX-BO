# 3c delta: batched-start acquisition multi-start (2026-08-02)

Paired before/after recording for issue #32. The change lives at
`compute_next_point_lbfgs`, so the delta shows up on `bench_next_point.py`
(added with the change); the eight pre-existing benches ride along as
controls, since 3c should not move any of them.

## What was compared

| Side | Worktree | Commit |
|---|---|---|
| base | `C:\git\JAX-BO` | `2e2bb87` (main, 3a merged, 3b NOT merged) |
| head | `C:\git\JAX-BO-3c` | `8bb80bb` (`perf/3c-acquisition-path`) |

Base excludes 3b (PR #70, open at recording time), so both sides train
through the same scipy path and this table isolates 3c alone. If 3b lands
first the absolute numbers shift, but the compared quantity does not.

`bench_next_point.py` exists only on the branch, so the base runs were made
by copying that one file into the base worktree unmodified. Provenance is
verifiable: every JSON records the commit it ran at in `commit_info.id`,
`2e2bb87` for the four base runs and `8bb80bb` for the four head runs.

## Protocol

Four pairs in one session, order-balanced in ABBA blocks
(base/head/head/base) so a monotone within-session drift cannot favor either
side, on a machine held at a steady state (suite wall time 41 to 44 s per run
for the 11 benches, against 21 to 29 s for the 8-bench suite plus 10 to 13 s
for the three next-point benches measured separately). Both venvs verified
identical by `uv pip list` except the editable path (Python 3.11.14,
jax/jaxlib 0.10.2, numpy 2.4.6, scipy 1.17.1, pytest-benchmark 5.2.3).

Raw artifacts: `2026-08-02-3c-{base,head}-run{1,2,3,4}.json`.

Two earlier same-day sessions are not the basis of this table. The first used
unbalanced ordering (base always first) and was discarded outright: its
control rows moved on untouched code (`ei_fused_acquisition` 0.45x), the tell
that within-pair drift was biasing toward whichever side ran second. The
second was a clean ABBA session at `d96f019`, superseded only because the
all-polishes-failed fix (`8bb80bb`) landed afterward and the numbers must come
from the code being merged. It agreed closely, and that agreement is the
reproduction evidence below.

## Result

Ratio is head/base, so below 1.00 is faster. Medians are the median of four
per-run medians; the ratio column is the median of the four paired ratios.

| Bench | base median (ms) | head median (ms) | paired ratio | spread |
|---|---|---|---|---|
| `bench_next_point_lbfgs_lcb` | 500 | 126 | **0.25x** | 0.24 to 0.26 |
| `bench_next_point_lbfgs_early_ei` | 73.8 | 34.6 | **0.46x** | 0.41 to 0.54 |
| `bench_next_point_lbfgs` (late EI) | 78.3 | 61.1 | **0.80x** | 0.67 to 0.83 |
| `bench_train_warm_instance[32]` (control) | 123 | 104 | 0.82x | 0.79 to 0.92 |
| `bench_train_warm_instance[128]` (control) | 336 | 354 | 1.06x | 0.89 to 1.12 |
| `bench_train_warm_instance[512]` (control) | 2643 | 2659 | 1.00x | 0.98 to 1.08 |
| `bench_train_fresh_instance` (control) | 332 | 318 | 0.95x | 0.81 to 1.23 |
| `bench_predict_batch256` (control) | 3.22 | 3.31 | 1.03x | 0.61 to 1.17 |
| `bench_ei_consumer_path_256` (control) | 86.9 | 86.5 | 1.02x | 0.98 to 1.04 |
| `bench_ei_fused_acquisition_256` (control) | 54.7 | 55.3 | 1.01x | 0.98 to 1.89 |
| `bench_ei_score_candidates_256` (control) | 2.51 | 2.21 | 1.02x | 0.27 to 2.99 |

### Reproduction across two independent sessions

The same three benches, recorded in two separate ABBA sessions hours apart at
two different commits of the branch:

| Bench | session at `d96f019` | session at `8bb80bb` |
|---|---|---|
| next point, LCB | 0.24x (0.18 to 0.25) | 0.25x (0.24 to 0.26) |
| next point, early-BO EI | 0.45x (0.41 to 0.50) | 0.46x (0.41 to 0.54) |
| next point, late-stage EI | 0.84x (0.82 to 0.89) | 0.80x (0.67 to 0.83) |

### On the controls

Seven of eight controls sit at 0.95x to 1.06x, which is what says the pairing
worked. `train_warm_instance[32]` is the exception at 0.82x, and it is noise
rather than signal: the same untouched bench posted 1.06x in the `d96f019`
session, so it flips sign between sessions. Taking both sessions together it
sizes the residual noise on the small train bench at roughly 20 percent, which
is still far inside the margin on all three changed benches.

Two control spreads have a wide tail (`ei_fused_acquisition` to 1.89,
`ei_score_candidates` 0.27 to 2.99). Both come from single-run outliers on
sub-3 ms or warm-up-sensitive benches, and both medians sit at 1.01x and 1.02x.
3c does not touch either path.

### Why the three surfaces differ so much

Reduction in polished starts is the whole mechanism (10 blind polishes
become one batched scan of 320 candidates plus k=2 polishes), so the win
tracks how much each surface charges per polish.

- **LCB, trained n=128 (0.25x).** Every start carries gradient signal, so a
  polish costs real L-BFGS-B iterations, each paying a host round trip.
  Cutting 10 polishes to 2 removes most of the work.
- **Early-BO EI, n=16 (0.46x).** A wiggly EI surface, same mechanism, fewer
  iterations per polish.
- **Late-stage EI, n=128 (0.80x).** EI is flat over most of the domain here,
  so most blind starts terminate at zero iterations and cost almost nothing.
  Only the scan's improved starting quality is left, which is why this is
  the smallest of the three and why it is the surface the naive "EI is
  already cheap" reading comes from.

## Cost: first-call latency

`first_call_latency_s`, per run:

| Bench | base | head |
|---|---|---|
| `bench_next_point_lbfgs` (late EI) | 1.72 / 1.63 / 1.67 / 1.67 | 3.02 / 2.88 / 3.01 / 3.01 |
| `bench_next_point_lbfgs_early_ei` | 0.70 / 0.70 / 0.77 / 0.73 | 1.15 / 1.18 / 1.16 / 1.21 |
| `bench_next_point_lbfgs_lcb` | 0.92 / 1.01 / 1.00 / 0.98 | 1.15 / 1.15 / 1.02 / 1.08 |

The batched scan adds one more traced shape, costing about **+1.3 s once on
the first EI next-point call of a process** and about +0.45 s on the early-BO
shape, with LCB about +0.1 s. It is repaid on the second call at every surface
measured, and a BO run makes one such call per iteration.
