# 3b delta: on-device multi-start train, against the post-3c base (2026-08-05)

Paired before/after recording for issue #31 / PR #70, re-recorded against the
base the PR will actually merge onto. This supersedes
`2026-08-02-3b-train-delta.md`, which used base `2e2bb87` (pre-3c), per the
issue #18 rule that the second PR to land rebases on the first and re-records
its paired delta before merge.

The re-record was not a formality. The old base predates
`benchmarks/bench_next_point.py`, which arrived with 3c, so the three
next-point benches had never been measured on both sides of 3b. They surface
a real side effect of this change that the earlier table structurally could
not show. See "Disclosed side effect" below.

## What was compared

| Side | Worktree | Commit |
|---|---|---|
| base | `C:\git\JAX-BO` | `27f607f` (main, 3a and 3c merged) |
| head | `C:\git\JAX-BO-3b` | `4dc26ce` (`perf/3b-train-path`, rebased onto `27f607f`) |

All 11 benches now exist on both sides, so nothing had to be copied across
worktrees. Every JSON records `commit_info.id` and, unlike the v0.2.0
baseline, **every run recorded `dirty = false`**: both trees were clean.

## Protocol

Four pairs in one session, order balanced in two ABBA blocks
(base/head/head/base, then base/head/head/base) so a monotone within-session
drift cannot favor either side. Machine verified quiet before recording, by
the tell documented in the v0.2.0 baseline:

- CPU sampled five times at 14 to 17 percent idle, 13 GB RAM free, on AC;
- probe suite wall time 27 s against the about 30 s clean threshold;
- probe medians at or below the clean band on every gated bench
  (`ei_consumer_path` 86.9 ms, `train_warm[512]` 1040 ms, `train_warm[32]`
  33.2 ms).

For contrast, an attempt earlier the same evening was **aborted without
recording**: the same probe read 67 s wall time, `ei_consumer_path` at 363 ms
(3x its clean value) and `train_warm[512]` at 2801 ms. That is the same
signature that caused two runs to be discarded on 2026-08-01, and it is why
nothing was recorded then.

The eight recording runs came in at 24.3 to 27.7 s wall time, a 14 percent
spread, the tightest of any session recorded on this machine.

Both venvs verified identical by `uv pip list` except the editable path
(Python 3.11.14, jax/jaxlib 0.10.2, numpy 2.4.6, scipy 1.17.1,
pytest-benchmark 5.2.3).

Raw artifacts: `2026-08-05-3b-{base,head}-run{1,2,3,4}.json`.

## Result

Ratio is head/base, so below 1.00 is faster. Median columns are the median of
the four per-run medians; the ratio column is the median of the four paired
ratios.

| Bench | base median (ms) | head median (ms) | paired ratio | spread |
|---|---|---|---|---|
| `bench_train_warm_instance[32]` | 44.32 | 12.67 | **0.29x** | 0.27 to 0.38 |
| `bench_train_warm_instance[128]` | 105.3 | 81.25 | **0.79x** | 0.74 to 0.98 |
| `bench_train_warm_instance[512]` | 1502 | 637.7 | **0.42x** | 0.38 to 0.46 |
| `bench_train_fresh_instance` | 110.9 | 88.46 | **0.80x** | 0.71 to 1.01 |
| `bench_next_point_lbfgs_lcb` | 38.95 | 30.84 | 0.83x | 0.77 to 0.94 |
| `bench_next_point_lbfgs` (late EI) | 16.05 | 35.05 | **2.03x** | 1.79 to 2.30 |
| `bench_next_point_lbfgs_early_ei` | 10.16 | 17.87 | **1.88x** | 1.55 to 2.10 |
| `bench_predict_batch256` (control) | 1.166 | 1.126 | 0.97x | 0.92 to 1.40 |
| `bench_ei_consumer_path_256` (control) | 88.07 | 86.83 | 0.99x | 0.95 to 1.03 |
| `bench_ei_fused_acquisition_256` (control) | 56.34 | 56.16 | 1.00x | 0.97 to 1.03 |
| `bench_ei_score_candidates_256` (control) | 1.087 | 1.081 | 1.00x | 0.98 to 1.00 |

**Train is 1.25x to 3.4x faster and every one of the sixteen paired train
ratios is below 1.02**, with the single worst pair at 1.01 on the noisy
fresh-instance bench. The win reproduces the direction and rough size of the
2026-08-02 recording against the older base (0.33x / 0.71x / 0.42x / 0.78x
there), which is the cross-session agreement that matters.

Head IQRs are also much tighter on the train benches (n=512: 36 ms head
against 125 ms base; n=32: 0.90 ms against 5.0 ms), which is what removing
about 85 host-device round trips per train should look like.

The four true controls are flat at 0.97x to 1.00x. That is what says the
pairing worked.

## Disclosed side effect: EI next-point selection is about 2x slower

`bench_next_point_lbfgs` (2.03x) and `bench_next_point_lbfgs_early_ei`
(1.88x) are **slower on head**, well outside their spreads. This is a real
regression and it is not machine noise: the four paired ratios never overlap
1.00 on either bench.

It is not a code change. 3b touches `GP.train`, its docstrings, and adds
`_train_multistart`; `compute_next_point_lbfgs` is byte identical on both
sides. Traced to its cause:

1. **The trained hyperparameters differ.** Base selects
   `logsigma_f 2.7638`, lengthscales `1.0955 / 1.7012 / 12.2877 / 11.6405`;
   head selects `2.5739` and `1.0440 / 1.6305 / 10.8373 / 10.9956`. Both
   reach essentially the same NLML (32.6439 base, 32.6660 head) and the
   next-point call finds essentially the same acquisition value
   (-0.2732 base, -0.2719 head), so neither answer is worse.
2. **The polish does less work on head, not more.** On the actual batched
   path, the two polishes take 50 objective calls and 13 L-BFGS-B iterations
   on head against 101 calls and 15 iterations on base.
3. **But each objective call is more expensive.** `acq_value_and_grad`
   measured over 200 warm calls: **0.362 ms on head against 0.254 ms on
   base, 1.42x**, on identical code with only the params differing.

So head's optimizer lands on a hyperparameter set whose EI evaluation is
individually costlier, and that per-call cost outweighs the halved call
count. Reproduced outside pytest-benchmark: one `compute_next_point_lbfgs`
call, each worktree's own fixture, 16.3 ms base against 28.4 ms head.

Why LCB moves the other way (0.83x, faster) is consistent with the same
mechanism: that surface charges much more per polish, so head's halved call
count dominates its higher per-call price.

**Consequence for a consumer.** A BO iteration is one train plus one
next-point call. On the benched shapes the train saving (about 24 ms at
n=128 warm, about 865 ms at n=512) is larger than the next-point cost (about
19 ms on late EI, about 8 ms on early EI), so the iteration is still net
faster, and increasingly so as n grows. It is not free, and a workload that
trains rarely and selects often would see less benefit than the train table
alone suggests.

This effect is a downstream consequence of which optimum the multi-start
selects, which is the same root the disclosed gap-posterior regression comes
from (issue #71, the fixed 1e-8 jitter at cond 1e15 to 1e17). It is recorded
here rather than fixed inside a perf PR.

## Cost: first-call latency

`first_call_latency_s`, per run:

| Bench | base | head |
|---|---|---|
| `train_warm_instance[32]` (first train in the process) | 0.327 / 0.406 / 0.415 / 0.399 | 1.326 / 1.319 / 1.382 / 1.323 |
| `train_warm_instance[128]` | 0.085 / 0.100 / 0.110 / 0.104 | 0.079 / 0.081 / 0.078 / 0.099 |
| `train_warm_instance[512]` | 1.463 / 1.795 / 2.010 / 1.990 | 1.846 / 1.796 / 1.920 / 1.934 |
| `next_point_lbfgs` (late EI) | 1.482 / 1.479 / 1.377 / 1.378 | 1.439 / 1.442 / 1.436 / 1.478 |

The vmapped BFGS `while_loop` costs about **+0.92 s of one-time trace and
compile on the first train of a process**, and nothing after: n=128 comes in
below base and n=512 is within noise. The next-point first call is unchanged
between sides, which is further evidence its slowdown is per-call work rather
than compilation.

For the consumer (a fresh GP per BO iteration, many iterations per process)
the one-time cost is repaid by the second train. A single-train script pays
it and does not.
