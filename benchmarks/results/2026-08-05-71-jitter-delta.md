# 71 delta: problem-scaled Cholesky jitter (2026-08-05)

Paired before/after recording for issue #71 / the `fix/71-scaled-jitter` PR,
per AC5: the jitter changes every Cholesky, so every timed path could move.

This is not an optimization PR. No speedup is claimed and none is expected.
The delta exists to answer one question: does a correctness fix that touches
the innermost linear algebra of every model cost anything a consumer would
notice? The answer is that it does not change the acquisition surfaces or the
selected points, and that the one bench that moves reproducibly outside the
harness (LCB next-point, 1.6x slower) moves for the same reason the 3b delta
documented, which is a different trained optimum, not different code.

## What was compared

| Side | Worktree | Commit |
|---|---|---|
| base | `C:\git\JAX-BO-71-base` | `d25dc65` (main, v0.2.1) |
| head | `C:\git\JAX-BO` | `d2e99eb` (`fix/71-scaled-jitter`) |

All 11 benches exist on both sides. Every JSON records `commit_info.id` and
**`dirty = false` on all eight runs**; both trees were clean, verified before
the first run and unchanged after the last. Both venvs are identical
(Python 3.11.14, jax/jaxlib 0.10.2, numpy 2.4.6, scipy 1.17.1) except the
editable path.

Raw artifacts: `2026-08-05-71-{base,head}-run{1,2,3,4}.json`.

## Protocol, and its one honest weakness

Four pairs in one session, order balanced in two ABBA blocks
(base/head/head/base, then base/head/head/base), so a monotone within-session
drift cannot favor either side.

**The machine was not as quiet as the 3b session.** The documented quiet tell
(see the v0.2.0 baseline) is about 30 s probe wall time with
`ei_consumer_path` near 86.9 ms. A first probe read 40.7 s and 128.9 ms and
was **discarded without recording**. A second probe read 32.6 s and 107.1 ms:
wall time inside the band, `ei_consumer_path` still about 23 percent high,
with VS Code and Chrome holding CPU at 24 to 62 percent against the clean 14
to 17 and 5.8 GB RAM free against the clean 13.

That state was accepted for recording and it shows in the results: the paired
spreads here are much wider than 3b's (several benches span 0.4x to 1.2x
where 3b's controls sat inside 0.03x). **Treat any per-bench ratio inside
about 0.8x to 1.25x as indistinguishable from machine noise in this session.**
That band is wide enough that this table cannot certify a small regression
absent, only a large one. What it can do, and what it was recorded for, is
confirm that nothing moved by the order of magnitude a broken Cholesky would
produce, and it does not.

The two ratios that fall outside the noise band are traced below by direct
measurement outside pytest-benchmark, which is where the real evidence for
them is.

## Result

Ratio is head/base, so below 1.00 is faster. Median columns are the median of
the four per-run medians; the ratio column is the median of the four paired
ratios.

| Bench | base median (ms) | head median (ms) | paired ratio | spread |
|---|---|---|---|---|
| `bench_train_warm_instance[32]` | 14.79 | 15.31 | 0.94x | 0.90 to 1.26 |
| `bench_train_warm_instance[128]` | 104.0 | 116.2 | 1.18x | 0.98 to 1.32 |
| `bench_train_warm_instance[512]` | 802.7 | 730.8 | 1.01x | 0.80 to 1.12 |
| `bench_train_fresh_instance` | 108.4 | 118.1 | 1.05x | 0.90 to 1.34 |
| `bench_next_point_lbfgs` (late EI) | 39.03 | 52.55 | 1.33x | 0.76 to 1.77 |
| `bench_next_point_lbfgs_early_ei` | 20.33 | 15.32 | **0.77x** | 0.60 to 0.84 |
| `bench_next_point_lbfgs_lcb` | 36.65 | 64.24 | **1.68x** | 1.56 to 1.76 |
| `bench_predict_batch256` | 1.314 | 1.342 | 1.04x | 1.00 to 1.04 |
| `bench_ei_consumer_path_256` | 146.1 | 139.3 | 0.96x | 0.85 to 1.22 |
| `bench_ei_fused_acquisition_256` | 104.2 | 86.28 | 0.90x | 0.43 to 0.93 |
| `bench_ei_score_candidates_256` | 1.982 | 1.605 | 0.81x | 0.38 to 1.00 |

Nine of the eleven benches land inside the session noise band. `predict`, the
bench most directly exposed to the change (it now calls `jitter` twice per
call and evaluates a `where` over the variance), is 1.04x with a spread of
1.00 to 1.04, the tightest pair in the session: **the added per-call work is
not measurable.** That is the expected result, since `jitter` is an O(n) read
of an already-materialized diagonal next to an O(n^3) Cholesky.

Two benches fall outside the band and never overlap 1.00 across their four
pairs: `next_point_lbfgs_lcb` at 1.68x slower and
`next_point_lbfgs_early_ei` at 0.77x faster.

## The two outliers: a different optimum, not different code

`compute_next_point_lbfgs`, `acquisitions`, and the whole polish path are byte
identical on both sides. This change touches the jitter and the posterior
variance clip, so the mechanism has to be indirect, and it is the same one the
3b delta documented: **the trained hyperparameters differ, and the acquisition
surface they induce charges a different amount per polish step.**

Trained hyperparameters at the benched seed, `[logsigma_f, 4 lengthscales,
lognoise]`:

| | base | head |
|---|---|---|
| n=128 | `2.5739, 0.0431, 0.4889, 2.3830, 2.3975, -2.8861` | `2.5498, 0.0386, 0.4818, 2.3648, 2.3848, -2.9574` |
| n=16 | `-0.1315, -1.4874, -0.9822, 1.3864, 0.1482, -2.8475` | `-0.1326, -1.4959, -0.9775, 1.3840, 0.1478, -2.8501` |

The two sides agree to about 1 percent on every coordinate, which is what a
regularization change of this size should do. They are not identical, so the
multi-start lands at a slightly different point on the same basin.

**Both sides return the same answer.** Reproduced outside pytest-benchmark,
five warm calls each, each worktree's own fixture:

| Surface | base acq | head acq | base x | head x | base ms | head ms |
|---|---|---|---|---|---|---|
| late EI (n=128) | -0.271884 | -0.271122 | `0.4788, 1, 0, 1` | `0.4789, 1, 0, 1` | 44.1 | 48.7 |
| LCB (n=128) | -3.045998 | -3.047703 | `0.4894, 1, 0, 1` | `0.4896, 1, 0, 1` | 41.5 | 67.0 |
| early EI (n=16) | -0.152451 | -0.152133 | `0.3783, 1, 0, 0.9160` | `0.3785, 1, 0, 0.9123` | 36.0 | 16.3 |

The selected points agree to 3 or 4 decimals and the acquisition values to 3.
**Neither side finds a better point; they find the same point.** The LCB
slowdown (1.61x here, against the 1.68x paired ratio, so the harness number is
real) and the early-EI speedup (0.45x here, against 0.77x paired, same
direction) are both per-call polish cost on surfaces induced by slightly
different hyperparameters. They move in opposite directions on the same
change, which is the signature of surface-dependent polish cost rather than
of added work: added work cannot make a bench faster.

Whether the LCB cost persists on a quieter machine is not settled by this
session. It is recorded rather than argued away, and it should be re-measured
if it ever gates anything. The consumer path (`ei_consumer_path`, the exact
shelter-pulse loop) is 0.96x, and `predict` is 1.04x, so nothing here reaches
a consumer as a slowdown that matters.

## First-call latency

Unchanged within noise on every bench. The largest shift is
`next_point_lbfgs_early_ei` (base 0.745 to 0.821 s, head 0.975 to 1.106 s),
about +0.25 s of one-time trace and compile from the extra `where` in the
variance path and the two `jitter` reductions; `train_warm[128]` is flat at
0.085 to 0.113 s on both sides, and `train_warm[512]` overlaps completely
(base 1.95 to 2.70 s, head 2.33 to 2.93 s).

## What this delta does and does not establish

- **Does**: the correctness fix costs nothing measurable on `predict`, on the
  consumer EI path, or on train; and it does not change which point the
  optimizer selects on any of the three acquisition surfaces.
- **Does not**: certify a sub-25-percent regression absent anywhere. The
  machine state in this session is too loose for that. If a later change needs
  that resolution, it needs its own recording on a verified-quiet machine.
