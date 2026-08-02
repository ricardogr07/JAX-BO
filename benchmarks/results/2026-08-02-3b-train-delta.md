# 3b delta: on-device multi-start train (2026-08-02)

Paired before/after recording for issue #31 / PR #70. This is the recording
the PR's perf claim rests on; the earlier table in the PR body (taken at
`a57cbf1`, before the line search was replaced) is stale and superseded.

## What was compared

| Side | Worktree | Commit |
|---|---|---|
| base | `C:\git\JAX-BO` | `2e2bb87` (main, 3a merged) |
| head | `C:\git\JAX-BO-3b` | `1df12b3` (`perf/3b-train-path`) |

Base is main-with-3a, not the v0.2.0 baseline file: 3a already moved
`train_fresh_instance` (516 ms at v0.2.0 to about 132 ms here), so comparing
3b against v0.2.0 would credit 3a's jit-cache fix to this PR.

## Protocol

Three pairs, strict alternation base/head/base/head/base/head, one session,
same machine, same seeds. Both venvs verified byte-identical by
`uv pip list` except the editable path (Python 3.11.14, jax/jaxlib 0.10.2,
numpy 2.4.6, scipy 1.17.1, pytest-benchmark 5.2.3). Machine verified quiet by
the documented tell: suite wall time 20 to 29 s per run against the about 30 s
clean threshold, on AC.

Raw artifacts: `2026-08-02-3b-{base,head}-run{1,2,3}.json`.

## Result

Ratio is head/base, so below 1.00 is faster. Median column is the median of
the three per-run medians; the ratio column is the median of the three
paired ratios, with the full paired spread beside it.

| Bench | base median (ms) | head median (ms) | paired ratio | spread |
|---|---|---|---|---|
| `bench_train_warm_instance[32]` | 54.9 | 17.4 | 0.33x | 0.23 to 0.38 |
| `bench_train_warm_instance[128]` | 141 | 92.0 | 0.71x | 0.62 to 0.79 |
| `bench_train_warm_instance[512]` | 1738 | 723 | 0.42x | 0.37 to 0.45 |
| `bench_train_fresh_instance` | 132 | 101 | 0.78x | 0.76 to 1.02 |
| `bench_predict_batch256` | 1.20 | 1.38 | 1.15x | 1.01 to 1.19 |
| `bench_ei_consumer_path_256` | 130 | 131 | 1.00x | 0.96 to 1.93 |
| `bench_ei_fused_acquisition_256` | 90.9 | 86.9 | 0.99x | 0.96 to 1.15 |
| `bench_ei_score_candidates_256` | 1.33 | 1.42 | 1.06x | 1.06 to 1.33 |

**Train is 1.3x to 3.0x faster and every paired train ratio is below 1.0**,
except one fresh-instance pair at 1.02. The three warm cases separate
cleanly: all nine paired ratios sit below 0.80, and head IQRs are 3 to 100x
tighter than base (n=512: 16 to 41 ms head against 107 to 1610 ms base), which
is what removing about 85 host-device round trips per train should look like.

The four untouched benches are noise. `predict` at 1.15x is a 0.18 ms move on
a 1.2 ms bench whose per-run IQR is 0.11 to 0.33 ms, so the shift is inside
one IQR; `ei_consumer_path` spans 0.96 to 1.93 across pairs on unchanged code,
which sizes the residual machine noise on this suite. No claim is made on any
of these four.

## Cost: first-call latency on the first train

`first_call_latency_s`, per run:

| Bench | base | head |
|---|---|---|
| `train_warm_instance[32]` (first train in the process) | 0.52 / 0.47 / 0.54 | 1.81 / 1.51 / 1.56 |
| `train_warm_instance[128]` | 0.19 / 0.17 / 0.16 | 0.12 / 0.10 / 0.13 |
| `train_warm_instance[512]` | 2.86 / 1.97 / 2.05 | 2.26 / 2.25 / 2.16 |

The vmapped BFGS `while_loop` costs about **+1.05 s of one-time trace and
compile on the first train of a process**, and nothing after: the later
shapes come in at or below base. For the consumer (a fresh GP per BO
iteration, many iterations per process) that is repaid by the second train.
A single-train script pays it and does not.
