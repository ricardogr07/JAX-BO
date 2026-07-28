"""GP.predict baseline: one batched call over 256 4D points (n=128 train).

This is the batched reference a future score_candidates would use; the
per-candidate (1, 4) predict path the consumer runs today is measured in
bench_acquisition.py. First-call latency (trace + compile + execute for
this shape) is timed separately; the benchmark measures the warm path
with jax.block_until_ready.
"""

import time

import jax
import jax.numpy as jnp
import numpy as onp

from conftest import DIM, SEED


def bench_predict_batch256(benchmark, trained_128):
    gp, kwargs = trained_128
    rng = onp.random.default_rng(SEED + 2)
    X_star = jnp.asarray(rng.uniform(size=(256, DIM)))

    t0 = time.perf_counter()
    jax.block_until_ready(gp.predict(X_star, **kwargs))
    first_s = time.perf_counter() - t0

    benchmark.extra_info["points"] = int(X_star.shape[0])
    benchmark.extra_info["first_call_latency_s"] = round(first_s, 4)

    benchmark(lambda: jax.block_until_ready(gp.predict(X_star, **kwargs)))
