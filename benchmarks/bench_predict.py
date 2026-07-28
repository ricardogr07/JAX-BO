"""GP.predict baseline: 256-point grid after one train (n=128).

Cold call (JIT compile of predict for the grid shape) is timed separately;
the benchmark measures the warm path with jax.block_until_ready.
"""

import time

import jax
import jax.numpy as jnp


def bench_predict_grid256(benchmark, trained_128):
    gp, kwargs = trained_128
    g = jnp.linspace(0.0, 1.0, 16)
    xx, yy = jnp.meshgrid(g, g)
    X_star = jnp.column_stack([xx.ravel(), yy.ravel()])  # (256, 2)

    t0 = time.perf_counter()
    jax.block_until_ready(gp.predict(X_star, **kwargs))
    cold_s = time.perf_counter() - t0

    benchmark.extra_info["grid_points"] = int(X_star.shape[0])
    benchmark.extra_info["cold_first_call_s"] = round(cold_s, 4)

    benchmark(lambda: jax.block_until_ready(gp.predict(X_star, **kwargs)))
