"""GP.train baseline: multi-start L-BFGS, num_restarts=3, 2D input.

The first call per (n, dim) shape includes JIT compilation of the
likelihood; it is timed separately and reported as cold_first_call_s in
extra_info. The benchmarked path is the warm one (same model instance,
jit cache hot), synchronized with jax.block_until_ready.
"""

import time

import jax
import pytest
from jax import random

from conftest import NUM_RESTARTS, SEED, make_problem


@pytest.mark.parametrize("n", [32, 128, 512])
def bench_train(benchmark, n):
    gp, batch, _, _ = make_problem(n)
    key = random.PRNGKey(SEED)

    t0 = time.perf_counter()
    jax.block_until_ready(gp.train(batch, key, num_restarts=NUM_RESTARTS))
    cold_s = time.perf_counter() - t0

    benchmark.extra_info["n"] = n
    benchmark.extra_info["cold_first_call_s"] = round(cold_s, 4)

    benchmark.pedantic(
        lambda: jax.block_until_ready(
            gp.train(batch, key, num_restarts=NUM_RESTARTS)
        ),
        rounds=5,
        iterations=1,
    )
