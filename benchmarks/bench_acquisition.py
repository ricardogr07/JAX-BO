"""EI evaluated per candidate in a Python loop over 256 candidates.

This is the shelter-pulse usage pattern and the exact case a future
batched score_candidates will replace: one gp.acquisition call per
candidate, each paying Python-side jit dispatch. The first call (JIT
compile of the fused predict + EI graph) is timed separately.
"""

import time

import jax
import jax.numpy as jnp
import numpy as onp

from conftest import SEED

N_CANDIDATES = 256


def bench_ei_python_loop_256(benchmark, trained_128):
    gp, kwargs = trained_128
    rng = onp.random.default_rng(SEED + 1)
    X_cand = jnp.asarray(rng.uniform(size=(N_CANDIDATES, 2)))

    t0 = time.perf_counter()
    jax.block_until_ready(gp.acquisition(X_cand[0], **kwargs))
    cold_s = time.perf_counter() - t0

    benchmark.extra_info["candidates"] = N_CANDIDATES
    benchmark.extra_info["cold_first_call_s"] = round(cold_s, 4)

    def ei_loop():
        # ponytail: intentional per-candidate Python loop, this is the
        # baseline that a single vmapped score_candidates call replaces.
        vals = [gp.acquisition(X_cand[i], **kwargs) for i in range(N_CANDIDATES)]
        return jax.block_until_ready(vals)

    benchmark(ei_loop)
