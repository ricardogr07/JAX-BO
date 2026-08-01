"""Next-point selection through compute_next_point_lbfgs.

The consumer's other per-iteration cost besides train: multi-start
optimization of the acquisition surface. One call with the default
num_restarts=10 on the trained n=128 4D problem, seeded. Before 3c this
is 10 serial bounded L-BFGS-B polishes, each optimizer step paying a
Python to jit to host round trip; after 3c it is one batched
score_candidates start scan plus top-k serial polishes. The bench only
uses the public API present on both sides, so the same file records both
halves of the paired delta.

First-call latency (trace + compile + execute) is timed separately and
reported as first_call_latency_s in extra_info.
"""

import time

import numpy as onp
from jax import random

from conftest import SEED

NUM_RESTARTS_NP = 10


def bench_next_point_lbfgs(benchmark, trained_128):
    gp, kwargs = trained_128
    call_kwargs = {**kwargs, "rng_key": random.PRNGKey(SEED + 2)}

    def next_point():
        x_new, acq, loc = gp.compute_next_point_lbfgs(
            num_restarts=NUM_RESTARTS_NP, **call_kwargs
        )
        # Host conversion forces the sync, like the consumer using the point.
        return onp.asarray(x_new)

    t0 = time.perf_counter()
    next_point()
    first_s = time.perf_counter() - t0

    benchmark.extra_info["num_restarts"] = NUM_RESTARTS_NP
    benchmark.extra_info["first_call_latency_s"] = round(first_s, 4)

    benchmark(next_point)
