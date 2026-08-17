"""Next-point selection through compute_next_point_lbfgs.

The consumer's other per-iteration cost besides train: multi-start
optimization of the acquisition surface. Each case makes one call with
the default num_restarts=10, seeded. Before 3c every case is a serial
loop of bounded L-BFGS-B polishes, each optimizer step paying a Python
to jit to host round trip; after 3c it is one batched score_candidates
start scan plus top-k serial polishes. The benches only use the public
API present on both sides, so the same file records both halves of the
paired delta.

Three surfaces, because the polish cost is surface-dependent (3c
diagnostics): on late-stage flat EI (trained n=128) most blind starts
terminate at zero iterations, so start reduction is structurally a wash;
starts carry gradient signal on early-BO EI (n=16) and on LCB
(everywhere), which is where the multi-start actually costs iterations.

First-call latency (trace + compile + execute) is timed separately and
reported as first_call_latency_s in extra_info.
"""

import time

import numpy as onp
import pytest
from jax import random

from conftest import OPTIONS, SEED, make_problem
from jaxbo.models import GP

NUM_RESTARTS_NP = 10
"""Number of random starts in next-point benchmarks."""
LCB_KAPPA = 2.0
"""Exploration coefficient for the LCB benchmark."""


@pytest.fixture(scope="session")
def trained_16():
    """Early-BO-shaped problem: 16 points, wiggly EI surface."""
    gp, batch, bounds, norm_const = make_problem(16)
    params = gp.train(batch, random.PRNGKey(SEED), num_restarts=3)
    kwargs = {
        "params": params,
        "batch": batch,
        "bounds": bounds,
        "norm_const": norm_const,
    }
    return gp, kwargs


def _time_first_call(benchmark, fn):
    t0 = time.perf_counter()
    fn()
    benchmark.extra_info["first_call_latency_s"] = round(time.perf_counter() - t0, 4)
    benchmark.extra_info["num_restarts"] = NUM_RESTARTS_NP


def bench_next_point_lbfgs(benchmark, trained_128):
    """Run the bench next point lbfgs benchmark."""
    gp, kwargs = trained_128
    call_kwargs = {**kwargs, "rng_key": random.PRNGKey(SEED + 2)}

    def next_point():
        x_new, acq, loc = gp.compute_next_point_lbfgs(
            num_restarts=NUM_RESTARTS_NP, **call_kwargs
        )
        # Host conversion forces the sync, like the consumer using the point.
        return onp.asarray(x_new)

    _time_first_call(benchmark, next_point)
    benchmark(next_point)


def bench_next_point_lbfgs_early_ei(benchmark, trained_16):
    """Run the bench next point lbfgs early ei benchmark."""
    gp, kwargs = trained_16
    call_kwargs = {**kwargs, "rng_key": random.PRNGKey(SEED + 2)}

    def next_point():
        x_new, acq, loc = gp.compute_next_point_lbfgs(
            num_restarts=NUM_RESTARTS_NP, **call_kwargs
        )
        return onp.asarray(x_new)

    _time_first_call(benchmark, next_point)
    benchmark(next_point)


def bench_next_point_lbfgs_lcb(benchmark, trained_128):
    """Run the bench next point lbfgs lcb benchmark."""
    # Same trained hyperparameters (training is criterion-independent), LCB
    # dispatch: every start carries gradient signal on this surface.
    _, kwargs = trained_128
    gp = GP({**OPTIONS, "criterion": "LCB"})
    call_kwargs = {**kwargs, "kappa": LCB_KAPPA, "rng_key": random.PRNGKey(SEED + 2)}

    def next_point():
        x_new, acq, loc = gp.compute_next_point_lbfgs(
            num_restarts=NUM_RESTARTS_NP, **call_kwargs
        )
        return onp.asarray(x_new)

    _time_first_call(benchmark, next_point)
    benchmark(next_point)
