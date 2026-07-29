"""EI scored per candidate over 256 candidates.

The primary case mirrors the real consumer (shelter-pulse
jaxbo_optimizer.py, EI loop) exactly: for each candidate, a (1, 4)
gp.predict with params/batch/bounds only (the consumer passes no
norm_const), then acquisitions.EI(mu, std, best), then a host-side
float(). Each candidate pays two jit dispatches plus a device to host
transfer. This is the loop a future batched score_candidates replaces.

A secondary case keeps the fused gp.acquisition per-candidate variant
for comparison (single fused predict + EI graph per candidate).

The batched case scores all 256 candidates in one
acquisitions.score_candidates call (issue #28), the replacement for the
consumer loop above; its delta against bench_ei_consumer_path_256 is the
number that gates the claim.

First-call latency (trace + compile + execute) is timed separately and
reported as first_call_latency_s in extra_info.
"""

import time

import jax
import jax.numpy as jnp
import numpy as onp

from conftest import DIM, SEED
from jaxbo import acquisitions

N_CANDIDATES = 256


def bench_ei_consumer_path_256(benchmark, trained_128):
    gp, kwargs = trained_128
    # The consumer passes only params/batch/bounds to predict.
    pk = {k: kwargs[k] for k in ("params", "batch", "bounds")}
    best = float(kwargs["batch"]["y"].min())
    rng = onp.random.default_rng(SEED + 1)
    # Dirichlet: 4 budget shares summing to 1, like the consumer candidates.
    X_cand = rng.dirichlet(onp.ones(DIM), size=N_CANDIDATES)

    def score(cand):
        c = jnp.array(cand).reshape(1, DIM)
        mu, std = gp.predict(c, **pk)
        # float() forces the device to host sync, as in the consumer.
        return float(acquisitions.EI(mu, std, best)[0])

    t0 = time.perf_counter()
    score(X_cand[0])
    first_s = time.perf_counter() - t0

    benchmark.extra_info["candidates"] = N_CANDIDATES
    benchmark.extra_info["first_call_latency_s"] = round(first_s, 4)

    def ei_loop():
        # ponytail: intentional per-candidate Python loop, this is the
        # consumer baseline that a batched score_candidates call replaces.
        return [score(X_cand[i]) for i in range(N_CANDIDATES)]

    benchmark(ei_loop)


def bench_ei_fused_acquisition_256(benchmark, trained_128):
    gp, kwargs = trained_128
    rng = onp.random.default_rng(SEED + 1)
    X_cand = jnp.asarray(rng.dirichlet(onp.ones(DIM), size=N_CANDIDATES))

    t0 = time.perf_counter()
    jax.block_until_ready(gp.acquisition(X_cand[0], **kwargs))
    first_s = time.perf_counter() - t0

    benchmark.extra_info["candidates"] = N_CANDIDATES
    benchmark.extra_info["first_call_latency_s"] = round(first_s, 4)

    def ei_loop():
        vals = [gp.acquisition(X_cand[i], **kwargs) for i in range(N_CANDIDATES)]
        return jax.block_until_ready(vals)

    benchmark(ei_loop)


def bench_ei_score_candidates_256(benchmark, trained_128):
    gp, kwargs = trained_128
    # Same inputs as the consumer-path bench: params/batch/bounds only,
    # same seed, same Dirichlet candidates.
    pk = {k: kwargs[k] for k in ("params", "batch", "bounds")}
    best = float(kwargs["batch"]["y"].min())
    rng = onp.random.default_rng(SEED + 1)
    X_cand = jnp.asarray(rng.dirichlet(onp.ones(DIM), size=N_CANDIDATES))

    def ei_batched():
        return jax.block_until_ready(
            acquisitions.score_candidates(gp, X_cand, **pk, best=best)
        )

    t0 = time.perf_counter()
    ei_batched()
    first_s = time.perf_counter() - t0

    benchmark.extra_info["candidates"] = N_CANDIDATES
    benchmark.extra_info["first_call_latency_s"] = round(first_s, 4)

    benchmark(ei_batched)
