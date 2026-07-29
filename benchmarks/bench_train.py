"""GP.train baselines: multi-start L-BFGS, num_restarts=3, 4D input.

Two cases:

- train_warm_instance: one GP instance is constructed and trained once
  (untimed first call), then re-trained in the timed rounds. The jit cache
  (keyed on instance identity via static_argnums=(0,)) is hot, so this
  isolates the optimizer-loop cost.
- train_fresh_instance: a NEW GP(...) is constructed INSIDE every timed
  round and trained once. This mirrors the real consumer (shelter-pulse
  jaxbo_optimizer.py), which builds a fresh GP each BO iteration, so every
  round pays the instance-keyed recompilation. This case is the direct
  gate for issue #30 (jit cache keying).

The first call per (n, dim) shape is timed separately and reported as
first_call_latency_s in extra_info. First-call latency conflates trace +
compile + execute (plus process-cold init when jax is fresh); see the
results doc for the composition. Timed paths are synchronized with
jax.block_until_ready.
"""

import time

import jax
import pytest
from jax import random

from conftest import NUM_RESTARTS, OPTIONS, SEED, make_problem
from jaxbo.models import GP


@pytest.mark.parametrize("n", [32, 128, 512])
def bench_train_warm_instance(benchmark, n):
    gp, batch, _, _ = make_problem(n)
    key = random.PRNGKey(SEED)

    t0 = time.perf_counter()
    jax.block_until_ready(gp.train(batch, key, num_restarts=NUM_RESTARTS))
    first_s = time.perf_counter() - t0

    benchmark.extra_info["n"] = n
    benchmark.extra_info["first_call_latency_s"] = round(first_s, 4)

    benchmark.pedantic(
        lambda: jax.block_until_ready(
            gp.train(batch, key, num_restarts=NUM_RESTARTS)
        ),
        rounds=5,
        iterations=1,
    )


def bench_train_fresh_instance(benchmark):
    """Consumer path: GP constructed inside the timed call, trained once.

    Every round constructs a new instance, so every round re-pays the
    instance-keyed jit compilation. Run in-session (after the other
    benches), so process-cold costs that are NOT instance-keyed (jax
    init, the module-level random_init_GP trace) are already paid; the
    per-round cost is what one consumer BO iteration pays today.
    """
    _, batch, _, _ = make_problem(128)
    key = random.PRNGKey(SEED)

    def fresh_train():
        gp = GP(dict(OPTIONS))
        return jax.block_until_ready(
            gp.train(batch, key, num_restarts=NUM_RESTARTS)
        )

    benchmark.extra_info["n"] = 128
    benchmark.pedantic(fresh_train, rounds=5, iterations=1)
