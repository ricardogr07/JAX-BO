"""jit compilation-cache keying tests (3a, issue #30).

The ``GP`` methods decorated with ``static_argnums=(0,)`` key jax's
compilation cache on the instance. ``GP`` therefore defines config-based
``__eq__`` and ``__hash__``: two instances built from an equivalent options
dict are interchangeable static arguments, so a second ``GP(...)``
construction hits the caches the first one populated instead of paying the
per-instance recompilation the fresh-instance bench measures.

Retracing is the compile proxy here: jax only compiles what it has traced,
and tracing calls the kernel function at Python level, so a counting kernel
wrapper observes cache misses without touching jax private APIs.
"""

from jax import random

import jaxbo.gp as gp_module
import jaxbo.kernels as kernels
from jaxbo import initializers
from jaxbo.gp import GP

CONFIG = {"kernel": "RBF", "criterion": "EI", "input_prior": None}


def _demo_problem(n=16, d=2, seed=0):
    key_x, key_y, key_p = random.split(random.PRNGKey(seed), 3)
    X = random.uniform(key_x, (n, d))
    y = random.normal(key_y, (n, 1))
    params = initializers.random_init_GP(key_p, d)
    return {"X": X, "y": y}, params


def test_same_config_instances_are_equal():
    assert GP(dict(CONFIG)) == GP(dict(CONFIG))
    assert hash(GP(dict(CONFIG))) == hash(GP(dict(CONFIG)))


def test_config_differences_break_equality():
    base = GP(dict(CONFIG))
    assert base != GP({**CONFIG, "kernel": "Matern52"})
    assert base != GP({**CONFIG, "criterion": "LCB"})
    # input_prior compares by object identity: a different prior object is a
    # different cache line even if it would behave identically.
    assert base != GP({**CONFIG, "input_prior": object()})
    assert base != "not a GP"


class _UnhashablePrior:
    """Value-equality prior stand-in: eq without hash, like a mutable dataclass."""

    def __init__(self, tag):
        self.tag = tag

    def __eq__(self, other):
        return isinstance(other, _UnhashablePrior) and self.tag == other.tag

    __hash__ = None


def test_unhashable_prior_keeps_identity_semantics():
    prior = _UnhashablePrior("p")
    gp = GP({**CONFIG, "input_prior": prior})
    hash(gp)  # the prior must never be hashed itself
    assert gp == GP({**CONFIG, "input_prior": prior})
    # Equal by value but a different object: identity semantics, distinct
    # cache line.
    assert gp != GP({**CONFIG, "input_prior": _UnhashablePrior("p")})
    batch, params = _demo_problem()
    gp.likelihood(params, batch)  # jit static-arg hashing must accept it


def test_criterion_mutation_rekeys(monkeypatch):
    calls = {"n": 0}

    def counting_rbf(x1, x2, params):
        calls["n"] += 1
        return kernels.RBF(x1, x2, params)

    monkeypatch.setitem(gp_module.SUPPORTED_KERNELS, "RBF", counting_rbf)

    batch, params = _demo_problem()
    warm = GP(dict(CONFIG))
    warm.likelihood(params, batch)

    # Mutating the criterion after construction must move the instance off
    # the warmed cache line: the key is read live, so the mutated instance
    # retraces instead of silently reusing code traced for the old config.
    mutated = GP(dict(CONFIG))
    assert mutated == warm
    mutated.options["criterion"] = "LCB"
    assert mutated != warm

    calls["n"] = 0
    mutated.likelihood(params, batch)
    assert calls["n"] > 0


def test_second_instance_pays_zero_retraces(monkeypatch):
    calls = {"n": 0}

    # Defined inside the test so its identity is unique per run: the cache
    # lines this test exercises cannot collide with entries other tests
    # already compiled against the real RBF.
    def counting_rbf(x1, x2, params):
        calls["n"] += 1
        return kernels.RBF(x1, x2, params)

    monkeypatch.setitem(gp_module.SUPPORTED_KERNELS, "RBF", counting_rbf)

    batch, params = _demo_problem()
    bounds = {"lb": batch["X"].min(0), "ub": batch["X"].max(0)}
    predict_kwargs = {"params": params, "batch": batch, "bounds": bounds}

    first = GP(dict(CONFIG))
    first.likelihood(params, batch)
    first.predict(batch["X"], **predict_kwargs)
    assert calls["n"] > 0, "first instance must trace"

    calls["n"] = 0
    second = GP(dict(CONFIG))
    second.likelihood(params, batch)
    second.predict(batch["X"], **predict_kwargs)
    assert calls["n"] == 0, "equal-config instance must hit the jit cache"

    # Granularity sanity: a different criterion is a different cache key, so
    # even likelihood (which never reads the criterion) retraces. That is
    # the price of keying conservatively on the whole config.
    calls["n"] = 0
    third = GP({**CONFIG, "criterion": "LCB"})
    third.likelihood(params, batch)
    assert calls["n"] > 0
