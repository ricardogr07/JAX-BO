"""Shared fixtures for the jaxbo baseline benchmarks. Seeded, CPU only.

Only the public API is used (jaxbo.models.GP), so this suite must keep
running unchanged against the refactored package.
"""

import numpy as onp
import jax.numpy as jnp
import pytest
from jax import random

from jaxbo.models import GP

SEED = 0
# 4D to match the real consumer (shelter-pulse jaxbo_optimizer.py: 4 budget
# shares on [0, 1]^4, (1, 4) predict inputs).
DIM = 4
NUM_RESTARTS = 3
# Current GPmodel.__init__ requires the input_prior key even when unused (EI).
OPTIONS = {"kernel": "Matern52", "criterion": "EI", "input_prior": None}


def make_dataset(n, dim=DIM, seed=SEED):
    """Deterministic synthetic regression set on [0, 1]^dim."""
    rng = onp.random.default_rng(seed)
    X = rng.uniform(size=(n, dim))
    y = onp.sin(3.0 * X[:, :1]) * onp.cos(2.0 * X[:, 1:2])
    y = y + 0.25 * (X[:, 2:3] - X[:, 3:4]) if dim >= 4 else y
    y = y + 0.1 * rng.standard_normal((n, 1))
    return jnp.asarray(X), jnp.asarray(y)


def make_problem(n):
    """Fresh model plus normalized batch, bounds, norm_const (untrained)."""
    X, y = make_dataset(n)
    bounds = {"lb": jnp.zeros(DIM), "ub": jnp.ones(DIM)}
    # X already lives in [0, 1]^dim; standardize y inline so the suite only
    # depends on the model API, not on jaxbo.utils surviving the refactor.
    batch = {"X": X, "y": (y - y.mean(0)) / y.std(0)}
    norm_const = {"mu_y": y.mean(0), "sigma_y": y.std(0)}
    gp = GP(dict(OPTIONS))
    return gp, batch, bounds, norm_const


@pytest.fixture(scope="session")
def trained_128():
    """One trained GP (n=128) shared by the predict and acquisition benches."""
    gp, batch, bounds, norm_const = make_problem(128)
    params = gp.train(batch, random.PRNGKey(SEED), num_restarts=NUM_RESTARTS)
    kwargs = {
        "params": params,
        "batch": batch,
        "bounds": bounds,
        "norm_const": norm_const,
    }
    return gp, kwargs
