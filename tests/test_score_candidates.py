"""Tests for acquisitions.score_candidates (issue #28, slice 2c).

Parity is checked against the exact serial consumer loop from shelter-pulse
jaxbo_optimizer.py: per candidate, reshape to (1, D), gp.predict with
params/batch/bounds, acquisitions.EI, host float().
"""

import jax.numpy as jnp
import numpy as onp
import pytest
from jax import random

from jaxbo import acquisitions
from jaxbo.gp import GP
from jaxbo.utils import normalize

N_CAND = 32


def _trained_gp(dim, n=24, seed=0):
    """Small trained GP on [0, 1]^dim with the normalized-batch contract."""
    rng = onp.random.default_rng(seed)
    X = jnp.asarray(rng.uniform(size=(n, dim)))
    y = jnp.sin(3.0 * X[:, 0]) + 0.1 * jnp.asarray(rng.standard_normal(n))
    bounds = {"lb": jnp.zeros(dim), "ub": jnp.ones(dim)}
    batch, _ = normalize(X, y[:, None], bounds)
    gp = GP({"kernel": "Matern52", "criterion": "EI", "input_prior": None})
    params = gp.train(batch, random.PRNGKey(seed), num_restarts=2)
    X_cand = jnp.asarray(rng.uniform(size=(N_CAND, dim)))
    return gp, params, batch, bounds, X_cand


@pytest.fixture(scope="module")
def gp1d():
    return _trained_gp(1)


@pytest.fixture(scope="module")
def gp4d():
    return _trained_gp(4)


def _serial_ei_loop(gp, params, batch, bounds, X_cand, best):
    """The verbatim shelter-pulse consumer pattern."""
    out = []
    for i in range(X_cand.shape[0]):
        c = jnp.asarray(X_cand[i]).reshape(1, -1)
        mu, std = gp.predict(c, params=params, batch=batch, bounds=bounds)
        out.append(float(acquisitions.EI(mu, std, best)[0]))
    return onp.asarray(out)


@pytest.mark.parametrize("problem", ["gp1d", "gp4d"])
def test_shape_and_parity_with_serial_loop(problem, request):
    gp, params, batch, bounds, X_cand = request.getfixturevalue(problem)
    best = float(jnp.min(batch["y"]))

    scores = acquisitions.score_candidates(
        gp, X_cand, params=params, batch=batch, bounds=bounds, best=best
    )

    assert scores.shape == (N_CAND,)
    assert bool(jnp.all(jnp.isfinite(scores)))

    serial = _serial_ei_loop(gp, params, batch, bounds, X_cand, best)
    # float32: batched XLA fusion reorders reductions vs the serial graph,
    # so bit-exactness is not expected; 1e-4 still catches any real bug.
    onp.testing.assert_allclose(onp.asarray(scores), serial, rtol=1e-4, atol=1e-6)
    # Same argmin: the point the consumer would pick next is unchanged.
    assert int(jnp.argmin(scores)) == int(onp.argmin(serial))


def test_acq_fn_and_kwargs_forwarding(gp1d):
    """acq_fn is pluggable and extra kwargs reach it (LCB with kappa)."""
    gp, params, batch, bounds, X_cand = gp1d

    scores = acquisitions.score_candidates(
        gp,
        X_cand,
        params=params,
        batch=batch,
        bounds=bounds,
        acq_fn=acquisitions.LCB,
        kappa=3.0,
    )

    serial = []
    for i in range(X_cand.shape[0]):
        c = X_cand[i].reshape(1, -1)
        mu, std = gp.predict(c, params=params, batch=batch, bounds=bounds)
        serial.append(float(acquisitions.LCB(mu, std, kappa=3.0)[0]))

    assert scores.shape == (N_CAND,)
    onp.testing.assert_allclose(
        onp.asarray(scores), onp.asarray(serial), rtol=1e-4, atol=1e-6
    )


def test_rejects_non_2d_candidates(gp1d):
    gp, params, batch, bounds, X_cand = gp1d
    with pytest.raises(ValueError, match=r"\(N, D\)"):
        acquisitions.score_candidates(
            gp, X_cand[0], params=params, batch=batch, bounds=bounds, best=0.0
        )
