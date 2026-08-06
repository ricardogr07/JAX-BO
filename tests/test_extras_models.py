"""Seeded round trips for every extras model (issue #71, PR 3).

``tests/test_extras.py`` proves the extras *import*; nothing proved they
*compute*. The #71 jitter sweep edited the Cholesky of all 13 multifidelity
models plus ``jaxbo.mcmc``, so that gap is what made the sweep a mechanical
change rather than a verified one. This module closes it: one seeded
train-then-predict per model class, asserting the posterior is finite,
non-negative, and NaN-free.

The NaN assert is the load-bearing one. :func:`jaxbo.gp._std_from_variance`
returns NaN when a posterior variance is negative beyond rounding scale, so
a NaN here is not a vague failure: it is that model's regularization failing
to hold its own Cholesky together.

Writing these found that every extras model's ``train`` raised on its first
call, on ``main`` as much as here: all 15 sites passed a ``(value, grad)``
objective to :func:`jaxbo.optimizers.minimize_lbfgs`, which sets
``jac="2-point"`` and therefore reads that tuple as the function value.
scipy rejects it with "The user-provided objective function must return a
scalar value." The fix is the sibling with the matching contract,
:func:`jaxbo.optimizers.minimize_lbfgs_grad` (``jac=True``), which the core
:class:`jaxbo.gp.GP` has always used. That the bug survived to now is the
argument for this module: import tests cannot see it.

Each builder returns ``(mu, std)`` rather than a model, because the models
disagree on almost everything upstream of that pair (constructor arity,
whether ``train`` takes a batch or a list of them, which ``normalize_*``
helper applies, whether ``predict`` takes ``bounds``). A shared fixture
would be a bigger lie than a builder each. Problems are deliberately tiny
and use 2 restarts: these ``train`` implementations still run the serial
scipy :func:`jaxbo.optimizers.minimize_lbfgs_grad` loop, once per restart,
through the host.
"""

import jax.numpy as jnp
import pytest
from jax import random

from jaxbo import multifidelity as mf
from jaxbo.input_priors import uniform_prior
from jaxbo.utils import (
    normalize,
    normalize_GradientGP,
    normalize_HeterogeneousMultifidelityGP,
    normalize_MultifidelityGP,
)

from .test_extras import MODEL_SHIMS

RESTARTS = 2


def _options(dim, **extra):
    """Minimal GPmodel options for a `dim`-dimensional unit-cube problem."""
    lb, ub = jnp.zeros(dim), jnp.ones(dim)
    opts = {
        "kernel": "RBF",
        "input_prior": uniform_prior(lb, ub),
        "criterion": "LCB",
    }
    opts.update(extra)
    return opts, {"lb": lb, "ub": ub}


def _single_fidelity_data(n=8, dim=1):
    """Smooth 1D-in-`dim` problem on the unit cube."""
    X = jnp.linspace(0.0, 1.0, n)[:, None]
    if dim > 1:
        X = jnp.tile(X, (1, dim))
    y = jnp.sin(3.0 * X[:, 0]) + 0.5 * X[:, 0]
    return X, y


def _two_fidelity_data(nL=8, nH=5, dim=1):
    """Correlated low/high fidelity pair: yH is a warped, shifted yL."""
    XL = jnp.linspace(0.0, 1.0, nL)[:, None]
    XH = jnp.linspace(0.05, 0.95, nH)[:, None]
    if dim > 1:
        XL, XH = jnp.tile(XL, (1, dim)), jnp.tile(XH, (1, dim))
    yL = jnp.sin(3.0 * XL[:, 0])
    yH = 1.5 * jnp.sin(3.0 * XH[:, 0]) + 0.2
    return XL, yL, XH, yH


def _multifidelity_gp(key):
    opts, bounds = _options(1)
    XL, yL, XH, yH = _two_fidelity_data()
    batch, _ = normalize_MultifidelityGP(XL, yL, XH, yH, bounds)
    model = mf.MultifidelityGP(opts)
    params = model.train(batch, key, num_restarts=RESTARTS)
    return model.predict(XH, params=params, batch=batch, bounds=bounds)


def _deep_multifidelity_gp(key):
    opts, bounds = _options(1, net_arch="MLP")
    XL, yL, XH, yH = _two_fidelity_data()
    batch, _ = normalize_MultifidelityGP(XL, yL, XH, yH, bounds)
    model = mf.DeepMultifidelityGP(opts, layers=[1, 4, 1])
    params = model.train(batch, key, num_restarts=RESTARTS)
    return model.predict(XH, params=params, batch=batch, bounds=bounds)


def _deep_multifidelity_gp_multioutputs(key):
    opts, bounds = _options(1, net_arch="MLP", constrained_criterion="LCBC")
    XL, yL, XH, yH = _two_fidelity_data()
    batch, _ = normalize_MultifidelityGP(XL, yL, XH, yH, bounds)
    model = mf.DeepMultifidelityGP_MultiOutputs(opts, layers=[1, 4, 1])
    params = model.train([batch, batch], key, num_restarts=RESTARTS)
    return model.predict(XH, params=params[0], batch=batch, bounds=bounds)


def _heterogeneous_multifidelity_gp(key):
    # Low fidelity lives in 2D and is projected onto the 1D high-fidelity
    # input space by the network, hence layers[0] = 2 and layers[-1] = 1.
    opts, bounds = _options(1)
    XL, yL, XH, yH = _two_fidelity_data()
    XL = jnp.hstack([XL, XL**2])
    batch, _ = normalize_HeterogeneousMultifidelityGP(XL, yL, XH, yH, bounds)
    model = mf.HeterogeneousMultifidelityGP(opts, layers=[2, 4, 1])
    params = model.train(batch, key, num_restarts=RESTARTS)
    return model.predict(XH, params=params, batch=batch, bounds=bounds)


def _manifold_gp(key):
    opts, bounds = _options(1)
    X, y = _single_fidelity_data()
    batch, _ = normalize(X, y, bounds)
    model = mf.ManifoldGP(opts, layers=[1, 4, 1])
    params = model.train(batch, key, num_restarts=RESTARTS)
    return model.predict(X, params=params, batch=batch, bounds=bounds)


def _manifold_gp_multioutputs(key):
    opts, bounds = _options(1, constrained_criterion="LCBC")
    X, y = _single_fidelity_data()
    batch, _ = normalize(X, y, bounds)
    model = mf.ManifoldGP_MultiOutputs(opts, layers=[1, 4, 1])
    params = model.train([batch, batch], key, num_restarts=RESTARTS)
    return model.predict(X, params=params[0], batch=batch, bounds=bounds)


def _gradient_gp(key):
    # The only model that consumes predict inputs as given: no bounds.
    opts, _ = _options(1)
    XF = jnp.linspace(0.0, 1.0, 6)[:, None]
    XG = jnp.linspace(0.1, 0.9, 4)[:, None]
    yF = jnp.sin(3.0 * XF[:, 0])
    yG = 3.0 * jnp.cos(3.0 * XG[:, 0])
    batch, _ = normalize_GradientGP(XF, yF, XG, yG)
    model = mf.GradientGP(opts)
    params = model.train(batch, key, num_restarts=RESTARTS)
    return model.predict(XF, params=params, batch=batch)


def _multiple_independent_mfgp(key):
    opts, bounds = _options(1, constrained_criterion="LCBC")
    XL, yL, XH, yH = _two_fidelity_data()
    batch, _ = normalize_MultifidelityGP(XL, yL, XH, yH, bounds)
    model = mf.MultipleIndependentMFGP(opts)
    params = model.train([batch, batch], key, num_restarts=RESTARTS)
    return model.predict(XH, params=params[0], batch=batch, bounds=bounds)


def _multiple_independent_heterogeneous_mfgp(key):
    opts, bounds = _options(1, constrained_criterion="LCBC")
    XL, yL, XH, yH = _two_fidelity_data()
    XL = jnp.hstack([XL, XL**2])
    batch, _ = normalize_HeterogeneousMultifidelityGP(XL, yL, XH, yH, bounds)
    model = mf.MultipleIndependentHeterogeneousMFGP(opts, layers=[2, 4, 1])
    params = model.train([batch, batch], key, num_restarts=RESTARTS)
    return model.predict(XH, params=params[0], batch=batch, bounds=bounds)


def _multiple_independent_outputs_gp(key):
    opts, bounds = _options(1, constrained_criterion="LCBC")
    X, y = _single_fidelity_data()
    batch, _ = normalize(X, y, bounds)
    model = mf.MultipleIndependentOutputsGP(opts)
    params = model.train([batch, batch], key, num_restarts=RESTARTS)
    return model.predict(X, params=params[0], batch=batch, bounds=bounds)


# Keyed by the class names in tests/test_extras.py MODEL_SHIMS, so a model
# added there without a round trip here fails test_every_model_is_covered.
BUILDERS = {
    "MultifidelityGP": _multifidelity_gp,
    "DeepMultifidelityGP": _deep_multifidelity_gp,
    "DeepMultifidelityGP_MultiOutputs": _deep_multifidelity_gp_multioutputs,
    "HeterogeneousMultifidelityGP": _heterogeneous_multifidelity_gp,
    "ManifoldGP": _manifold_gp,
    "ManifoldGP_MultiOutputs": _manifold_gp_multioutputs,
    "GradientGP": _gradient_gp,
    "MultipleIndependentMFGP": _multiple_independent_mfgp,
    "MultipleIndependentHeterogeneousMFGP": _multiple_independent_heterogeneous_mfgp,
    "MultipleIndependentOutputsGP": _multiple_independent_outputs_gp,
}


def test_every_model_is_covered():
    """No extras model ships without a round trip in this module."""
    assert set(BUILDERS) == set(MODEL_SHIMS)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_extras_model_round_trip(name):
    """Train then predict: the posterior must be usable, not just importable.

    ``std`` NaN means :func:`jaxbo.gp._std_from_variance` saw a posterior
    variance negative beyond the jitter scale, which is the #71 failure this
    module exists to catch on the extras models.
    """
    mu, std = BUILDERS[name](random.PRNGKey(0))
    assert not bool(jnp.any(jnp.isnan(std))), f"{name}: negative posterior variance"
    assert bool(jnp.all(jnp.isfinite(mu))), f"{name}: non-finite mean"
    assert bool(jnp.all(jnp.isfinite(std))), f"{name}: non-finite std"
    assert bool(jnp.all(std >= 0.0)), f"{name}: negative std"
