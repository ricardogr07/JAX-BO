"""GP.train on-device multi-start path tests (3b, issue #31).

``GP.train`` runs its restarts as a single jitted, vmapped BFGS computation
on device. These tests pin the contracts that move must preserve: seeded
determinism of the public API, the (dim + 2,) result shape and default
dtype, NaN-restart robustness (a failed restart must not poison the
selection and must not perturb the healthy restarts), the guard against
every restart failing, and the decided equivalence criterion from issue
#31: the new path reaches an equal-or-better final NLML than the scipy
L-BFGS-B path (still available through
:func:`jaxbo.optimizers.minimize_lbfgs_grad`) on a seeded problem, rather
than bit-identical trajectories.
"""

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from jax import random

from jaxbo import initializers
from jaxbo.gp import GP
from jaxbo.optimizers import minimize_lbfgs_grad
from jaxbo.priors import uniform_prior
from jaxbo.utils import normalize


def _make_problem(n=12):
    """Seeded 1D quadratic bowl on [-2, 3] with a normalized batch."""
    lb, ub = jnp.array([-2.0]), jnp.array([3.0])
    bounds = {"lb": lb, "ub": ub}
    X = jnp.linspace(-2.0, 3.0, n)[:, None]
    y = (X.flatten() - 1.5) ** 2
    batch, _ = normalize(X, y, bounds)
    prior = uniform_prior(lb, ub)
    gp = GP({"kernel": "RBF", "input_prior": prior, "criterion": "EI"})
    return gp, batch


def _scipy_train(gp, batch, rng_key, num_restarts):
    """The pre-3b train loop: scipy L-BFGS-B per restart through the host."""

    def objective(params):
        value, grads = gp.likelihood_value_and_grad(params, batch)
        return onp.array(value), onp.array(grads)

    dim = batch["X"].shape[1]
    keys = random.split(rng_key, num_restarts)
    results = [
        minimize_lbfgs_grad(objective, initializers.random_init_GP(k, dim))
        for k in keys
    ]
    values = onp.array([value for _, value in results])
    return jnp.asarray(results[int(onp.nanargmin(values))][0])


def test_train_nlml_matches_scipy_path_within_tolerance():
    """The on-device path reaches an equal-or-better NLML than scipy.

    Decided equivalence criterion (issue #31): compare the final likelihood
    of the returned optimum on a seeded problem, not trajectories. Runs in
    float64 so both optimizers converge tightly enough for the comparison
    to be meaningful; restored to float32 afterwards.
    """
    jax.config.update("jax_enable_x64", True)
    try:
        gp, batch = _make_problem()
        key = random.PRNGKey(0)
        new_params = gp.train(batch, key, num_restarts=10)
        old_params = _scipy_train(gp, batch, key, num_restarts=10)
        assert new_params.shape == old_params.shape
        assert bool(jnp.all(jnp.isfinite(new_params)))
        nlml_new = float(gp.likelihood(new_params, batch))
        nlml_old = float(gp.likelihood(old_params, batch))
        assert nlml_new <= nlml_old + 1e-2
    finally:
        jax.config.update("jax_enable_x64", False)


def test_train_is_deterministic_per_seed():
    """Same rng_key, same batch: bitwise identical hyperparameters."""
    gp, batch = _make_problem()
    first = gp.train(batch, random.PRNGKey(3), num_restarts=5)
    second = gp.train(batch, random.PRNGKey(3), num_restarts=5)
    assert first.shape == (3,)
    assert first.dtype == jnp.float32
    assert bool(jnp.array_equal(first, second))
    assert bool(jnp.all(jnp.isfinite(first)))


def test_failed_restart_does_not_poison_selection():
    """A restart with a non-finite NLML is skipped and perturbs nothing.

    Overflowing the log signal variance makes that restart's likelihood
    non-finite from the first evaluation, the on-device analogue of the
    Cholesky-failure NaNs the scipy path absorbed via nanargmin. The other
    restarts must return bitwise the same result as without the bad lane.
    """
    gp, batch = _make_problem()
    keys = random.split(random.PRNGKey(0), 4)
    good = jnp.stack([initializers.random_init_GP(k, 1) for k in keys])
    # exp(1000.0) overflows in float32 and float64 alike.
    bad = good.at[1, 0].set(1000.0)

    params_good, values_good = gp._train_multistart(good, batch)
    params_bad, values_bad = gp._train_multistart(bad, batch)

    assert not bool(jnp.isfinite(values_bad[1]))
    keep = onp.array([True, False, True, True])
    assert onp.array_equal(
        onp.asarray(values_bad)[keep], onp.asarray(values_good)[keep], equal_nan=True
    )
    idx = int(jnp.nanargmin(values_bad))
    assert idx != 1
    assert bool(jnp.all(jnp.isfinite(params_bad[idx])))


def test_train_raises_when_every_restart_fails():
    """All-restarts-failed surfaces as an error, not silent garbage."""
    gp, batch = _make_problem()
    poisoned = {"X": batch["X"], "y": jnp.full_like(batch["y"], jnp.nan)}
    with pytest.raises(RuntimeError, match="non-finite likelihood"):
        gp.train(poisoned, random.PRNGKey(0), num_restarts=3)
