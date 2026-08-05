"""GP.train on-device multi-start path tests (3b, issue #31).

``GP.train`` runs its restarts as a single jitted, vmapped BFGS computation
on device. These tests pin the contracts that move must preserve: seeded
determinism of the public API, the (dim + 2,) result shape and default
dtype, NaN-restart robustness (a failed restart must not poison the
selection and must not perturb the healthy restarts), the guard against
every restart failing, and the decided equivalence criterion from issue
#31: the new path is not systematically worse in final NLML than the
scipy L-BFGS-B path (still available through
:func:`jaxbo.optimizers.minimize_lbfgs_grad`), compared across seeds at
basin scale rather than as bit-identical trajectories or per-seed value
parity.
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
    """The on-device path is not systematically worse in NLML than scipy.

    Decided equivalence criterion (issue #31): compare the final likelihood
    of the returned optimum, not trajectories. Runs in float64 so both
    optimizers converge tightly enough for the comparison to be meaningful;
    restored to float32 afterwards.

    Checked across seeds, not at PRNGKey(0) alone, and at basin scale, not
    at value-parity scale. The original single-seed 1e-2 assert was never
    sound and failed on the 3.13-latest and 3.14-floor lanes (by 0.63 and
    0.10 nats) on code that passes it locally. Measuring seeds 0 to 19 at
    10 restarts shows why: the selected-NLML difference spans -2.42 to
    +2.71 nats in BOTH directions and 9 of 20 seeds exceed 1e-2, so
    per-seed dominance is not a property either path has.

    That spread is basin selection rather than optimizer noise. This
    surface has a dominant attractor near -49.5 that most restarts fall
    into and a better endpoint near -52 that only some restarts find, so
    the per-seed difference mostly records which plateau each path's best
    restart reached, about 2.7 nats apart. A per-seed tolerance therefore
    has to exceed the plateau gap to be platform-stable, which leaves it
    unable to discriminate anyway. Two asserts split the job instead:

    - the MEDIAN difference over seeds 0 to 7 carries the equal-or-better
      criterion (measured -0.30, bounded at +1.0), so a systematic
      degradation fails loudly while one seed landing on the other plateau
      does not;
    - the per-seed bound stays at basin scale (+4.0, above the 2.7 nat
      plateau gap) to catch the on-device path diverging where scipy did
      not: real blowups on this fixture land near +1030, not near +3.
    """
    jax.config.update("jax_enable_x64", True)
    try:
        gp, batch = _make_problem()
        diffs = []
        for seed in range(8):
            key = random.PRNGKey(seed)
            new_params = gp.train(batch, key, num_restarts=10)
            old_params = _scipy_train(gp, batch, key, num_restarts=10)
            assert new_params.shape == old_params.shape
            assert bool(jnp.all(jnp.isfinite(new_params)))
            nlml_new = float(gp.likelihood(new_params, batch))
            nlml_old = float(gp.likelihood(old_params, batch))
            diffs.append(nlml_new - nlml_old)
        assert max(diffs) <= 4.0, f"a seed diverged from the basin: {diffs}"
        assert (
            float(onp.median(onp.array(diffs))) <= 1.0
        ), f"on-device path systematically worse than scipy: {diffs}"
    finally:
        jax.config.update("jax_enable_x64", False)


def test_selected_nlml_matches_scipy_on_gap_fixture():
    """The issue #31 criterion, encoded on the exact canary problem.

    Mirrors the tests/test_gp.py ``gp_1d`` fixture (data-gap problem,
    PRNGKey(0), 10 restarts, float64): the multi-start's SELECTED optimum
    must land in the same basin as the scipy L-BFGS-B path's on the same
    seeds. The CI failure mode this pins down is every restart collapsing
    into the degenerate tiny-lengthscale optimum on some platform (the
    tests/test_gp.py module docstring records it at NLML +12.8 against
    -28; on this fixture such restarts land near +3.0), which shows up
    here as a selected NLML tens of nats above scipy's.

    The tolerance is deliberately basin-scale, not value-parity scale.
    Both paths terminate somewhere on the zero-noise ridge of this
    noiseless likelihood, where the surface is nearly flat and the last
    nat of NLML only records how far each optimizer rode the ridge before
    its own stopping rule fired. Measured over seeds 0 to 19 at 10
    restarts, the selected-value difference spans -0.51 to +1.02 nats in
    both directions (12 of 20 seeds differ by more than 0.05), and it does
    not shrink usefully with more restarts (+0.71 max at 20, +0.27 at 30).
    Deeper is also not better here: the seed whose NLML beat scipy by the
    largest margin (-28.73 against -28.22) is the one seed of the 20 whose
    trained model FAILS the gap-uncertainty contract that
    ``test_gp_1d_uncertainty_grows_in_data_gap`` asserts, because riding
    the ridge to zero noise is exactly the improper optimum
    :func:`jaxbo.optimizers.minimize_bfgs_jax` refuses to chase. A
    sub-nat parity assert would therefore be both platform-flaky and
    pointed the wrong way; 1.2 nats still catches the 31 nat collapse
    this test exists for.
    """
    jax.config.update("jax_enable_x64", True)
    try:
        lb, ub = jnp.array([-2.0]), jnp.array([3.0])
        X = jnp.concatenate([jnp.linspace(-2.0, 0.8, 8), jnp.array([3.0])])[:, None]
        y = (X.flatten() - 1.5) ** 2
        batch, _ = normalize(X, y, {"lb": lb, "ub": ub})
        prior = uniform_prior(lb, ub)
        gp = GP({"kernel": "RBF", "input_prior": prior, "criterion": "EI"})
        key = random.PRNGKey(0)
        new_params = gp.train(batch, key, num_restarts=10)
        old_params = _scipy_train(gp, batch, key, num_restarts=10)
        nlml_new = float(jnp.sum(gp.likelihood(new_params, batch)))
        nlml_old = float(jnp.sum(gp.likelihood(old_params, batch)))
        assert bool(jnp.all(jnp.isfinite(new_params)))
        assert nlml_new <= nlml_old + 1.2
    finally:
        jax.config.update("jax_enable_x64", False)


def test_gap_posterior_stays_usable_across_seeds():
    """Bound a KNOWN, DISCLOSED regression of the on-device path.

    The property is the one ``test_gp_1d_uncertainty_grows_in_data_gap``
    asserts, checked across seeds instead of at PRNGKey(0) alone:
    predictive std in the unsampled gap must exceed the std at the
    training inputs. Measured over seeds 0 to 19 at 10 restarts, this path
    breaks it on 1 seed (12), where the selected optimum has zero
    predictive std at 5 of 9 training points; the scipy L-BFGS-B path it
    replaces breaks it on 0 of 20. The regression is disclosed on the PR
    for issue #31 rather than fixed, because no cheap selection-time
    screen separates the pathological endpoint from a healthy one: the
    kernel matrices here run at condition number 1e15 to 1e17 as their
    NORMAL state on both paths, so neither a zero-variance check (the
    scipy path trips it on 12 of 20 seeds of the 4D fixture) nor a
    conditioning threshold (usable endpoints reach log10 cond 17.3,
    broken ones start at 14.4) discriminates. The underlying fragility is
    the model's fixed 1e-8 jitter at that conditioning, which predates
    this work and is tracked separately.

    So this test does not assert the regression away. It pins the RATE, so
    1 in 20 cannot silently become 10 in 20. Seeds 0 to 7 all pass today;
    the bound of 1 leaves room for a platform that lands differently on
    one seed while still failing loudly on a systemic worsening.
    """
    jax.config.update("jax_enable_x64", True)
    try:
        lb, ub = jnp.array([-2.0]), jnp.array([3.0])
        bounds = {"lb": lb, "ub": ub}
        X = jnp.concatenate([jnp.linspace(-2.0, 0.8, 8), jnp.array([3.0])])[:, None]
        y = (X.flatten() - 1.5) ** 2
        batch, _ = normalize(X, y, bounds)
        prior = uniform_prior(lb, ub)
        gp = GP({"kernel": "RBF", "input_prior": prior, "criterion": "EI"})
        # Midpoint of the unsampled (0.8, 3.0) gap.
        probe = jnp.array([[1.9]])

        broken = []
        for seed in range(8):
            params = gp.train(batch, random.PRNGKey(seed), num_restarts=10)
            _, std_train = gp.predict(X, params=params, batch=batch, bounds=bounds)
            _, std_gap = gp.predict(probe, params=params, batch=batch, bounds=bounds)
            if not float(std_gap[0]) > float(jnp.max(std_train)):
                broken.append(seed)
        assert len(broken) <= 1, f"gap posterior unusable on seeds {broken} of 0..7"
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
