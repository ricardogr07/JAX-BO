"""GP numerical conditioning contracts (issue #71).

Before this work the model added a fixed ``1e-8`` to every kernel diagonal.
That constant is absolute while the matrix scale is free: every supported
kernel is stationary, so ``diag(K)`` is the signal amplitude, and on the
zero-noise ridge of a noiseless likelihood the optimizer inflates that
amplitude until the jitter's relative size collapses toward machine epsilon.
The kernel matrices then ran at condition number 1e15 to 1e17 as their normal
state, the posterior variance came out negative, and ``predict``'s
``sqrt(clip(var, 0.0))`` turned that into an exactly-zero std: perfect
reported confidence exactly where the arithmetic had broken down.

These tests pin the three properties that fix rests on:

1. :func:`jaxbo.gp.jitter` is scale invariant (the mechanism, no training
   needed, so this test is the one that localizes a future regression);
2. a trained model never reports a zero predictive std at its own training
   inputs, across seeds, on all three fixtures from the issue;
3. ``predict`` never returns NaN, which is the evidence that the loud branch
   of the negative-variance band is unreachable in practice rather than merely
   present.

Property 2 is asserted on BOTH the on-device path and the scipy L-BFGS-B path
the issue measured: the issue's table reports the scipy path failing 12 of 20
seeds on the 4D fixture, so a fix that only repaired the on-device path would
pass a one-path test while leaving the released behavior broken.

Runs in float64 for the same reason ``tests/test_gp.py`` does (see its module
docstring): in float32 the multi-start NLML landscape is a platform lottery.
The float32 path is covered by ``test_jitter_is_dtype_aware`` and by
``test_compat``.
"""

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from jax import random

from jaxbo import initializers
from jaxbo.gp import GP, jitter
from jaxbo.optimizers import minimize_lbfgs_grad
from jaxbo.priors import uniform_prior
from jaxbo.utils import normalize

# The issue measured 20 seeds. Each seed is a 10-restart train, so the full
# grid is 3 fixtures x 2 paths x 20 seeds and far too slow for CI. 6 seeds
# keeps the failing seeds the issue named in range on every fixture (4D failed
# on seeds 1, 2, 4, 5 among the first 6) while running in about a minute.
SEEDS = range(6)
NUM_RESTARTS = 10


@pytest.fixture(scope="module", autouse=True)
def _float64_mode():
    """Run this module in float64, restore float32 for the rest of the suite."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def _scipy_train(gp, batch, rng_key, num_restarts):
    """The pre-3b train loop: scipy L-BFGS-B per restart through the host.

    Kept in sync with the copy in ``tests/test_train_multistart.py``; this is
    the path issue #71 measured its 12 of 20 on, so it must be asserted here
    and not only the on-device one.
    """

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


def _on_device_train(gp, batch, rng_key, num_restarts):
    """The shipped path."""
    return gp.train(batch, rng_key, num_restarts=num_restarts)


def _fixture_1d(gap):
    """1D quadratic bowl on [-2, 3]. ``gap`` leaves (0.8, 3.0) unsampled."""
    lb, ub = jnp.array([-2.0]), jnp.array([3.0])
    bounds = {"lb": lb, "ub": ub}
    if gap:
        X = jnp.concatenate([jnp.linspace(-2.0, 0.8, 8), jnp.array([3.0])])[:, None]
    else:
        X = jnp.linspace(-2.0, 3.0, 12)[:, None]
    y = (X.flatten() - 1.5) ** 2
    batch, _ = normalize(X, y, bounds)
    gp = GP({"kernel": "RBF", "input_prior": uniform_prior(lb, ub), "criterion": "EI"})
    return gp, batch, bounds, X


def _fixture_4d():
    """4D quadratic bowl, n=32. Mirrors the ``gp_4d`` fixture in test_gp.py."""
    lb, ub = -1.0 * jnp.ones(4), 2.0 * jnp.ones(4)
    bounds = {"lb": lb, "ub": ub}
    prior = uniform_prior(lb, ub)
    X = prior.sample(random.PRNGKey(1), 32)
    y = jnp.sum((X - jnp.array([0.2, 0.5, 0.8, 1.1])) ** 2, axis=-1)
    batch, _ = normalize(X, y, bounds)
    gp = GP({"kernel": "RBF", "input_prior": prior, "criterion": "EI"})
    return gp, batch, bounds, X


FIXTURES = {
    "1d_gap": lambda: _fixture_1d(gap=True),
    "1d_dense": lambda: _fixture_1d(gap=False),
    "4d": _fixture_4d,
}


def test_jitter_is_scale_invariant():
    """The mechanism, asserted directly: no training loop, no seeds.

    Rescaling the kernel matrix must rescale the jitter by the same factor,
    so the RELATIVE regularization is constant. The old fixed 1e-8 fails this
    by six orders of magnitude on the same input, which is the whole bug.
    """
    K = jnp.eye(9) * 2.0 + 0.5
    scaled = K * 1e6

    ratio_new = float(jitter(scaled) / jitter(K))
    assert ratio_new == pytest.approx(1e6, rel=1e-6)

    # Relative jitter: what the Cholesky actually sees.
    rel = float(jitter(K) / jnp.mean(jnp.diag(K)))
    rel_scaled = float(jitter(scaled) / jnp.mean(jnp.diag(scaled)))
    assert rel_scaled == pytest.approx(rel, rel=1e-6)

    # The old constant, for contrast: its relative size collapses by 1e6.
    old_rel = 1e-8 / float(jnp.mean(jnp.diag(K)))
    old_rel_scaled = 1e-8 / float(jnp.mean(jnp.diag(scaled)))
    assert old_rel_scaled == pytest.approx(old_rel / 1e6, rel=1e-6)


def test_jitter_is_dtype_aware():
    """float32 gets a proportionally larger floor than float64.

    sqrt(eps) of the matrix dtype, so about 3.5e-4 against about 1.5e-8. A
    dtype-blind constant would regularize float32 far too weakly.
    """
    K64 = jnp.eye(4, dtype=jnp.float64)
    K32 = jnp.eye(4, dtype=jnp.float32)
    j64, j32 = float(jitter(K64)), float(jitter(K32))
    assert j64 == pytest.approx(float(jnp.sqrt(jnp.finfo(jnp.float64).eps)), rel=1e-6)
    assert j32 == pytest.approx(float(jnp.sqrt(jnp.finfo(jnp.float32).eps)), rel=1e-4)
    assert j32 > j64 * 1e3


def test_jitter_reproduces_the_old_constant_at_unit_amplitude():
    """A well-conditioned float64 problem sees essentially no change.

    The fix must repair the pathological corner without silently altering the
    regularization every healthy problem was already getting. At unit
    amplitude sqrt(eps) is 1.49e-8 against the old 1e-8: the same order.
    """
    K = jnp.eye(16)
    assert 1e-8 <= float(jitter(K)) < 2e-8


@pytest.mark.parametrize("fixture", list(FIXTURES))
@pytest.mark.parametrize("path", ["scipy", "on_device"])
def test_trained_model_has_no_zero_or_nan_std_at_training_inputs(fixture, path):
    """AC3 and AC2: no exactly-zero std, no NaN, on either training path.

    An exactly-zero predictive std at a TRAINING input is the signature the
    issue measured: it means the posterior variance came out negative and was
    clipped. Before the fix this fired on 12 of 20 seeds on the 4D fixture on
    the scipy path (which is the RELEASED behavior) and on 20 of 20 on the
    on-device path. After it, 0 of 20 on both, on all three fixtures.

    NaN is the loud branch of the negative-variance band in
    :func:`jaxbo.gp._std_from_variance`. It must stay unfired here: the band
    exists so a future regression is visible, not because the current code is
    expected to reach it.
    """
    gp, batch, bounds, X = FIXTURES[fixture]()
    train = _scipy_train if path == "scipy" else _on_device_train

    zero_seeds, nan_seeds = [], []
    for seed in SEEDS:
        params = train(gp, batch, random.PRNGKey(seed), NUM_RESTARTS)
        _, std = gp.predict(X, params=params, batch=batch, bounds=bounds)
        std = onp.asarray(std)
        if onp.isnan(std).any():
            nan_seeds.append(seed)
        elif (std == 0.0).any():
            zero_seeds.append(seed)

    assert not nan_seeds, f"predict returned NaN std on seeds {nan_seeds}"
    assert not zero_seeds, f"zero predictive std at training inputs: {zero_seeds}"


@pytest.mark.parametrize("fixture", list(FIXTURES))
def test_conditioning_is_bounded_and_seed_independent(fixture):
    """The regularized kernel matrix is no longer near singular.

    Direct evidence for the issue's headline number. Before the fix the
    condition number ran at 1e15 to 1e17 and varied by two orders of magnitude
    across seeds, because it depended on which optimum the multi-start
    happened to reach. A jitter scaled to the matrix pins it: measured at
    about 1e9 on all three fixtures with no spread across seeds.

    The ceiling of 1e12 is loose on purpose, well below the 1e15 the issue
    reports and well above the 1e9 observed, so this catches a return to
    near-singular without being a platform tripwire.
    """
    gp, batch, bounds, X = FIXTURES[fixture]()
    conds = []
    for seed in SEEDS:
        params = gp.train(batch, random.PRNGKey(seed), num_restarts=NUM_RESTARTS)
        # cond(K) from its Cholesky factor: cond(K) = cond(L)^2.
        L = onp.asarray(gp.compute_cholesky(params, batch))
        conds.append(onp.linalg.cond(L) ** 2)
    assert max(conds) < 1e12, f"kernel matrix near singular: max cond {max(conds):.2e}"


def test_negative_variance_beyond_rounding_becomes_nan():
    """AC2's loud branch, exercised directly since training never reaches it.

    A variance negative within rounding scale clips to zero (a genuine float
    artifact of subtracting two nearly equal quantities). Anything more
    negative is not representable as a variance under any rounding story and
    must surface rather than masquerade as certainty.
    """
    from jaxbo.gp import _std_from_variance

    k_pp = jnp.eye(4)
    tol = float(jitter(k_pp))

    variance = jnp.array([1.0, 0.0, -tol * 0.5, -tol * 100.0])
    std = onp.asarray(_std_from_variance(variance, k_pp))

    assert std[0] == pytest.approx(1.0)
    assert std[1] == 0.0
    assert std[2] == 0.0, "a rounding-scale negative must clip quietly"
    assert onp.isnan(std[3]), "a large negative variance must surface as NaN"
