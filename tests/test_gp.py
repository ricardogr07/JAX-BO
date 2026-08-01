"""GP train/predict round-trip and acquisition behavior tests (slice 2d).

Covers the model surface that previously had zero tests: exact ``GP``
training convergence, predictive round-trips on 1D and 4D synthetic
functions, the normalization contract (normalized batch in, raw ``X_star``
to ``predict``; SCOPE.md section 2), and EI acquisition sanity at both the
function level and through the model dispatch.

All randomness is seeded; the trained models are module-scoped fixtures so
the L-BFGS restarts run once per dimensionality and every test reuses them.

This module runs in float64 (see ``_float64_mode``): in float32 the NLML
multi-start is a platform lottery. On several CI lanes all 10 restarts of
the 1D fixture landed in the degenerate tiny-lengthscale optimum (NLML
12.77 instead of -28) that interpolates the data and predicts the prior
mean everywhere else, while the same seeds escaped it locally. Under
float64 every calibration seed converges to the same broad optimum, so
value asserts are meaningful cross-platform. The consumer-default float32
path stays covered by ``test_compat``.
"""

import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from jax import random, vmap
from jax.scipy.stats import norm
from scipy.stats import qmc

from jaxbo import acquisitions, initializers
from jaxbo.gp import GP
from jaxbo.optimizers import minimize_lbfgs_grad
from jaxbo.priors import uniform_prior
from jaxbo.utils import normalize

# 1D problem: quadratic bowl on the RAW domain [-2, 3], true minimum at 1.5.
# Training points cover [-2, 0.8] densely plus the right edge, leaving a
# deliberate gap (0.8, 3.0) that contains the true minimum: the GP must be
# uncertain there and EI must want to explore it.
LB_1D, UB_1D = -2.0, 3.0
TRUE_MIN_1D = 1.5
GAP_1D = (0.8, 3.0)


def f_1d(x):
    """Quadratic test objective with its global minimum at ``TRUE_MIN_1D``."""
    return (x - TRUE_MIN_1D) ** 2


# 4D problem: quadratic bowl on the RAW domain [-1, 2]^4. Kept as plain
# python constants so all arrays are created inside fixtures, under the
# float64 mode below, never at import time.
LB_4D, UB_4D = -1.0, 2.0
CENTER_4D = (0.2, 0.5, 0.8, 1.1)


def f_4d(x):
    """Separable 4D quadratic bowl with its minimum at ``CENTER_4D``."""
    return jnp.sum((x - jnp.array(CENTER_4D)) ** 2, axis=-1)


def make_gp(lb, ub, criterion="EI"):
    """Build a core ``GP`` with the shelter-pulse style options dict."""
    prior = uniform_prior(lb, ub)
    return GP({"kernel": "RBF", "input_prior": prior, "criterion": criterion})


@pytest.fixture(scope="module", autouse=True)
def _float64_mode():
    """Run this module in float64, restore float32 for the rest of the suite.

    See the module docstring: float32 makes the multi-start NLML landscape
    a platform lottery, float64 makes the value asserts reproducible.
    """
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


@pytest.fixture(scope="module")
def gp_1d(_float64_mode):
    """Train a 1D GP once and share model, data, and params across tests."""
    lb, ub = jnp.array([LB_1D]), jnp.array([UB_1D])
    bounds = {"lb": lb, "ub": ub}
    X = jnp.concatenate([jnp.linspace(LB_1D, GAP_1D[0], 8), jnp.array([UB_1D])])
    X = X[:, None]
    y = f_1d(X.flatten())
    batch, norm_const = normalize(X, y, bounds)
    gp = make_gp(lb, ub)
    # 10 restarts plus float64 on purpose: with fewer restarts, or in
    # float32, the multi-start can land in the degenerate tiny-lengthscale
    # NLML optimum that interpolates the data but predicts the prior mean
    # everywhere else (module docstring).
    opt_params = gp.train(batch, random.PRNGKey(0), num_restarts=10)
    return {
        "gp": gp,
        "X": X,
        "y": y,
        "batch": batch,
        "norm_const": norm_const,
        "bounds": bounds,
        "opt_params": opt_params,
        "rng_key": random.PRNGKey(0),
        "num_restarts": 10,
    }


@pytest.fixture(scope="module")
def gp_4d(_float64_mode):
    """Train a 4D GP once on 32 seeded samples and share it across tests."""
    lb, ub = LB_4D * jnp.ones(4), UB_4D * jnp.ones(4)
    bounds = {"lb": lb, "ub": ub}
    prior = uniform_prior(lb, ub)
    X = prior.sample(random.PRNGKey(1), 32)
    y = f_4d(X)
    batch, norm_const = normalize(X, y, bounds)
    gp = make_gp(lb, ub)
    opt_params = gp.train(batch, random.PRNGKey(2), num_restarts=10)
    X_test = prior.sample(random.PRNGKey(3), 64)
    return {
        "gp": gp,
        "X": X,
        "y": y,
        "batch": batch,
        "norm_const": norm_const,
        "bounds": bounds,
        "opt_params": opt_params,
        "X_test": X_test,
        "y_test": f_4d(X_test),
        "rng_key": random.PRNGKey(2),
        "num_restarts": 10,
    }


def denorm(mu, norm_const):
    """Map a normalized predictive mean back to the raw target scale."""
    return mu * norm_const["sigma_y"] + norm_const["mu_y"]


def assert_train_converged(fitted, dim):
    """Assert training improved on every restart's own starting NLML.

    Rebuilds the exact initializations ``GP.train`` used (same key split,
    same initializer) and requires the returned optimum to beat the best of
    them, so a train that silently returns an init would fail here.
    """
    gp, batch, opt = fitted["gp"], fitted["batch"], fitted["opt_params"]
    assert jnp.all(jnp.isfinite(opt))
    nlml_opt = gp.likelihood(opt, batch)
    init_keys = random.split(fitted["rng_key"], fitted["num_restarts"])
    nlml_inits = jnp.array(
        [gp.likelihood(initializers.random_init_GP(k, dim), batch) for k in init_keys]
    )
    assert nlml_opt < jnp.min(nlml_inits)


def test_gp_1d_train_beats_every_restart_init(gp_1d):
    """1D training must land strictly below the best restart init's NLML."""
    # dim + 2 hyperparameters: signal variance, one lengthscale, noise.
    assert gp_1d["opt_params"].shape[0] == 3
    assert_train_converged(gp_1d, dim=1)


def test_gp_1d_training_is_seed_robust(gp_1d):
    """A different training key must reach the same fit quality.

    Guards against the fixture seed being a lucky draw: the NLML landscape
    is nonconvex, so convergence and held-out accuracy must hold for a
    fresh restart key too (observed held-out max error 7e-5 to 2e-4 in
    float64 across keys 0, 1, 2, 5 during calibration).
    """
    gp, batch = gp_1d["gp"], gp_1d["batch"]
    opt_other = gp.train(batch, random.PRNGKey(5), num_restarts=10)
    X_dense = gp_1d["X"][:8]
    X_mid = (X_dense[:-1] + X_dense[1:]) / 2.0
    mu, _ = gp.predict(X_mid, params=opt_other, batch=batch, bounds=gp_1d["bounds"])
    y_pred = denorm(mu, gp_1d["norm_const"])
    assert float(jnp.max(jnp.abs(y_pred - f_1d(X_mid.flatten())))) < 0.02


def test_gp_1d_predict_reproduces_training_targets(gp_1d):
    """On noiseless data the posterior mean must interpolate the training set.

    ``predict`` receives the RAW training locations and normalizes them
    internally against the non-unit bounds; getting the raw targets back is
    the round-trip proof that the normalization contract holds end to end.
    """
    mu, std = gp_1d["gp"].predict(
        gp_1d["X"],
        params=gp_1d["opt_params"],
        batch=gp_1d["batch"],
        bounds=gp_1d["bounds"],
    )
    y_pred = denorm(mu, gp_1d["norm_const"])
    # Observed max error ~2e-4 (float64) on the 11.76 target range; 0.02
    # keeps a ~100x margin while rejecting the degenerate fit (error ~6).
    assert float(jnp.max(jnp.abs(y_pred - gp_1d["y"]))) < 0.02
    # Nearly noiseless interpolation: the model is confident at its own data
    # (observed max std ~1.4e-4; the degenerate fit sits at ~0.06).
    assert jnp.all(std >= 0.0)
    assert float(jnp.max(std)) < 0.01


def test_gp_1d_predict_interpolates_held_out(gp_1d):
    """Held-out midpoints in the densely sampled region match the truth."""
    X_dense = gp_1d["X"][:8]
    X_mid = (X_dense[:-1] + X_dense[1:]) / 2.0
    mu, _ = gp_1d["gp"].predict(
        X_mid,
        params=gp_1d["opt_params"],
        batch=gp_1d["batch"],
        bounds=gp_1d["bounds"],
    )
    y_pred = denorm(mu, gp_1d["norm_const"])
    y_true = f_1d(X_mid.flatten())
    # Observed max error ~2e-4 across seeds (float64); see the round-trip
    # test note for the margin rationale.
    assert float(jnp.max(jnp.abs(y_pred - y_true))) < 0.02


def test_gp_1d_uncertainty_grows_in_data_gap(gp_1d):
    """Predictive std inside the unsampled gap exceeds std at the data."""
    X_all = jnp.vstack([gp_1d["X"], jnp.array([[sum(GAP_1D) / 2.0]])])
    _, std = gp_1d["gp"].predict(
        X_all,
        params=gp_1d["opt_params"],
        batch=gp_1d["batch"],
        bounds=gp_1d["bounds"],
    )
    std_train, std_gap = std[:-1], std[-1]
    # The broad float64 optimum is confident everywhere (std ~1e-4), so the
    # contrast is modest but consistent: observed ratio 1.1 to 1.2 across
    # seeds. Require both strict ordering and a floor on the ratio.
    assert std_gap > jnp.max(std_train)
    assert float(std_gap / jnp.max(std_train)) > 1.05


def test_gp_1d_normalization_contract_is_asymmetric(gp_1d):
    """normalize() output feeds train; predict wants RAW points, not unit cube.

    The batch really is on the unit cube with targets standardized, the
    norm constants round-trip the raw targets, and feeding predict
    pre-normalized inputs (the documented silent failure) is measurably
    wrong while raw inputs are near exact.
    """
    batch, norm_const, bounds = gp_1d["batch"], gp_1d["norm_const"], gp_1d["bounds"]
    assert float(jnp.min(batch["X"])) >= 0.0
    assert float(jnp.max(batch["X"])) <= 1.0
    assert jnp.allclose(denorm(batch["y"], norm_const), gp_1d["y"], atol=1e-5)

    kwargs = {"params": gp_1d["opt_params"], "batch": batch, "bounds": bounds}
    mu_raw, _ = gp_1d["gp"].predict(gp_1d["X"], **kwargs)
    X_wrong = (gp_1d["X"] - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
    mu_wrong, _ = gp_1d["gp"].predict(X_wrong, **kwargs)
    err_raw = jnp.max(jnp.abs(denorm(mu_raw, norm_const) - gp_1d["y"]))
    err_wrong = jnp.max(jnp.abs(denorm(mu_wrong, norm_const) - gp_1d["y"]))
    # The wrong input space must be at least an order of magnitude worse.
    assert float(err_wrong / err_raw) > 10.0


def test_ei_matches_closed_form():
    """EI agrees with the textbook closed form and a hand-computed constant.

    jaxbo returns the NEGATIVE expected improvement (minimization), so the
    improvement itself is the negated return value. The implementation is the
    textbook closed form, delta * Phi(z) + std * phi(z) with delta = best -
    mean and z = delta / std, on both sides of mean = best (issue #55).
    """
    # mean == best: improvement reduces to std * phi(0) = 0.5 / sqrt(2*pi).
    val = acquisitions.EI(jnp.array([1.0]), jnp.array([0.5]), 1.0)
    assert float(-val) == pytest.approx(0.19947114020071635, abs=1e-6)

    for mean, std, best in [
        (0.5, 1.0, 1.0),
        (-0.3, 0.7, 0.2),
        (1.0, 0.4, 2.0),
        (1.5, 0.5, 1.0),  # mean > best: same closed form, no special branch
    ]:
        delta = best - mean
        z = delta / std
        expected = delta * norm.cdf(z) + std * norm.pdf(z)
        got = -acquisitions.EI(jnp.array([mean]), jnp.array([std]), best)
        assert jnp.allclose(got, expected, atol=1e-6)


def test_ei_worse_than_best_matches_closed_form():
    """For mean > best, EI is the textbook closed form, not an upper bound.

    With mean=1, std=1, best=0 (minimization) the textbook expected
    improvement is delta * Phi(z) + std * phi(z) with delta = z = -1:
    -1 * 0.15865525393145707 + 0.24197072451914337 = 0.0833154705876863.
    The pre-fix Frazier tutorial form dropped the negative delta * Phi(z)
    term and returned the std * phi(z) = 0.24197 upper bound; fixed in
    issue #55, locked here so any regression is deliberate.
    """
    got = -acquisitions.EI(jnp.array([1.0]), jnp.array([1.0]), 0.0)
    assert float(got) == pytest.approx(0.0833154705876863, abs=1e-6)


def test_ei_nonnegative_and_monotone_in_mean():
    """Improvement is nonnegative and strictly falls as the mean worsens."""
    means = jnp.linspace(-2.0, 2.0, 21)
    improvements = jnp.array(
        [-acquisitions.EI(jnp.array([m]), jnp.array([0.5]), 0.0) for m in means]
    )
    assert float(jnp.min(improvements)) > -1e-6
    assert float(jnp.max(jnp.diff(improvements))) < 0.0


def test_ei_limits_as_uncertainty_vanishes():
    """With std ~ 0, EI collapses to max(best - mean, 0)."""
    worse = -acquisitions.EI(jnp.array([1.0]), jnp.array([1e-8]), 0.0)
    better = -acquisitions.EI(jnp.array([-1.0]), jnp.array([1e-8]), 0.0)
    assert float(jnp.abs(worse)) < 1e-6
    assert float(better) == pytest.approx(1.0, abs=1e-6)

    # Exactly std = 0 hits the explicit exact-knowledge branch: no division
    # by zero, value is exactly max(best - mean, 0) on both sides and at best.
    assert float(-acquisitions.EI(jnp.array([1.0]), jnp.array([0.0]), 0.0)) == 0.0
    assert float(-acquisitions.EI(jnp.array([-1.0]), jnp.array([0.0]), 0.0)) == 1.0
    assert float(-acquisitions.EI(jnp.array([0.0]), jnp.array([0.0]), 0.0)) == 0.0


def test_eic_deterministic_constraint_is_not_nan():
    """A zero-variance constraint gives feasibility exactly 1 or 0, not NaN.

    The objective row (mean 0.5, std 1.0, best 0.0) is shared, so a surely
    feasible deterministic constraint (mean 1, std 0) must reproduce the
    unconstrained EI, and a surely infeasible one (mean -1, std 0) must
    zero it out; 0/0 NaN from the constraint row would poison both.
    """
    ei = acquisitions.EI(jnp.array([0.5]), jnp.array([1.0]), 0.0)
    feasible = acquisitions.EIC(
        jnp.array([[0.5], [1.0]]), jnp.array([[1.0], [0.0]]), 0.0
    )
    infeasible = acquisitions.EIC(
        jnp.array([[0.5], [-1.0]]), jnp.array([[1.0], [0.0]]), 0.0
    )
    assert float(feasible) == pytest.approx(float(ei), abs=1e-12)
    assert float(infeasible) == 0.0


def test_gp_1d_ei_acquisition_explores_promising_gap(gp_1d):
    """Model-level EI is a valid improvement and steers into the data gap.

    The unsampled gap contains the true minimum, so grid search over the
    acquisition must propose a point inside it, and EI near the true
    minimum must beat EI in the well-sampled far-from-optimum region.
    """
    kwargs = {
        "params": gp_1d["opt_params"],
        "batch": gp_1d["batch"],
        "bounds": gp_1d["bounds"],
    }
    gp = gp_1d["gp"]
    X_cand = jnp.linspace(LB_1D, UB_1D, 101)[:, None]
    acq_vals = vmap(lambda x: gp.acquisition(x, **kwargs))(X_cand)
    # Negated EI: every value must be a nonpositive number (improvement >= 0).
    assert float(jnp.max(acq_vals)) <= 1e-8
    x_next = gp.compute_next_point_gs(X_cand, **kwargs)
    assert x_next.shape == (1, 1)
    assert GAP_1D[0] < float(x_next[0, 0]) < GAP_1D[1]
    acq_at_min = gp.acquisition(jnp.array([TRUE_MIN_1D]), **kwargs)
    acq_far = gp.acquisition(jnp.array([LB_1D]), **kwargs)
    assert acq_at_min < acq_far


def test_gp_4d_train_beats_every_restart_init(gp_4d):
    """4D training must land strictly below the best restart init's NLML."""
    # dim + 2 hyperparameters: signal variance, four lengthscales, noise.
    assert gp_4d["opt_params"].shape[0] == 6
    assert_train_converged(gp_4d, dim=4)


def test_gp_4d_predict_reproduces_training_targets(gp_4d):
    """Posterior mean interpolates the 32 noiseless 4D training targets."""
    mu, std = gp_4d["gp"].predict(
        gp_4d["X"],
        params=gp_4d["opt_params"],
        batch=gp_4d["batch"],
        bounds=gp_4d["bounds"],
    )
    y_pred = denorm(mu, gp_4d["norm_const"])
    # Observed max error ~2e-5 (float64) on the 8.42 target range; 0.05
    # keeps a huge margin while rejecting a prior-mean-only fit.
    assert float(jnp.max(jnp.abs(y_pred - gp_4d["y"]))) < 0.05
    assert jnp.all(std >= 0.0)


def test_gp_4d_training_is_seed_robust(gp_4d):
    """A different 4D training key must reach the same held-out quality.

    Same rationale as the 1D twin: observed relative RMSE 1e-4 to 4e-4
    (float64) across training keys 2, 4, 8 during calibration, all far
    under the 0.05 bound.
    """
    gp, batch = gp_4d["gp"], gp_4d["batch"]
    opt_other = gp.train(batch, random.PRNGKey(8), num_restarts=10)
    mu, _ = gp.predict(
        gp_4d["X_test"], params=opt_other, batch=batch, bounds=gp_4d["bounds"]
    )
    y_pred = denorm(mu, gp_4d["norm_const"])
    rel_rmse = jnp.sqrt(jnp.mean((y_pred - gp_4d["y_test"]) ** 2))
    rel_rmse = rel_rmse / gp_4d["y_test"].std()
    assert float(rel_rmse) < 0.05


def test_gp_4d_predict_generalizes_to_held_out(gp_4d):
    """Held-out predictions carry real signal: shapes right, errors small.

    The relative RMSE bound (against the spread of the true targets) is the
    value assert: predicting the mean everywhere would score ~1.0.
    """
    mu, std = gp_4d["gp"].predict(
        gp_4d["X_test"],
        params=gp_4d["opt_params"],
        batch=gp_4d["batch"],
        bounds=gp_4d["bounds"],
    )
    assert mu.shape == (64,)
    assert std.shape == (64,)
    assert jnp.all(jnp.isfinite(mu)) and jnp.all(jnp.isfinite(std))
    assert jnp.all(std >= 0.0)
    y_pred = denorm(mu, gp_4d["norm_const"])
    rel_rmse = jnp.sqrt(jnp.mean((y_pred - gp_4d["y_test"]) ** 2))
    rel_rmse = rel_rmse / gp_4d["y_test"].std()
    # Observed relative RMSE ~1e-4 across seeds (float64); 0.05 keeps a
    # huge margin while a mean-only predictor would score ~1.0.
    assert float(rel_rmse) < 0.05


def _serial_next_point(gp, num_restarts, **kwargs):
    """The pre-batched-start serial multi-start loop, kept as the parity
    reference: LHS-sample num_restarts starts with the same seeding as
    ``compute_next_point_lbfgs``, polish every one with bounded L-BFGS-B,
    return the best polished point and its acquisition value.
    """

    def objective(x):
        value, grads = gp.acq_value_and_grad(x, **kwargs)
        return onp.array(value), onp.array(grads)

    lb, ub = kwargs["bounds"]["lb"], kwargs["bounds"]["ub"]
    rng_key = kwargs["rng_key"]
    onp.random.seed(rng_key[0])
    sampler = qmc.LatinHypercube(d=lb.shape[0], seed=int(rng_key[0]))
    inits = lb + (ub - lb) * sampler.random(num_restarts)
    dom_bounds = tuple(map(tuple, jnp.vstack((lb, ub)).T))
    solutions, values = [], []
    for i in range(num_restarts):
        pos, val = minimize_lbfgs_grad(objective, inits[i, :], bnds=dom_bounds)
        solutions.append(pos)
        values.append(val)
    loc = jnp.vstack(solutions)
    acq = jnp.vstack(values)
    idx_best = jnp.argmin(acq)
    return loc[idx_best : idx_best + 1, :]


def _next_point_kwargs(fitted, seed=7):
    return {
        "params": fitted["opt_params"],
        "batch": fitted["batch"],
        "bounds": fitted["bounds"],
        "rng_key": random.PRNGKey(seed),
    }


@pytest.mark.parametrize("problem", ["gp_1d", "gp_4d"])
def test_next_point_lbfgs_batched_start_shapes_and_bounds(problem, request):
    """Batched-start EI path: valid shapes, in-bounds results, and the
    documented restart mapping (k = min(10, max(2, 10 // 5)) = 2 polishes
    for the default budget of 10)."""
    fitted = request.getfixturevalue(problem)
    kwargs = _next_point_kwargs(fitted)
    lb, ub = fitted["bounds"]["lb"], fitted["bounds"]["ub"]
    dim = lb.shape[0]

    x_new, acq, loc = fitted["gp"].compute_next_point_lbfgs(num_restarts=10, **kwargs)

    assert x_new.shape == (1, dim)
    assert acq.shape == (2, 1)
    assert loc.shape == (2, dim)
    # A polish may return NaN where the EI gradient NaNs at a
    # variance-clipped point; at least one must survive and be selected.
    assert int(jnp.sum(jnp.isfinite(acq))) >= 1
    assert bool(jnp.all(jnp.isfinite(x_new)))
    assert bool(jnp.all(loc >= lb)) and bool(jnp.all(loc <= ub))
    assert bool(jnp.all(x_new >= lb)) and bool(jnp.all(x_new <= ub))
    # x_new is the best finite polished start, per the return contract.
    assert bool(jnp.all(x_new[0] == loc[int(jnp.nanargmin(acq))]))


@pytest.mark.parametrize("problem", ["gp_1d", "gp_4d"])
def test_next_point_lbfgs_batched_start_matches_or_beats_serial(problem, request):
    """The batched-start point is at least as good as the old serial path's.

    Equal-or-better acquisition value within a small tolerance is the bar,
    not identical points: both paths polish with the same bounded L-BFGS-B
    but from different (equally seeded) start sets.
    """
    fitted = request.getfixturevalue(problem)
    kwargs = _next_point_kwargs(fitted)
    gp = fitted["gp"]

    x_new, _, _ = gp.compute_next_point_lbfgs(num_restarts=10, **kwargs)
    x_serial = _serial_next_point(gp, 10, **kwargs)

    # ravel: acquisition returns a 0-d value here (the fixtures carry 1-D
    # targets, so predict's mean is 1-D and the [0] convention consumes it).
    acq_new = float(jnp.ravel(gp.acquisition(x_new[0], **kwargs))[0])
    acq_serial = float(jnp.ravel(gp.acquisition(x_serial[0], **kwargs))[0])
    assert acq_new <= acq_serial + 1e-6


def test_next_point_lbfgs_is_deterministic(gp_1d):
    """Same rng_key, same result: seeding survives the batched-start path."""
    kwargs = _next_point_kwargs(gp_1d)
    x_a, acq_a, loc_a = gp_1d["gp"].compute_next_point_lbfgs(num_restarts=10, **kwargs)
    x_b, acq_b, loc_b = gp_1d["gp"].compute_next_point_lbfgs(num_restarts=10, **kwargs)
    onp.testing.assert_array_equal(onp.asarray(x_a), onp.asarray(x_b))
    onp.testing.assert_array_equal(onp.asarray(acq_a), onp.asarray(acq_b))
    onp.testing.assert_array_equal(onp.asarray(loc_a), onp.asarray(loc_b))


def test_next_point_lbfgs_serial_fallback_keeps_restart_count(gp_1d):
    """A criterion without a batched scorer (IMSE) keeps the serial contract:
    one polish per restart, so acq and loc have num_restarts rows."""
    lb, ub = gp_1d["bounds"]["lb"], gp_1d["bounds"]["ub"]
    gp_imse = make_gp(lb, ub, criterion="IMSE")
    kwargs = _next_point_kwargs(gp_1d)

    x_new, acq, loc = gp_imse.compute_next_point_lbfgs(num_restarts=2, **kwargs)

    assert x_new.shape == (1, 1)
    assert acq.shape == (2, 1)
    assert loc.shape == (2, 1)
    assert bool(jnp.all(loc >= lb)) and bool(jnp.all(loc <= ub))


def test_batched_start_scorer_capability_map(gp_1d):
    """EI maps to a batched scorer; per-candidate-state criteria map to None."""
    lb, ub = gp_1d["bounds"]["lb"], gp_1d["bounds"]["ub"]
    assert gp_1d["gp"]._batched_start_scorer(batch=gp_1d["batch"]) is not None
    for criterion in ["TS", "IMSE", "IMSE_L"]:
        assert make_gp(lb, ub, criterion=criterion)._batched_start_scorer() is None
