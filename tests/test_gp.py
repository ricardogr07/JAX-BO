"""GP train/predict round-trip and acquisition behavior tests (slice 2d).

Covers the model surface that previously had zero tests: exact ``GP``
training convergence, predictive round-trips on 1D and 4D synthetic
functions, the normalization contract (normalized batch in, raw ``X_star``
to ``predict``; SCOPE.md section 2), and EI acquisition sanity at both the
function level and through the model dispatch.

All randomness is seeded; the trained models are module-scoped fixtures so
the L-BFGS restarts run once per dimensionality and every test reuses them.
"""

import jax.numpy as jnp
import pytest
from jax import random, vmap
from jax.scipy.stats import norm

from jaxbo import acquisitions, initializers
from jaxbo.gp import GP
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


# 4D problem: quadratic bowl on the RAW domain [-1, 2]^4.
LB_4D = -1.0 * jnp.ones(4)
UB_4D = 2.0 * jnp.ones(4)
CENTER_4D = jnp.array([0.2, 0.5, 0.8, 1.1])


def f_4d(x):
    """Separable 4D quadratic bowl with its minimum at ``CENTER_4D``."""
    return jnp.sum((x - CENTER_4D) ** 2, axis=-1)


def make_gp(lb, ub, criterion="EI"):
    """Build a core ``GP`` with the shelter-pulse style options dict."""
    prior = uniform_prior(lb, ub)
    return GP({"kernel": "RBF", "input_prior": prior, "criterion": criterion})


@pytest.fixture(scope="module")
def gp_1d():
    """Train a 1D GP once and share model, data, and params across tests."""
    lb, ub = jnp.array([LB_1D]), jnp.array([UB_1D])
    bounds = {"lb": lb, "ub": ub}
    X = jnp.concatenate([jnp.linspace(LB_1D, GAP_1D[0], 8), jnp.array([UB_1D])])
    X = X[:, None]
    y = f_1d(X.flatten())
    batch, norm_const = normalize(X, y, bounds)
    gp = make_gp(lb, ub)
    # 10 restarts on purpose: fewer restarts can land in the degenerate
    # tiny-lengthscale NLML optimum that interpolates the data but predicts
    # the prior mean everywhere else.
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
def gp_4d():
    """Train a 4D GP once on 32 seeded samples and share it across tests."""
    bounds = {"lb": LB_4D, "ub": UB_4D}
    prior = uniform_prior(LB_4D, UB_4D)
    X = prior.sample(random.PRNGKey(1), 32)
    y = f_4d(X)
    batch, norm_const = normalize(X, y, bounds)
    gp = make_gp(LB_4D, UB_4D)
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
    fresh restart key too (observed held-out max error 0.005 to 0.009
    across keys 0, 1, 2, 5 during calibration).
    """
    gp, batch = gp_1d["gp"], gp_1d["batch"]
    opt_other = gp.train(batch, random.PRNGKey(5), num_restarts=10)
    X_dense = gp_1d["X"][:8]
    X_mid = (X_dense[:-1] + X_dense[1:]) / 2.0
    mu, _ = gp.predict(X_mid, params=opt_other, batch=batch, bounds=gp_1d["bounds"])
    y_pred = denorm(mu, gp_1d["norm_const"])
    assert float(jnp.max(jnp.abs(y_pred - f_1d(X_mid.flatten())))) < 0.12


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
    # Observed max error ~5e-3 on the 11.76 target range; 0.12 (about 1
    # percent of the range) keeps a wide cross-platform float32 margin while
    # still rejecting a degenerate fit.
    assert float(jnp.max(jnp.abs(y_pred - gp_1d["y"]))) < 0.12
    # Nearly noiseless interpolation: the model is confident at its own data.
    assert jnp.all(std >= 0.0)
    assert float(jnp.max(std)) < 0.05


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
    # Observed max error ~5e-3 across seeds on the 11.76 target range; see
    # the round-trip test note for the margin rationale.
    assert float(jnp.max(jnp.abs(y_pred - y_true))) < 0.12


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
    # Observed ratio 2.9 to 3.9 across seeds; 2.0 keeps margin while still
    # requiring the gap to be clearly less certain than the data.
    assert float(std_gap / jnp.max(std_train)) > 2.0


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
    improvement itself is the negated return value. The implemented Frazier
    tutorial form equals the textbook EI, delta * Phi(z) + std * phi(z) with
    delta = best - mean, whenever mean <= best; for mean > best it returns
    the std * phi(z) upper bound instead, so that branch is locked by the
    vanishing-uncertainty limit and monotonicity tests below, not here.
    """
    # mean == best: improvement reduces to std * phi(0) = 0.5 / sqrt(2*pi).
    val = acquisitions.EI(jnp.array([1.0]), jnp.array([0.5]), 1.0)
    assert float(-val) == pytest.approx(0.19947114020071635, abs=1e-6)

    for mean, std, best in [(0.5, 1.0, 1.0), (-0.3, 0.7, 0.2), (1.0, 0.4, 2.0)]:
        delta = best - mean
        z = delta / std
        expected = delta * norm.cdf(z) + std * norm.pdf(z)
        got = -acquisitions.EI(jnp.array([mean]), jnp.array([std]), best)
        assert jnp.allclose(got, expected, atol=1e-6)


def test_ei_worse_than_best_is_frazier_upper_bound():
    """Characterization: for mean > best, EI returns std * phi(z), not textbook.

    With mean=1, std=1, best=0 (minimization) the textbook expected
    improvement is delta * Phi(z) + std * phi(z) = -0.15866 + 0.24197 =
    0.08332, but the implemented Frazier tutorial form drops the negative
    delta * Phi(z) term and returns the std * phi(z) = 0.24197 upper bound.
    Locked here so any change to that branch is deliberate; the deviation is
    tracked in issue #55 (the acquisitions implementation is owned by the
    2b/2c slices, not this test suite).
    """
    got = -acquisitions.EI(jnp.array([1.0]), jnp.array([1.0]), 0.0)
    assert float(got) == pytest.approx(0.24197072451914337, abs=1e-6)


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
    # Observed max error ~0.017 on the 8.42 target range across seeds; 0.17
    # (2 percent of the range) keeps a ~10x margin while rejecting a
    # prior-mean-only fit.
    assert float(jnp.max(jnp.abs(y_pred - gp_4d["y"]))) < 0.17
    assert jnp.all(std >= 0.0)


def test_gp_4d_training_is_seed_robust(gp_4d):
    """A different 4D training key must reach the same held-out quality.

    Same rationale as the 1D twin: observed relative RMSE 0.04 to 0.05
    across training keys 2, 4, 8 during calibration, all far under the
    0.15 bound.
    """
    gp, batch = gp_4d["gp"], gp_4d["batch"]
    opt_other = gp.train(batch, random.PRNGKey(8), num_restarts=10)
    mu, _ = gp.predict(
        gp_4d["X_test"], params=opt_other, batch=batch, bounds=gp_4d["bounds"]
    )
    y_pred = denorm(mu, gp_4d["norm_const"])
    rel_rmse = jnp.sqrt(jnp.mean((y_pred - gp_4d["y_test"]) ** 2))
    rel_rmse = rel_rmse / gp_4d["y_test"].std()
    assert float(rel_rmse) < 0.15


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
    # Observed relative RMSE ~0.05 across seeds; 0.15 keeps a 3x margin
    # while a mean-only predictor would score ~1.0.
    assert float(rel_rmse) < 0.15
