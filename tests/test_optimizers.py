import jax.numpy as jnp
import numpy as np
from jaxbo.optimizers import minimize_bfgs_jax, minimize_lbfgs, minimize_lbfgs_grad


def quad(x):
    return np.sum((x - 3.0) ** 2)


def quad_grad(x):
    loss = quad(x)
    grad = 2 * (x - 3.0)
    return loss, grad


def test_minimize_lbfgs():
    x_opt, f_opt = minimize_lbfgs(quad, np.array([0.0]))
    assert np.allclose(x_opt, 3.0, atol=1e-3)
    assert f_opt < 1e-6


def test_minimize_lbfgs_grad():
    x_opt, f_opt = minimize_lbfgs_grad(quad_grad, np.array([0.0]))
    assert np.allclose(x_opt, 3.0, atol=1e-3)
    assert f_opt < 1e-6


def _jax_quad_vg(x):
    return jnp.sum((x - 3.0) ** 2), 2 * (x - 3.0)


def test_minimize_bfgs_jax():
    # Runs in the suite's default float32, exercising the dtype-aware
    # gtol/ftol termination defaults.
    x_opt, f_opt = minimize_bfgs_jax(_jax_quad_vg, jnp.zeros(2))
    assert jnp.allclose(x_opt, 3.0, atol=1e-3)
    assert float(f_opt) < 1e-5


def test_minimize_bfgs_jax_non_finite_start_is_reported():
    # A non-finite objective at x0 must come back as-is for the caller
    # to discard, never as a fabricated finite value.
    x_opt, f_opt = minimize_bfgs_jax(_jax_quad_vg, jnp.array([jnp.nan]))
    assert jnp.isnan(f_opt)


def test_minimize_bfgs_jax_non_finite_start_gradient_is_reported():
    """A finite value with a non-finite gradient at x0 is a failed run.

    The line search cannot accept any step from a non-finite direction, so
    the run optimizes nothing. Reporting its finite starting value would
    let GP.train's nanargmin select the lane as though it had converged.
    """

    def finite_value_nan_grad(x):
        return jnp.sum(x**2), jnp.full_like(x, jnp.nan)

    _, f_opt = minimize_bfgs_jax(finite_value_nan_grad, jnp.array([1.0, 2.0]))
    assert not bool(jnp.isfinite(f_opt))
