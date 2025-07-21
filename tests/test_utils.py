import jax.numpy as jnp
from jaxbo import utils


def test_standardize():
    X = jnp.array([[1.0], [2.0]])
    y = jnp.array([1.0, 3.0])
    batch, norm = utils.standardize(X, y)
    assert jnp.allclose(batch['X'].mean(), 0.0)
    assert jnp.allclose(batch['y'].mean(), 0.0)
    assert 'mu_X' in norm and 'sigma_X' in norm


def test_compute_w_gmm():
    x = jnp.array([0.5, 0.5])
    bounds = {'lb': jnp.zeros(2), 'ub': jnp.ones(2)}
    weights = jnp.array([1.0])
    means = jnp.array([[0.5, 0.5]])
    covs = jnp.eye(2).reshape(1,2,2)
    val = utils.compute_w_gmm(x, bounds=bounds, gmm_vars=(weights, means, covs))
    assert val > 0

