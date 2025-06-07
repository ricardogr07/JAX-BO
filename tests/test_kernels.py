import jax.numpy as jnp
from jaxbo import kernels


def test_pairwise_diff_squared():
    x1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    x2 = jnp.array([[0.0, 0.0]])
    length = jnp.array([1.0, 1.0])
    d2 = kernels._pairwise_diff_squared(x1, x2, length)
    assert d2.shape == (2, 1)
    assert jnp.allclose(d2, jnp.array([[0.0], [2.0]]))


def test_rbf_kernel():
    x = jnp.array([[0.0], [1.0]])
    params = jnp.array([1.0, 1.0])
    K = kernels.RBF(x, x, params)
    assert K.shape == (2, 2)
    assert jnp.isclose(K[0, 0], 1.0)

