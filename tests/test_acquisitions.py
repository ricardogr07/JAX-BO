import jax.numpy as jnp
from jax import random
from jaxbo import acquisitions


def test_ei_basic():
    mean = jnp.array([0.5])
    std = jnp.array([1.0])
    best = 0.0
    # EI returns negative improvement
    val = acquisitions.EI(mean, std, best)
    assert val < 0


def test_eic_constraints():
    mean = jnp.array([[0.5], [1.0]])
    std = jnp.array([[1.0], [1.0]])
    best = 0.0
    val = acquisitions.EIC(mean, std, best)
    assert val < 0


def test_lcb_basic():
    mean = jnp.array([1.0])
    std = jnp.array([0.5])
    val = acquisitions.LCB(mean, std, kappa=2.0)
    # LCB = mean - kappa*std
    assert jnp.isclose(val, mean - 2.0 * std)


def test_us():
    std = jnp.array([0.5])
    val = acquisitions.US(std)
    assert jnp.isclose(val, -0.5)

