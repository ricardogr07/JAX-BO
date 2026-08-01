import jax.numpy as jnp
from jaxbo import acquisitions


def test_ei_basic():
    mean = jnp.array([0.5])
    std = jnp.array([1.0])
    best = 0.0
    val = acquisitions.EI(mean, std, best)
    assert val < 0


def test_eic_constraints():
    mean = jnp.array([[0.5], [1.0]])
    std = jnp.array([[1.0], [1.0]])
    best = 0.0
    val = acquisitions.EIC(mean, std, best)
    assert val < 0


def test_eic_tiny_nonzero_constraint_std():
    # A valid tiny std must keep the real Z = mean/std, so feasibility is
    # Phi(1), not the eps-clamped Phi(mean/eps) (Codex review on PR 61).
    mean = jnp.array([[0.5], [1e-13]])
    std = jnp.array([[1.0], [1e-13]])
    val = acquisitions.EIC(mean, std, 0.0)
    ei = -acquisitions.EI(mean[0], std[0], 0.0)
    assert jnp.isclose(-val, ei * 0.8413447, rtol=1e-4)


def test_eic_deterministic_constraint():
    # std == 0 takes the step limit: infeasible kills EIC exactly, no NaN.
    mean = jnp.array([[0.5], [-1.0]])
    std = jnp.array([[1.0], [0.0]])
    assert acquisitions.EIC(mean, std, 0.0) == 0.0
    feasible_mean = jnp.array([[0.5], [1.0]])
    val = acquisitions.EIC(feasible_mean, std, 0.0)
    ei = -acquisitions.EI(feasible_mean[0], std[0, :1], 0.0)
    assert jnp.isclose(-val, ei)


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
