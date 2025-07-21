import numpy as np
from jaxbo.optimizers import minimize_lbfgs, minimize_lbfgs_grad


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

