from functools import partial
from jax import jit, random
from jax.scipy.linalg import cholesky, solve_triangular
import jax.numpy as np
import numpy as onp

from jaxbo.models.base_gpmodel import GPmodel
from jaxbo.optimizers import minimize_lbfgs_grad
from jaxbo import initializers


class GP(GPmodel):
    """
    Concrete implementation of a Gaussian Process (GP) regression model
    for Bayesian optimization tasks.
    """

    def __init__(self, options):
        """Initialize a standard Gaussian Process model."""
        super().__init__(options)

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        """
        Compute the Cholesky decomposition of the kernel matrix.

        Args:
            params: Log-transformed kernel parameters (including noise term).
            batch: Dictionary containing normalized training inputs 'X'.

        Returns:
            Lower-triangular matrix from Cholesky decomposition.
        """
        X = batch['X']
        N, D = X.shape
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])
        K = self.kernel(X, X, theta) + np.eye(N) * (sigma_n + 1e-8)
        return cholesky(K, lower=True)

    def train(self, batch, rng_key, num_restarts=10):
        """
        Optimize GP hyperparameters using multi-start L-BFGS-B.

        Args:
            batch: Dictionary with 'X' and 'y'.
            rng_key: PRNGKey for reproducibility.
            num_restarts: Number of random initializations.

        Returns:
            Best hyperparameters found (array).
        """
        def objective(params):
            value, grads = self.likelihood_value_and_grad(params, batch)
            return onp.array(value), onp.array(grads)

        dim = batch['X'].shape[1]
        rng_keys = random.split(rng_key, num_restarts)

        params_list, values = [], []
        for i in range(num_restarts):
            init = initializers.random_init_GP(rng_keys[i], dim)
            p, val = minimize_lbfgs_grad(objective, init)
            params_list.append(p)
            values.append(val)

        params_stack = np.vstack(params_list)
        values_stack = np.vstack(values)
        idx_best = np.nanargmin(values_stack)
        return params_stack[idx_best, :]

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        """
        Predict mean and standard deviation for new input points.

        Args:
            X_star: New input points, shape (N, D).
            kwargs: Must include 'params', 'batch', 'bounds', and 'norm_const'.

        Returns:
            Tuple (mean, std): Predictive posterior mean and standard deviation.
        """
        params, batch, bounds = kwargs['params'], kwargs['batch'], kwargs['bounds']
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        X, y = batch['X'], batch['y']
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])

        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0]) * (sigma_n + 1e-8)
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T, solve_triangular(L, k_pX.T, lower=True))

        mu = k_pX @ alpha
        cov = k_pp - k_pX @ beta
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        return mu, std

    @partial(jit, static_argnums=(0,))
    def posterior_covariance(self, x, xp, **kwargs):
        """
        Compute the posterior covariance between two sets of points.

        Args:
            x, xp: Input arrays of shape (N, D).
            kwargs: Must include 'params', 'batch', and 'bounds'.

        Returns:
            Posterior covariance matrix between x and xp.
        """
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']

        x = (x - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        xp = (xp - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        X, y = batch['X'], batch['y']
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])

        k_pp = self.kernel(x, xp, theta)
        k_pX = self.kernel(x, X, theta)
        k_Xp = self.kernel(X, xp, theta)
        L = self.compute_cholesky(params, batch)
        beta = solve_triangular(L.T, solve_triangular(L, k_Xp, lower=True))
        cov = k_pp - np.matmul(k_pX, beta)
        return cov

    @partial(jit, static_argnums=(0,))
    def draw_posterior_sample(self, X_star, **kwargs):
        """
        Draw a single sample from the GP posterior at new input locations.

        Args:
            X_star: Input locations, shape (N, D).
            kwargs: Must include 'params', 'batch', 'bounds', 'rng_key'.

        Returns:
            Sample drawn from the multivariate normal posterior.
        """
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        rng_key = kwargs['rng_key']

        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        X, y = batch['X'], batch['y']
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])

        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0]) * (sigma_n + 1e-8)
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T, solve_triangular(L, k_pX.T, lower=True))

        mu = k_pX @ alpha
        cov = k_pp - k_pX @ beta
        return random.multivariate_normal(rng_key, mu, cov)
