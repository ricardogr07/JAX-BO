import jax.numpy as np
from jax import random
from jax.scipy.stats import multivariate_normal, uniform

class uniform_prior:
    """
    Uniform prior distribution over a hyper-rectangle.

    Args:
        lb (np.ndarray): Lower bounds of the uniform distribution (shape: [dim]).
        ub (np.ndarray): Upper bounds of the uniform distribution (shape: [dim]).
    """
    def __init__(self, lb: np.ndarray = 0, ub: np.ndarray = 1):
        self.lb = lb
        self.ub = ub
        self.dim = lb.shape[0]
    
    def sample(self, rng_key: random.PRNGKey, N: int) -> np.ndarray:
        """
        Draw N samples from the uniform prior.

        Args:
            rng_key (random.PRNGKey): JAX random key.
            N (int): Number of samples.

        Returns:
            np.ndarray: Samples of shape (N, dim).
        """
        return self.lb + (self.ub - self.lb) * random.uniform(rng_key, (N, self.dim))
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the probability density function at points x.

        Args:
            x (np.ndarray): Points to evaluate, shape (..., dim).

        Returns:
            np.ndarray: PDF values at x, shape (...,).
        """
        return np.sum(uniform.pdf(x, self.lb, self.ub - self.lb), axis=-1)

class gaussian_prior:
    """
    Gaussian (multivariate normal) prior distribution.

    Args:
        mu (np.ndarray): Mean vector of the Gaussian (shape: [dim]).
        cov (np.ndarray): Covariance matrix of the Gaussian (shape: [dim, dim]).
    """
    def __init__(self, mu: np.ndarray, cov: np.ndarray):
        self.mu = mu
        self.cov = cov
        self.dim = mu.shape[0]

    def sample(self, rng_key: random.PRNGKey, N: int) -> np.ndarray:
        """
        Draw N samples from the Gaussian prior.

        Args:
            rng_key (random.PRNGKey): JAX random key.
            N (int): Number of samples.

        Returns:
            np.ndarray: Samples of shape (N, dim).
        """
        return random.multivariate_normal(rng_key, self.mu, self.cov, (N,))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the probability density function at points x.

        Args:
            x (np.ndarray): Points to evaluate, shape (..., dim).

        Returns:
            np.ndarray: PDF values at x, shape (...,).
        """
        return multivariate_normal.pdf(x, self.mu, self.cov)
