"""GP regression on joint function and gradient observations.

Part of the ``[multifidelity]`` extra (SCOPE.md section 3). The covariance
between gradient observations is obtained by differentiating the kernel with
forward-mode autodiff (``jvp``), so any core kernel works unchanged.
"""

from functools import partial
from typing import Any, Dict, Tuple

import jax.numpy as np
import numpy as onp
from jax import jit, jvp, random
from jax.scipy.linalg import cholesky, solve_triangular

import jaxbo.initializers as initializers
from jaxbo.gp import GPmodel, _std_from_variance, jitter
from jaxbo.optimizers import minimize_lbfgs


class GradientGP(GPmodel):
    """GP over function values XF/yF and gradient values XG/yG.

    Batches carry the keys ``XF``, ``XG``, ``yF``, ``yG``, and the
    concatenated targets ``y`` (see
    :func:`jaxbo.utils.normalize_GradientGP`). The parameter vector layout is
    ``[theta, log sigma_n_F, log sigma_n_G]``. Unlike the other models,
    ``predict`` consumes inputs as given: no bounds normalization is applied.
    """

    def __init__(self, options: Dict[str, Any]):
        """Initialize the model from a GPmodel options dictionary."""
        super().__init__(options)

    @partial(jit, static_argnums=(0,))
    def k_dx2(self, x1: np.ndarray, x2: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Kernel differentiated once, against its second argument."""

        def fun(x2):
            return self.kernel(x1, x2, params)

        g = jvp(fun, (x2,), (np.ones_like(x2),))[1]
        return g

    @partial(jit, static_argnums=(0,))
    def k_dx1dx2(
        self, x1: np.ndarray, x2: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        """Kernel differentiated against both arguments (gradient block)."""

        def fun(x1_):
            return self.k_dx2(x1_, x2, params)

        g = jvp(fun, (x1,), (np.ones_like(x1),))[1]
        return g

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(
        self, params: np.ndarray, batch: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Cholesky factor of the joint function/gradient covariance.

        Args:
            params: Packed hyperparameters (see class docstring for layout).
            batch: Training data with keys 'XF' and 'XG'.

        Returns:
            Lower-triangular Cholesky factor of the (NF+NG, NF+NG) kernel.
        """
        XF, XG = batch["XF"], batch["XG"]
        NF, NG = XF.shape[0], XG.shape[0]
        # Fetch params
        sigma_n_F = np.exp(params[-2])
        sigma_n_G = np.exp(params[-1])
        theta = np.exp(params[:-2])
        # Compute kernels
        K_FF = self.kernel(XF, XF, theta) + np.eye(NF) * sigma_n_F
        K_FG = self.k_dx2(XF, XG, theta)
        K_GG = self.k_dx1dx2(XG, XG, theta) + np.eye(NG) * sigma_n_G
        K = np.vstack((np.hstack((K_FF, K_FG)), np.hstack((K_FG.T, K_GG))))
        # Jitter on the assembled joint matrix: the value and gradient blocks
        # carry different scales, so neither alone sets the regularization.
        K = K + np.eye(NF + NG) * jitter(K)
        L = cholesky(K, lower=True)
        return L

    def train(
        self,
        batch: Dict[str, np.ndarray],
        rng_key: np.ndarray,
        num_restarts: int = 10,
    ) -> np.ndarray:
        """Optimize hyperparameters by multi-start L-BFGS on the NLML.

        Args:
            batch: Training data, keys 'XF', 'XG', 'y'.
            rng_key: PRNGKey seeding the restarts.
            num_restarts: Number of random initializations.

        Returns:
            The best packed hyperparameter vector found (nan restarts are
            skipped).
        """

        # Define objective that returns NumPy arrays
        def objective(params: np.ndarray) -> Tuple[onp.ndarray, onp.ndarray]:
            value, grads = self.likelihood_value_and_grad(params, batch)
            out = (onp.array(value), onp.array(grads))
            return out

        # Optimize with random restarts
        params = []
        likelihood = []
        dim = batch["XF"].shape[1]
        rng_key = random.split(rng_key, num_restarts)
        for i in range(num_restarts):
            init = initializers.random_init_GradientGP(rng_key[i], dim)
            p, val = minimize_lbfgs(objective, init)
            params.append(p)
            likelihood.append(val)
        params = np.vstack(params)
        likelihood = np.vstack(likelihood)

        #### find the best likelihood besides nan ####
        bestlikelihood = np.nanmin(likelihood)
        idx_best = np.where(likelihood == bestlikelihood)
        idx_best = idx_best[0][0]
        best_params = params[idx_best, :]
        return best_params

    @partial(jit, static_argnums=(0,))
    def predict(
        self, X_star: np.ndarray, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predictive mean and std of the function values at X_star.

        Args:
            X_star: Query points, shape (N, D), consumed as given (this model
                applies NO bounds normalization).
            **kwargs: Must include 'params' and 'batch'.

        Returns:
            Tuple (mu, std) of the posterior over function values.
        """
        params = kwargs["params"]
        batch = kwargs["batch"]
        # (do not Normalize!)
        # X_star = (X_star - norm_const['mu_X'])/norm_const['sigma_X']
        # Fetch training data
        XF, XG = batch["XF"], batch["XG"]
        y = batch["y"]
        # Fetch params
        sigma_n_F = np.exp(params[-2])
        theta = np.exp(params[:-2])
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta)
        k_pp = k_pp + np.eye(X_star.shape[0]) * (sigma_n_F + jitter(k_pp))
        psi1 = self.kernel(X_star, XF, theta)
        psi2 = self.k_dx2(X_star, XG, theta)
        k_pX = np.hstack((psi1, psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T, solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = _std_from_variance(np.diag(cov), k_pp)

        return mu, std
