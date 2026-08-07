"""Manifold GP: a neural feature map composed with a GP kernel.

Part of the ``[multifidelity]`` extra (SCOPE.md section 3). Inputs are warped
through an MLP (:func:`jaxbo.multifidelity.nn.init_NN`) before entering the
kernel; the network weights and the GP hyperparameters are optimized jointly
on the marginal likelihood.
"""

from functools import partial
from typing import Any, Dict, List, Tuple

import jax.numpy as np
import numpy as onp
from jax import jit, random
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cholesky, solve_triangular

import jaxbo.initializers as initializers
from jaxbo.gp import GPmodel, _std_from_variance, jitter
from jaxbo.multifidelity.nn import init_NN
from jaxbo.optimizers import minimize_lbfgs_grad


class ManifoldGP(GPmodel):
    """GP regression over MLP-warped inputs.

    The packed parameter vector concatenates the GP hyperparameters (indexed
    by ``gp_params_ids``) with the flattened network weights (indexed by
    ``nn_params_ids``); ``unravel`` restores the network pytree.
    """

    def __init__(self, options: Dict[str, Any], layers: List[int]):
        """Initialize the feature map and parameter index bookkeeping.

        Args:
            options: GPmodel options dictionary (kernel, input_prior,
                criterion).
            layers: MLP layer widths; layers[-1] is the manifold dimension
                seen by the kernel.
        """
        super().__init__(options)
        self.layers = layers
        self.net_init, self.net_apply = init_NN(layers)
        # Determine parameter IDs
        nn_params = self.net_init(random.PRNGKey(0), (-1, layers[0]))[1]
        nn_params_flat, self.unravel = ravel_pytree(nn_params)
        num_nn_params = len(nn_params_flat)
        num_gp_params = initializers.random_init_GP(
            random.PRNGKey(0), layers[-1]
        ).shape[0]
        self.gp_params_ids = np.arange(num_gp_params)
        self.nn_params_ids = np.arange(num_nn_params) + num_gp_params

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(
        self, params: np.ndarray, batch: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Cholesky factor of the kernel over the warped training inputs.

        Args:
            params: Packed GP plus network parameters.
            batch: Normalized training data with key 'X'.

        Returns:
            Lower-triangular Cholesky factor.
        """
        # Warp inputs
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        X = self.net_apply(nn_params, batch["X"])
        N = X.shape[0]
        # Fetch params
        sigma_n = np.exp(gp_params[-1])
        theta = np.exp(gp_params[:-1])
        # Compute kernel
        K = self.kernel(X, X, theta)
        K = K + np.eye(N) * (sigma_n + jitter(K))
        L = cholesky(K, lower=True)
        return L

    def train(
        self,
        batch: Dict[str, np.ndarray],
        rng_key: np.ndarray,
        num_restarts: int = 10,
    ) -> np.ndarray:
        """Jointly optimize network and GP parameters by multi-start L-BFGS.

        Args:
            batch: ALREADY NORMALIZED training data, keys 'X' and 'y'.
            rng_key: PRNGKey seeding the restarts.
            num_restarts: Number of random initializations.

        Returns:
            The best packed parameter vector found (nan restarts are
            skipped).
        """

        # Define objective that returns NumPy arrays
        def objective(params):
            value, grads = self.likelihood_value_and_grad(params, batch)
            out = (onp.array(value), onp.array(grads))
            return out

        # Optimize with random restarts
        params = []
        likelihood = []
        dim = batch["X"].shape[1]
        rng_key = random.split(rng_key, num_restarts)
        for i in range(num_restarts):
            key1, key2 = random.split(rng_key[i])
            gp_params = initializers.random_init_GP(key1, dim)
            nn_params = self.net_init(key2, (-1, self.layers[0]))[1]
            init_params = np.concatenate([gp_params, ravel_pytree(nn_params)[0]])
            p, val = minimize_lbfgs_grad(objective, init_params)
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
        """Predictive mean and std at RAW domain points.

        Args:
            X_star: Query points in the RAW domain, shape (N, D); normalized
                against kwargs['bounds'], then warped through the network.
            **kwargs: Must include 'params', 'batch', and 'bounds'.

        Returns:
            Tuple (mu, std) of the posterior.
        """
        params = kwargs["params"]
        batch = kwargs["batch"]
        bounds = kwargs["bounds"]
        # Normalize to [0,1]
        X_star = (X_star - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
        # Fetch normalized training data
        X, y = batch["X"], batch["y"]
        # Warp inputs
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        X = self.net_apply(nn_params, X)
        X_star = self.net_apply(nn_params, X_star)
        # Fetch params
        sigma_n = np.exp(gp_params[-1])
        theta = np.exp(gp_params[:-1])
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta)
        k_pp = k_pp + np.eye(X_star.shape[0]) * (sigma_n + jitter(k_pp))
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T, solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = _std_from_variance(np.diag(cov), k_pp)

        return mu, std
