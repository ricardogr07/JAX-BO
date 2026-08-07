"""Heterogeneous two-fidelity GP: fidelities living on different input spaces.

Part of the ``[multifidelity]`` extra (SCOPE.md section 3). The low-fidelity
inputs are mapped into the high-fidelity unit cube by a sigmoid-squashed MLP
(:func:`jaxbo.multifidelity.nn.init_NN`) before the autoregressive
two-fidelity kernel is applied.
"""

from functools import partial
from typing import Any, Dict, List, Tuple

import jax.numpy as np
import numpy as onp
from jax import jit, random
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cholesky, solve_triangular
from jax.scipy.special import expit as sigmoid

import jaxbo.initializers as initializers
from jaxbo.gp import GPmodel, _std_from_variance, jitter
from jaxbo.multifidelity.nn import init_NN
from jaxbo.optimizers import minimize_lbfgs_grad


class HeterogeneousMultifidelityGP(GPmodel):
    """Two-fidelity GP whose low-fidelity inputs are learned projections.

    The packed parameter vector concatenates the multifidelity GP
    hyperparameters (indexed by ``gp_params_ids``) with the flattened network
    weights (indexed by ``nn_params_ids``). Unlike
    :class:`jaxbo.multifidelity.MultifidelityGP`, the kernel hyperparameters
    ``theta_L``/``theta_H`` are used WITHOUT exponentiation.
    """

    def __init__(self, options: Dict[str, Any], layers: List[int]):
        """Initialize the projection network and parameter bookkeeping.

        Args:
            options: GPmodel options dictionary (kernel, input_prior,
                criterion).
            layers: MLP layer widths mapping the low-fidelity input dimension
                to the high-fidelity one (layers[-1]).
        """
        super().__init__(options)
        self.layers = layers
        self.net_init, self.net_apply = init_NN(layers)
        # Determine parameter IDs
        nn_params = self.net_init(random.PRNGKey(0), (-1, layers[0]))[1]
        nn_params_flat, self.unravel = ravel_pytree(nn_params)
        num_nn_params = len(nn_params_flat)
        num_gp_params = initializers.random_init_MultifidelityGP(
            random.PRNGKey(0), layers[-1]
        ).shape[0]
        self.gp_params_ids = np.arange(num_gp_params)
        self.nn_params_ids = np.arange(num_nn_params) + num_gp_params

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(
        self, params: np.ndarray, batch: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Cholesky factor of the joint covariance with warped XL.

        Args:
            params: Packed GP plus network parameters.
            batch: Normalized training data with keys 'XL' and 'XH'.

        Returns:
            Lower-triangular Cholesky factor of the (NL+NH, NL+NH) kernel.
        """
        XL, XH = batch["XL"], batch["XH"]
        NL, NH = XL.shape[0], XH.shape[0]
        D = XH.shape[1]
        # Warp low-fidelity inputs to [0,1]^D_H
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        XL = sigmoid(self.net_apply(nn_params, XL))
        # Fetch params
        rho = gp_params[-3]
        sigma_n_L = np.exp(gp_params[-2])
        sigma_n_H = np.exp(gp_params[-1])
        theta_L = gp_params[: D + 1]
        theta_H = gp_params[D + 1 : -3]
        # Compute kernels
        K_LL = self.kernel(XL, XL, theta_L) + np.eye(NL) * sigma_n_L
        K_LH = rho * self.kernel(XL, XH, theta_L)
        K_HH = (
            rho**2 * self.kernel(XH, XH, theta_L)
            + self.kernel(XH, XH, theta_H)
            + np.eye(NH) * sigma_n_H
        )
        K = np.vstack((np.hstack((K_LL, K_LH)), np.hstack((K_LH.T, K_HH))))
        # Jitter on the assembled joint matrix, so it scales with the block
        # structure rather than with either fidelity alone.
        K = K + np.eye(NL + NH) * jitter(K)
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
            batch: ALREADY NORMALIZED training data, keys 'XL', 'XH', 'y'
                (see :func:`jaxbo.utils.normalize_HeterogeneousMultifidelityGP`).
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
        dim = batch["XH"].shape[1]
        rng_key = random.split(rng_key, num_restarts)
        for i in range(num_restarts):
            key1, key2 = random.split(rng_key[i])
            gp_params = initializers.random_init_MultifidelityGP(key1, dim)
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
        """High-fidelity predictive mean and std at RAW domain points.

        Args:
            X_star: Query points in the RAW high-fidelity domain, shape
                (N, D); normalized internally against kwargs['bounds'].
            **kwargs: Must include 'params', 'batch', and 'bounds'.

        Returns:
            Tuple (mu, std) of the high-fidelity posterior.
        """
        params = kwargs["params"]
        batch = kwargs["batch"]
        bounds = kwargs["bounds"]
        # Normalize to [0,1]
        X_star = (X_star - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
        # Fetch normalized training data
        XL, XH = batch["XL"], batch["XH"]
        D = batch["XH"].shape[1]
        y = batch["y"]
        # Warp low-fidelity inputs
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        XL = sigmoid(self.net_apply(nn_params, XL))
        # Fetch params
        rho = gp_params[-3]
        sigma_n_H = np.exp(gp_params[-1])
        theta_L = gp_params[: D + 1]
        theta_H = gp_params[D + 1 : -3]
        # Compute kernels
        k_pp = rho**2 * self.kernel(X_star, X_star, theta_L) + self.kernel(
            X_star, X_star, theta_H
        )
        k_pp = k_pp + np.eye(X_star.shape[0]) * (sigma_n_H + jitter(k_pp))
        psi1 = rho * self.kernel(X_star, XL, theta_L)
        psi2 = rho**2 * self.kernel(X_star, XH, theta_L) + self.kernel(
            X_star, XH, theta_H
        )
        k_pX = np.hstack((psi1, psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T, solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = _std_from_variance(np.diag(cov), k_pp)

        return mu, std
