"""Two-fidelity autoregressive GP regression (Kennedy and O'Hagan style).

Part of the ``[multifidelity]`` extra (SCOPE.md section 3). The low- and
high-fidelity processes are linked by a scalar correlation ``rho``:
``f_H(x) = rho * f_L(x) + delta(x)``, each with its own kernel
hyperparameters and noise.
"""

from functools import partial
from typing import Any, Dict, Tuple

import jax.numpy as np
import numpy as onp
from jax import jit, random
from jax.scipy.linalg import cholesky, solve_triangular

import jaxbo.initializers as initializers
from jaxbo.gp import GPmodel, _std_from_variance, jitter
from jaxbo.optimizers import minimize_lbfgs_grad


class MultifidelityGP(GPmodel):
    """Minimal two-fidelity GP regression model on the shared GPmodel base.

    The parameter vector layout is ``[theta_L, theta_H, rho, log sigma_n_L,
    log sigma_n_H]`` where each ``theta`` holds a log-variance plus D
    log-lengthscales. Batches carry the keys ``XL``, ``XH``, and the
    concatenated targets ``y`` (see
    :func:`jaxbo.utils.normalize_MultifidelityGP`).
    """

    def __init__(self, options: Dict[str, Any]):
        """Initialize the model from a GPmodel options dictionary."""
        super().__init__(options)

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(
        self, params: np.ndarray, batch: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Cholesky factor of the joint two-fidelity covariance matrix.

        Args:
            params: Packed hyperparameters (see class docstring for layout).
            batch: Normalized training data with keys 'XL' and 'XH'.

        Returns:
            Lower-triangular Cholesky factor of the (NL+NH, NL+NH) kernel.
        """
        XL, XH = batch["XL"], batch["XH"]
        NL, NH = XL.shape[0], XH.shape[0]
        D = XH.shape[1]
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[: D + 1])
        theta_H = np.exp(params[D + 1 : -3])
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
        """Optimize hyperparameters by multi-start L-BFGS on the NLML.

        Args:
            batch: ALREADY NORMALIZED training data (same contract as
                :meth:`jaxbo.gp.GP.train`), keys 'XL', 'XH', 'y'.
            rng_key: PRNGKey seeding the restarts.
            num_restarts: Number of random initializations.

        Returns:
            The best packed hyperparameter vector found (nan restarts are
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
            init = initializers.random_init_MultifidelityGP(rng_key[i], dim)
            p, val = minimize_lbfgs_grad(objective, init)
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
            X_star: Query points in the RAW domain, shape (N, D); normalized
                internally against kwargs['bounds'].
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
        # Fetch params
        rho = params[-3]
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[: D + 1])
        theta_H = np.exp(params[D + 1 : -3])
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

    @partial(jit, static_argnums=(0,))
    def predict_L(
        self, X_star: np.ndarray, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Low-fidelity predictive mean and std at RAW domain points.

        Same contract as :meth:`predict` but for the low-fidelity process
        (used by the IMSE_L acquisition criterion).
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
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        theta_L = np.exp(params[: D + 1])
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta_L)
        k_pp = k_pp + np.eye(X_star.shape[0]) * (sigma_n_L + jitter(k_pp))
        psi1 = self.kernel(X_star, XL, theta_L)
        psi2 = rho * self.kernel(X_star, XH, theta_L)
        k_pX = np.hstack((psi1, psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T, solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = _std_from_variance(np.diag(cov), k_pp)
        return mu, std

    @partial(jit, static_argnums=(0,))
    def posterior_covariance_L(
        self, x: np.ndarray, xp: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Low-fidelity posterior covariance between RAW points x and xp.

        Args:
            x, xp: RAW domain point sets; normalized internally.
            **kwargs: Must include 'params', 'batch', and 'bounds'.

        Returns:
            Posterior covariance matrix of the low-fidelity process.
        """
        params = kwargs["params"]
        batch = kwargs["batch"]
        bounds = kwargs["bounds"]
        # Normalize to [0,1]
        x = (x - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
        xp = (xp - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
        # Fetch normalized training data
        XL, XH = batch["XL"], batch["XH"]
        D = batch["XH"].shape[1]
        # Fetch params
        rho = params[-3]
        theta_L = np.exp(params[: D + 1])
        # Compute kernels
        k_pp = self.kernel(x, xp, theta_L)
        psi1 = self.kernel(x, XL, theta_L)
        psi2 = rho * self.kernel(x, XH, theta_L)
        k_pX = np.hstack((psi1, psi2))
        psi1 = self.kernel(XL, xp, theta_L)
        psi2 = rho * self.kernel(XH, xp, theta_L)
        k_Xp = np.hstack((psi1, psi2))
        L = self.compute_cholesky(params, batch)
        # Compute predictive mean, std
        beta = solve_triangular(L.T, solve_triangular(L, k_Xp, lower=True))
        cov = k_pp - np.matmul(k_pX, beta)
        return cov

    @partial(jit, static_argnums=(0,))
    def posterior_covariance_H(
        self, x: np.ndarray, xp: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """High-fidelity posterior covariance between RAW points x and xp.

        Same contract as :meth:`posterior_covariance_L` but for the
        high-fidelity process (used by the IMSE acquisition criterion).
        """
        params = kwargs["params"]
        batch = kwargs["batch"]
        bounds = kwargs["bounds"]
        # Normalize to [0,1]
        x = (x - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
        xp = (xp - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
        # Fetch normalized training data
        XL, XH = batch["XL"], batch["XH"]
        D = batch["XH"].shape[1]
        # Fetch params
        rho = params[-3]
        theta_L = np.exp(params[: D + 1])
        theta_H = np.exp(params[D + 1 : -3])
        # Compute kernels
        k_pp = rho**2 * self.kernel(x, xp, theta_L) + self.kernel(x, xp, theta_H)
        psi1 = rho * self.kernel(x, XL, theta_L)
        psi2 = rho**2 * self.kernel(x, XH, theta_L) + self.kernel(x, XH, theta_H)
        k_pX = np.hstack((psi1, psi2))
        psi1 = rho * self.kernel(XL, xp, theta_L)
        psi2 = rho**2 * self.kernel(XH, xp, theta_L) + self.kernel(XH, xp, theta_H)
        k_Xp = np.hstack((psi1, psi2))
        L = self.compute_cholesky(params, batch)
        # Compute predictive mean, std
        beta = solve_triangular(L.T, solve_triangular(L, k_Xp, lower=True))
        cov = k_pp - np.matmul(k_pX, beta)
        return cov
