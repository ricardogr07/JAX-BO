"""Multiple independent heterogeneous two-fidelity GPs for constrained BO.

Part of the ``[multifidelity]`` extra (SCOPE.md section 3). One
:class:`HeterogeneousMultifidelityGP`-style model per output (objective plus
constraints), each warping its low-fidelity inputs through a shared
sigmoid-squashed MLP, plus the constrained acquisition machinery (EIC, LCBC,
and the ``[weighted]``-gated LW_LCBC) and a constrained ``fit_gmm`` that
delegates to :func:`jaxbo.weights.fit_gmm_constrained`.
"""

from functools import partial
from typing import Any, Dict, List, Tuple

import jax.numpy as np
import numpy as onp
from jax import jit, random, vjp
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cholesky, solve_triangular
from jax.scipy.special import expit as sigmoid
from scipy.stats import qmc

import jaxbo.acquisitions as acquisitions
import jaxbo.initializers as initializers
from jaxbo.gp import GPmodel, _std_from_variance, jitter
from jaxbo.multifidelity.nn import init_NN
from jaxbo.optimizers import minimize_lbfgs


class MultipleIndependentHeterogeneousMFGP(GPmodel):
    """Independent heterogeneous two-fidelity GPs, one per output.

    List-valued kwargs ('params', 'batch', 'norm_const') carry one entry per
    output; entry 0 is the objective and the rest are constraints. Kernel
    hyperparameters ``theta_L``/``theta_H`` are used WITHOUT exponentiation,
    matching :class:`jaxbo.multifidelity.HeterogeneousMultifidelityGP`.
    """

    def __init__(self, options: Dict[str, Any], layers: List[int]):
        """Initialize the projection network and parameter bookkeeping.

        Args:
            options: GPmodel options dictionary; must carry
                'constrained_criterion' for the constrained acquisition.
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
        """Cholesky factor of one output's joint covariance with warped XL.

        Args:
            params: Packed GP plus network parameters for one output.
            batch: That output's normalized data with keys 'XL' and 'XH'.

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
        batch_list: List[Dict[str, np.ndarray]],
        rng_key: np.ndarray,
        num_restarts: int = 10,
    ) -> List[np.ndarray]:
        """Train one independent model per output batch.

        Args:
            batch_list: One ALREADY NORMALIZED batch per output, each with
                keys 'XL', 'XH', 'y'.
            rng_key: PRNGKey seeding the restarts (shared across outputs).
            num_restarts: Number of random initializations per output.

        Returns:
            One best packed parameter vector per output.
        """
        best_params = []
        for _, batch in enumerate(batch_list):
            # Define objective that returns NumPy arrays
            def objective(params):
                value, grads = self.likelihood_value_and_grad(params, batch)
                out = (onp.array(value), onp.array(grads))
                return out

            # Optimize with random restarts
            params = []
            likelihood = []
            dim = batch["XH"].shape[1]
            rng_keys = random.split(rng_key, num_restarts)
            for i in range(num_restarts):
                key1, key2 = random.split(rng_keys[i])
                gp_params = initializers.random_init_MultifidelityGP(key1, dim)
                nn_params = self.net_init(key2, (-1, self.layers[0]))[1]
                init_params = np.concatenate([gp_params, ravel_pytree(nn_params)[0]])
                p, val = minimize_lbfgs(objective, init_params)
                params.append(p)
                likelihood.append(val)
            params = np.vstack(params)
            likelihood = np.vstack(likelihood)
            #### find the best likelihood besides nan ####
            bestlikelihood = np.nanmin(likelihood)
            idx_best = np.where(likelihood == bestlikelihood)
            idx_best = idx_best[0][0]
            best_params.append(params[idx_best, :])
        return best_params

    # Predict all high fidelity prediction (objective + constraints)
    @partial(jit, static_argnums=(0,))
    def predict_all(
        self, X_star: np.ndarray, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict every output's high-fidelity posterior at RAW points.

        Outputs after the first are de-normalized with their own
        ``norm_const`` entry (the objective row stays normalized).

        Args:
            X_star: Query points in the RAW domain, shape (N, D).
            **kwargs: List-valued 'params', 'batch', 'norm_const' plus
                'bounds'.

        Returns:
            Tuple of stacked means and stds, each of shape (num_outputs, N).
        """
        mu_list = []
        std_list = []
        params_list = kwargs["params"]
        batch_list = kwargs["batch"]
        bounds = kwargs["bounds"]
        norm_const_list = kwargs["norm_const"]
        zipped_args = zip(params_list, batch_list, norm_const_list)
        # Normalize once to [0,1] before the per-output loop
        X_star = (X_star - bounds["lb"]) / (bounds["ub"] - bounds["lb"])

        for k, (params, batch, norm_const) in enumerate(zipped_args):
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
            if k > 0:
                mu = mu * norm_const["sigma_y"] + norm_const["mu_y"]
                std = std * norm_const["sigma_y"]
            mu_list.append(mu)
            std_list.append(std)
        return np.array(mu_list), np.array(std_list)

    @partial(jit, static_argnums=(0,))
    def predict(
        self, X_star: np.ndarray, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict a single output's high-fidelity posterior at RAW points.

        Args:
            X_star: Query points in the RAW domain, shape (N, D).
            **kwargs: Scalar 'params' and 'batch' for one output plus
                'bounds'.

        Returns:
            Tuple (mu, std) of that output's posterior.
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

    @partial(jit, static_argnums=(0,))
    def constrained_acquisition(self, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Constrained acquisition value at a single point x.

        Dispatches on options['constrained_criterion']: 'EIC', 'LCBC', or
        'LW_LCBC'. The LW_LCBC branch consumes ``gmm_vars`` through
        :func:`jaxbo.weights.compute_w_gmm` and therefore needs the
        [weighted] extra installed.
        """
        x = x[None, :]
        mean, std = self.predict_all(x, **kwargs)
        if self.options["constrained_criterion"] == "EIC":
            batch_list = kwargs["batch"]
            best = np.min(batch_list[0]["y"])
            return acquisitions.EIC(mean, std, best)
        elif self.options["constrained_criterion"] == "LCBC":
            kappa = kwargs["kappa"]
            return acquisitions.LCBC(mean, std, kappa)
        elif self.options["constrained_criterion"] == "LW_LCBC":
            # Lazy import: gmm_vars flows live behind the [weighted] extra
            # (SCOPE.md decision 7) and raise its install hint when missing.
            from jaxbo.weights import compute_w_gmm

            kappa = kwargs["kappa"]
            weights = compute_w_gmm(x, **kwargs)
            return acquisitions.LW_LCBC(mean, std, weights, kappa)
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def constrained_acq_value_and_grad(
        self, x: np.ndarray, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Constrained acquisition value and gradient at x (reverse mode)."""

        def fun(x):
            return self.constrained_acquisition(x, **kwargs)

        primals, f_vjp = vjp(fun, x)
        grads = f_vjp(np.ones_like(primals))[0]
        return primals, grads

    def constrained_compute_next_point_lbfgs(
        self, num_restarts: int = 10, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Minimize the constrained acquisition with multi-start L-BFGS-B.

        Args:
            num_restarts: Number of LHS-seeded restarts.
            **kwargs: Must include 'bounds' and 'rng_key' plus everything the
                constrained acquisition needs.

        Returns:
            Tuple (x_new, acq, loc): best point (1, D), acquisition values
            per restart, and the tested locations.
        """

        # Define objective that returns NumPy arrays
        def objective(x):
            value, grads = self.constrained_acq_value_and_grad(x, **kwargs)
            out = (onp.array(value), onp.array(grads))
            return out

        # Optimize with random restarts
        loc = []
        acq = []
        bounds = kwargs["bounds"]
        lb = bounds["lb"]
        ub = bounds["ub"]
        rng_key = kwargs["rng_key"]
        dim = lb.shape[0]

        onp.random.seed(rng_key[0])
        sampler = qmc.LatinHypercube(d=dim, seed=int(rng_key[0]))
        x0 = lb + (ub - lb) * sampler.random(num_restarts)
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))
        for i in range(num_restarts):
            pos, val = minimize_lbfgs(objective, x0[i, :], bnds=dom_bounds)
            loc.append(pos)
            acq.append(val)
        loc = np.vstack(loc)
        acq = np.vstack(acq)
        idx_best = np.argmin(acq)
        x_new = loc[idx_best : idx_best + 1, :]
        return x_new, acq, loc

    def fit_gmm(
        self, num_comp: int = 2, N_samples: int = 10000, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit the constraint-weighted GMM behind the LW_LCBC criterion.

        Delegates to :func:`jaxbo.weights.fit_gmm_constrained` (the
        [weighted] extra); calling it without scikit-learn and KDEpy
        installed raises an ImportError naming ``pip install
        jaxbo[weighted]``. See that function for the full contract.
        """
        from jaxbo.weights import fit_gmm_constrained

        return fit_gmm_constrained(self, num_comp, N_samples, **kwargs)
