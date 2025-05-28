import numpy as onp
import jax.numpy as np
from jax import vmap, jit, vjp, random
from jax.scipy.linalg import cholesky, solve_triangular
from functools import partial
from sklearn import mixture
from pyDOE import lhs
from jax.scipy.stats import norm
from typing import List, Dict, Tuple, Any

import jaxbo.kernels as kernels
import jaxbo.acquisitions as acquisitions
import jaxbo.utils as utils
import jaxbo.initializers as initializers
from jaxbo.optimizers import minimize_lbfgs
from jaxbo.models import GPmodel

class MultipleIndependentOutputsGP(GPmodel):
    def __init__(self, options: Dict):
        """
        Gaussian Process model supporting multiple independent outputs (constraints).
        Inherits from the base GPmodel class and handles per-output hyperparameter training.

        Args:
            options (Dict): Configuration dictionary, including kernel choice and priors.
        """
        super().__init__(options)

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params: np.ndarray, batch: Dict[str, np.ndarray]) -> np.ndarray:
        X = batch['X']
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])
        K = self.kernel(X, X, theta) + np.eye(X.shape[0]) * (sigma_n + 1e-8)
        return cholesky(K, lower=True)

    def train(self, batch_list: List[Dict[str, np.ndarray]], rng_key, num_restarts: int = 10) -> List[np.ndarray]:
        """
        Train independent GP models for each output in the list of batches.
        """
        best_params = []
        for batch in batch_list:
            def objective(params):
                val, grad = self.likelihood_value_and_grad(params, batch)
                return onp.array(val), onp.array(grad)

            likelihoods = []
            all_params = []
            dim = batch['X'].shape[1]
            rng_keys = random.split(rng_key, num_restarts)

            for i in range(num_restarts):
                init = initializers.random_init_GP(rng_keys[i], dim)
                p, val = minimize_lbfgs(objective, init)
                all_params.append(p)
                likelihoods.append(val)

            all_params = np.vstack(all_params)
            likelihoods = np.vstack(likelihoods)
            idx_best = np.argmin(likelihoods)
            best_params.append(all_params[idx_best])

        return best_params

    @partial(jit, static_argnums=(0,))
    def predict_all(self, X_star: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict all outputs independently and return lists of means and stds.
        """
        params_list = kwargs['params']
        batch_list = kwargs['batch']
        norm_const_list = kwargs['norm_const']
        bounds = kwargs['bounds']
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])

        mu_list, std_list = [], []
        for k, (params, batch, norm_const) in enumerate(zip(params_list, batch_list, norm_const_list)):
            X, y = batch['X'], batch['y']
            sigma_n = np.exp(params[-1])
            theta = np.exp(params[:-1])
            k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0]) * (sigma_n + 1e-8)
            k_pX = self.kernel(X_star, X, theta)
            L = self.compute_cholesky(params, batch)
            alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
            beta = solve_triangular(L.T, solve_triangular(L, k_pX.T, lower=True))
            mu = np.matmul(k_pX, alpha)
            cov = k_pp - np.matmul(k_pX, beta)
            std = np.sqrt(np.clip(np.diag(cov), a_min=0.))

            if k > 0:
                mu = mu * norm_const['sigma_y'] + norm_const['mu_y']
                std = std * norm_const['sigma_y']
            mu_list.append(mu)
            std_list.append(std)

        return np.array(mu_list), np.array(std_list)

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation for a single-output GP.
        """
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']

        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        X, y = batch['X'], batch['y']
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0]) * (sigma_n + 1e-8)
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T, solve_triangular(L, k_pX.T, lower=True))
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))

        return mu, std

    @partial(jit, static_argnums=(0,))
    def constrained_acquisition(self, x: np.ndarray, **kwargs: Any) -> float:
        x = x[None, :]
        mean, std = self.predict_all(x, **kwargs)
        criterion = self.options['constrained_criterion']

        if criterion == 'EIC':
            best = np.min(kwargs['batch'][0]['y'])
            return acquisitions.EIC(mean, std, best)
        elif criterion == 'LCBC':
            return acquisitions.LCBC(mean, std, kwargs['kappa'])
        elif criterion == 'LW_LCBC':
            weights = utils.compute_w_gmm(x, **kwargs)
            return acquisitions.LW_LCBC(mean, std, weights, kwargs['kappa'])
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def constrained_acq_value_and_grad(self, x: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        fun = lambda x: self.constrained_acquisition(x, **kwargs)
        primals, f_vjp = vjp(fun, x)
        grads = f_vjp(np.ones_like(primals))[0]
        return primals, grads

    def constrained_compute_next_point_lbfgs(self, num_restarts: int = 10, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        def objective(x):
            value, grads = self.constrained_acq_value_and_grad(x, **kwargs)
            return onp.array(value), onp.array(grads)

        bounds = kwargs['bounds']
        lb, ub = bounds['lb'], bounds['ub']
        dim = lb.shape[0]
        rng_key = kwargs['rng_key']
        onp.random.seed(rng_key[0])
        x0 = lb + (ub - lb) * lhs(dim, num_restarts)
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))

        loc, acq = [], []
        for i in range(num_restarts):
            pos, val = minimize_lbfgs(objective, x0[i, :], bnds=dom_bounds)
            loc.append(pos)
            acq.append(val)

        loc = np.vstack(loc)
        acq = np.vstack(acq)
        idx_best = np.argmin(acq)
        x_new = loc[idx_best:idx_best + 1, :]
        return x_new, acq, loc

    @partial(jit, static_argnums=(0,))
    def draw_posterior_sample(self, X_star: np.ndarray, **kwargs: Any) -> np.ndarray:
        params_list = kwargs['params']
        batch_list = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const_list = kwargs['norm_const']
        rng_key_list = kwargs['rng_key']

        sample_list = []
        for params, batch, norm_const, rng_key in zip(params_list, batch_list, norm_const_list, rng_key_list):
            X_star_scaled = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
            X, y = batch['X'], batch['y']
            sigma_n = np.exp(params[-1])
            theta = np.exp(params[:-1])
            k_pp = self.kernel(X_star_scaled, X_star_scaled, theta) + np.eye(X_star.shape[0]) * (sigma_n + 1e-8)
            k_pX = self.kernel(X_star_scaled, X, theta)
            L = self.compute_cholesky(params, batch)
            alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
            beta = solve_triangular(L.T, solve_triangular(L, k_pX.T, lower=True))
            mu = np.matmul(k_pX, alpha)
            cov = k_pp - np.matmul(k_pX, beta)
            sample = random.multivariate_normal(rng_key, mu, cov)
            sample_list.append(sample)

        return np.array(sample_list)
