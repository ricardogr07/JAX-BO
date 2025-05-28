import numpy as onp
import jax.numpy as np
from jax import jit, jvp, random
from jax.scipy.linalg import cholesky, solve_triangular
from functools import partial

import jaxbo.initializers as initializers
from jaxbo.optimizers import minimize_lbfgs
from jaxbo.models.base_gpmodel import GPmodel

class GradientGP(GPmodel):

    def __init__(self, options):
        super().__init__(options)

    @partial(jit, static_argnums=(0,))
    def k_dx2(self, x1, x2, params):
        fun = lambda x2: self.kernel(x1, x2, params)
        g = jvp(fun, (x2,), (np.ones_like(x2),))[1]
        return g

    @partial(jit, static_argnums=(0,))
    def k_dx1dx2(self, x1, x2, params):
        fun = lambda x1: self.k_dx2(x1, x2, params)
        g = jvp(fun, (x1,), (np.ones_like(x1),))[1]
        return g

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        XF, XG = batch['XF'], batch['XG']
        NF, NG = XF.shape[0], XG.shape[0]
        # Fetch params
        sigma_n_F = np.exp(params[-2])
        sigma_n_G = np.exp(params[-1])
        theta = np.exp(params[:-2])
        # Compute kernels
        K_FF = self.kernel(XF, XF, theta) + np.eye(NF)*(sigma_n_F + 1e-8)
        K_FG = self.k_dx2(XF, XG, theta)
        K_GG = self.k_dx1dx2(XG, XG, theta) + np.eye(NG)*(sigma_n_G + 1e-8)
        K = np.vstack((np.hstack((K_FF,K_FG)),
                       np.hstack((K_FG.T,K_GG))))
        L = cholesky(K, lower=True)
        return L

    def train(self, batch, rng_key, num_restarts = 10):
        # Define objective that returns NumPy arrays
        def objective(params):
            value, grads = self.likelihood_value_and_grad(params, batch)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        params = []
        likelihood = []
        dim = batch['XF'].shape[1]
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
        best_params = params[idx_best,:]
        return best_params

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        norm_const = kwargs['norm_const']
        # (do not Normalize!)
        # X_star = (X_star - norm_const['mu_X'])/norm_const['sigma_X']
        # Fetch training data
        XF, XG = batch['XF'], batch['XG']
        y = batch['y']
        # Fetch params
        sigma_n_F = np.exp(params[-2])
        sigma_n_G = np.exp(params[-1])
        theta = np.exp(params[:-2])
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0])*(sigma_n_F + 1e-8)
        psi1 = self.kernel(X_star, XF, theta)
        psi2 = self.k_dx2(X_star, XG, theta)
        k_pX = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))

        return mu, std
