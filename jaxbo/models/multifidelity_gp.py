import numpy as onp
import jax.numpy as np
from jax import jit, random
from jax.scipy.linalg import cholesky, solve_triangular
from functools import partial

import jaxbo.initializers as initializers
from jaxbo.optimizers import minimize_lbfgs
from jaxbo.models.base_gpmodel import GPmodel

# A minimal MultifidelityGP regression class (inherits from GPmodel)
class MultifidelityGP(GPmodel):
    # Initialize the class
    def __init__(self, options):
        super().__init__(options)

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        XL, XH = batch['XL'], batch['XH']
        NL, NH = XL.shape[0], XH.shape[0]
        D = XH.shape[1]
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[:D+1])
        theta_H = np.exp(params[D+1:-3])
        # Compute kernels
        K_LL = self.kernel(XL, XL, theta_L) + np.eye(NL)*(sigma_n_L + 1e-8)
        K_LH = rho*self.kernel(XL, XH, theta_L)
        K_HH = rho**2 * self.kernel(XH, XH, theta_L) + \
                        self.kernel(XH, XH, theta_H) + np.eye(NH)*(sigma_n_H + 1e-8)
        K = np.vstack((np.hstack((K_LL,K_LH)),
                       np.hstack((K_LH.T,K_HH))))
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
        dim = batch['XH'].shape[1]
        rng_key = random.split(rng_key, num_restarts)
        for i in range(num_restarts):
            init = initializers.random_init_MultifidelityGP(rng_key[i], dim)
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
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb'])/(bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        XL, XH = batch['XL'], batch['XH']
        D = batch['XH'].shape[1]
        y = batch['y']
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[:D+1])
        theta_H = np.exp(params[D+1:-3])
        # Compute kernels
        k_pp = rho**2 * self.kernel(X_star, X_star, theta_L) + \
                        self.kernel(X_star, X_star, theta_H) + \
                        np.eye(X_star.shape[0])*(sigma_n_H + 1e-8)
        psi1 = rho*self.kernel(X_star, XL, theta_L)
        psi2 = rho**2 * self.kernel(X_star, XH, theta_L) + \
                        self.kernel(X_star, XH, theta_H)
        k_pX = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        return mu, std
    
    @partial(jit, static_argnums=(0,))
    def predict_L(self, X_star, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb'])/(bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        XL, XH = batch['XL'], batch['XH']
        D = batch['XH'].shape[1]
        y = batch['y']
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[:D+1])
        theta_H = np.exp(params[D+1:-3])
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta_L) + \
               np.eye(X_star.shape[0])*(sigma_n_L + 1e-8)
        psi1 = self.kernel(X_star, XL, theta_L)
        psi1 = rho*self.kernel(X_star, XH, theta_L)
        k_pX = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        return mu, std
    
    @partial(jit, static_argnums=(0,))
    def posterior_covariance_L(self, x, xp, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        x = (x - bounds['lb'])/(bounds['ub'] - bounds['lb'])
        xp = (xp - bounds['lb'])/(bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        XL, XH = batch['XL'], batch['XH']
        D = batch['XH'].shape[1]
        y = batch['y']
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[:D+1])
        theta_H = np.exp(params[D+1:-3])
        # Compute kernels
        k_pp = self.kernel(x, xp, theta_L)
        psi1 = self.kernel(x, XL, theta_L)
        psi2 = rho*self.kernel(x, XH, theta_L)
        k_pX = np.hstack((psi1,psi2))
        psi1 = self.kernel(XL, xp, theta_L)
        psi2 = rho*self.kernel(XH, xp, theta_L)
        k_Xp = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        # Compute predictive mean, std
        beta  = solve_triangular(L.T,solve_triangular(L, k_Xp, lower=True))
        cov = k_pp - np.matmul(k_pX, beta)
        return cov
    
    @partial(jit, static_argnums=(0,))
    def posterior_covariance_H(self, x, xp, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        x = (x - bounds['lb'])/(bounds['ub'] - bounds['lb'])
        xp = (xp - bounds['lb'])/(bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        XL, XH = batch['XL'], batch['XH']
        D = batch['XH'].shape[1]
        y = batch['y']
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[:D+1])
        theta_H = np.exp(params[D+1:-3])
        # Compute kernels
        k_pp = rho**2 * self.kernel(x, xp, theta_L) + \
                        self.kernel(x, xp, theta_H)
        psi1 = rho*self.kernel(x, XL, theta_L)
        psi2 = rho**2 * self.kernel(x, XH, theta_L) + \
                        self.kernel(x, XH, theta_H)
        k_pX = np.hstack((psi1,psi2))
        psi1 = rho*self.kernel(XL, xp, theta_L)
        psi2 = rho**2 * self.kernel(XH, xp, theta_L) + \
                        self.kernel(XH, xp, theta_H)
        k_Xp = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        # Compute predictive mean, std
        beta  = solve_triangular(L.T,solve_triangular(L, k_Xp, lower=True))
        cov = k_pp - np.matmul(k_pX, beta)
        return cov
  