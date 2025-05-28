import numpy as onp
import jax.numpy as np
from jax import jit, random
from jax.scipy.linalg import cholesky, solve_triangular
from jax.flatten_util import ravel_pytree
from jax.scipy.special import expit as sigmoid
from functools import partial

import jaxbo.initializers as initializers
import jaxbo.utils as utils
from jaxbo.optimizers import minimize_lbfgs
from jaxbo.models.base_gpmodel import GPmodel

class HeterogeneousMultifidelityGP(GPmodel):

    def __init__(self, options, layers):
        super().__init__(options)
        self.layers = layers
        self.net_init, self.net_apply = utils.init_NN(layers)
        # Determine parameter IDs
        nn_params = self.net_init(random.PRNGKey(0), (-1, layers[0]))[1]
        nn_params_flat, self.unravel = ravel_pytree(nn_params)
        num_nn_params = len(nn_params_flat)
        num_gp_params = initializers.random_init_MultifidelityGP(random.PRNGKey(0), layers[-1]).shape[0]
        self.gp_params_ids = np.arange(num_gp_params)
        self.nn_params_ids = np.arange(num_nn_params) + num_gp_params

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        XL, XH = batch['XL'], batch['XH']
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
        theta_L = gp_params[:D+1]
        theta_H = gp_params[D+1:-3]
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
            key1, key2 = random.split(rng_key[i])
            gp_params = initializers.random_init_MultifidelityGP(key1, dim)
            nn_params = self.net_init(key2,  (-1, self.layers[0]))[1]
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
#        idx_best = np.argmin(likelihood)
        #print("best index: {}".format(idx_best))
        best_params = params[idx_best,:]
        #print("shape of parameters: {}".format(best_params.shape))

        #### modify it according to the jupyter notebook
        #### Maybe jax will not stop when it see NaNs or Infs
        #print('Likelihoods for current iteration: {}'.format(likelihood.flatten()))
        #print('Best likelihood for current iteration: {}'.format(likelihood[idx_best]))
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
        # Warp low-fidelity inputs
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        XL = sigmoid(self.net_apply(nn_params, XL))
        # Fetch params
        rho = gp_params[-3]
        sigma_n_L = np.exp(gp_params[-2])
        sigma_n_H = np.exp(gp_params[-1])
        theta_L = gp_params[:D+1]
        theta_H = gp_params[D+1:-3]
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
