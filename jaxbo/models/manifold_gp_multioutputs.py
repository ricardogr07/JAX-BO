import numpy as onp
import jax.numpy as np
from jax import jit, vjp, random
from jax.scipy.linalg import cholesky, solve_triangular
from jax.flatten_util import ravel_pytree
from functools import partial

import jaxbo.acquisitions as acquisitions
import jaxbo.initializers as initializers
import jaxbo.utils as utils
from jaxbo.optimizers import minimize_lbfgs
from pyDOE import lhs

from jaxbo.models.base_gpmodel import GPmodel

class ManifoldGP_MultiOutputs(GPmodel):
    # Initialize the class
    def __init__(self, options, layers):
        super().__init__(options)
        self.layers = layers
        self.net_init, self.net_apply = utils.init_NN(layers)
        # Determine parameter IDs
        nn_params = self.net_init(random.PRNGKey(0), (-1, layers[0]))[1]
        nn_params_flat, self.unravel = ravel_pytree(nn_params)
        num_nn_params = len(nn_params_flat)
        num_gp_params = initializers.random_init_GP(random.PRNGKey(0), layers[-1]).shape[0]
        self.gp_params_ids = np.arange(num_gp_params)
        self.nn_params_ids = np.arange(num_nn_params) + num_gp_params

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        # Warp inputs
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        X = self.net_apply(nn_params, batch['X'])
        N = X.shape[0]
        # Fetch params
        sigma_n = np.exp(gp_params[-1])
        theta = np.exp(gp_params[:-1])
        # Compute kernel
        K = self.kernel(X, X, theta) + np.eye(N)*(sigma_n + 1e-8)
        L = cholesky(K, lower=True)
        return L

    def train(self, batch_list, rng_key, num_restarts = 10):
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
            dim = batch['X'].shape[1]
            rng_keys = random.split(rng_key, num_restarts)
            for i in range(num_restarts):
                key1, key2 = random.split(rng_keys[i])
                gp_params = initializers.random_init_GP(key1, dim)
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
            best_params.append(params[idx_best,:])

        return best_params
    
    @partial(jit, static_argnums=(0,))
    def predict_all(self, X_star, **kwargs):
        mu_list = []
        std_list = []
        
        params_list =  kwargs['params']
        batch_list = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const_list = kwargs['norm_const']
        zipped_args = zip(params_list, batch_list, norm_const_list)
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb'])/(bounds['ub'] - bounds['lb'])
        
        for k, (params, batch, norm_const) in enumerate(zipped_args):
            # Fetch normalized training data
            X, y = batch['X'], batch['y']
            # Warp inputs
            gp_params = params[self.gp_params_ids]
            nn_params = self.unravel(params[self.nn_params_ids])
            X = self.net_apply(nn_params, X)
            X_star_nn = self.net_apply(nn_params, X_star)
            # Fetch params
            sigma_n = np.exp(gp_params[-1])
            theta = np.exp(gp_params[:-1])
            # Compute kernels
            k_pp = self.kernel(X_star_nn, X_star_nn, theta) + np.eye(X_star_nn.shape[0])*(sigma_n + 1e-8)
            k_pX = self.kernel(X_star_nn, X, theta)
            L = self.compute_cholesky(params, batch)
            alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
            beta  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
            # Compute predictive mean, std
            mu = np.matmul(k_pX, alpha)
            cov = k_pp - np.matmul(k_pX, beta)
            std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
            
            mu_list.append(mu)
            std_list.append(std)
            
        return np.array(mu_list), np.array(std_list)
    
    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        
        params =  kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb'])/(bounds['ub'] - bounds['lb'])
        
        # Fetch normalized training data
        X, y = batch['X'], batch['y']
        # Warp inputs
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        X = self.net_apply(nn_params, X)
        X_star_nn = self.net_apply(nn_params, X_star)
        # Fetch params
        sigma_n = np.exp(gp_params[-1])
        theta = np.exp(gp_params[:-1])
        # Compute kernels
        k_pp = self.kernel(X_star_nn, X_star_nn, theta) + np.eye(X_star_nn.shape[0])*(sigma_n + 1e-8)
        k_pX = self.kernel(X_star_nn, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
            
        return mu, std
    
    @partial(jit, static_argnums=(0,))
    def constrained_acquisition(self, x, **kwargs):
        x = x[None,:]
        mean, std = self.predict_all(x, **kwargs)
        if self.options['constrained_criterion'] == 'EIC':
            batch_list = kwargs['batch']
            best = np.min(batch_list[0]['y'])
            return acquisitions.EIC(mean, std, best)
        elif self.options['constrained_criterion'] == 'LCBC':
            kappa = kwargs['kappa']
            ##### normalize the mean and std again and subtract the mean by 2*sigma
            #norm_const = kwargs['norm_const'][0]
            #mean[0,:] = (mean[0,:] - norm_const['mu_y']) / norm_const['sigma_y'] - 3 * norm_const['sigma_y']
            #std[0,:] = std[0,:] / norm_const['sigma_y']
            #####
            return acquisitions.LCBC(mean, std, kappa)
        elif self.options['constrained_criterion'] == 'LW_LCBC':
            kappa = kwargs['kappa']
            ##### normalize the mean and std again and subtract the mean by 3*sigma
            weights = utils.compute_w_gmm(x, **kwargs)
            return acquisitions.LW_LCBC(mean, std, weights, kappa)
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def constrained_acq_value_and_grad(self, x, **kwargs):
        fun = lambda x: self.constrained_acquisition(x, **kwargs)
        primals, f_vjp = vjp(fun, x)
        grads = f_vjp(np.ones_like(primals))[0]
        return primals, grads

    def constrained_compute_next_point_lbfgs(self, num_restarts = 10, **kwargs):
        # Define objective that returns NumPy arrays
        def objective(x):
            value, grads = self.constrained_acq_value_and_grad(x, **kwargs)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        loc = []
        acq = []
        bounds = kwargs['bounds']
        lb = bounds['lb']
        ub = bounds['ub']
        rng_key = kwargs['rng_key']
        dim = lb.shape[0]

        onp.random.seed(rng_key[0])
        x0 = lb + (ub-lb)*lhs(dim, num_restarts)
        #print("x0 for bfgs", x0)
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))
        for i in range(num_restarts):
            pos, val = minimize_lbfgs(objective, x0[i,:], bnds = dom_bounds)
            loc.append(pos)
            acq.append(val)
        loc = np.vstack(loc)
        acq = np.vstack(acq)
        idx_best = np.argmin(acq)
        x_new = loc[idx_best:idx_best+1,:]
        return x_new, acq, loc
 