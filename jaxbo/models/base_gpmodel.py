from typing import Any, Callable, Dict, Optional, Tuple

import numpy as onp
import jax.numpy as np
from jax import jit, vjp, vmap, random
from jax.random import split
from jax.scipy.linalg import solve_triangular
from functools import partial
from sklearn import mixture
from pyDOE import lhs
from abc import ABC

import jaxbo.kernels as kernels
import jaxbo.acquisitions as acquisitions
import jaxbo.utils as utils
from jaxbo.utils import fit_kernel_density
from jaxbo.optimizers import minimize_lbfgs_grad

SUPPORTED_KERNELS: Dict[str, Callable] = {
    'RBF': kernels.RBF,
    'Matern52': kernels.Matern52,
    'Matern32': kernels.Matern32,
    'Matern12': kernels.Matern12,
    'RatQuad': kernels.RatQuad
}

class GPmodel(ABC):
    def __init__(self, options: Dict):
        """
        Abstract base class for Gaussian Process models.

        This constructor initializes shared configuration and kernel selection logic
        for all derived Gaussian Process models. It assigns the input prior and
        selects the kernel function to be used based on the provided options.

        Args:
            options (dict): Dictionary of model configuration parameters. Must include:
                - 'input_prior': a prior distribution over the input space.
                - 'kernel': string specifying the kernel type to use. One of:
                    'RBF', 'Matern52', 'Matern32', 'Matern12', 'RatQuad', or None.
                    If None, defaults to 'RBF'.

        Raises:
            NotImplementedError: If the kernel name is not among the supported options.
        """

        self.options = options
        self.input_prior = options['input_prior']
        kernel_name = options.get('kernel', 'RBF')  # fallback to 'RBF' if None or missing

        if kernel_name not in SUPPORTED_KERNELS:
            raise NotImplementedError(
                f"Kernel '{kernel_name}' is not supported. "
                f"Choose from: {', '.join(SUPPORTED_KERNELS.keys())}"
            )
        self.kernel = SUPPORTED_KERNELS[kernel_name]

    @partial(jit, static_argnums=(0,))
    def likelihood(self, params: np.ndarray, batch: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute the negative log-marginal likelihood (NLML) for a given set of hyperparameters.

        This function evaluates how well a Gaussian Process with the given kernel parameters 
        explains the observed data. It uses the Cholesky decomposition of the covariance matrix 
        for numerical stability and efficiency.

        Args:
            params (np.ndarray): Log-transformed kernel parameters, including the noise term.
            batch (dict): A dictionary containing:
                - 'y' (np.ndarray): Training targets of shape (N, 1).
                - Any other data needed by `compute_cholesky`.

        Returns:
            np.ndarray: Scalar NLML value representing the data fit and model complexity.
        """

        y = batch['y']                     # Target observations
        N = y.shape[0]                     # Number of observations

        # Compute Cholesky decomposition of the kernel matrix
        L = self.compute_cholesky(params, batch)

        # Solve for alpha = K⁻¹y using triangular solver
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))

        # Compute the Negative Log-Marginal Likelihood (NLML)
        # Terms: data fit, log determinant, and normalization constant
        NLML = 0.5 * np.matmul(y.T, alpha) \
             + np.sum(np.log(np.diag(L))) \
             + 0.5 * N * np.log(2.0 * np.pi)

        return NLML

    @partial(jit, static_argnums=(0,))
    def likelihood_value_and_grad(
        self,
        params: np.ndarray,
        batch: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute both the value and the gradient of the negative log-marginal likelihood (NLML).

        This function uses reverse-mode automatic differentiation (via JAX's vjp)
        to obtain gradients of the NLML with respect to the kernel parameters.
        It is useful for hyperparameter optimization using gradient-based methods.

        Args:
            params (np.ndarray): Log-transformed kernel parameters (including noise term).
            batch (dict): Dictionary with training data, must include key 'y'.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - NLML value (scalar as 1-element array)
                - Gradient array with same shape as `params`

        Notes:
            - We use `vjp` instead of `value_and_grad` to reduce issues with NaNs in some cases.
            - If instability persists, consider clipping gradients or using `check_grads` for debugging.
        """

        # Define a closure for NLML computation
        fun = lambda p: self.likelihood(p, batch)

        # Compute the value and the backward pass function
        primals, f_vjp = vjp(fun, params)

        # Apply the VJP (vector-Jacobian product) to compute the gradient
        grads = f_vjp(np.ones_like(primals))[0]

        return primals, grads

    def fit_gmm(
            self,
            num_comp: int = 2,
            N_samples: int = 10000,
            **kwargs
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Fit a Gaussian Mixture Model (GMM) to reweight the prior distribution over inputs
            based on current model predictions. Used to enable prior-informed acquisition functions
            such as LW-LCB or LW-US.

            This is done by:
            1. Sampling uniformly over the input space.
            2. Evaluating model predictions.
            3. Estimating a kernel density for outputs.
            4. Reweighting by prior / posterior to prioritize informative regions.
            5. Sampling a new input set using the resulting importance weights.
            6. Fitting a GMM to this resampled set.

            Args:
                num_comp (int): Number of Gaussian components in the GMM.
                N_samples (int): Number of samples to use for reweighting and training the GMM.
                **kwargs:
                    - bounds (dict): Keys 'lb' and 'ub' defining lower and upper bounds of input domain.
                    - rng_key (jax.random.PRNGKey): JAX random seed key for reproducibility.
                    - norm_const (optional): Normalization constants, not used here but passed through.
                    - All other kwargs are forwarded to `self.predict`.

            Returns:
                Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    - weights: GMM component weights.
                    - means: Component means.
                    - covariances: Component covariance matrices.

            Notes:
                - This assumes self.predict returns only the predictive mean as [0]-th element.
                - LHS is used for more uniform coverage of the input domain.
            """

            bounds = kwargs['bounds']
            lb = bounds['lb']
            ub = bounds['ub']
            rng_key = kwargs['rng_key']
            dim = lb.shape[0]

            # Sample data uniformly over the full input space
            onp.random.seed(rng_key[0])
            X = lb + (ub - lb) * lhs(dim, N_samples)
            y = self.predict(X, **kwargs)[0]

            # Sample inputs according to the prior distribution
            rng_key = split(rng_key)[0]
            onp.random.seed(rng_key[0])
            X_samples = lb + (ub - lb) * lhs(dim, N_samples)
            y_samples = self.predict(X_samples, **kwargs)[0]

            # Estimate output densities from both prior and uniform samples
            p_x = self.input_prior.pdf(X)
            p_x_samples = self.input_prior.pdf(X_samples)
            p_y = fit_kernel_density(y_samples, y, weights=p_x_samples)

            # Importance weighting based on p(x)/p(y)
            weights = p_x / p_y
            weights /= np.sum(weights)  # Normalize to a valid probability distribution

            # Resample data points using computed weights
            indices = np.arange(N_samples)
            resample_idx = onp.random.choice(indices, N_samples, p=weights.flatten())
            X_train = (X[resample_idx] - lb) / (ub - lb)  # Scale to [0, 1]^D

            # Fit GMM to resampled inputs
            clf = mixture.GaussianMixture(n_components=num_comp, covariance_type='full')
            clf.fit(X_train)

            return clf.weights_, clf.means_, clf.covariances_

    @partial(jit, static_argnums=(0,))
    def acquisition(self, x: np.ndarray, **kwargs: Any) -> float:
        """
        Compute the acquisition value for a given input x using the specified criterion.

        Supports various acquisition strategies, both standard and prior-weighted.
        Normalization and denormalization of predictions is applied as needed.

        Args:
            x (np.ndarray): Input point (1D array).
            **kwargs:
                - params: Optimized hyperparameters of the model.
                - batch: Training data (normalized).
                - norm_const: Normalization constants.
                - bounds: Dictionary with 'lb' and 'ub' keys for domain.
                - rng_key: PRNGKey for random sampling.
                - gmm_vars (optional): GMM weights, means, covariances.
                - kappa (optional): Trade-off parameter for UCB-type criteria.

        Returns:
            float: Acquisition value (to be minimized).
        """
        # Expand x to match (1, D) expected shape
        x = x[None, :]

        # Predict mean and std for current input
        mean, std = self.predict(x, **kwargs)
        criterion = self.options['criterion']
        
        def lcb_wrapped():
            kappa = kwargs['kappa']
            return acquisitions.LCB(mean, std, kappa)

        def lw_lcb_wrapped():
            kappa = kwargs['kappa']
            weights = utils.compute_w_gmm(x, **kwargs)
            return acquisitions.LW_LCB(mean, std, weights, kappa)

        def ei_wrapped():
            y_batch = kwargs['batch']['y']
            best = np.min(y_batch)
            return acquisitions.EI(mean, std, best)

        def us_wrapped():
            return acquisitions.US(std)

        def ts_wrapped():
            return self.draw_posterior_sample(x, **kwargs)

        def lw_us_wrapped():
            weights = utils.compute_w_gmm(x, **kwargs)
            return acquisitions.LW_US(std, weights)

        def clsf_wrapped():
            kappa = kwargs['kappa']
            norm_const = kwargs['norm_const']
            denorm_mean = mean * norm_const['sigma_y'] + norm_const['mu_y']
            denorm_std = std * norm_const['sigma_y']
            return acquisitions.CLSF(denorm_mean, denorm_std, kappa)

        def lw_clsf_wrapped():
            kappa = kwargs['kappa']
            norm_const = kwargs['norm_const']
            denorm_mean = mean * norm_const['sigma_y'] + norm_const['mu_y']
            denorm_std = std * norm_const['sigma_y']
            weights = utils.compute_w_gmm(x, **kwargs)
            return acquisitions.LW_CLSF(denorm_mean, denorm_std, weights, kappa)

        def imse_wrapped():
            rng_key = kwargs['rng_key']
            bounds = kwargs['bounds']
            lb, ub = bounds['lb'], bounds['ub']
            dim = lb.shape[0]
            xp = lb + (ub - lb) * random.uniform(rng_key, (10000, dim))
            cov = self.posterior_covariance(x, xp, **kwargs)
            return np.mean(cov**2) / std**2

        def imse_l_wrapped():
            rng_key = kwargs['rng_key']
            bounds = kwargs['bounds']
            lb, ub = bounds['lb'], bounds['ub']
            dim = lb.shape[0]
            _, std_L = self.predict_L(x, **kwargs)
            xp = lb + (ub - lb) * random.uniform(rng_key, (10000, dim))
            cov = self.posterior_covariance_L(x, xp, **kwargs)
            return np.mean(cov**2) / std_L**2

        # Dispatch table
        ACQUISITION_HANDLERS: Dict[str, Callable[[], float]] = {
            'LCB': lcb_wrapped,
            'LW-LCB': lw_lcb_wrapped,
            'EI': ei_wrapped,
            'US': us_wrapped,
            'TS': ts_wrapped,
            'LW-US': lw_us_wrapped,
            'CLSF': clsf_wrapped,
            'LW_CLSF': lw_clsf_wrapped,
            'IMSE': imse_wrapped,
            'IMSE_L': imse_l_wrapped,
        }

        if criterion not in ACQUISITION_HANDLERS:
            raise NotImplementedError(f"Acquisition criterion '{criterion}' is not supported.")

        return ACQUISITION_HANDLERS[criterion]()

    @partial(jit, static_argnums=(0,))
    def acq_value_and_grad(self, x: np.ndarray, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the acquisition function value and its gradient at a given input point x.

        This method uses reverse-mode autodiff (via `vjp`) to efficiently compute the gradient
        of the acquisition function with respect to input `x`.

        Args:
            x (np.ndarray): Input array of shape (D,), representing a single point in input space.
            **kwargs (dict): Additional arguments required by the acquisition function,
                            e.g., model parameters, bounds, priors, etc.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - primals: The acquisition function value at `x`.
                - grads: Gradient of the acquisition function with respect to `x`.
        """
        # Define acquisition function as a function of x
        def acquisition_fn(xi: np.ndarray) -> np.ndarray:
            return self.acquisition(xi, **kwargs)

        # Compute value and vector-Jacobian product (reverse-mode gradient)
        primals, f_vjp = vjp(acquisition_fn, x)
        grads = f_vjp(np.ones_like(primals))[0]

        return primals, grads

    def compute_next_point_lbfgs(self, num_restarts: int = 10, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Optimize the acquisition function using L-BFGS-B with multiple random restarts.

        This method searches for the input location that minimizes the acquisition function
        by performing multiple L-BFGS-B optimizations from different initializations
        within the input bounds.

        Args:
            num_restarts (int): Number of random initializations for multi-start optimization.
            **kwargs (dict): Dictionary containing required elements such as:
                - 'bounds': {'lb': np.ndarray, 'ub': np.ndarray}
                - 'rng_key': random key for reproducibility
                - other parameters required by the acquisition function

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - x_new: Best point found, shape (1, D)
                - acq: Acquisition values for each restart, shape (num_restarts, 1)
                - loc: Locations tested, shape (num_restarts, D)
        """

        def objective(x: np.ndarray) -> Tuple[onp.ndarray, onp.ndarray]:
            """Objective function wrapper that converts JAX arrays to NumPy for the optimizer."""
            value, grads = self.acq_value_and_grad(x, **kwargs)
            return onp.array(value), onp.array(grads)

        # Extract bounds and dimensionality
        bounds = kwargs['bounds']
        lb, ub = bounds['lb'], bounds['ub']
        dim = lb.shape[0]

        # Generate initial points using Latin Hypercube Sampling
        rng_key = kwargs['rng_key']
        onp.random.seed(rng_key[0])  # Deterministic initialization
        initial_points = lb + (ub - lb) * lhs(dim, num_restarts)

        # Format bounds for SciPy optimizer
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))

        # Perform L-BFGS-B optimization from each starting point
        solutions = []
        scores = []
        for i in range(num_restarts):
            pos, val = minimize_lbfgs_grad(objective, initial_points[i, :], bnds=dom_bounds)
            solutions.append(pos)
            scores.append(val)

        loc = np.vstack(solutions)  # Shape: (num_restarts, D)
        acq = np.vstack(scores)     # Shape: (num_restarts, 1)

        # Select the point with the best acquisition score
        idx_best = np.argmin(acq)
        x_new = loc[idx_best:idx_best + 1, :]  # Shape: (1, D)

        return x_new, acq, loc

    def compute_next_point_gs(self, X_cand: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Selects the next point to evaluate by evaluating the acquisition function
        over a grid or set of candidate points and picking the one with the minimum value.

        This method is useful when working with a precomputed candidate set (e.g., grid search).

        Args:
            X_cand (np.ndarray): Array of candidate points, shape (N, D).
            **kwargs (dict): Additional arguments passed to the acquisition function.

        Returns:
            np.ndarray: The candidate point with the best acquisition value, shape (1, D).
        """

        # Vectorize acquisition function over candidate points
        acq_values = vmap(lambda x: self.acquisition(x, **kwargs))(X_cand)

        # Select the candidate with the lowest acquisition value
        best_index = np.argmin(acq_values)
        x_new = X_cand[best_index:best_index + 1, :]  # Keep 2D shape

        return x_new
