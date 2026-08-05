"""Core Gaussian process models: the ``GPmodel`` base class and the exact ``GP``.

This module is the heart of the jaxbo core (SCOPE.md section 3). It must stay
importable with only the core dependencies (jax, jaxlib, numpy, scipy): the
weighted-sampling surface (``fit_gmm``, the ``LW_*`` acquisition branches)
lives in :mod:`jaxbo.weights` (the ``[weighted]`` extra) and is imported
lazily inside the methods that need it, never at module level.
"""

from abc import ABC
from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax.numpy as np
import numpy as onp
from jax import jit, random, vjp, vmap
from jax.random import PRNGKey
from jax.scipy.linalg import cholesky, solve_triangular
from scipy.stats import qmc

import jaxbo.acquisitions as acquisitions
import jaxbo.kernels as kernels
from jaxbo import initializers
from jaxbo.optimizers import minimize_bfgs_jax, minimize_lbfgs_grad


SUPPORTED_KERNELS: Dict[str, Callable] = {
    "RBF": kernels.RBF,
    "Matern52": kernels.Matern52,
    "Matern32": kernels.Matern32,
    "Matern12": kernels.Matern12,
    "RatQuad": kernels.RatQuad,
}


class GPmodel(ABC):
    """Abstract base class shared by every jaxbo Gaussian process model.

    Provides the negative log-marginal likelihood, acquisition dispatch, and
    next-point selection machinery; concrete subclasses implement
    ``compute_cholesky``, ``train``, and ``predict``.
    """

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
        self.input_prior = options["input_prior"]
        kernel_name = options.get(
            "kernel", "RBF"
        )  # fallback to 'RBF' if None or missing

        if kernel_name not in SUPPORTED_KERNELS:
            raise NotImplementedError(
                f"Kernel '{kernel_name}' is not supported. "
                f"Choose from: {', '.join(SUPPORTED_KERNELS.keys())}"
            )
        self.kernel = SUPPORTED_KERNELS[kernel_name]

    @partial(jit, static_argnums=(0,))
    def likelihood(
        self, params: np.ndarray, batch: Dict[str, np.ndarray]
    ) -> np.ndarray:
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

        y = batch["y"]  # Target observations
        N = y.shape[0]  # Number of observations

        # Compute Cholesky decomposition of the kernel matrix
        L = self.compute_cholesky(params, batch)

        # Solve for alpha = K⁻¹y using triangular solver
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))

        # Compute the Negative Log-Marginal Likelihood (NLML)
        # Terms: data fit, log determinant, and normalization constant
        NLML = (
            0.5 * np.matmul(y.T, alpha)
            + np.sum(np.log(np.diag(L)))
            + 0.5 * N * np.log(2.0 * np.pi)
        )

        return NLML

    @partial(jit, static_argnums=(0,))
    def likelihood_value_and_grad(
        self, params: np.ndarray, batch: Dict[str, np.ndarray]
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
        def fun(p):
            return self.likelihood(p, batch)

        # Compute the value and the backward pass function
        primals, f_vjp = vjp(fun, params)

        # Apply the VJP (vector-Jacobian product) to compute the gradient
        grads = f_vjp(np.ones_like(primals))[0]

        return primals, grads

    def fit_gmm(
        self, num_comp: int = 2, N_samples: int = 10000, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit a GMM that reweights the input prior by the model's predictions.

        Delegates to :func:`jaxbo.weights.fit_gmm` (the [weighted] extra);
        calling it without scikit-learn and KDEpy installed raises an
        ImportError naming ``pip install jaxbo[weighted]``. See that function
        for the full contract.

        Args:
            num_comp (int): Number of Gaussian components in the GMM.
            N_samples (int): Number of samples used for reweighting and
                training the GMM.
            **kwargs: 'bounds' and 'rng_key' plus everything ``self.predict``
                needs.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: GMM component weights,
            means, and covariance matrices.
        """
        # Lazy import: the whole gmm_vars surface lives behind the [weighted]
        # extra (SCOPE.md decision 7); the core import graph never reaches
        # scikit-learn or KDEpy.
        from jaxbo.weights import fit_gmm

        return fit_gmm(self, num_comp, N_samples, **kwargs)

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
        criterion = self.options["criterion"]

        def lcb_wrapped():
            kappa = kwargs["kappa"]
            return acquisitions.LCB(mean, std, kappa)

        def lw_lcb_wrapped():
            # Lazy import: gmm_vars flows live behind the [weighted] extra
            # (SCOPE.md decision 7) and raise its install hint when missing.
            from jaxbo.weights import compute_w_gmm

            kappa = kwargs["kappa"]
            weights = compute_w_gmm(x, **kwargs)
            return acquisitions.LW_LCB(mean, std, weights, kappa)

        def ei_wrapped():
            y_batch = kwargs["batch"]["y"]
            best = np.min(y_batch)
            return acquisitions.EI(mean, std, best)

        def us_wrapped():
            return acquisitions.US(std)

        def ts_wrapped():
            return self.draw_posterior_sample(x, **kwargs)

        def lw_us_wrapped():
            # Lazy import: see lw_lcb_wrapped.
            from jaxbo.weights import compute_w_gmm

            weights = compute_w_gmm(x, **kwargs)
            return acquisitions.LW_US(std, weights)

        def clsf_wrapped():
            kappa = kwargs["kappa"]
            norm_const = kwargs["norm_const"]
            denorm_mean = mean * norm_const["sigma_y"] + norm_const["mu_y"]
            denorm_std = std * norm_const["sigma_y"]
            return acquisitions.CLSF(denorm_mean, denorm_std, kappa)

        def lw_clsf_wrapped():
            # Lazy import: see lw_lcb_wrapped.
            from jaxbo.weights import compute_w_gmm

            kappa = kwargs["kappa"]
            norm_const = kwargs["norm_const"]
            denorm_mean = mean * norm_const["sigma_y"] + norm_const["mu_y"]
            denorm_std = std * norm_const["sigma_y"]
            weights = compute_w_gmm(x, **kwargs)
            return acquisitions.LW_CLSF(denorm_mean, denorm_std, weights, kappa)

        def imse_wrapped():
            rng_key = kwargs["rng_key"]
            bounds = kwargs["bounds"]
            lb, ub = bounds["lb"], bounds["ub"]
            dim = lb.shape[0]
            xp = lb + (ub - lb) * random.uniform(rng_key, (10000, dim))
            cov = self.posterior_covariance(x, xp, **kwargs)
            return np.mean(cov**2) / std**2

        def imse_l_wrapped():
            rng_key = kwargs["rng_key"]
            bounds = kwargs["bounds"]
            lb, ub = bounds["lb"], bounds["ub"]
            dim = lb.shape[0]
            _, std_L = self.predict_L(x, **kwargs)
            xp = lb + (ub - lb) * random.uniform(rng_key, (10000, dim))
            cov = self.posterior_covariance_L(x, xp, **kwargs)
            return np.mean(cov**2) / std_L**2

        # Dispatch table
        ACQUISITION_HANDLERS: Dict[str, Callable[[], float]] = {
            "LCB": lcb_wrapped,
            "LW-LCB": lw_lcb_wrapped,
            "EI": ei_wrapped,
            "US": us_wrapped,
            "TS": ts_wrapped,
            "LW-US": lw_us_wrapped,
            "CLSF": clsf_wrapped,
            "LW_CLSF": lw_clsf_wrapped,
            "IMSE": imse_wrapped,
            "IMSE_L": imse_l_wrapped,
        }

        if criterion not in ACQUISITION_HANDLERS:
            raise NotImplementedError(
                f"Acquisition criterion '{criterion}' is not supported."
            )

        return ACQUISITION_HANDLERS[criterion]()

    @partial(jit, static_argnums=(0,))
    def acq_value_and_grad(
        self, x: np.ndarray, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def _batched_start_scorer(self, **kwargs: Any):
        """Map the model criterion to a batched candidate scorer, if one exists.

        :func:`jaxbo.acquisitions.score_candidates` can rank a whole candidate
        batch in one dispatch for criteria that depend only on the shared
        predictive mean and std (EI, LCB, US, CLSF). Criteria that need
        per-candidate state it cannot express (the LW-* weights, TS posterior
        draws, the IMSE covariance scans) return None and keep the serial
        multi-start path. Only the exact :class:`GP` model qualifies: the
        multifidelity and MCMC subclasses have ``predict`` contracts beyond
        the ``params, batch, bounds`` signature score_candidates calls.

        Args:
            **kwargs: The compute_next_point_lbfgs kwargs; the same keys the
                serial acquisition path reads for the criterion (e.g. 'kappa'
                for LCB, 'norm_const' for CLSF) are required here.

        Returns:
            Tuple[Callable, Dict] of (acq_fn, shared acq kwargs) for
            score_candidates, or None when the criterion (or model class)
            needs the serial path.
        """
        # Exact type, not isinstance: a user subclass may override predict
        # (score_candidates forwards only params, batch, bounds) or
        # acquisition (its starts would be ranked by the stock criterion
        # instead of the override). Subclasses keep the serial path.
        if type(self) is not GP:
            return None
        criterion = self.options["criterion"]
        if criterion == "EI":
            return acquisitions.EI, {"best": np.min(kwargs["batch"]["y"])}
        if criterion == "LCB":
            return acquisitions.LCB, {"kappa": kwargs["kappa"]}
        if criterion == "US":
            return lambda mean, std: acquisitions.US(std), {}
        if criterion == "CLSF":
            norm_const = kwargs["norm_const"]

            def clsf_denorm(mean, std, kappa):
                denorm_mean = mean * norm_const["sigma_y"] + norm_const["mu_y"]
                denorm_std = std * norm_const["sigma_y"]
                return acquisitions.CLSF(denorm_mean, denorm_std, kappa)

            return clsf_denorm, {"kappa": kwargs["kappa"]}
        return None

    def compute_next_point_lbfgs(
        self, num_restarts: int = 10, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Optimize the acquisition function using batched-start L-BFGS-B.

        ``num_restarts`` remains the multi-start budget. When the criterion
        supports batched scoring (see :meth:`_batched_start_scorer`), the
        budget maps to ``32 * num_restarts`` Latin hypercube candidates
        scored in ONE vectorized dispatch, of which only the
        ``k = min(num_restarts, max(2, num_restarts // 5))`` best become
        L-BFGS-B polish starts. That preserves (and widens) the exploration
        the serial path bought with ``num_restarts`` blind polishes while
        paying the expensive per-step host round-trip of the scipy optimizer
        k times instead of num_restarts times. Criteria without a batched
        scorer keep the exact previous behavior: num_restarts LHS starts,
        one polish each.

        The polish loop itself stays serial on purpose: the bounded
        optimizer is scipy L-BFGS-B (:func:`jaxbo.optimizers.minimize_lbfgs_grad`),
        which runs on the host and cannot be vmapped on device; a
        device-native bounded quasi-Newton would need a new dependency.
        Reducing the number of polished starts via one batched scoring
        dispatch is the vectorization this path gets instead.

        Args:
            num_restarts (int): Multi-start budget. On the batched path it
                sets the scoring batch (32x) and the polish count k as above;
                on the serial path it is the number of polishes, as before.
            **kwargs (dict): Dictionary containing required elements such as:
                - 'bounds': {'lb': np.ndarray, 'ub': np.ndarray}
                - 'rng_key': random key for reproducibility
                - other parameters required by the acquisition function

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - x_new: Best point found, shape (1, D)
                - acq: Acquisition values per polished start, shape (m, 1)
                - loc: Polished locations, shape (m, D)
                where m == k on the batched-start path and
                m == num_restarts on the serial path.
        """

        def objective(x: np.ndarray) -> Tuple[onp.ndarray, onp.ndarray]:
            """Objective function wrapper that converts JAX arrays to NumPy for the optimizer."""
            value, grads = self.acq_value_and_grad(x, **kwargs)
            return onp.array(value), onp.array(grads)

        # Extract bounds and dimensionality
        bounds = kwargs["bounds"]
        lb, ub = bounds["lb"], bounds["ub"]
        dim = lb.shape[0]

        # Generate starting points using Latin Hypercube Sampling
        rng_key = kwargs["rng_key"]
        onp.random.seed(rng_key[0])  # Deterministic initialization
        sampler = qmc.LatinHypercube(d=dim, seed=int(rng_key[0]))

        scorer = self._batched_start_scorer(**kwargs)
        if scorer is None:
            starts = lb + (ub - lb) * sampler.random(num_restarts)
        else:
            acq_fn, acq_kwargs = scorer
            candidates = lb + (ub - lb) * sampler.random(32 * num_restarts)
            cand_scores = acquisitions.score_candidates(
                self,
                candidates,
                params=kwargs["params"],
                batch=kwargs["batch"],
                bounds=bounds,
                acq_fn=acq_fn,
                **acq_kwargs,
            )
            k = min(num_restarts, max(2, num_restarts // 5))
            # Never rank a non-finite score as best. On a near-singular model
            # the criterion NaNs at variance-clipped candidates, and those are
            # exactly the starts a polish cannot recover from; the serial path
            # diluted that risk across num_restarts blind starts, k of them
            # cannot. Sorting them last costs nothing when every score is
            # finite, and when none is, the order is arbitrary either way and
            # the all-failed guard below catches it.
            cand_scores = np.where(np.isfinite(cand_scores), cand_scores, np.inf)
            starts = candidates[np.argsort(cand_scores)[:k]]

        # Format bounds for SciPy optimizer
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))

        # Perform L-BFGS-B optimization from each starting point
        solutions = []
        scores = []
        for i in range(starts.shape[0]):
            pos, val = minimize_lbfgs_grad(objective, starts[i, :], bnds=dom_bounds)
            solutions.append(pos)
            scores.append(val)

        loc = np.vstack(solutions)  # Shape: (m, D)
        acq = np.vstack(scores)  # Shape: (m, 1)

        # Select the best acquisition score. nanargmin, like train(): a
        # polish can return NaN where the acquisition gradient NaNs at a
        # variance-clipped point, and argmin would select that row.
        #
        # EVERY polish failing is reachable, and more so on the batched path,
        # which polishes k starts instead of num_restarts (k = 2 at the
        # default budget, so two NaNs is all it takes). nanargmin has no
        # answer there: it returns an out-of-range index, and the caller gets
        # a silently EMPTY (0, D) point rather than an error. Fall back to
        # the best start instead, which is finite and in bounds by
        # construction: the top-ranked scored candidate on the batched path,
        # the first LHS draw on the serial one. A start beats both a NaN and
        # an empty array, and a BO loop keeps running.
        if bool(np.any(np.isfinite(acq))):
            idx_best = np.nanargmin(acq)
            x_new = loc[idx_best : idx_best + 1, :]  # Shape: (1, D)
        else:
            x_new = starts[0:1, :]  # Shape: (1, D)

        return x_new, acq, loc

    def compute_next_point_gs(self, X_cand: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Selects the next point to evaluate by evaluating the acquisition function
        over a grid or set of candidate points and picking the one with the minimum value.

        This method is useful when working with a precomputed candidate set (e.g., grid search).

        Note:
            This is deliberately NOT routed through
            :func:`jaxbo.acquisitions.score_candidates`: the vmap over the
            jitted ``self.acquisition`` is already a single batched dispatch
            (candidate-independent work such as the Cholesky factor is not
            batched by vmap), and it supports every criterion, including the
            LW-*, TS, and IMSE ones score_candidates cannot express.

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
        x_new = X_cand[best_index : best_index + 1, :]  # Keep 2D shape

        return x_new


class GP(GPmodel):
    """Exact Gaussian process regression model for Bayesian optimization.

    Hyperparameters are optimized by multi-start BFGS on the negative
    log-marginal likelihood, run fully on device as one jitted, vmapped
    computation across restarts.

    Warning:
        Normalization contract (SCOPE.md section 2), the two halves are
        asymmetric on purpose and mixing them up fails silently:

        - ``train`` consumes ``batch`` exactly as given: pass an ALREADY
          NORMALIZED batch, typically the output of
          :func:`jaxbo.utils.normalize` (inputs scaled to the unit cube
          against the domain bounds, targets standardized).
        - ``predict`` takes RAW domain points ``X_star`` and normalizes them
          internally against ``bounds``.

    Note:
        Instances compare equal when their live configuration matches: same
        concrete class, kernel function, ``criterion``, and ``input_prior``
        object (priors compare by identity and are never hashed, so
        unhashable priors keep working). The jitted methods take ``self``
        as a static argument, so jax keys its compilation cache on this
        equality: a second ``GP(...)`` with the same configuration reuses
        every compilation the first one paid for instead of recompiling per
        instance. The key is read live at every call, so rebinding
        ``options["criterion"]`` after construction moves the instance to
        its own fresh cache line instead of hitting a stale one.
    """

    def __init__(self, options: Dict[str, Any]):
        """Initialize a standard Gaussian Process model.

        Args:
            options: Model configuration dictionary (see :class:`GPmodel`),
                with keys such as 'kernel', 'input_prior', and 'criterion'.
        """
        super().__init__(options)

    def _jit_cache_key(self) -> Tuple[Any, ...]:
        # Everything the traced methods read from the instance, resolved
        # live at call time so post-construction rebinding rekeys instead
        # of hitting a stale compiled cache line. This must stay in sync
        # with what the jitted methods read from self, or equal instances
        # with divergent behavior would silently share compiled code. The
        # prior participates by identity and is never hashed: it enters no
        # trace, and id() keeps unhashable priors usable. Cache entries
        # hold their GP (and so its prior) alive, so a live id cannot be
        # recycled into a false match.
        return (
            type(self),
            self.kernel,
            self.options.get("criterion"),
            None if self.input_prior is None else id(self.input_prior),
        )

    def __hash__(self) -> int:
        return hash(self._jit_cache_key())

    def __eq__(self, other: object) -> Any:
        if not isinstance(other, GP):
            return NotImplemented
        return self._jit_cache_key() == other._jit_cache_key()

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(
        self, params: np.ndarray, batch: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute the Cholesky decomposition of the kernel matrix.

        Args:
            params: Log-transformed kernel parameters (including noise term).
            batch: Dictionary containing normalized training inputs 'X'.

        Returns:
            Lower-triangular matrix from Cholesky decomposition.
        """
        X = batch["X"]
        N, D = X.shape
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])
        K = self.kernel(X, X, theta) + np.eye(N) * (sigma_n + 1e-8)
        return cholesky(K, lower=True)

    def train(
        self,
        batch: Dict[str, np.ndarray],
        rng_key: PRNGKey,
        num_restarts: int = 10,
    ) -> np.ndarray:
        """
        Optimize GP hyperparameters using multi-start BFGS, fully on device.

        Each restart draws a random initialization and minimizes the negative
        log-marginal likelihood with unbounded BFGS. All restarts run inside
        a single jitted, vmapped computation (:meth:`_train_multistart`), so
        no host-device round trip happens per optimizer step. The best
        restart is selected with a NaN-aware argmin, so a restart that fails
        (for example a Cholesky breakdown at a degenerate initialization)
        cannot poison the result.

        Args:
            batch: Dictionary with 'X' and 'y'. See the warning below for the
                normalization this data must already carry.
            rng_key: PRNGKey for reproducibility. The same key always yields
                the same initializations and therefore the same result.
            num_restarts: Number of random initializations.

        Returns:
            Best hyperparameters found (array of shape (dim + 2,)).

        Raises:
            RuntimeError: If no restart produced a finite likelihood value,
                which would otherwise silently return meaningless parameters.

        Warning:
            ``batch`` is consumed exactly as given: it must be ALREADY
            NORMALIZED by the caller, typically via
            :func:`jaxbo.utils.normalize` (inputs 'X' scaled to the unit cube
            against the domain bounds, targets 'y' standardized). This is
            asymmetric with :meth:`predict`, which takes RAW domain points and
            normalizes them internally against ``bounds``. Training on raw
            inputs produces silently wrong results: no error is raised.
        """
        dim = batch["X"].shape[1]
        rng_keys = random.split(rng_key, num_restarts)
        inits = np.stack([initializers.random_init_GP(k, dim) for k in rng_keys])
        params, values = self._train_multistart(inits, batch)
        idx_best = np.nanargmin(values)
        if not bool(np.isfinite(values[idx_best])):
            raise RuntimeError(
                "GP.train: every restart produced a non-finite likelihood; "
                "check the scaling and normalization of the training data."
            )
        return params[idx_best, :]

    @partial(jit, static_argnums=(0,))
    def _train_multistart(
        self, inits: np.ndarray, batch: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run one unbounded BFGS likelihood minimization per initialization.

        The whole multi-start optimization is one jitted computation: the
        solver is vmapped over the stacked initializations, so every optimizer
        step of every restart runs on device. The scipy path this replaces
        paid one host-device round trip per objective evaluation. The train
        problem is unbounded and tiny (dim + 2 parameters), exactly the
        regime :func:`jaxbo.optimizers.minimize_bfgs_jax` covers; its
        dtype-aware gtol and ftol defaults mirror scipy L-BFGS-B's
        termination model in both float32 and float64, and its line search
        only accepts finite decreasing steps, so the returned value of
        every restart is the true likelihood at its returned parameters.

        Args:
            inits: Stacked initializations, shape (num_restarts, dim + 2).
            batch: Normalized training data with 'X' and 'y'.

        Returns:
            Tuple (params, values): per-restart optimized hyperparameters of
            shape (num_restarts, dim + 2) and final NLML values of shape
            (num_restarts,). A failed restart reports a non-finite value
            and is skipped by the caller's selection.

        Note:
            The objective is domain-guarded: hyperparameters whose kernel
            matrix lacks numerical positive-definiteness margin evaluate
            to NaN, so the line search brackets away from them and a
            restart headed that way stops at the boundary with its last
            usable iterate. The guard requires the guaranteed eigenvalue
            floor of K (sigma_n plus the 1e-8 jitter) to exceed the
            rounding noise of the kernel entries themselves, about
            eps * sqrt(N) * K[0, 0] (Weyl bound). Past that point the
            computed K is PSD in some compiled programs and indefinite in
            others: a restart converging there (the zero-noise limit that
            noiseless data invites) would report a winning NLML while the
            same parameters evaluate to NaN in ``predict``. The safety
            factor is dtype-calibrated against measurements: in float32 a
            real PSD flip was observed at 0.17x the bound (factor 1.0
            keeps 6x margin); in float64 the reference scipy path's
            historical winners sit as low as 0.12x the bound and years of
            green predict tests prove them safe, so 0.05 admits every
            endpoint the reference path produced while still excluding
            the far zero-noise ridge.
        """
        eps = float(np.finfo(inits.dtype).eps)
        factor = 1.0 if eps > 1e-10 else 0.05
        margin = factor * eps * float(batch["X"].shape[0]) ** 0.5

        def value_and_grad(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            value, grads = self.likelihood_value_and_grad(p, batch)
            amplitude = np.exp(p[0])
            noise_floor = np.exp(p[-1]) + 1e-8
            ok = noise_floor > margin * (amplitude + noise_floor)
            bad = np.asarray(np.nan, dtype=inits.dtype)
            # likelihood values carry shape (1, 1); the solver wants a scalar.
            return np.where(ok, np.sum(value), bad), np.where(ok, grads, bad)

        def run(x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return minimize_bfgs_jax(value_and_grad, x0)

        return vmap(run)(inits)

    @partial(jit, static_argnums=(0,))
    def predict(
        self, X_star: np.ndarray, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and standard deviation for new input points.

        Args:
            X_star: New input points in the RAW (unnormalized) domain,
                shape (N, D). See the warning below.
            kwargs: Must include 'params', 'batch', 'bounds', and 'norm_const'.

        Returns:
            Tuple (mean, std): Predictive posterior mean and standard deviation.

        Warning:
            ``X_star`` is RAW domain input: it is normalized internally
            against ``bounds`` before evaluation. The ``batch`` passed through
            ``kwargs`` must be the same ALREADY NORMALIZED batch used by
            :meth:`train` (unit-cube inputs). Passing pre-normalized
            ``X_star``, or a raw ``batch``, fails silently: no error is
            raised, the predictions are just wrong.
        """
        params, batch, bounds = kwargs["params"], kwargs["batch"], kwargs["bounds"]
        X_star = (X_star - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
        X, y = batch["X"], batch["y"]
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])

        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0]) * (
            sigma_n + 1e-8
        )
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T, solve_triangular(L, k_pX.T, lower=True))

        mu = k_pX @ alpha
        cov = k_pp - k_pX @ beta
        std = np.sqrt(np.clip(np.diag(cov), 0.0))
        return mu, std

    @partial(jit, static_argnums=(0,))
    def posterior_covariance(
        self, x: np.ndarray, xp: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """
        Compute the posterior covariance between two sets of points.

        Args:
            x, xp: Input arrays of shape (N, D), in the RAW domain (both are
                normalized internally against 'bounds').
            kwargs: Must include 'params', 'batch', and 'bounds'.

        Returns:
            Posterior covariance matrix between x and xp.
        """
        params = kwargs["params"]
        batch = kwargs["batch"]
        bounds = kwargs["bounds"]

        x = (x - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
        xp = (xp - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
        X = batch["X"]
        theta = np.exp(params[:-1])

        k_pp = self.kernel(x, xp, theta)
        k_pX = self.kernel(x, X, theta)
        k_Xp = self.kernel(X, xp, theta)
        L = self.compute_cholesky(params, batch)
        beta = solve_triangular(L.T, solve_triangular(L, k_Xp, lower=True))
        cov = k_pp - np.matmul(k_pX, beta)
        return cov

    @partial(jit, static_argnums=(0,))
    def draw_posterior_sample(self, X_star: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Draw a single sample from the GP posterior at new input locations.

        Args:
            X_star: Input locations in the RAW domain, shape (N, D);
                normalized internally against 'bounds'.
            kwargs: Must include 'params', 'batch', 'bounds', 'rng_key'.

        Returns:
            Sample drawn from the multivariate normal posterior.
        """
        params = kwargs["params"]
        batch = kwargs["batch"]
        bounds = kwargs["bounds"]
        rng_key = kwargs["rng_key"]

        X_star = (X_star - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
        X, y = batch["X"], batch["y"]
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])

        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0]) * (
            sigma_n + 1e-8
        )
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T, solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T, solve_triangular(L, k_pX.T, lower=True))

        mu = k_pX @ alpha
        cov = k_pp - k_pX @ beta
        return random.multivariate_normal(rng_key, mu, cov)
