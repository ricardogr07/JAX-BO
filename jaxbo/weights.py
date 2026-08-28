"""Weighted-sampling machinery: the ``[weighted]`` optional extra.

Owns the whole GMM/KDE surface extracted from the model classes (the
decision 7): :func:`fit_kernel_density`, :func:`compute_w_gmm`, and the two
importance-reweighted GMM fitters :func:`fit_gmm` (single objective) and
:func:`fit_gmm_constrained` (objective plus constraints). The ``LW_*``
acquisition functions themselves stay in core :mod:`jaxbo.acquisitions` and
take precomputed weights as plain arguments; everything that produces or
consumes ``gmm_vars`` lives here.

Nothing in the jaxbo core imports this module eagerly. Model methods such as
:meth:`jaxbo.gp.GPmodel.fit_gmm` and the ``LW_*`` acquisition branches load
it lazily, so the guard below is the single place a missing dependency
surfaces.
"""

import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

try:
    from KDEpy import FFTKDE
    from sklearn import mixture
except ImportError as err:  # pragma: no cover - exercised in a subprocess test
    raise ImportError(
        "jaxbo.weights needs scikit-learn and KDEpy, which are part of the "
        "[weighted] extra. Install them with: pip install jaxbo[weighted]"
    ) from err

import jax.numpy as np
import numpy as onp
from jax import jit, vmap
from jax.random import split
from jax.scipy.stats import multivariate_normal, norm
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, qmc

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime import
    from jaxbo.gp import GPmodel

__all__ = [
    "compute_w_gmm",
    "fit_gmm",
    "fit_gmm_constrained",
    "fit_kernel_density",
]


def fit_kernel_density(
    X: onp.ndarray,
    xi: onp.ndarray,
    weights: Optional[onp.ndarray] = None,
    bw: Optional[Union[float, onp.ndarray]] = None,
) -> np.ndarray:
    """Fit a kernel density estimator and evaluate its PDF at ``xi``.

    Parameters
    ----------
    X : array-like
        Input data used to fit the kernel density estimator. Should be a 1D
        array.
    xi : array-like
        Points at which to evaluate the estimated PDF.
    weights : array-like, optional
        Weights for each data point in ``X``. If ``None``, equal weighting is
        assumed.
    bw : float, optional
        Bandwidth for the KDE. If ``None``, the bandwidth is estimated
        automatically.

    Returns
    -------
    pdf : ndarray
        The estimated PDF values at the points specified by ``xi``. Values are
        clipped to be non-negative and a small epsilon (``1e-8``) is added for
        numerical stability.

    Notes
    -----
    - Uses :class:`FFTKDE` for fast kernel density estimation.
    - If bandwidth estimation fails or results in a very small value, a default
      of ``1.0`` is used.
    - The output PDF is interpolated using a linear interpolation and
      extrapolated as needed.
    """

    X, weights = onp.array(X), onp.array(weights)
    X = X.flatten()
    if bw is None:
        try:
            sc = gaussian_kde(X, weights=weights)
            bw = onp.sqrt(sc.covariance).flatten()[0]
        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(
                f"KDE bandwidth estimation failed: {e}. Falling back to bw=1.0."
            )
            bw = 1.0
        if bw < 1e-8:
            warnings.warn(
                f"Estimated bandwidth {bw:.2e} is too small. Using bw=1.0 instead."
            )
            bw = 1.0

    kde_pdf_x, kde_pdf_y = FFTKDE(bw=bw).fit(X, weights).evaluate()

    # Define the interpolation function
    interp1d_fun = interp1d(
        kde_pdf_x, kde_pdf_y, kind="linear", fill_value="extrapolate"
    )

    # Evaluate the weights on the input data
    pdf = interp1d_fun(xi)
    return np.clip(pdf, 0.0) + 1e-8


@jit
def compute_w_gmm(x: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Evaluate a Gaussian mixture density at ``x`` in the unit cube.

    This is the ``gmm_vars`` consumer behind the ``LW_*`` acquisition
    criteria: it normalizes ``x`` against the domain bounds and evaluates the
    weighted sum of multivariate normal PDFs parameterized by ``gmm_vars``
    (typically the output of :func:`fit_gmm`).

    Args:
        x (np.ndarray): The input point(s) at which to evaluate the GMM,
            shape (..., D), in the RAW domain.
        **kwargs: Additional keyword arguments, including:
            - 'bounds' (dict): Dictionary with keys 'lb' and 'ub' for lower
              and upper bounds (arrays).
            - 'gmm_vars' (tuple): Tuple containing (weights, means,
              covariances) of the GMM:
                - weights (np.ndarray): Component weights, shape (K,).
                - means (np.ndarray): Component means, shape (K, D).
                - covs (np.ndarray): Covariance matrices, shape (K, D, D).

    Returns:
        float or np.ndarray: The weighted sum of GMM component PDFs at ``x``.
    """
    bounds = kwargs["bounds"]
    lb = bounds["lb"]
    ub = bounds["ub"]
    x = (x - lb) / (ub - lb)
    weights, means, covs = kwargs["gmm_vars"]

    def gmm_mode(w, mu, cov):
        return w * multivariate_normal.pdf(x, mu, cov)

    w = np.sum(vmap(gmm_mode)(weights, means, covs), axis=0)
    return w


def fit_gmm(
    model: "GPmodel", num_comp: int = 2, N_samples: int = 10000, **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a GMM that reweights the input prior by the model's predictions.

    Extracted from ``GPmodel.fit_gmm``; the method now
    delegates here. Used to enable prior-informed acquisition functions such
    as LW-LCB or LW-US:

    1. Sample uniformly over the input space (LHS).
    2. Evaluate model predictions.
    3. Estimate a kernel density for outputs.
    4. Reweight by prior / posterior to prioritize informative regions.
    5. Resample the inputs using the resulting importance weights.
    6. Fit a GMM to this resampled set.

    Args:
        model: Any trained jaxbo GP model exposing ``predict`` and
            ``input_prior``.
        num_comp (int): Number of Gaussian components in the GMM.
        N_samples (int): Number of samples used for reweighting and training
            the GMM.
        **kwargs:
            - bounds (dict): Keys 'lb' and 'ub' bounding the input domain.
            - rng_key (jax.random.PRNGKey): Random key for reproducibility.
            - All other kwargs are forwarded to ``model.predict``.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: GMM component weights,
        means, and covariance matrices, in the ``gmm_vars`` layout
        :func:`compute_w_gmm` expects.

    Notes:
        - Assumes ``model.predict`` returns the predictive mean as its first
          element.
        - Each LHS draw is seeded from its own split subkey so the two designs
          never coincide (PR #44 pattern).
    """
    bounds = kwargs["bounds"]
    lb = bounds["lb"]
    ub = bounds["ub"]
    rng_key = kwargs["rng_key"]
    dim = lb.shape[0]

    # Sample data uniformly over the full input space
    onp.random.seed(rng_key[0])
    sampler = qmc.LatinHypercube(d=dim, seed=int(rng_key[0]))
    X = lb + (ub - lb) * sampler.random(N_samples)
    y = model.predict(X, **kwargs)[0]

    # Sample inputs according to the prior distribution
    rng_key = split(rng_key)[0]
    onp.random.seed(rng_key[0])
    sampler = qmc.LatinHypercube(d=dim, seed=int(rng_key[0]))
    X_samples = lb + (ub - lb) * sampler.random(N_samples)
    y_samples = model.predict(X_samples, **kwargs)[0]

    # Estimate output densities from both prior and uniform samples
    p_x = model.input_prior.pdf(X)
    p_x_samples = model.input_prior.pdf(X_samples)
    p_y = fit_kernel_density(y_samples, y, weights=p_x_samples)

    # Importance weighting based on p(x)/p(y), normalized in float64 because
    # onp.random.choice rejects the residual error of a float32 normalization
    weights = onp.asarray(p_x / p_y, dtype=onp.float64).flatten()
    weights /= weights.sum()

    # Resample data points using computed weights
    indices = np.arange(N_samples)
    resample_idx = onp.random.choice(indices, N_samples, p=weights)
    X_train = (X[resample_idx] - lb) / (ub - lb)  # Scale to [0, 1]^D

    # Fit GMM to resampled inputs
    clf = mixture.GaussianMixture(n_components=num_comp, covariance_type="full")
    clf.fit(X_train)

    return clf.weights_, clf.means_, clf.covariances_


def fit_gmm_constrained(
    model: "GPmodel", num_comp: int = 2, N_samples: int = 10000, **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constrained variant of :func:`fit_gmm` for multi-output models.

    Extracted from the byte-identical ``fit_gmm`` methods of
    ``MultipleIndependentMFGP`` and ``MultipleIndependentHeterogeneousMFGP``
   ; both now delegate here. Row 0 of
    ``model.predict_all`` is treated as the objective and every following row
    as a constraint; the importance weights are multiplied by the probability
    of constraint satisfaction ``Phi(mu_c / std_c)``.

    Args:
        model: A multi-output jaxbo GP model exposing ``predict_all`` and
            ``input_prior``.
        num_comp (int): Number of Gaussian components in the GMM.
        N_samples (int): Number of samples used for reweighting and training
            the GMM.
        **kwargs:
            - bounds (dict): Keys 'lb' and 'ub' bounding the input domain.
            - rng_key (jax.random.PRNGKey): Random key for reproducibility.
            - All other kwargs are forwarded to ``model.predict_all``.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: GMM component weights,
        means, and covariance matrices, in the ``gmm_vars`` layout
        :func:`compute_w_gmm` expects.
    """
    bounds = kwargs["bounds"]
    lb = bounds["lb"]
    ub = bounds["ub"]

    rng_key = kwargs["rng_key"]
    dim = lb.shape[0]
    # Sample data across the entire domain, seeded from the caller's rng_key
    onp.random.seed(rng_key[0])
    sampler = qmc.LatinHypercube(d=dim, seed=int(rng_key[0]))
    X = lb + (ub - lb) * sampler.random(N_samples)

    # Row 0 of predict_all is the objective; the rest are constraints
    mu, std = model.predict_all(X, **kwargs)
    y = mu[0, :]
    mu_c, std_c = mu[1:, :], std[1:, :]

    constraint_w = np.ones((std_c.shape[1], 1)).flatten()
    for k in range(std_c.shape[0]):
        constraint_w_temp = norm.cdf(mu_c[k, :] / std_c[k, :])
        if np.sum(constraint_w_temp) > 1e-8:
            constraint_w = constraint_w * constraint_w_temp

    # Second LHS design from a split subkey so the two draws never coincide
    rng_key = split(rng_key)[0]
    onp.random.seed(rng_key[0])
    sampler = qmc.LatinHypercube(d=dim, seed=int(rng_key[0]))
    X_samples = lb + (ub - lb) * sampler.random(N_samples)
    y_samples = model.predict_all(X_samples, **kwargs)[0][0, :]

    # Compute p_x and p_y from samples across the entire domain
    p_x = model.input_prior.pdf(X)
    p_x_samples = model.input_prior.pdf(X_samples)
    p_y = fit_kernel_density(y_samples, y, weights=p_x_samples)

    # Constraint-weighted importance weights, normalized in float64 because
    # onp.random.choice rejects the residual error of a float32 normalization
    weights = onp.asarray(p_x / p_y * constraint_w, dtype=onp.float64).flatten()
    weights /= weights.sum()
    indices = np.arange(N_samples)
    # Scale inputs to [0, 1]^D
    X = (X - lb) / (ub - lb)
    # Resample according to the analytical weights
    idx = onp.random.choice(indices, N_samples, p=weights)
    X_train = X[idx]
    # Fit GMM to the resampled inputs
    clf = mixture.GaussianMixture(n_components=num_comp, covariance_type="full")
    clf.fit(X_train)
    return clf.weights_, clf.means_, clf.covariances_
