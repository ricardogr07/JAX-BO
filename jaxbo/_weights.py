"""Private staging module for the weighted-sampling machinery.

Holds the KDE surface that the ``[weighted]`` optional extra (scikit-learn,
KDEpy) will own after slice 2b turns it into ``jaxbo/weights.py`` (SCOPE.md
decision 7). Nothing in the jaxbo core may import this module eagerly:
importing it pulls KDEpy, so it must only ever load from inside a method
call. ``jaxbo.utils`` forwards the historical import path lazily.
"""

import warnings
from typing import Optional, Union

import jax.numpy as np
import numpy as onp
from KDEpy import FFTKDE
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde


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
