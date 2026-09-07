"""Core normalization and standardization helpers for jaxbo models.

This module is part of the jaxbo core and depends only
on jax and numpy. The weighted-sampling machinery (``fit_kernel_density``,
``compute_w_gmm``) moved to :mod:`jaxbo.weights` (the ``[weighted]`` extra)
and the stax neural network initializers (``init_NN``, ``init_ResNet``,
``init_MomentumResNet``) to :mod:`jaxbo.multifidelity.nn` (the
``[multifidelity]`` extra); their historical import paths on this module
keep working through the lazy ``__getattr__`` below, without the core import
graph ever reaching KDEpy, scikit-learn, or ``jax.example_libraries.stax``.
"""

from typing import Dict, List, Tuple

import jax.numpy as np
from jax import jit

# Names that moved out of the core with the extras split (the
# decision 7): attribute name to its new home. Accessing a [weighted] name
# without scikit-learn and KDEpy installed raises the jaxbo.weights
# ImportError naming pip install jaxbo[weighted].
_MOVED = {
    "fit_kernel_density": "jaxbo.weights",
    "compute_w_gmm": "jaxbo.weights",
    "init_NN": "jaxbo.multifidelity.nn",
    "init_ResNet": "jaxbo.multifidelity.nn",
    "init_MomentumResNet": "jaxbo.multifidelity.nn",
}


def __getattr__(name: str):
    """Forward moved attributes to their staging modules lazily (PEP 562)."""
    target = _MOVED.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    obj = getattr(importlib.import_module(target), name)
    globals()[name] = obj  # cache so __getattr__ runs once per name
    return obj


def __dir__() -> List[str]:
    """Advertise the core helpers plus the lazily forwarded names."""
    return sorted(set(globals()) | set(_MOVED))


@jit
def normalize(
    X: np.ndarray, y: np.ndarray, bounds: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Normalizes input features X and target values y using provided bounds and statistics.

    Args:
        X (jax.numpy.ndarray): Input features to be normalized. Shape: (n_samples, n_features).
        y (jax.numpy.ndarray): Target values to be normalized. Shape: (n_samples,) or (n_samples, n_targets).
        bounds (dict): Dictionary containing 'lb' (lower bounds) and 'ub' (upper bounds) for each feature in X.
            - 'lb' (jax.numpy.ndarray): Lower bounds for X. Shape: (n_features,).
            - 'ub' (jax.numpy.ndarray): Upper bounds for X. Shape: (n_features,).

    Returns:
        tuple:
            - batch (dict): Dictionary containing normalized 'X' and 'y'.
            - norm_const (dict): Dictionary containing normalization constants:
                - 'mu_y': Mean of y before normalization.
                - 'sigma_y': Standard deviation of y before normalization.

    Notes:
        - X is normalized to the [0, 1] range using the provided bounds.
        - y is normalized to have zero mean and unit variance.
        - The returned batch is what :meth:`jaxbo.gp.GP.train` expects; see the
          normalization contract in :class:`jaxbo.gp.GP`.
    """
    mu_y, sigma_y = y.mean(0), y.std(0)
    X = (X - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
    y = (y - mu_y) / sigma_y
    batch = {"X": X, "y": y}
    norm_const = {"mu_y": mu_y, "sigma_y": sigma_y}
    return batch, norm_const


@jit
def normalize_MultifidelityGP(
    XL: np.ndarray,
    yL: np.ndarray,
    XH: np.ndarray,
    yH: np.ndarray,
    bounds: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Normalizes input and output data for a multi-fidelity Gaussian Process (GP) model.

    This function takes in low-fidelity and high-fidelity input/output pairs, along with their bounds,
    and normalizes both the input features and output targets. The inputs are scaled to the [0, 1] range
    based on the provided bounds, and the outputs are standardized to have zero mean and unit variance
    using statistics computed from the concatenated outputs.

    Args:
        XL (np.ndarray): Low-fidelity input data of shape (n_L, d).
        yL (np.ndarray): Low-fidelity output data of shape (n_L,).
        XH (np.ndarray): High-fidelity input data of shape (n_H, d).
        yH (np.ndarray): High-fidelity output data of shape (n_H,).
        bounds (dict): Dictionary with keys 'lb' and 'ub' representing lower and upper bounds
            for input normalization. Each should be an array of shape (d,).

    Returns:
        batch (dict): Dictionary containing normalized data:
            - 'XL': Normalized low-fidelity inputs.
            - 'XH': Normalized high-fidelity inputs.
            - 'y':  Normalized concatenated outputs.
            - 'yL': Normalized low-fidelity outputs.
            - 'yH': Normalized high-fidelity outputs.
        norm_const (dict): Dictionary containing normalization constants:
            - 'mu_y': Mean of concatenated outputs before normalization.
            - 'sigma_y': Standard deviation of concatenated outputs before normalization.
    """
    y = np.concatenate([yL, yH], axis=0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    XL = (XL - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
    XH = (XH - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
    yL = (yL - mu_y) / sigma_y
    yH = (yH - mu_y) / sigma_y
    y = (y - mu_y) / sigma_y
    batch = {"XL": XL, "XH": XH, "y": y, "yL": yL, "yH": yH}
    norm_const = {"mu_y": mu_y, "sigma_y": sigma_y}
    return batch, norm_const


@jit
def normalize_GradientGP(
    XF: np.ndarray, yF: np.ndarray, XG: np.ndarray, yG: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Normalizes the inputs and outputs for a Gradient Gaussian Process (GP) model.

    Args:
        XF (array-like): Feature matrix for function observations.
        yF (array-like): Output vector for function observations.
        XG (array-like): Feature matrix for gradient observations.
        yG (array-like): Output vector for gradient observations.

    Returns:
        tuple: A tuple containing:
            - batch (dict): Dictionary with keys 'XF', 'XG', 'yF', 'yG', and 'y' (concatenated outputs).
            - norm_const (dict): Dictionary with normalization constants for inputs and outputs,
            including 'mu_X', 'sigma_X', 'mu_y', and 'sigma_y'.
    """
    y = np.concatenate([yF, yG], axis=0)
    batch = {"XF": XF, "XG": XG, "yF": yF, "yG": yG, "y": y}
    norm_const = {"mu_X": 0.0, "sigma_X": 1.0, "mu_y": 0.0, "sigma_y": 1.0}
    return batch, norm_const


@jit
def normalize_HeterogeneousMultifidelityGP(
    XL: np.ndarray,
    yL: np.ndarray,
    XH: np.ndarray,
    yH: np.ndarray,
    bounds: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Normalizes input and output data for heterogeneous multifidelity Gaussian Process (GP) models.

    This function standardizes the low-fidelity inputs (XL) and outputs (yL), and high-fidelity outputs (yH)
    using the mean and standard deviation of the low-fidelity data. The high-fidelity inputs (XH) are normalized
    to the [0, 1] range using the provided bounds.

    Args:
        XL (np.ndarray): Low-fidelity input data of shape (n_low, d).
        yL (np.ndarray): Low-fidelity output data of shape (n_low, 1) or (n_low,).
        XH (np.ndarray): High-fidelity input data of shape (n_high, d).
        yH (np.ndarray): High-fidelity output data of shape (n_high, 1) or (n_high,).
        bounds (dict): Dictionary with keys 'lb' and 'ub' representing the lower and upper bounds for normalization
                       of high-fidelity inputs. Each should be an array of shape (d,).

    Returns:
        batch (dict): Dictionary containing normalized data with keys:
            - 'XL': Normalized low-fidelity inputs.
            - 'XH': Normalized high-fidelity inputs.
            - 'y':  Concatenated and normalized outputs.
            - 'yL': Normalized low-fidelity outputs.
            - 'yH': Normalized high-fidelity outputs.
        norm_const (dict): Dictionary containing normalization constants with keys:
            - 'mu_X': Mean of low-fidelity inputs.
            - 'sigma_X': Standard deviation of low-fidelity inputs.
            - 'mu_y': Mean of concatenated outputs.
            - 'sigma_y': Standard deviation of concatenated outputs.
    """
    y = np.concatenate([yL, yH], axis=0)
    mu_X, sigma_X = XL.mean(0), XL.std(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    XL = (XL - mu_X) / sigma_X
    XH = (XH - bounds["lb"]) / (bounds["ub"] - bounds["lb"])
    yL = (yL - mu_y) / sigma_y
    yH = (yH - mu_y) / sigma_y
    y = (y - mu_y) / sigma_y
    batch = {"XL": XL, "XH": XH, "y": y, "yL": yL, "yH": yH}
    norm_const = {"mu_X": mu_X, "sigma_X": sigma_X, "mu_y": mu_y, "sigma_y": sigma_y}
    return batch, norm_const


@jit
def standardize(
    X: np.ndarray, y: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Standardizes the input features X and target y to have zero mean and unit variance.

    Parameters
    ----------
    X : np.ndarray
        Input features, where each row is a sample and each column is a feature.
    y : np.ndarray
        Target values, can be a 1D or 2D array.

    Returns
    -------
    batch : dict
        Dictionary containing the standardized 'X' and 'y'.
    norm_const : dict
        Dictionary containing the means ('mu_X', 'mu_y') and standard deviations ('sigma_X', 'sigma_y')
        used for standardization.
    """
    mu_X, sigma_X = X.mean(0), X.std(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    X = (X - mu_X) / sigma_X
    y = (y - mu_y) / sigma_y
    batch = {"X": X, "y": y}
    norm_const = {"mu_X": mu_X, "sigma_X": sigma_X, "mu_y": mu_y, "sigma_y": sigma_y}
    return batch, norm_const


@jit
def standardize_MultifidelityGP(
    XL: np.ndarray, yL: np.ndarray, XH: np.ndarray, yH: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Standardizes input and output data for multi-fidelity Gaussian Process modeling.

    This function concatenates low-fidelity (XL, yL) and high-fidelity (XH, yH) datasets,
    computes the mean and standard deviation for both inputs and outputs, and standardizes
    each dataset accordingly. The standardized datasets and normalization constants are returned.

    Args:
        XL (np.ndarray): Low-fidelity input data of shape (n_low, d).
        yL (np.ndarray): Low-fidelity output data of shape (n_low,).
        XH (np.ndarray): High-fidelity input data of shape (n_high, d).
        yH (np.ndarray): High-fidelity output data of shape (n_high,).

    Returns:
        batch (dict): Dictionary containing standardized datasets:
            - 'XL': Standardized low-fidelity inputs.
            - 'XH': Standardized high-fidelity inputs.
            - 'y': Standardized concatenated outputs.
            - 'yL': Standardized low-fidelity outputs.
            - 'yH': Standardized high-fidelity outputs.
        norm_const (dict): Dictionary containing normalization constants:
            - 'mu_X': Mean of concatenated inputs.
            - 'sigma_X': Standard deviation of concatenated inputs.
            - 'mu_y': Mean of concatenated outputs.
            - 'sigma_y': Standard deviation of concatenated outputs.
    """
    X = np.concatenate([XL, XH], axis=0)
    y = np.concatenate([yL, yH], axis=0)
    mu_X, sigma_X = X.mean(0), X.std(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    XL = (XL - mu_X) / sigma_X
    XH = (XH - mu_X) / sigma_X
    yL = (yL - mu_y) / sigma_y
    yH = (yH - mu_y) / sigma_y
    y = (y - mu_y) / sigma_y
    batch = {"XL": XL, "XH": XH, "y": y, "yL": yL, "yH": yH}
    norm_const = {"mu_X": mu_X, "sigma_X": sigma_X, "mu_y": mu_y, "sigma_y": sigma_y}
    return batch, norm_const


@jit
def standardize_HeterogeneousMultifidelityGP(
    XL: np.ndarray, yL: np.ndarray, XH: np.ndarray, yH: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Standardizes and normalizes input and output data for heterogeneous multifidelity Gaussian Process models.

    This function applies standardization to low-fidelity inputs (XL) and normalization to high-fidelity inputs (XH).
    The outputs (yL, yH) are standardized using the mean and standard deviation computed from the concatenated outputs.
    The function returns the standardized/normalized data and the normalization constants used.

    Args:
        XL (np.ndarray): Low-fidelity input data of shape (n_L, d).
        yL (np.ndarray): Low-fidelity output data of shape (n_L,).
        XH (np.ndarray): High-fidelity input data of shape (n_H, d).
        yH (np.ndarray): High-fidelity output data of shape (n_H,).

    Returns:
        batch (dict): Dictionary containing standardized/normalized data:
            - 'XL': Standardized low-fidelity inputs.
            - 'XH': Normalized high-fidelity inputs.
            - 'y': Standardized concatenated outputs.
            - 'yL': Standardized low-fidelity outputs.
            - 'yH': Standardized high-fidelity outputs.
        norm_const (dict): Dictionary containing normalization constants:
            - 'mu_XL': Mean of XL.
            - 'sigma_XL': Standard deviation of XL.
            - 'min_XH': Minimum of XH.
            - 'max_XH': Maximum of XH.
            - 'mu_y': Mean of concatenated outputs.
            - 'sigma_y': Standard deviation of concatenated outputs.
    """
    y = np.concatenate([yL, yH], axis=0)
    mu_XL, sigma_XL = XL.mean(0), XL.std(0)
    min_XH, max_XH = XH.min(0), XH.max(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    XL = (XL - mu_XL) / sigma_XL
    XH = (XH - min_XH) / (max_XH - min_XH)
    yL = (yL - mu_y) / sigma_y
    yH = (yH - mu_y) / sigma_y
    y = (y - mu_y) / sigma_y
    batch = {"XL": XL, "XH": XH, "y": y, "yL": yL, "yH": yH}
    norm_const = {
        "mu_XL": mu_XL,
        "sigma_XL": sigma_XL,
        "min_XH": min_XH,
        "max_XH": max_XH,
        "mu_y": mu_y,
        "sigma_y": sigma_y,
    }
    return batch, norm_const
