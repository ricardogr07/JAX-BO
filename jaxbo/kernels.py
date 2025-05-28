import jax.numpy as np
from jax import jit

def _pairwise_diff_squared(x1: np.ndarray, x2: np.ndarray, lengthscales: np.ndarray) -> np.ndarray:
    """
    Computes squared differences between all pairs of inputs scaled by the lengthscales.

    Args:
        x1 (np.ndarray): First input array of shape (N, D).
        x2 (np.ndarray): Second input array of shape (M, D).
        lengthscales (np.ndarray): Array of lengthscales per dimension (D,).

    Returns:
        np.ndarray: Pairwise squared distances of shape (N, M).
    """
    diffs = np.expand_dims(x1 / lengthscales, 1) - np.expand_dims(x2 / lengthscales, 0)
    return np.sum(diffs**2, axis=2)

@jit
def RBF(x1: np.ndarray, x2: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Radial Basis Function (Gaussian) kernel.

    Args:
        x1 (np.ndarray): Input array of shape (N, D).
        x2 (np.ndarray): Input array of shape (M, D).
        params (np.ndarray): Parameters where params[0] is output scale, params[1:] are lengthscales.

    Returns:
        np.ndarray: Kernel matrix of shape (N, M).
    """
    output_scale = params[0]
    lengthscales = params[1:]
    r2 = _pairwise_diff_squared(x1, x2, lengthscales)
    return output_scale * np.exp(-0.5 * r2)

@jit
def Matern52(x1: np.ndarray, x2: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Matern 5/2 kernel.

    Args:
        x1 (np.ndarray): Input array of shape (N, D).
        x2 (np.ndarray): Input array of shape (M, D).
        params (np.ndarray): Parameters where params[0] is output scale, params[1:] are lengthscales.

    Returns:
        np.ndarray: Kernel matrix of shape (N, M).
    """
    output_scale = params[0]
    lengthscales = params[1:]
    r2 = _pairwise_diff_squared(x1, x2, lengthscales)
    return output_scale * (1.0 + np.sqrt(5.0*r2 + 1e-12) + 5.0*r2/3.0) * np.exp(-np.sqrt(5.0*r2 + 1e-12))

@jit
def Matern32(x1: np.ndarray, x2: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Matern 3/2 kernel.

    Args:
        x1 (np.ndarray): Input array of shape (N, D).
        x2 (np.ndarray): Input array of shape (M, D).
        params (np.ndarray): Parameters where params[0] is output scale, params[1:] are lengthscales.

    Returns:
        np.ndarray: Kernel matrix of shape (N, M).
    """
    output_scale = params[0]
    lengthscales = params[1:]
    r2 = _pairwise_diff_squared(x1, x2, lengthscales)
    return output_scale * (1.0 + np.sqrt(3.0*r2 + 1e-12)) * np.exp(-np.sqrt(3.0*r2 + 1e-12))

@jit
def Matern12(x1: np.ndarray, x2: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Matern 1/2 kernel (equivalent to exponential kernel).

    Args:
        x1 (np.ndarray): Input array of shape (N, D).
        x2 (np.ndarray): Input array of shape (M, D).
        params (np.ndarray): Parameters where params[0] is output scale, params[1:] are lengthscales.

    Returns:
        np.ndarray: Kernel matrix of shape (N, M).
    """
    output_scale = params[0]
    lengthscales = params[1:]
    r2 = _pairwise_diff_squared(x1, x2, lengthscales)
    return output_scale * np.exp(-np.sqrt(r2 + 1e-12))

@jit
def RatQuad(x1: np.ndarray, x2: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Rational Quadratic kernel.

    Args:
        x1 (np.ndarray): Input array of shape (N, D).
        x2 (np.ndarray): Input array of shape (M, D).
        params (np.ndarray): Parameters where params[0] is output scale, params[1:] are lengthscales.

    Returns:
        np.ndarray: Kernel matrix of shape (N, M).
    """
    alpha = 1.0  # Shape parameter
    output_scale = params[0]
    lengthscales = params[1:]
    r2 = _pairwise_diff_squared(x1, x2, lengthscales)
    return output_scale * np.power(1.0 + (0.5 / alpha) * r2, -alpha)
