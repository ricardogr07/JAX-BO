import jax.numpy as np
from jax import jit
from jax.scipy.stats import norm

# Caution: all functions are designed for single point evaluation (use vmap to vectorize)
# See derivation in:
# https://people.orie.cornell.edu/pfrazier/Presentations/2011.11.INFORMS.Tutorial.pdf

@jit
def EI(mean: np.ndarray, std: np.ndarray, best: float) -> float:
    """
    Computes the Expected Improvement (EI) acquisition function.

    Parameters:
    mean (np.ndarray): Predictive mean of the objective function at the point of interest.
    std (np.ndarray): Predictive standard deviation.
    best (float): Best observed value so far.

    Returns:
    float: Negative expected improvement (for minimization).
    """

    delta = -(mean - best)
    deltap = np.clip(delta, a_min=0.)
    Z = delta / std
    EI = deltap - np.abs(deltap) * norm.cdf(-Z) + std * norm.pdf(Z)
    return -EI[0]

@jit
def EIC(mean: np.ndarray, std: np.ndarray, best: float) -> float:
    """
    Computes the Constrained Expected Improvement (EIC) acquisition function.

    Parameters:
    mean (np.ndarray): Predictive means (first row is objective, remaining are constraints).
    std (np.ndarray): Predictive standard deviations.
    best (float): Best observed objective value.

    Returns:
    float: Negative constrained expected improvement.
    """
    delta = -(mean[0, :] - best)
    deltap = np.clip(delta, a_min=0.)
    Z = delta / std[0, :]
    EI = deltap - np.abs(deltap) * norm.cdf(-Z) + std[0, :] * norm.pdf(Z)
    constraints = np.prod(norm.cdf(mean[1:, :] / std[1:, :]), axis=0)
    return -EI[0] * constraints[0]

@jit
def LCBC(mean: np.ndarray, std: np.ndarray, kappa: float = 2.0, threshold: float = 3.0) -> float:
    """
    Lower Confidence Bound with Constraints.

    Parameters:
    mean (np.ndarray): Predictive means (first row is objective).
    std (np.ndarray): Predictive standard deviations.
    kappa (float): Confidence interval parameter.
    threshold (float): Threshold value for constraint.

    Returns:
    float: Constrained LCB acquisition value.
    """
    lcb = mean[0, :] - threshold - kappa * std[0, :]
    constraints = np.prod(norm.cdf(mean[1:, :] / std[1:, :]), axis=0)
    return lcb[0] * constraints[0]

@jit
def LW_LCBC(mean: np.ndarray, std: np.ndarray, weights: np.ndarray, kappa: float = 2.0, threshold: float = 3.0) -> float:
    """
    Log-Weighted Lower Confidence Bound with Constraints.

    Parameters:
    mean (np.ndarray): Predictive means.
    std (np.ndarray): Predictive standard deviations.
    weights (np.ndarray): Log-weighted factors.
    kappa (float): Confidence interval parameter.
    threshold (float): Constraint threshold.

    Returns:
    float: Weighted constrained LCB acquisition value.
    """
    lcb = mean[0, :] - threshold - kappa * std[0, :] * weights
    constraints = np.prod(norm.cdf(mean[1:, :] / std[1:, :]), axis=0)
    return lcb[0] * constraints[0]

@jit
def LCB(mean: np.ndarray, std: np.ndarray, kappa: float = 2.0) -> float:
    """
    Lower Confidence Bound (LCB) acquisition function.

    Parameters:
    mean (np.ndarray): Predictive mean.
    std (np.ndarray): Predictive standard deviation.
    kappa (float): Confidence parameter.

    Returns:
    float: LCB value.
    """
    lcb = mean - kappa * std
    return lcb[0]

@jit
def US(std: np.ndarray) -> float:
    """
    Uncertainty Sampling acquisition function.

    Parameters:
    std (np.ndarray): Predictive standard deviation.

    Returns:
    float: Negative uncertainty value.
    """
    return -std[0]

@jit
def LW_LCB(mean: np.ndarray, std: np.ndarray, weights: np.ndarray, kappa: float = 2.0) -> float:
    """
    Log-Weighted Lower Confidence Bound.

    Parameters:
    mean (np.ndarray): Predictive mean.
    std (np.ndarray): Predictive standard deviation.
    weights (np.ndarray): Importance weights.
    kappa (float): Confidence parameter.

    Returns:
    float: LW-LCB value.
    """
    lw_lcb = mean - kappa * std * weights
    return lw_lcb[0]

@jit
def LW_US(std: np.ndarray, weights: np.ndarray) -> float:
    """
    Log-Weighted Uncertainty Sampling.

    Parameters:
    std (np.ndarray): Predictive standard deviation.
    weights (np.ndarray): Importance weights.

    Returns:
    float: Weighted negative uncertainty.
    """
    lw_us = std * weights
    return -lw_us[0]

@jit
def CLSF(mean: np.ndarray, std: np.ndarray, kappa: float = 1.0) -> float:
    """
    Classification Surrogate Function acquisition.

    Parameters:
    mean (np.ndarray): Predictive mean.
    std (np.ndarray): Predictive standard deviation.
    kappa (float): Regularization coefficient.

    Returns:
    float: CLSF value.
    """
    acq = np.log(np.abs(mean) + 1e-8) - kappa * np.log(std + 1e-8)
    return acq[0]

@jit
def LW_CLSF(mean: np.ndarray, std: np.ndarray, weights: np.ndarray, kappa: float = 1.0) -> float:
    """
    Log-Weighted Classification Surrogate Function acquisition.

    Parameters:
    mean (np.ndarray): Predictive mean.
    std (np.ndarray): Predictive standard deviation.
    weights (np.ndarray): Importance weights.
    kappa (float): Regularization coefficient.

    Returns:
    float: Weighted CLSF value.
    """
    acq = np.log(np.abs(mean) + 1e-8) - kappa * (np.log(std + 1e-8) + np.log(weights + 1e-8))
    return acq[0]
