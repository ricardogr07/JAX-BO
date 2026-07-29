import jax.numpy as np
from jax import jit, vmap
from jax.scipy.stats import norm

# Caution: all functions below are designed for single point evaluation; use
# score_candidates (or vmap directly) to score a batch of candidates.
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
    deltap = np.clip(delta, 0.0)
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
    deltap = np.clip(delta, 0.0)
    Z = delta / std[0, :]
    EI = deltap - np.abs(deltap) * norm.cdf(-Z) + std[0, :] * norm.pdf(Z)
    constraints = np.prod(norm.cdf(mean[1:, :] / std[1:, :]), axis=0)
    return -EI[0] * constraints[0]


@jit
def LCBC(
    mean: np.ndarray, std: np.ndarray, kappa: float = 2.0, threshold: float = 3.0
) -> float:
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
def LW_LCBC(
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    kappa: float = 2.0,
    threshold: float = 3.0,
) -> float:
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
def LW_LCB(
    mean: np.ndarray, std: np.ndarray, weights: np.ndarray, kappa: float = 2.0
) -> float:
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
def LW_CLSF(
    mean: np.ndarray, std: np.ndarray, weights: np.ndarray, kappa: float = 1.0
) -> float:
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
    acq = np.log(np.abs(mean) + 1e-8) - kappa * (
        np.log(std + 1e-8) + np.log(weights + 1e-8)
    )
    return acq[0]


def score_candidates(model, X_cand, *, params, batch, bounds, acq_fn=EI, **acq_kwargs):
    """
    Scores a batch of candidate points in one vmapped pass.

    Replaces the per-candidate Python loop over ``model.predict`` plus a
    single-point acquisition (the shelter-pulse pattern: reshape each
    candidate to (1, D), predict, score, ``float()``), which pays two jit
    dispatches and a device to host sync per candidate. Here the whole
    candidate set is scored in a single batched call; the Cholesky factor
    of the training covariance does not depend on the candidate, so vmap
    computes it once, not N times.

    Shape conventions:
    - X_cand has shape (N, D) and lives in the RAW domain. Like
      ``model.predict``, each candidate is normalized internally against
      ``bounds``; do NOT pre-normalize it. ``batch``, by contrast, must be
      the ALREADY NORMALIZED training batch, exactly as passed to
      ``model.train`` (mixing these up fails silently, see jaxbo.gp.GP).
    - Each candidate is scored as a single (1, D) point, so ``acq_fn``
      sees the single-point shapes used throughout this module: ``mean``
      of shape (1, 1), ``std`` of shape (1,), the ``[0]`` indexing
      convention.
    - Returns an (N,) array of scores, one per candidate, in candidate
      order. Lower is better for every acquisition in this module (EI is
      returned negated), so the next point is ``X_cand[np.argmin(scores)]``.

    Parameters:
    model: Trained GP model exposing ``predict(X_star, params=, batch=, bounds=)``,
        e.g. ``jaxbo.gp.GP``.
    X_cand (np.ndarray): Candidate points, shape (N, D), raw domain.
    params (np.ndarray): Trained hyperparameters, as returned by ``model.train``.
    batch (dict): The already normalized training batch ({'X', 'y'}) used to train.
    bounds (dict): {'lb', 'ub'} domain bounds predict normalizes against.
    acq_fn (callable): Single-point acquisition taking (mean, std, **acq_kwargs).
        Defaults to EI.
    **acq_kwargs: Extra arguments forwarded to ``acq_fn``, e.g.
        ``best=float(np.min(batch['y']))`` for EI (best observed target in
        the same normalized space as ``batch['y']``), or ``kappa`` for LCB.

    Returns:
    np.ndarray: Acquisition scores, shape (N,).
    """
    X_cand = np.asarray(X_cand)
    if X_cand.ndim != 2:
        raise ValueError(
            f"X_cand must have shape (N, D); got shape {X_cand.shape}"
        )

    def score_one(x):
        mean, std = model.predict(
            x[None, :], params=params, batch=batch, bounds=bounds
        )
        return acq_fn(mean, std, **acq_kwargs)

    return np.reshape(vmap(score_one)(X_cand), (-1,))
