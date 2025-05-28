import jax.numpy as np
from jax import random
from jax.random import PRNGKey
import numpy as onp

def random_init_GP(
    rng_key: PRNGKey,
    dim: int
    ) -> onp.ndarray:
    """
    Initializes random hyperparameters for a Gaussian Process (GP) model.

    Args:
    rng_key (jax.random.PRNGKey): A JAX random number generator key.
    dim (int): The dimensionality of the input space.

    Returns:
    np.ndarray: A 1D array containing the concatenated log signal variance,
            log lengthscales, and log noise variance for the GP.
    """
    logsigma_f = np.log(50.0 * random.uniform(rng_key, (1,)))
    loglength = np.log(random.uniform(rng_key, (dim,)) + 1e-8)
    logsigma_n = np.array([-4.0]) + random.normal(rng_key, (1,))
    hyp = np.concatenate([logsigma_f, loglength, logsigma_n])
    return hyp

def random_init_MultifidelityGP(
    rng_key: PRNGKey,
    dim: int
) -> onp.ndarray:
    """
    Initializes hyperparameters for a multifidelity Gaussian Process (GP) model with random values.

    Parameters
    ----------
    rng_key : jax.random.PRNGKey
        A JAX random number generator key for reproducibility.
    dim : int
        The dimensionality of the input space.

    Returns
    -------
    hyp : numpy.ndarray
        A 1D array containing the concatenated hyperparameters:
            - logsigma_fL: Log signal variance for the low-fidelity GP.
            - loglength_L: Log lengthscales for the low-fidelity GP (one per dimension).
            - logsigma_fH: Log signal variance for the high-fidelity GP.
            - loglength_H: Log lengthscales for the high-fidelity GP (one per dimension).
            - rho: Correlation parameter between fidelities.
            - logsigma_nL: Log noise variance for the low-fidelity GP.
            - logsigma_nH: Log noise variance for the high-fidelity GP.

    Notes
    -----
    The hyperparameters are initialized using random draws to encourage exploration during optimization.
    """
    key1, key2 = random.split(rng_key)
    logsigma_fL = np.log(50.0 * random.uniform(key1, (1,)))
    loglength_L = np.log(random.uniform(key1, (dim,)) + 1e-8)
    logsigma_fH = np.log(50.0 * random.uniform(key2, (1,)))
    loglength_H = np.log(random.uniform(key2, (dim,)) + 1e-8)
    rho = 5.0 * random.normal(rng_key, (1,))
    logsigma_nL = np.array([-4.0]) + random.normal(key1, (1,))
    logsigma_nH = np.array([-4.0]) + random.normal(key2, (1,))
    hyp = np.concatenate([
        logsigma_fL, loglength_L,
        logsigma_fH, loglength_H,
        rho, logsigma_nL, logsigma_nH
    ])
    return hyp

def random_init_GradientGP(
    rng_key: PRNGKey,
    dim: int
) -> onp.ndarray:
    """
    Initializes random hyperparameters for a Gaussian Process (GP) model with gradient observations.

    Args:
        rng_key (jax.random.PRNGKey): A JAX random number generator key.
        dim (int): The dimensionality of the input space.

    Returns:
        np.ndarray: A 1D array containing the concatenated log signal variance,
            log lengthscales, log noise variance for function values,
            and log noise variance for gradients.
    """
    logsigma_f = np.log(50.0 * random.uniform(rng_key, (1,)))
    loglength = np.log(random.uniform(rng_key, (dim,)) + 1e-8)
    logsigma_n_F = np.array([-4.0]) + random.normal(rng_key, (1,))
    logsigma_n_G = np.array([-4.0]) + random.normal(rng_key, (1,))
    hyp = np.concatenate([logsigma_f, loglength, logsigma_n_F, logsigma_n_G])
    return hyp

def random_init_SparseGP(
    rng_key: PRNGKey,
    dim: int
) -> onp.ndarray:
    """
    Initializes random hyperparameters for a Sparse Gaussian Process (GP) model.

    Args:
        rng_key (jax.random.PRNGKey): A JAX random number generator key.
        dim (int): The dimensionality of the input space.

    Returns:
        np.ndarray: A 1D array containing the concatenated log signal variance,
            log lengthscales, and log noise variance for the Sparse GP.
    """
    logsigma_f = np.log(50.0 * random.uniform(rng_key, (1,)))
    loglength = np.log(random.uniform(rng_key, (dim,)) + 1e-8)
    logsigma_n = np.array([-4.0]) + random.normal(rng_key, (1,))
    hyp = np.concatenate([logsigma_f, loglength, logsigma_n])
    return hyp
