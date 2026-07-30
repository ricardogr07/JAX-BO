"""MCMC-based Gaussian process models: the ``[mcmc]`` optional extra.

Everything here runs full Bayesian inference over the GP hyperparameters
with numpyro's NUTS instead of the core's point-estimate L-BFGS training
(SCOPE.md section 3). The jaxbo core never imports this package eagerly;
``jaxbo.mcmc_models`` remains as a compatibility shim for the historical
import path.
"""

try:
    import numpyro  # noqa: F401  presence check for the [mcmc] extra
except ImportError as err:  # pragma: no cover - exercised in a subprocess test
    raise ImportError(
        "jaxbo.mcmc needs numpyro, which is part of the [mcmc] extra. "
        "Install it with: pip install jaxbo[mcmc]"
    ) from err

from jaxbo.mcmc.models import (
    BayesianMLP,
    GPclassifier,
    MCMCGP,
    MCMCmodel,
    MissingInputsGP,
    MultifidelityGPclassifier,
)

__all__ = [
    "BayesianMLP",
    "GPclassifier",
    "MCMCGP",
    "MCMCmodel",
    "MissingInputsGP",
    "MultifidelityGPclassifier",
]
