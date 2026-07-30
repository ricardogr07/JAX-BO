"""Compatibility shim: the MCMC models moved to :mod:`jaxbo.mcmc`.

Importing this path requires the [mcmc] extra (numpyro); without it the
package raises an ImportError naming ``pip install jaxbo[mcmc]``. New code
should import from :mod:`jaxbo.mcmc` directly.
"""

from jaxbo.mcmc import (
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
