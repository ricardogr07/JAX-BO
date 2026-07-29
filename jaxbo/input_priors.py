"""Compatibility shim: the input priors now live in :mod:`jaxbo.priors`.

Kept so the historical import path ``from jaxbo.input_priors import
uniform_prior`` keeps working (it is part of the 0.2.0 compatibility
promise, SCOPE.md section 2). New code should import from
:mod:`jaxbo.priors`.
"""

from jaxbo.priors import Prior, gaussian_prior, uniform_prior

__all__ = ["Prior", "uniform_prior", "gaussian_prior"]
