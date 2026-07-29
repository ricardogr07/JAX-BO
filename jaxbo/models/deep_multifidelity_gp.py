"""Compatibility shim: ``DeepMultifidelityGP`` now lives in :mod:`jaxbo.multifidelity`.

Kept so the historical import path
``from jaxbo.models.deep_multifidelity_gp import DeepMultifidelityGP`` keeps working.
New code should import from :mod:`jaxbo.multifidelity`.
"""

from jaxbo.multifidelity.deep_multifidelity_gp import DeepMultifidelityGP

__all__ = ["DeepMultifidelityGP"]
